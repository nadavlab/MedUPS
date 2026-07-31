#!/usr/bin/env python3
"""
Merge a midstream SFT-CoT LoRA adapter into its base model's weights, so the
result can be used as a plain local-directory "base model" for a subsequent
GRPO run (GRPO_MODEL_OVERRIDE in grpo_training_new_code.py).

Reuses grpo_training_new_code's MODEL/MODEL_FAMILY resolution and per-family
chat-template kwargs so the merged checkpoint's tokenizer/chat_template files
are byte-identical to what GRPO/eval already expect for this family -- only
the base weights change (SFT-adapted instead of the released checkpoint).

Why merge-and-serve rather than continuing the same adapter under GRPO:
grpo_training_new_code.py always constructs GRPOTrainer(model=MODEL,
peft_config=peft_config) from a bare model string/path, with no existing path
to seed that fresh LoraConfig from another adapter's weights. Merging the SFT
adapter into the base once, then pointing MODEL at the merged directory, lets
GRPO's own untouched fresh-LoRA-on-a-base-model code path run exactly as it
does for every other family -- no GRPOTrainer construction changes needed.

Example
-------
GRPO_MODEL_FAMILY=qwen python merge_sft_adapter.py \
    --adapter ./checkpoints/medical-sft-qwen35-9b-midstream_task/checkpoint-600 \
    --output-dir ./models/qwen35-9b-sftcot-merged-ckpt600
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=str, required=True, help="Path to the SFT LoRA adapter checkpoint dir.")
    ap.add_argument("--output-dir", type=str, required=True, help="Where to save the merged full-weight model.")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    from grpo_training_new_code import MODEL, MODEL_FAMILY

    logger.info("MODEL_FAMILY=%s base MODEL=%s adapter=%s", MODEL_FAMILY, MODEL, args.adapter)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # qwen/qwen36: the release repo's OWN config.json declares architectures=
    # ["Qwen3_5ForConditionalGeneration"] (model_type "qwen3_5") -- a genuine
    # multimodal wrapper (vision_config + a "visual.*" tower), not the plain
    # Qwen3ForCausalLM every other family here uses. AutoModelForCausalLM.
    # from_pretrained(MODEL) still "works" on this repo, but only because
    # transformers registers a deliberate "VLM compatibility" text-only sibling
    # class (Qwen3_5ForCausalLM, model_type "qwen3_5_text") for exactly this
    # model_type -- it silently loads JUST the language_model.* submodule,
    # dropping the entire vision tower. transformers/SFT/GRPO's own HF-side
    # training load is fine with that (never touches vision), but vLLM is NOT:
    # its weight loader hard-checks that every parameter the *checkpoint's own
    # declared config* implies must exist actually has a matching tensor, and
    # raises ValueError("weights were not initialized from checkpoint") for
    # the ~100+ missing visual.* tensors otherwise (confirmed job 152116) --
    # this is not a lenient "leave it randomly initialized" path. An earlier,
    # incomplete fix (restoring the wrapper config.json alone, without the
    # actual vision weights) got past vLLM's config-type check but still hit
    # this exact hard failure (jobs 151566/151570 -> config fix -> 152103/
    # 152116). The only correct fix is to never drop the vision tower in the
    # first place: detect the wrapper architecture up front and load with
    # AutoModelForImageTextToText instead, which returns the FULL
    # Qwen3_5ForConditionalGeneration model (language_model + visual tower +
    # everything else in the original checkpoint's weight_map, ~775 keys vs.
    # the causal-LM-only class's 427). The SFT LoRA's target_modules were
    # attention/MLP proj *leaf names* (q_proj/k_proj/.../down_proj), which PEFT
    # matches by suffix regardless of nesting depth, so merging against this
    # fuller model still finds and merges the exact same language_model.*
    # linear layers -- the vision tower is simply carried through unmodified
    # end to end, then saved back out complete.
    base_config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    is_vlm_wrapper = any(a.endswith("ForConditionalGeneration") for a in (base_config.architectures or []))
    model_cls = AutoModelForImageTextToText if is_vlm_wrapper else AutoModelForCausalLM
    logger.info("Base config architectures=%s -> loading with %s", base_config.architectures, model_cls.__name__)

    logger.info("Loading base model in bf16...")
    base_model = model_cls.from_pretrained(
        MODEL, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda",
    )

    logger.info("Loading LoRA adapter and merging...")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    merged_model = peft_model.merge_and_unload()

    logger.info("Saving merged model to %s", output_dir)
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Done. Merged checkpoint ready at %s", output_dir)


if __name__ == "__main__":
    main()

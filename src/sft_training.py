#!/usr/bin/env python3
"""
LoRA SFT baseline on train.jsonl, using TRL's SFTTrainer, for direct
comparison against the GRPO-trained checkpoints of the same base models.

Reuses grpo_training_new_code.py's MODEL/MODEL_FAMILY resolution,
build_messages() (the exact same system prompt + user template used by
GRPO training and by every eval script), and per-family chat-template
kwargs -- so the SFT *input* side is byte-identical to GRPO/eval. LoRA rank
is read from the same GRPO_LORA_R/GRPO_LORA_ALPHA env vars GRPO training
uses, so passing the values a model was actually GRPO-trained with (e.g.
GRPO_LORA_R=64 for qwen/qwen36, the unset default 16 for huatuo) keeps the
adapter capacity identical between the SFT and GRPO checkpoints being
compared -- the comparison isolates "training algorithm", not "how much
capacity the LoRA adapter had".

Target (completion) construction
---------------------------------
train.jsonl's own `training_format` field (`<think>{cot}</think>\\n{final_answer}`)
is NOT used directly -- it omits the <final_answer> tag wrapper that the
production system prompt instructs the model to use and that the eval
extractors (_answer_for_judge / _extract_final_answer) require. Instead we
build the assistant turn as:

    <think>
    {cot}
    </think>

    <final_answer>{final_answer}</final_answer>

matching the QWEN_SYSTEM_PROMPT example format exactly, so the SFT model is
taught to emit the same tag structure GRPO rewards and eval scores.

Prompt/completion split is derived from the tokenizer's OWN chat template
rather than any hardcoded EOS assumption: we render the full 3-turn
conversation (system, user, assistant) and diff it against the
prompt-only render (system, user, add_generation_prompt=True). The
suffix is used verbatim as the "completion" string, so whatever the
template's own turn-closing tokens are (<|im_end|> etc.) for this specific
model/family, they're captured correctly without guessing.

Loss masking: TRL SFTConfig(completion_only_loss=True) on a
{"prompt": ..., "completion": ...} dataset sets prompt-token labels to
-100 automatically -- loss is computed on the assistant completion only.

Long-tail filtering: examples whose full (prompt+completion) token count
exceeds --max-seq-length are DROPPED, not truncated -- truncation defaults
to keep_start (keeps the front, cuts the tail), which would silently chop
off the <final_answer> for the rare very-long-CoT record. Dropping is safer
than training on a corrupted target; the expected drop rate is small (p99
training_format length is ~18.5K chars =~ 4-5K tokens, well under either
model's cap).

Example
-------
GRPO_MODEL_FAMILY=huatuo python sft_training.py \
    --train-jsonl ./data/train.jsonl \
    --output-dir ./checkpoints/medical-sft-huatuogpt3-8b-midstream_task \
    --num-epochs 3 --learning-rate 1e-4 --max-seq-length 12288
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
                         handlers=[logging.StreamHandler(sys.stdout),
                                   logging.FileHandler(output_dir / "sft_train.log")])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-jsonl", type=str,
                     default="./data/train.jsonl")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--num-epochs", type=float, default=3.0)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=16384,
                     help="Examples with prompt+completion longer than this (in tokens) are DROPPED.")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="Cap number of training examples (smoke tests).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cot", action="store_true",
                     help="Train the assistant target as <final_answer>...</final_answer> only, "
                          "omitting the <think>...</think> block entirely. Records still require a "
                          "non-empty cot field for inclusion (same example set as the CoT variant), "
                          "the cot text is just excluded from the assistant target. Loss is still "
                          "assistant-only via completion_only_loss.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    logger.info("Args: %s", vars(args))

    # Deferred imports: heavy (torch/transformers/trl), and grpo_training_new_code
    # reads GRPO_MODEL_FAMILY / GRPO_LORA_R / GRPO_LORA_ALPHA from the environment
    # at import time -- must already be set by the launching sbatch before this.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from grpo_training_new_code import (
        LORA_ALPHA,
        LORA_DROPOUT,
        LORA_R,
        MODEL,
        MODEL_FAMILY,
        _CHAT_TEMPLATE_KWARGS_BY_FAMILY,
        build_messages,
        load_jsonl_records,
    )

    logger.info("MODEL_FAMILY=%s MODEL=%s LORA_R=%d LORA_ALPHA=%d",
                MODEL_FAMILY, MODEL, LORA_R, LORA_ALPHA)

    chat_kwargs = _CHAT_TEMPLATE_KWARGS_BY_FAMILY[MODEL_FAMILY]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    logger.info("Loading train records from %s", args.train_jsonl)
    records = load_jsonl_records(args.train_jsonl)
    if args.limit:
        records = records[: args.limit]
    logger.info("Loaded %d raw records", len(records))

    def _render(messages: list[dict], **kw) -> str:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, **kw)
        except Exception:
            if messages and messages[0].get("role") == "system":
                folded = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
                folded += messages[2:]
                return tokenizer.apply_chat_template(folded, tokenize=False, **kw)
            raise

    examples: list[dict[str, str]] = []
    turn_closer: str | None = None  # e.g. "<|im_end|>" -- derived below, not hardcoded
    dropped_missing = dropped_template_mismatch = dropped_too_long = 0
    for r in records:
        ctx = (r.get("context_chunks") or "").strip()
        q = (r.get("question") or "").strip()
        cot = (r.get("cot") or "").strip()
        final_answer = (r.get("final_answer") or "").strip()
        if not (ctx and q and cot and final_answer):
            dropped_missing += 1
            continue

        messages = build_messages(ctx, q)
        if args.no_cot:
            assistant_content = f"<final_answer>{final_answer}</final_answer>"
        else:
            assistant_content = f"<think>\n{cot}\n</think>\n\n<final_answer>{final_answer}</final_answer>"

        prompt_text = _render(messages, add_generation_prompt=True, **chat_kwargs)
        full_text = _render(messages + [{"role": "assistant", "content": assistant_content}],
                             add_generation_prompt=False, **chat_kwargs)
        if not full_text.startswith(prompt_text):
            dropped_template_mismatch += 1
            continue
        completion_text = full_text[len(prompt_text):]

        if turn_closer is None:
            # The chat template's own turn-closing suffix (e.g. "<|im_end|>\n"):
            # whatever the template appends after assistant_content, verbatim.
            # This is the model's REAL stop token for this family, which is not
            # always the same as tokenizer.eos_token (verified: for huatuo,
            # tokenizer.eos_token == "<|endoftext|>" but the template actually
            # closes turns with "<|im_end|>" -- see _EOS_IDS_BY_FAMILY).
            #
            # TRIED passing this to SFTConfig(eos_token=...) so TRL's own
            # "does the completion already end in eos_token" dataset-prep
            # check would line up and skip its auto-append -- that instead
            # triggered TRL's clone_chat_template/embedding-resize code path,
            # which failed to load HuatuoGPT-3-8B's weights at all
            # ("does not appear to have a file named pytorch_model.bin or
            # model.safetensors"). Reverted: left at TRL's default, which DOES
            # append tokenizer.eos_token ("<|endoftext|>") after the
            # template's own closer, i.e. completions end in
            # "<|im_end|>\n<|endoftext|>". Accepted as harmless: real
            # generation always halts at the first EOS-equivalent id
            # (<|im_end|>, 151645), so the model is never actually asked to
            # produce that trailing token at inference -- it only ever
            # appears as one extra (correctly completion_only_loss-masked-in)
            # label during training. turn_closer is kept only as a log/sanity
            # value, not fed back into the trainer config.
            turn_closer = completion_text[len(assistant_content):].strip()
            logger.info("Detected turn-closer token from chat template: %r "
                        "(informational only, not passed to SFTConfig)", turn_closer)

        n_tokens = len(tokenizer(prompt_text + completion_text, add_special_tokens=False)["input_ids"])
        if n_tokens > args.max_seq_length:
            dropped_too_long += 1
            continue

        examples.append({"prompt": prompt_text, "completion": completion_text})

    logger.info(
        "Built %d training examples (dropped: %d missing-field, %d template-mismatch, %d too-long)",
        len(examples), dropped_missing, dropped_template_mismatch, dropped_too_long,
    )
    if examples:
        logger.info("Example[0] prompt (first 300 chars): %r", examples[0]["prompt"][:300])
        logger.info("Example[0] completion (first 300 chars): %r", examples[0]["completion"][:300])
    logger.info("tokenizer.eos_token=%r (vocab id %s) will be appended by TRL after each completion's own "
                "turn-closer (%r) -- see comment above", tokenizer.eos_token, tokenizer.eos_token_id, turn_closer)

    train_dataset = Dataset.from_list(examples)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_seq_length,
        completion_only_loss=True,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=True,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=MODEL,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting SFT training: %d examples, %.1f epochs, lr=%.2e, lora r=%d alpha=%d, no_cot=%s",
                len(examples), args.num_epochs, args.learning_rate, LORA_R, LORA_ALPHA, args.no_cot)
    trainer.train()
    trainer.save_model(str(output_dir))
    logger.info("Saved final SFT checkpoint to %s", output_dir)


if __name__ == "__main__":
    main()

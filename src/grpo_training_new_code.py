#!/usr/bin/env python3
"""
GRPO training for medical case-completion QA.

- Policy prompts: SYSTEM_PROMPT + user template (case chunk + question)
- Reward: DeepSeek LLM judge (4-criterion rubric, normalized 0–1)
- Generation: vLLM colocated in-process (use_vllm=True, vllm_mode="colocate")
  on every training GPU — no separate vllm-serve process, no dedicated
  generation GPU split.

Eval consists of two datasets, evaluated every EVAL_STEPS:
- "val":      held-out case-completion QA, scored by the rubric judge
- "ext_val":  external final-diagnosis task, scored by a separate
              diagnosis-equivalence judge (downstream accuracy)

Judge predictions (model completion, extracted answer, judge raw response,
score) are appended to a per-rank JSONL file under
OUTPUT_DIR/judge_predictions/ as training proceeds.

Run (single node, 8 GPUs, colocated vLLM on all of them):
  accelerate launch --num_processes=8 grpo_training_new_code.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sys
import textwrap
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from peft import LoraConfig
from tenacity import retry, stop_after_attempt, wait_exponential
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"


def setup_logging() -> Path:
    """Configure root logger to write to both stdout and a timestamped file.

    Per-rank file naming avoids contention when accelerate launches multiple
    processes. Configures the *root* logger so library logs (trl, transformers,
    peft, vllm, ...) are captured too.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Stable across ranks: use the launch timestamp from env if set, else now().
    ts = os.environ.get("GRPO_RUN_TS") or datetime.now().strftime("%Y%m%d_%H%M%S")
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    log_file = LOG_DIR / f"grpo_train_{ts}_rank{rank}.log"

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Quiet down noisy libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # Suppress the noisy transformers warning:
    #   "[transformers] Kwargs passed to `processor.__call__` have to be in
    #    `processor_kwargs` dict, not in `**kwargs`"
    # It's emitted once per generation by vLLM's tokenizer path, harmless,
    # and floods the log. Filter via warnings (for warnings-module emissions)
    # and via a logging filter (for logger-based emissions).
    warnings.filterwarnings("ignore", message=r".*processor_kwargs.*")
    warnings.filterwarnings("ignore", message=r".*Kwargs passed to `processor\.__call__`.*")

    class _DropProcessorKwargsMessages(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                return True
            return (
                "processor_kwargs" not in msg
                and "Kwargs passed to `processor" not in msg
            )

    _drop = _DropProcessorKwargsMessages()
    file_handler.addFilter(_drop)
    stream_handler.addFilter(_drop)

    # Belt-and-suspenders: if transformers' own logger emits these, lower it
    # to ERROR so warnings don't get through at all.
    try:
        import transformers  # noqa: F401
        transformers.logging.set_verbosity_error()
    except Exception:
        pass

    return log_file


# =============================================================================
# Config
# =============================================================================

# --- model family switch ---
# "qwen"  = Qwen/Qwen3.5-9B. Chat template auto-injects the opening <think>
#           into the prompt when enable_thinking=True; completion text only
#           ever contains the model-generated closing </think>.
# "gemma" = google/gemma-4-12b-it. With enable_thinking=True the template
#           injects nothing extra at generation time -- the model generates
#           both its own opening "<|channel>thought\n" AND its own closing
#           "<channel|>" as part of the completion (confirmed against the
#           real chat_template.jinja + a live generation smoke test, not
#           assumed from docs). <channel|> is therefore Gemma's functional
#           equivalent of Qwen's </think>.
# huatuo: FreedomIntelligence/HuatuoGPT-3-8B -- a medical model built on
#         Qwen3-8B-Base via an RL-only recipe ("SeedRL"), so it is architecturally
#         Qwen3ForCausalLM and uses the standard Qwen ChatML template with
#         model-emitted <think>...</think>. It therefore shares Qwen's reasoning
#         close-tag and sampling settings.
#
#         ONE CRITICAL DIFFERENCE from Qwen3.5: its tokenizer was inherited from
#         Qwen3-8B-*Base*, whose eos_token is <|endoftext|> (151643) -- but the
#         chat-tuned model actually ends every assistant turn with <|im_end|>
#         (151645), and its generation_config declares eos_token_id=[151645,
#         151643]. TRL 1.7.0 checks termination against the single scalar
#         self._tokenizer.eos_token_id (=151643), so EVERY completion would be
#         flagged as truncated -- exactly the Gemma multi-EOS bug (see the
#         _EOS_IDS_BY_FAMILY block below, which is why that fix is no longer
#         Gemma-only). Qwen3.5-9B is unaffected: its tokenizer eos_token IS
#         <|im_end|>, which is why the qwen path never needed this patch.
# deepseek_llama: deepseek-ai/DeepSeek-R1-Distill-Llama-8B -- a Llama-3.1-8B
#                  base R1-distilled into an always-reasoning model (no
#                  enable_thinking toggle: its chat template pre-injects the
#                  opening <think>\n into the generation prompt unconditionally,
#                  same as Qwen3.5 -- confirmed byte-identical render with/
#                  without enable_thinking=True passed, see
#                  download_deepseek_r1_llama8b.sbatch job 112148). Its own
#                  tokenizer.eos_token IS <｜end▁of▁sentence｜> (128001), which
#                  matches generation_config.eos_token_id (128001) exactly --
#                  a single scalar, unlike Gemma/huatuo's multi-token EOS
#                  scheme -- so it does NOT need the _EOS_IDS_BY_FAMILY patch
#                  (same exemption as qwen). <think>/</think> are real,
#                  dedicated vocab tokens (128013/128014), not reconstructed
#                  from BPE pieces.
# qwen36: Qwen/Qwen3.6-27B -- a newer, larger dense Qwen (April 2026, Apache
#         2.0). Verified job 118458 (download_qwen36_27b.sbatch): its tokenizer
#         eos_token IS <|im_end|> (248046), a single scalar matching the chat
#         template's turn-ender exactly -- so it does NOT need the
#         _EOS_IDS_BY_FAMILY patch (same exemption as qwen). Its chat template
#         renders BYTE-IDENTICALLY to Qwen3.5's under {"enable_thinking": True,
#         "thinking_budget": 6000} -- pre-injecting the opening <think> -- so it
#         reuses qwen's CHAT_TEMPLATE_KWARGS, thinking-close regex, sampling
#         params, and system prompt unchanged. <think>/</think> are dedicated
#         vocab tokens (248068/248069).
#         ONE OPEN RISK (does not affect generation/vLLM, only the training
#         load): its architecture is Qwen3_5ForConditionalGeneration
#         (model_type qwen3_5) -- a MULTIMODAL wrapper, not plain
#         Qwen3ForCausalLM. vocab_size/eos_token_id read None at the top-level
#         config (nested in a text sub-config). The transformers training-side
#         load (AutoModelForCausalLM + LoRA target-modules) must be verified
#         before a real GRPO launch; calibration (plain vLLM) only de-risks the
#         generation side.
MODEL_FAMILY = os.environ.get("GRPO_MODEL_FAMILY", "qwen")

# Opt-in only (default "0"): for evaluating the no-CoT SFT ablation, whose
# assistant target is JUST <final_answer>...</final_answer> with no <think>
# block at all. For qwen/qwen36, the chat template auto-opens "<think>\n" as
# part of add_generation_prompt itself (verified: prompt_text ends in
# "...assistant\n<think>\n"), so a no-cot-trained completion never contains a
# literal "</think>" -- _post_reasoning_region's strict gate then returns
# None for every single completion, and _extract_final_answer returns "" even
# though the completion has a perfectly well-formed <final_answer> tag
# (confirmed by direct inspection: job ci_sft_qwen36_27b_nocot_epoch1_n500
# scored a flat 0.000 despite completions like
# '<final_answer>The CT angiogram...</final_answer>' with sensible content).
# Gated behind this env var (never set for the production/CoT eval jobs) so
# it cannot change any existing GRPO-training or CoT-eval scoring behavior.
_NOCOT_EVAL_FALLBACK = os.environ.get("GRPO_EVAL_NOCOT_FALLBACK", "0") == "1"
if MODEL_FAMILY not in (
    "qwen", "gemma", "huatuo", "deepseek_llama", "qwen36", "gemma4_31b", "nemotron3_nano_omni",
    # openbiollm/medreason/glm52: EVAL-ONLY reference models (base-only 95%-CI
    # benchmarks against test.jsonl/ext_test.jsonl, no GRPO training planned).
    # Onboarded solely so eval_checkpoints.py/eval_ci_driver.py can reuse the
    # existing judge/system-prompt/parsing infra instead of a bespoke path.
    "openbiollm", "medreason", "glm52",
):
    raise ValueError(
        f"Unknown GRPO_MODEL_FAMILY={MODEL_FAMILY!r}; expected 'qwen', 'gemma', "
        f"'huatuo', 'deepseek_llama', 'qwen36', 'gemma4_31b', 'nemotron3_nano_omni', "
        f"'openbiollm', 'medreason' or 'glm52'"
    )

# --- paths ---
DATA_DIR = "./data"
TRAIN_JSONL = f"{DATA_DIR}/train.jsonl"
VAL_JSONL = f"{DATA_DIR}/val.jsonl"
EXT_VAL_JSONL = f"{DATA_DIR}/ext_val.jsonl"
_OUTPUT_DIR_DEFAULTS = {
    "qwen": "./checkpoints/medical-grpo-qwen3.5-9b-midstream_task-v2",
    "gemma": "./checkpoints/medical-grpo-gemma4-12b-midstream_task",
    "huatuo": "./checkpoints/medical-grpo-huatuogpt3-8b-midstream_task",
    "deepseek_llama": "./checkpoints/medical-grpo-deepseek-r1-llama8b-midstream_task",
    "qwen36": "./checkpoints/medical-grpo-qwen3.6-27b-midstream_task",
    "gemma4_31b": "./checkpoints/medical-grpo-gemma4-31b-midstream_task",
    "nemotron3_nano_omni": "./checkpoints/medical-grpo-nemotron3-nano-omni-midstream_task",
    # openbiollm/medreason/glm52: no training planned, so these are unused in
    # practice -- present only to satisfy the eager dict lookup on import.
    "openbiollm": "./checkpoints/reference-openbiollm-70b",
    "medreason": "./checkpoints/reference-medreason-8b",
    "glm52": "./checkpoints/reference-glm52-nvfp4",
}
OUTPUT_DIR = os.environ.get("GRPO_OUTPUT_DIR", _OUTPUT_DIR_DEFAULTS[MODEL_FAMILY])

# --- model ---
_MODEL_NAMES = {
    "qwen": "Qwen/Qwen3.5-9B",
    "gemma": "google/gemma-4-12b-it",
    "huatuo": "FreedomIntelligence/HuatuoGPT-3-8B",
    "deepseek_llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "qwen36": "Qwen/Qwen3.6-27B",
    # Same Gemma4ForConditionalGeneration architecture as "gemma" (12B), just
    # the 31B dense sibling. Chat template (chat_template.jinja), EOS scheme
    # (1, 106, 50), and thinking-channel behavior are inherited from "gemma"
    # below as a starting point -- NOT yet independently verified against this
    # size's actual generations. Run calibrate_gemma4_31b_thresholds.sbatch
    # before trusting the thinking-length thresholds for a real launch.
    "gemma4_31b": "google/gemma-4-31B-it",
    # NemotronH_Nano_Omni_Reasoning_V3's full wrapper is NOT usable for a
    # text-only task at all: its forward() takes `pixel_values` as a
    # required (non-optional) positional arg and unconditionally runs a
    # vision tower whose auto_map points to a SEPARATE remote repo
    # (nvidia/C-RADIOv4-H) needing timm/einops/open_clip AND a custom CUDA
    # op (indirect_grid_sample -- not even on PyPI, would need a source
    # build). Chasing that dependency chain is the wrong path for a model
    # we only ever call with text.
    #
    # The actual text backbone (NemotronHForCausalLM, hybrid Mamba2+
    # Attention+MoE, llm_config.model_type="nemotron_h") is fully
    # self-contained: configuration_nemotron_h.py/modeling_nemotron_h.py
    # only import stdlib + transformers.* (confirmed via grep -- no vision/
    # audio/CUDA-ext deps), and its weights are cleanly namespaced under
    # `language_model.*` in the omni checkpoint (6243 of 7349 total keys).
    # extract_nemotron_llm_only.py pulled just that subset (load_state_dict
    # matched with zero missing/unexpected keys) into this standalone local
    # dir, which also carries its own tokenizer/chat_template/generation_
    # config files copied from the omni repo (job 128283). Confirmed vLLM
    # 0.24.0 registers `NemotronHForCausalLM` (the bare architecture, not
    # just the omni wrapper) natively (job 128284), so this checkpoint is
    # loadable by both the trainer and the disaggregated vLLM rollout
    # server.
    #
    # MUST be a LOCAL DIRECTORY path, not a bare HF repo string -- same
    # dynamic_module_utils blobs/symlink bug that bit the original omni
    # download (job 128153) applies to any multi-file custom-code repo.
    "nemotron3_nano_omni": "./models/nemotron3-nano-omni-30b-a3b-llm-only",
    # openbiollm/medreason/glm52: base-only reference-model benchmarks (no
    # GRPO training) -- prompt/parse strategy validated via smoke tests
    # (_smoke_openbiollm_parse_v2.py, _smoke_medreason_parse.py,
    # _smoke_glm_parse.py) before onboarding here.
    "openbiollm": "aaditya/Llama3-OpenBioLLM-70B",
    "medreason": "UCSC-VLAA/MedReason-8B",
    "glm52": "nvidia/GLM-5.2-NVFP4",
}
# Env-overridable so a run can start from a local merged checkpoint (e.g. an
# SFT-CoT LoRA adapter merged into the base weights via merge_sft_adapter.py)
# instead of the family's released HF weights, while MODEL_FAMILY itself keeps
# driving every family-specific behavior below (LoRA target modules, chat
# template kwargs, EOS ids, sampling defaults) unchanged -- only the starting
# weights differ. Must be a local directory (same requirement as
# nemotron3_nano_omni's local-dir entry above).
MODEL = os.environ.get("GRPO_MODEL_OVERRIDE", "").strip() or _MODEL_NAMES[MODEL_FAMILY]
# Shared across families -- no GRPO/RL-specific hyperparameter guidance
# exists for Gemma 4 (its HF page only documents a supervised-QLoRA recipe,
# lr=2e-4, which is a different training paradigm from policy-gradient RL).
# Anchor to the already-tuned, empirically-stable Qwen RL settings instead.
# Env-overridable so exploration runs can raise capacity without editing code.
# Defaults (16 / 64) preserve the validated Qwen RL settings exactly. Keep the
# alpha/r scaling ratio constant (=4.0) when raising r, else the effective LoRA
# update magnitude shrinks (e.g. r=64 needs alpha=256 to hold scaling at 4.0).
LORA_R = int(os.environ.get("GRPO_LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("GRPO_LORA_ALPHA", "64"))
LORA_DROPOUT = 0.05

# --- eval data ---
VAL_SAMPLE_SIZE = int(os.environ.get("GRPO_VAL_SAMPLE_SIZE", "100"))
VAL_SAMPLE_SEED = 42
EXT_VAL_SAMPLE_SIZE = int(os.environ.get("GRPO_EXT_VAL_SAMPLE_SIZE", "100"))
EXT_VAL_SAMPLE_SEED = 42

# --- training ---
PER_DEVICE_TRAIN_BATCH_SIZE = 1
NUM_GENERATIONS = int(os.environ.get("GRPO_NUM_GENERATIONS", "8"))
NUM_EVAL_GENERATIONS = 1
MAX_COMPLETION_LENGTH = 8192
LEARNING_RATE = 1e-5
# Constant LR after a short warmup — cosine was decaying the LR while the
# policy was still trying to explore, contributing to the flat-reward plateau.
LR_SCHEDULER_TYPE = "constant_with_warmup"
WARMUP_STEPS = 10
NUM_TRAIN_EPOCHS = 1.0
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRPO_GRADIENT_ACCUMULATION_STEPS", "4"))
LOGGING_STEPS = 5
SAVE_STEPS = 25
EVAL_STEPS = 100
BF16 = True
GRADIENT_CHECKPOINTING = True

# --- generation ---
# Per-family recommended sampling settings (Qwen3.5 and Gemma-4 model cards).
# Gemma temperature dropped 1.0 -> 0.7 on 2026-07-06 (see
# GEMMA4_KL_ENTROPY_INSTABILITY_FIX.md): job 82280 (temperature=1.0) showed
# eval_entropy climbing 0.79 -> 1.84 over 400 steps with no reward
# improvement (Qwen, at temperature=0.6, stayed flat at ~0.3-0.5 entropy
# while reward climbed 0.54 -> 0.78 over the same step count) -- the higher
# temperature was letting the policy wander into a self-reinforcing
# high-entropy regime instead of sharpening around reward-earning behavior.
_GENERATION_PARAMS = {
    "qwen": dict(temperature=0.6, top_p=0.95, top_k=20),
    "gemma": dict(temperature=0.7, top_p=0.95, top_k=64),
    # Exactly HuatuoGPT-3-8B's own generation_config.json (temperature 0.6,
    # top_p 0.95, top_k 20) -- which is also identical to Qwen's validated
    # values, so no reconciliation was needed.
    "huatuo": dict(temperature=0.6, top_p=0.95, top_k=20),
    # DeepSeek-R1-Distill model card recommendation: temperature 0.5-0.7
    # (0.6 default), top_p 0.95, no greedy decoding, no top_k restriction
    # mentioned -- disabled (-1) rather than guessing an arbitrary value.
    "deepseek_llama": dict(temperature=0.6, top_p=0.95, top_k=-1),
    # qwen36 (Qwen3.6-27B) reuses Qwen3.5's validated sampling settings -- same
    # tokenizer/template lineage (see MODEL_FAMILY comment above).
    "qwen36": dict(temperature=0.6, top_p=0.95, top_k=20),
    # gemma4_31b (31B dense, same Gemma4ForConditionalGeneration architecture
    # as "gemma"/12B): default here is the 12B family's OWN validated-safe
    # value (0.7, not the risker 1.0 HF model-card default) as a starting
    # anchor -- NOT yet independently validated at 31B scale. A real launch
    # overrides via GRPO_TEMPERATURE regardless (e.g. testing Qwen's 0.9
    # exploration recipe); calibrate_gemma4_31b_thresholds.sbatch sweeps both.
    "gemma4_31b": dict(temperature=0.7, top_p=0.95, top_k=64),
    # nemotron3_nano_omni: exactly the model's own generation_config.json
    # (temperature 0.6, top_p 0.95) -- confirmed via
    # download_nemotron3_nano_omni.sbatch (job 128133). No top_k field is
    # present there at all, so disabled (-1) rather than guessing a value,
    # same rationale as deepseek_llama above.
    "nemotron3_nano_omni": dict(temperature=0.6, top_p=0.95, top_k=-1),
    # openbiollm/medreason/glm52: eval-only (no training), so these values are
    # dead code for our use case -- eval_checkpoints.py drives generation via
    # its own GEN_TEMPERATURE/GEN_TOP_P/GEN_TOP_K constants (0.6/0.95/20), not
    # this dict. Present only to satisfy the eager dict lookup on import.
    "openbiollm": dict(temperature=0.6, top_p=0.95, top_k=20),
    "medreason": dict(temperature=0.6, top_p=0.95, top_k=20),
    "glm52": dict(temperature=0.6, top_p=0.95, top_k=20),
}
# GRPO_TEMPERATURE override: lets a run test a different sampling temperature
# than the per-family default above (e.g. Gemma-4's own HF model-card value
# of 1.0, vs. the 0.7 this default was lowered to after job 82280 -- see the
# comment above _GENERATION_PARAMS) without editing this file.
TEMPERATURE = float(
    os.environ.get("GRPO_TEMPERATURE", str(_GENERATION_PARAMS[MODEL_FAMILY]["temperature"]))
)
TOP_P = _GENERATION_PARAMS[MODEL_FAMILY]["top_p"]
TOP_K = _GENERATION_PARAMS[MODEL_FAMILY]["top_k"]

# --- GRPO ---
NUM_ITERATIONS = 1
LOSS_TYPE = "dr_grpo"
# Was an unconditional False. Tried flipping this on for Gemma in job 82986
# to mask out the ~1.85% of eval completions that degenerate into >25k-char
# "let me reconsider" loops (see GEMMA4_KL_ENTROPY_INSTABILITY_FIX.md) --
# backfired immediately (loss/grad_norm/kl/entropy all pinned at 0 from step
# 1). Root cause: TRL 1.7.0's truncation check (grpo_trainer.py:1940 and
# :2076, `eos_and_pad = [self._tokenizer.eos_token_id,
# self._tokenizer.pad_token_id]`) uses a single scalar id. Gemma-4's
# generation_config.json declares THREE valid terminators
# (eos_token_id=[1, 106, 50]; 106 is <turn|>, the actual turn-end token the
# chat model emits -- plain <eos>/1 is essentially never generated
# mid-conversation), and transformers collapses this to just `1` when
# aligning tokenizer/generation_config at load time. So TRL flagged
# virtually every Gemma completion as "never terminated" -- including
# well-formed ones that correctly stopped on <turn|> -- not just the truly
# degenerate ones, zeroing out ~100% of every batch.
#
# Fixed 2026-07-06: the MODEL_FAMILY == "gemma" block below (near
# GRPO_KL_LOGRATIO_CLIP) now monkeypatches both call sites
# (GRPOTrainer._generate and ._generate_and_score_completions) to check
# membership in the real [1, 106, 50] id list instead of the tokenizer's
# scalar. Safe to re-enable for Gemma now via GRPO_MASK_TRUNCATED_COMPLETIONS
# =1 -- it will only mask genuinely non-terminating completions.
# Qwen is unaffected either way (single eos_token_id, no multi-token EOS
# scheme).
MASK_TRUNCATED_COMPLETIONS = os.environ.get(
    "GRPO_MASK_TRUNCATED_COMPLETIONS", "false"
).lower() in ("1", "true", "yes")
VLLM_IMPORTANCE_SAMPLING_CORRECTION = False
# KL coefficient against the reference (frozen base) model. Higher = stronger
# anchor against drift. TRL default is ~0.04. Previous runs with beta=0.1 and
# beta=0.05 stayed collapse-free but the policy barely drifted (KL ≈ 0.002 →
# flat reward). Dropping further to 0.01 lets the policy move much more per
# step, at the cost of a weaker anchor against drift/collapse.
#
# Was a hardcoded 0.01 (shared by both families). Gemma job 82280 at 0.01
# showed recurring batch-KL blowups (spikes to 49/74/151, see KL-DEBUG
# instrumentation from jobs 82044/82056) and climbing entropy with zero
# reward improvement over 400 steps -- 0.01 is too weak an anchor for
# Gemma's larger per-step drift. Raised back to the TRL default (0.04) for
# Gemma via GRPO_BETA; Qwen's sbatch doesn't set this env var so it keeps
# running at the validated 0.01.
BETA = float(os.environ.get("GRPO_BETA", "0.01"))

# --- thinking (chat template thinking mode + length penalty on reasoning chars) ---
# thinking_budget is a Qwen3.5-only soft cap kwarg; Gemma-4 has no equivalent
# (its template only checks the boolean enable_thinking, see MODEL_FAMILY
# switch above) so its chat_template_kwargs is just {"enable_thinking": True}.
THINKING_BUDGET_TOKENS = 6000
_CHAT_TEMPLATE_KWARGS_BY_FAMILY = {
    "qwen": {"enable_thinking": True, "thinking_budget": THINKING_BUDGET_TOKENS},
    "gemma": {"enable_thinking": True},
    # huatuo ships the stock Qwen3 ChatML template, which supports
    # enable_thinking but has NO thinking_budget variable (that is a Qwen3.5-only
    # extension) -- passing it would silently do nothing, so it is omitted.
    # Note the Qwen3 template only injects tags when enable_thinking is FALSE (it
    # emits an empty '<think>\n\n</think>' to suppress reasoning); with thinking
    # enabled the generation prompt just ends at '<|im_start|>assistant\n' and the
    # MODEL itself emits the opening <think>. That differs from Qwen3.5 (whose
    # template pre-injects the opening <think> into the prompt), but it does not
    # affect us: _thinking_char_count and _answer_for_judge both key off the
    # CLOSING </think>, which huatuo emits normally.
    "huatuo": {"enable_thinking": True},
    # deepseek_llama's chat template has NO enable_thinking check at all --
    # confirmed via job 112148: apply_chat_template() with and without
    # enable_thinking=True produced a byte-identical rendered prompt. The
    # template unconditionally pre-injects the opening '<think>\n' into every
    # generation prompt (same pre-injection style as Qwen3.5), so there is
    # nothing to toggle -- passing the kwarg would silently no-op.
    "deepseek_llama": {},
    # qwen36 (Qwen3.6-27B): its chat template honors BOTH enable_thinking and
    # thinking_budget identically to Qwen3.5 (verified byte-identical render,
    # job 118458), so it reuses qwen's exact kwargs.
    "qwen36": {"enable_thinking": True, "thinking_budget": THINKING_BUDGET_TOKENS},
    # gemma4_31b: same chat_template.jinja convention as "gemma" (boolean
    # enable_thinking only, no thinking_budget) -- confirmed via
    # download_gemma4_31b.sbatch's template render check.
    "gemma4_31b": {"enable_thinking": True},
    # nemotron3_nano_omni: template honors enable_thinking (confirmed via
    # download_nemotron3_nano_omni.sbatch render check -- True gives an open
    # '<think>\n', False gives an empty '<think></think>'). The template also
    # accepts an optional reasoning_budget kwarg but defaults it to None
    # (no-op) if omitted -- deliberately NOT set yet, same "start simple,
    # calibrate before adding extra knobs" approach used for every other new
    # family here.
    "nemotron3_nano_omni": {"enable_thinking": True},
    # openbiollm/medreason: plain Llama-3-lineage templates, no
    # enable_thinking/thinking_budget variables at all -- confirmed via smoke
    # test (no chat_template_kwargs passed, both loaded/generated fine).
    "openbiollm": {},
    "medreason": {},
    # glm52: template defaults to "Reasoning Effort: Max" unless reasoning_
    # effort="high" is explicitly passed (chat_template.jinja: `'high' if
    # reasoning_effort is defined and reasoning_effort == 'high' else 'max'`).
    # Max was left as the silent default through the first full-CI attempt
    # (job 134974 etc.) and measured (job 137077, N=60, both effort levels
    # generated at max_tokens=32768 so truncation never masked true length):
    #   Max : mean 12067 tokens, 33% would truncate at 16384, 20% STILL
    #         truncated even at 32768 -- and inspecting those isn't "just
    #         needs more budget": several are genuine indecision loops that
    #         never terminate (e.g. "This is my final answer... OK, I need
    #         to stop going back and forth... Let me settle on an answer"
    #         followed by MORE reconsideration, repeatedly).
    #   High: mean 2104 tokens (~6x shorter), only 2% would truncate at
    #         16384, 0% still truncated at 32768 -- every case in the
    #         sample actually terminates.
    # High effort is not just cheaper, it's the one that reliably converges.
    "glm52": {"reasoning_effort": "high"},
}
CHAT_TEMPLATE_KWARGS = _CHAT_TEMPLATE_KWARGS_BY_FAMILY[MODEL_FAMILY]

# Regex marking the end of the reasoning portion of a completion. Qwen:
# model-generated closing </think>. Gemma: model-generated closing <channel|>
# (confirmed via a live generation smoke test against the real chat
# template -- see MODEL_FAMILY comment above).
_THINKING_CLOSE_RE_BY_FAMILY = {
    "qwen": re.compile(r"</think>", re.IGNORECASE),
    "gemma": re.compile(r"<channel\|>", re.IGNORECASE),
    "huatuo": re.compile(r"</think>", re.IGNORECASE),  # stock Qwen3 ChatML
    # deepseek_llama: model-generated closing </think> -- <think>/</think> are
    # real dedicated vocab tokens (ids 128013/128014) for this model, not a
    # reconstructed BPE sequence (confirmed job 112148).
    "deepseek_llama": re.compile(r"</think>", re.IGNORECASE),
    # qwen36: same </think> close tag; dedicated vocab tokens 248068/248069
    # (confirmed job 118458).
    "qwen36": re.compile(r"</think>", re.IGNORECASE),
    # gemma4_31b: inherited from "gemma" (same template channel marker) --
    # NOT yet independently confirmed at 31B scale; the 12B sibling rarely
    # emits <channel|> in practice (see _post_reasoning_region) and falls back
    # to <final_answer>, which is why that fallback below also covers this
    # family. calibrate_gemma4_31b_thresholds.sbatch measures the real rate.
    "gemma4_31b": re.compile(r"<channel\|>", re.IGNORECASE),
    # nemotron3_nano_omni: real </think> closing tag, ChatML-style like
    # qwen/qwen36/huatuo/deepseek_llama (confirmed via chat template render).
    "nemotron3_nano_omni": re.compile(r"</think>", re.IGNORECASE),
    # openbiollm/medreason: non-reasoning-tag models -- smoke tests showed 0%
    # </think> emission for both (direct-answer prose or unstructured prose
    # reasoning, no dedicated close tag). Reuse gemma's never-matching marker
    # so the whole-completion fallback (see _post_reasoning_region) is the
    # one actually load-bearing, exactly as documented for gemma.
    "openbiollm": re.compile(r"<channel\|>", re.IGNORECASE),
    "medreason": re.compile(r"<channel\|>", re.IGNORECASE),
    # glm52: GENUINELY emits </think> (confirmed 8/8 in production-settings
    # smoke, job 134796) -- real ChatML-style close tag, not a channel marker.
    "glm52": re.compile(r"</think>", re.IGNORECASE),
}
_THINKING_CLOSE_RE = _THINKING_CLOSE_RE_BY_FAMILY[MODEL_FAMILY]

# Recalibrated per-family against each model's own actual reasoning-length
# distribution (chars before the answer, for completions that actually
# close -- i.e. the _thinking_char_count fixed-boundary case, not the
# whole-text cut-off fallback).
#
# Qwen: median ~21k chars for completions that close </think>, truncation
# dominates above ~30k -- the original 15k/25k values were tuned against a
# penalty that, due to the _thinking_char_count bug (now fixed), had never
# actually fired (see git history).
#
# Gemma: these were initially shared with Qwen's values as a starting point,
# but measured against a full training run's completions (2128 samples),
# Gemma's real reasoning-before-answer length is roughly 30x shorter than
# Qwen's -- median 722 chars, p90 1240 chars, max 8588 chars among
# completions that actually close with <final_answer>. At Qwen's 20k/30k
# thresholds this penalty was completely inert for Gemma (0/1595 closed
# completions, 0.0%, ever reached even the no-penalty floor) -- it provided
# no gradient signal in the range where Gemma's real behavior varies, and
# only ever fired on the ~26% of completions that never produce an answer
# at all (via the whole-text fallback), which is really a different,
# accidental total-length effect rather than the intended
# excessive-reasoning penalty. Recalibrated using the same methodology as
# Qwen's derivation: no-penalty floor a bit above the closed-completion
# median, linear-end further out past p90 where reasoning is genuinely
# excessive rather than just longer than typical.
#
# huatuo: CALIBRATED 2026-07-12 against 192 real completions (job 108053,
# calibrate_huatuo_thresholds.sbatch). HuatuoGPT-3-8B on our task truncates
# 0.0% of the time (longest completion 2967 of 8192 tokens -- 36% of the cap),
# closes </think> 100% of the time, and emits <final_answer> 100% of the time.
# Reasoning chars: median 4848, p75 6014, p90 7607, p99 9775, max 10417.
#
# The penalty's PURPOSE is to stop runaway reasoning from hitting the token cap.
# huatuo has no such problem, so these are set as a pure runaway GUARDRAIL --
# floor above the observed p99 (9775) so they never fire on any behavior this
# model actually exhibits, biting only on genuine degeneration. This is
# deliberately NOT the median-anchored rule used for qwen/gemma: that rule was
# derived for models whose reasoning length was already causing truncation, and
# applying it here (floor 7300) would penalize ~10-15% of perfectly healthy
# completions -- i.e. it would punish this model for thinking more, on the very
# run whose goal is to make it reason better.
#
# NOTE these are INERT by default anyway (shaping off => penalty fraction 0.0).
_THINKING_CHAR_NO_PENALTY_BY_FAMILY = {
    # qwen (Qwen3.5-9B): CALIBRATED 2026-07-23 against 192 real completions at
    # temp=0.6 (job 133807, calibrate_qwen35_9b_thresholds.sbatch). This family
    # was the LAST to be calibrated despite being the first onboarded -- its
    # previous 20_000 / 30_000 predated the standardized calibration practice
    # every later family received, and were meaningfully wrong: 20_000 sat
    # essentially AT the measured median (19_735), so enabling reward shaping
    # would have penalized ~half of all perfectly normal completions, and the
    # old 30_000 LINEAR_END (vs p90 29_047) would have applied the FULL penalty
    # to the top ~10%. Exactly the failure mode the huatuo note above warns
    # about. Harmless in practice only because shaping was never switched on
    # for either completed 9B run (THINKING_PENALTY_FRACTION -> 0.0 unless
    # GRPO_ENABLE_REWARD_SHAPING=1, which grpo_train_9b_8gpu.sbatch never sets).
    #
    # Reasoning chars: median 19_735, p75 24_691, p90 29_047, p99 31_906,
    # max 32_975. Floor set ~1.5x median, matching every other family's rule.
    #
    # Measured on the 168/192 completions that actually close </think>; the 24
    # that don't are largely the 15.6% truncated at the 8192-token cap. That
    # truncation is NOT treated as a problem to fix here -- it self-corrects
    # under training, since truncated completions emit no <final_answer>, score
    # poorly, and get pushed against by the gradient (observed directly in the
    # nemotron v1 run: eval clipped_ratio fell 0.08 -> 0.035 -> 0.025 over the
    # first 200 steps with no change to MAX_COMPLETION_LENGTH). Anchoring the
    # thresholds to completions that finish is therefore the right population.
    "qwen": 29_600,
    "gemma": 1_500,
    "huatuo": 12_000,  # above observed p99 (9775) / max (10417) -- guardrail only
    # deepseek_llama: CALIBRATED 2026-07-14 against 192 real completions (job
    # 112153, calibrate_deepseek_llama_thresholds.sbatch). Truncates 0.0% of
    # the time (longest completion 3387 of 8192 tokens -- 41% of the cap),
    # closes </think> 100% of the time, emits <final_answer> 99.5% of the
    # time. Reasoning chars: median 3138, p75 4523, p90 5866, p99 10773,
    # max 13300. Floor set to ~1.5x median (never fires on normal reasoning);
    # the placeholder 20_000 seeded from Qwen would have been ~6x looser than
    # this model's real median -- completely inert, the same silent-bug
    # pattern flagged for huatuo/gemma pre-calibration.
    "deepseek_llama": 4_700,
    # qwen36: CALIBRATED 2026-07-15 against 192 real completions (job 118470,
    # calibrate_qwen36_thresholds.sbatch). Qwen3.6-27B truncates 0.0% of the
    # time (longest completion 4685 of 8192 tokens = 57% of cap), closes
    # </think> 100%, emits <final_answer> 100% (fully well-formed 95.8%).
    # Reasoning chars: median 8602, p75 10546, p90 12493, p99 18261, max 18579
    # -- ~2.7x longer reasoning than DeepSeek-8B (median 3138), i.e. this model
    # actually deliberates on these cases. Floor set ~1.5x median.
    "qwen36": 12_900,
    # gemma4_31b: CALIBRATED 2026-07-19 against 384 real completions (192 each
    # at temp 0.7 and 0.9; job 123817, calibrate_gemma4_31b_thresholds.sbatch).
    # google/gemma-4-31B-it truncates 0.0% of the time at either temperature
    # (longest completion 6897 of 8192 tokens = 84% of cap), NEVER closes with
    # the <channel|> marker (0/384 -- unlike the 12B sibling's rare ~1%, this
    # is a full 0%, so _post_reasoning_region's whole-text fallback is load-
    # bearing here, not a rare edge case), and emits <final_answer> 100% of
    # the time (98.4-99.0% fully well-formed).
    #
    # Reasoning chars are ~14x LONGER than the 12B family (median 722 there
    # vs ~10.0-10.3k here) -- a genuinely new finding, not an assumption; the
    # inherited 1_500/5_000 placeholders would have fired on essentially every
    # completion. Using the temp=0.9 distribution (the intended launch
    # temperature): median 10_048, p75 13_536, p90 15_823, p99 21_232, max
    # 25_663 (temp=0.7 numbers were nearly identical: median 10_328, p90
    # 16_542). Floor set ~1.5x median, matching every other family's rule.
    "gemma4_31b": 15_100,
    # nemotron3_nano_omni: CALIBRATED 2026-07-20 against 192 real completions
    # at temp=0.6 (job 128206, calibrate_nemotron3_nano_omni_thresholds.sbatch).
    # Truncates 5.2% of the time (highest of any family calibrated so far --
    # every other family was 0.0%), closes </think> 94.8%, emits
    # <final_answer> 96.4% (94.8% fully well-formed). Reasoning chars are
    # extremely right-skewed: median 3280, p75 6929, p90 15458, p99 39534,
    # max 39578 -- a ~12x median-to-p99 spread, the widest of any family here,
    # almost certainly what drives the truncation rate. Floor set ~1.5x
    # median, matching every other family's rule.
    "nemotron3_nano_omni": 4_900,
    # openbiollm/medreason/glm52: NOT CALIBRATED -- eval-only reference
    # models, no training planned, no reward shaping ever applied to them.
    # These placeholders (gemma's values for the two whole-text-fallback
    # families, qwen36's for glm52 which genuinely closes </think>) exist
    # only to satisfy the eager dict lookup at import time; they are dead
    # code for the base-only CI eval path.
    "openbiollm": 1_500,
    "medreason": 1_500,
    "glm52": 12_900,
}
_THINKING_CHAR_LINEAR_END_BY_FAMILY = {
    "qwen": 59_200,  # past p90 (29047); CALIBRATED 2026-07-23, see NO_PENALTY comment
    "gemma": 5_000,
    "huatuo": 20_000,
    "deepseek_llama": 11_700,  # past p90 (5866); CALIBRATED, see comment above
    "qwen36": 25_800,  # past p90 (12493); CALIBRATED, see comment above
    "gemma4_31b": 31_600,  # past p90 (15823) at temp=0.9; CALIBRATED, see NO_PENALTY comment
    "nemotron3_nano_omni": 30_900,  # past p90 (15458), reaching toward p99 (39534); CALIBRATED, see NO_PENALTY comment
    # openbiollm/medreason/glm52: NOT CALIBRATED -- see NO_PENALTY comment above.
    "openbiollm": 5_000,
    "medreason": 5_000,
    "glm52": 25_800,
}
THINKING_CHAR_NO_PENALTY = _THINKING_CHAR_NO_PENALTY_BY_FAMILY[MODEL_FAMILY]
THINKING_CHAR_LINEAR_END = _THINKING_CHAR_LINEAR_END_BY_FAMILY[MODEL_FAMILY]
# Judge raw weighted sum is in [0, MAX_REWARD]; both compute_reward(normalize=True)
# and evaluate_answer["judge_score"] divide by MAX_REWARD -> [0, 1].
#
# Per-family: once a completion closes with <final_answer>, its reasoning
# length sits well under either family's NO_PENALTY floor (Qwen: ~21k median
# vs 20k floor is the exception, not the rule; Gemma: ~722-798 median vs
# 1500 floor), so in practice this cap is almost never reached by genuinely
# verbose-but-successful reasoning -- it's reached almost exclusively by the
# whole-text fallback for completions that never produce an answer at all
# (measured: 16.3% of all Gemma completions in job 100424, falling back to
# ~30k raw chars, past even the old 30k linear-end). That population is a
# distinct, more severe failure (no diagnosis produced) than "reasoned a bit
# long," so it deserves a bigger penalty than the shared 0.05 -- raised to
# 0.15 for Gemma (3x) to make that failure mode matter more to the gradient;
# Qwen's 0.05 is unchanged (validated, and Qwen reliably closes </think> so
# this population is much smaller there).
_THINKING_PENALTY_FRACTION_BY_FAMILY = {
    "qwen": 0.05,
    "gemma": 0.15,
    "huatuo": 0.05,  # inert unless GRPO_ENABLE_REWARD_SHAPING=1; see note above
    "deepseek_llama": 0.05,  # matches huatuo/qwen's rationale (near-100% close/answer rate)
    "qwen36": 0.05,  # reuses qwen's validated value; inert by default
    "gemma4_31b": 0.15,  # inherited from "gemma" (12B); inert unless GRPO_ENABLE_REWARD_SHAPING=1
    "nemotron3_nano_omni": 0.05,  # matches qwen36's rationale (real </think> tokens); inert by default
    # openbiollm/medreason/glm52: eval-only, no training -- inert placeholder.
    "openbiollm": 0.15,
    "medreason": 0.15,
    "glm52": 0.05,
}
# Experiment toggle (2026-07-12): the thinking-length penalty and format bonus
# were the levers that drove truncation-avoidance. With truncation already
# solved (ckpt-600 warm start) both are near-inert on finished completions --
# the format bonus lands on ~87% of them so it is a near-constant that cancels
# in the GRPO within-group advantage, and the penalty fires on <3% -- so they
# are DISABLED here to make reward == pure judge_score for the reasoning-focused
# resume. Re-enable (restoring the per-family penalty and the 0.05 bonus) by
# launching with GRPO_ENABLE_REWARD_SHAPING=1. See project-gemma-v6-reward-ceiling.
_ENABLE_REWARD_SHAPING = os.environ.get("GRPO_ENABLE_REWARD_SHAPING", "0") == "1"

# max subtraction from normalized judge score (0.0 when shaping disabled)
THINKING_PENALTY_FRACTION = (
    _THINKING_PENALTY_FRACTION_BY_FAMILY[MODEL_FAMILY] if _ENABLE_REWARD_SHAPING else 0.0
)

# Additive bonus when the completion follows the canonical
# reasoning-close-tag + <final_answer>...</final_answer> structure that
# ext_val also uses. Aligns training reward with eval format so the policy
# actually learns to emit the tag the eval parser looks for.
FORMAT_BONUS = 0.05 if _ENABLE_REWARD_SHAPING else 0.0

# --- judge ---
JUDGE_MAX_WORKERS = 100
JUDGE_MODEL = "deepseek-v4-flash"
JUDGE_BASE_URL = "https://api.deepseek.com"
# Was 600, sized for "deepseek-chat" (non-reasoning). After that model name
# was deprecated by the provider mid-run (job 133954 -> 137094 resume) and
# swapped to "deepseek-v4-pro", judge calls started returning EMPTY content
# with judge_api_status="parse_error" -- confirmed via a live test call:
# deepseek-v4-pro is a genuine reasoning model that spends real tokens on
# hidden chain-of-thought (returned separately as reasoning_content,
# counted against the same max_tokens budget as visible content -- a simple
# binary judgment already used 171 reasoning tokens). JUDGE_PROMPT's real
# 4-criteria rubric (a reason sentence + score per criterion) needs far more
# budget than that simple test, and 600 total left ~zero room for the actual
# JSON after reasoning -- every rubric judge call in the resumed run scored
# 0.0 via parse_error until this was raised.
JUDGE_MAX_TOKENS = 6000
JUDGE_TEMPERATURE = 0.1
JUDGE_RETRY_ATTEMPTS = 5

# --- vLLM ---
# Override with GRPO_USE_VLLM=0 to fall back to plain HF .generate() for
# rollouts.
USE_VLLM = os.environ.get("GRPO_USE_VLLM", "1").lower() not in ("0", "false", "no")

if MODEL_FAMILY in ("gemma", "gemma4_31b") and USE_VLLM:
    # vLLM's Gemma4UnifiedForConditionalGeneration weight loader can't map
    # HF's `model.embed_vision.*` parameter names (transformers nests both the
    # patch embedder and the projection under one `embed_vision` submodule)
    # onto vLLM's own reimplementation, which expects a separate
    # `vision_embedder.*` prefix for the patch-embedder weights -- confirmed
    # version-independent (vLLM 0.24.0 stable, nightly, and current git main
    # all raise the identical "no module or parameter named
    # 'embed_vision.pos_embedding'" ValueError) during TRL's periodic weight
    # re-sync (trl.generation.vllm_generation.VLLMGeneration._push_param_to_vllm).
    #
    # This is safe to skip entirely for this run: our data is text-only (no
    # images ever reach vLLM's generate() calls, so the vision path is dead
    # code), and our LoRA target_modules never touch embed_vision, so those
    # weights are frozen at their original checkpoint values the whole run --
    # values vLLM already loaded correctly at engine construction time, before
    # TRL's buggy re-sync ever runs. Verified end-to-end (0 failures pushing
    # all 619 non-vision params, generation still produces correct output)
    # against this exact environment (vLLM 0.24.0, TRL 1.7.0) in job 82020.
    from trl.generation.vllm_generation import VLLMGeneration

    _VISION_SKIP_PREFIXES = ("embed_vision.", "vision_embedder.")
    _orig_push_param_to_vllm = VLLMGeneration._push_param_to_vllm

    def _push_param_to_vllm_skip_vision(self, name: str, param) -> None:
        if name.removeprefix("model.").startswith(_VISION_SKIP_PREFIXES):
            return
        return _orig_push_param_to_vllm(self, name, param)

    VLLMGeneration._push_param_to_vllm = _push_param_to_vllm_skip_vision

if MODEL_FAMILY == "nemotron3_nano_omni" and USE_VLLM:
    # Same failure shape as the gemma4_31b vision-embedder skip above: TRL's
    # periodic weight re-sync (VLLMGeneration.sync_weights ->
    # _push_param_to_vllm) iterates model.named_parameters() and pushes EVERY
    # parameter, one at a time, via vllm.model_executor.models.nemotron_h's
    # load_weights([(name, param)]). For the 23 MoE layers, HF's live
    # NemotronHMoE module holds each proj as ONE fused/batched nn.Parameter
    # per layer (named e.g. "backbone.layers.1.mixer.experts.up_proj", no
    # per-expert index, no ".weight" suffix -- state_dict()/load_state_dict()
    # transparently expand/collapse this to the per-expert-indexed
    # checkpoint-format keys "experts.{0..127}.up_proj.weight" via custom
    # hooks, which is why our extracted checkpoint's safetensors DOES have
    # per-index keys and the initial full-checkpoint vLLM load -- which
    # consumes those exact indexed keys -- succeeds cleanly). vLLM's own
    # FusedMoE weight_loader only recognizes that per-expert-indexed pattern;
    # the bare fused name has no match, so the one-parameter-at-a-time push
    # hard KeyErrors (confirmed job 128350: 'layers.1.mixer.experts.up_proj').
    #
    # Safe to skip entirely: our LoRA target_modules (see the
    # MODEL_FAMILY == "nemotron3_nano_omni" branch below) never touch
    # .experts. at all -- all 2967 MoE expert modules are frozen for the
    # whole run, so vLLM's copy (already correct from the initial full
    # checkpoint load, before this buggy re-sync path ever runs) never needs
    # updating. Only attention (6 layers) + Mamba2 (23 layers) parameters
    # actually change during training and need re-syncing.
    from trl.generation.vllm_generation import VLLMGeneration

    _orig_push_param_to_vllm_nemotron = VLLMGeneration._push_param_to_vllm

    def _push_param_to_vllm_skip_moe_experts(self, name: str, param) -> None:
        if ".experts." in name:
            return
        return _orig_push_param_to_vllm_nemotron(self, name, param)

    VLLMGeneration._push_param_to_vllm = _push_param_to_vllm_skip_moe_experts

if MODEL_FAMILY == "nemotron3_nano_omni":
    # Fix for job 131210's crash on the FIRST-EVER GRPO_RESUME=<checkpoint>
    # attempt for this family (every prior run used GRPO_RESUME=none, so this
    # code path was never previously exercised): Trainer._load_from_checkpoint
    # -> model.load_adapter() -> peft's set_peft_model_state_dict ->
    # convert_peft_adapter_state_dict_for_transformers -> build_peft_weight_
    # mapping hard-crashes with "TypeError: WeightConverter.__init__() got an
    # unexpected keyword argument 'distributed_operation'" while building the
    # MoE-aware weight-conversion mapping (this code path is specifically for
    # MoE architectures, see peft's _MOE_TARGET_MODULE_MAPPING) -- a genuine
    # version-skew bug between the installed peft (0.19.1) and transformers
    # (5.12.1): peft constructs transformers' own WeightConverter with a
    # distributed_operation= kwarg that this transformers release's
    # WeightConverter.__init__ no longer/doesn't accept. Not related to the
    # KL-clip/beta guardrail change -- purely a dependency incompatibility
    # that only surfaces on adapter-checkpoint resume for MoE models.
    #
    # Fix: wrap WeightConverter.__init__ to silently drop any kwarg the real
    # constructor doesn't accept (inspected from its own signature), rather
    # than hardcoding "distributed_operation" specifically -- robust to
    # whatever the actual mismatch turns out to be across dependency versions.
    import inspect

    from transformers.core_model_loading import WeightConverter

    _orig_weight_converter_init = WeightConverter.__init__
    _wc_valid_params = set(inspect.signature(_orig_weight_converter_init).parameters) - {"self"}

    def _weight_converter_init_compat(self, *args, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in _wc_valid_params}
        return _orig_weight_converter_init(self, *args, **filtered)

    WeightConverter.__init__ = _weight_converter_init_compat

if MODEL_FAMILY in ("gemma", "gemma4_31b", "nemotron3_nano_omni"):
    # Clip the log-ratio feeding the KL exponential term to prevent rare
    # single-token outliers from producing catastrophic gradients.
    #
    # Added nemotron3_nano_omni 2026-07-21: the smoke test's first-ever real
    # training step (job 128815, 3-step 4-GPU smoke test, all three earlier
    # blockers already fixed) reported kl=3.887 -- ~8x the ABORT-EARLY
    # threshold (0.5) used for every family here, and the GRPO_KL_LOGRATIO_
    # CLIP=8 env var set in the launch script had ZERO effect, because this
    # whole patch block was gated to MODEL_FAMILY in ("gemma", "gemma4_31b")
    # only -- confirmed via the LIVE CONFIG log line printing "KL_LOGRATIO_
    # CLIP=n/a (qwen: unpatched)" even with the env var set. This patch
    # mechanism (source-inspection based, not architecture-specific) applies
    # equally well to any family using the standard non-liger K3 estimator,
    # so extending the gate is the correct fix -- not a change specific to
    # this architecture, just the first time a non-gemma family has actually
    # shown the failure mode this guards against.
    #
    # per_token_kl = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1
    # (the K3/Schulman estimator) is exponential in the logp gap. Debug
    # instrumentation (GRPO_DEBUG_KL, job 82044) found this gap reaching ~14
    # nats for a single token (the contraction "'s", where the disable_adapter()
    # reference is near-certain -- ref_logp close to 0 -- but the barely-
    # LoRA-adapted policy assigns it meaningfully lower probability),
    # producing a per-token KL of ~880k that still diluted into a healthy
    # batch mean. In the real run (job 82046) this recurred at higher
    # severity: 4 of the first 8 steps hit grad_norm in the 1e5-1e16 range
    # (kl up to 7.4e13), consistent with a gap of ~30-40 nats on some token
    # (exp(35) ~ 1.6e15) -- a single outlier token's gradient, once that
    # large, can dominate the entire batch's gradient norm even after the
    # advantage/policy-gradient term is well-behaved. This is a well-known
    # numerical failure mode of the K3 KL estimator with no built-in mitigation
    # exposed by TRL's GRPOConfig (epsilon_high/low and delta only clip the
    # policy-gradient importance ratio, not the KL term). Clamping the
    # logp gap to [-KL_LOGRATIO_CLIP, KL_LOGRATIO_CLIP] bounds per_token_kl to
    # exp(KL_LOGRATIO_CLIP) at worst (~4.9e8 at the default 20) -- large
    # enough to never affect a normal token (typical gaps here are 0-3 nats),
    # but eliminates the 1e13-1e16 blowups outright.
    #
    # 2026-07-06: the default of 20 turned out to only catch the most
    # catastrophic tail. Job 82280's recurring batch-KL spikes (49/74/151)
    # were driven by per-token gaps of 8-23 nats -- comfortably under a
    # 20-nat clip, so it never engaged on them. Set GRPO_KL_LOGRATIO_CLIP=8
    # (exp(8) ~ 2981 ceiling) for the Gemma relaunch so it actually bounds
    # the tokens observed causing the blowups; left at the 20 default for
    # Qwen, which never showed this failure mode.
    #
    # Applied via inspect.getsource + a targeted string substitution on the
    # ACTUALLY installed trl.trainer.grpo_trainer.GRPOTrainer._compute_loss
    # source (not a manually retyped copy), so it can never silently drift
    # from the real method -- if the expected expression isn't found (e.g.
    # after a TRL upgrade changes this code), it raises immediately instead
    # of applying a stale patch.
    #
    # NOTE: this block must run BEFORE the GRPO_DEBUG_KL block below, so the
    # clip patches the TRUE original TRL method. If GRPO_DEBUG_KL ran first,
    # inspect.getsource would retrieve the debug wrapper's source instead
    # (which happens to contain the same literal KL expression for its own
    # diagnostic recompute), and the clip would silently patch that wrapper
    # rather than the real loss computation -- leaving the actual training
    # loss unclipped while only the diagnostic copy got bounded.
    import inspect
    from trl.trainer.grpo_trainer import GRPOTrainer

    KL_LOGRATIO_CLIP = float(os.environ.get("GRPO_KL_LOGRATIO_CLIP", "20"))

    _ORIG_KL_EXPR = (
        "torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1"
    )
    _CLIPPED_KL_EXPR = (
        "torch.exp((ref_per_token_logps - per_token_logps).clamp(-_KL_LOGRATIO_CLIP, _KL_LOGRATIO_CLIP)) "
        "- (ref_per_token_logps - per_token_logps).clamp(-_KL_LOGRATIO_CLIP, _KL_LOGRATIO_CLIP) - 1"
    )

    _source = inspect.getsource(GRPOTrainer._compute_loss)
    if _ORIG_KL_EXPR not in _source:
        raise RuntimeError(
            "GRPO_KL_LOGRATIO clip patch: expected per_token_kl expression not found in the "
            "installed trl.trainer.grpo_trainer.GRPOTrainer._compute_loss source -- TRL's "
            "internals have likely changed since this patch was written. Refusing to apply a "
            "stale patch; update _ORIG_KL_EXPR/_CLIPPED_KL_EXPR in grpo_training_new_code.py "
            "to match the new source before proceeding."
        )
    _patched_source = _source.replace(_ORIG_KL_EXPR, _CLIPPED_KL_EXPR, 1)
    # Original method is indented as a class member (starts with leading
    # whitespace); dedent so it compiles as a standalone def, then re-indent
    # implicitly isn't needed since exec just needs a valid top-level def.
    _patched_source = textwrap.dedent(_patched_source)
    _patch_namespace = dict(GRPOTrainer._compute_loss.__globals__)
    _patch_namespace["_KL_LOGRATIO_CLIP"] = KL_LOGRATIO_CLIP
    exec(compile(_patched_source, "<grpo_kl_clip_patch>", "exec"), _patch_namespace)
    GRPOTrainer._compute_loss = _patch_namespace["_compute_loss"]

    # use_liger_kernel=True bypasses _compute_loss entirely (TRL's compute_loss
    # branches to compute_liger_loss -> LigerFusedLinearGRPOLoss instead), so
    # the clip patch above silently does nothing under Liger -- discovered the
    # hard way on job 126700 (v6-liger): step 1 reported kl=4.126, ~8x the
    # ABORT-EARLY threshold, because liger-kernel's own k3_loss_fn(log_p,
    # log_q) = exp(log_p-log_q) - (log_p-log_q) - 1.0 has NO clamp at all (not
    # even the insufficient 20-nat default our own history already found too
    # loose -- there's flat out nothing). Patch it the same way, reusing
    # KL_LOGRATIO_CLIP from above so both paths stay numerically consistent.
    # Verified empirically (not just from source structure) that a plain
    # module-attribute reassignment survives liger's compiled=True path:
    # patched with an obvious +1000 offset, called with both compiled=True and
    # compiled=False, got the identical inflated kl_metric (6747.26) either
    # way -- torch.compile traces the patched global at first call since we
    # patch before the trainer/loss module is ever constructed, so there's no
    # stale compiled graph to fight.
    if os.environ.get("GRPO_USE_LIGER_KERNEL", "0").lower() in ("1", "true", "yes"):
        import torch
        import liger_kernel.chunked_loss.grpo_loss as _liger_grpo_loss_mod

        def _clipped_liger_k3(log_p, log_q, _clip=KL_LOGRATIO_CLIP):
            gap = (log_p - log_q).clamp(-_clip, _clip)
            return torch.exp(gap) - gap - 1.0

        _liger_grpo_loss_mod.k3_loss_fn = _clipped_liger_k3
        logger.info(
            "Patched liger_kernel.chunked_loss.grpo_loss.k3_loss_fn with "
            "KL_LOGRATIO_CLIP=%s (unpatched liger has NO clamp on this term)",
            KL_LOGRATIO_CLIP,
        )


# --- multi-EOS termination fix (NOT Gemma-only: also required for huatuo) -----
# 2026-07-06: fixes completions/clipped_ratio and mask_truncated_completions for
# models with a multi-token EOS scheme. Both TRL 1.7.0 call sites decide whether
# a completion "terminated" with:
#   eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
#   is_truncated = ids[-1] not in eos_and_pad
# `self._tokenizer.eos_token_id` is a single scalar. Any model whose chat
# template ends assistant turns with a DIFFERENT token than tokenizer.eos_token
# therefore has ~100% of its completions misflagged as truncated.
#
# Gemma evidence (job 82986, mask_truncated_completions=True): a 973-char
# completion that cleanly ended with <final_answer>...</final_answer> and a
# genuinely-degenerate 35,772-char non-terminating "let me reconsider" loop were
# BOTH flagged truncated -- neither ends in raw token 1; the well-formed one ends
# in 106. Net effect: completions/clipped_ratio read a flat 1.0 on every batch
# (job 82280 steps 5-400, job 82986 steps 1-2), and enabling
# mask_truncated_completions zeroed ~100% of every batch's loss (loss/grad_norm/
# kl/entropy pinned at 0). See GEMMA4_KL_ENTROPY_INSTABILITY_FIX.md.
#
# Fix: patch both call sites (_generate, which logs the metric, and
# _generate_and_score_completions, which does the actual masking) to check
# membership in the model's REAL eos id list instead of the tokenizer's single
# scalar. The ids are hardcoded per family rather than re-read from
# self.model.generation_config at patch time, because by then transformers has
# already collapsed generation_config.eos_token_id down to match the tokenizer's
# scalar (see the "tokenizer has new PAD/BOS/EOS tokens... aligned" warning in
# the job logs) -- reading it live would just observe the same collapsed value.
#
# Families needing the fix, and their REAL terminator ids:
#
#   gemma  (1, 106, 50)          <eos>, <turn|>, <|tool_response>
#   huatuo (151645, 151643)      <|im_end|>, <|endoftext|>
#            HuatuoGPT-3-8B inherited its tokenizer from Qwen3-8B-*Base*, so
#            tokenizer.eos_token is <|endoftext|> (151643) -- but the chat-tuned
#            model ends every assistant turn with <|im_end|> (151645), and its
#            generation_config.json declares eos_token_id=[151645, 151643].
#            Without this fix TRL would flag ~100% of completions as truncated
#            (clipped_ratio pinned at 1.0), and mask_truncated_completions=True
#            would zero out every batch -- exactly the Gemma failure.
#
#   qwen   -- NOT affected, deliberately absent: Qwen3.5-9B's tokenizer
#            eos_token IS <|im_end|>, so TRL's scalar is already correct. Do not
#            add it here.
_EOS_IDS_BY_FAMILY = {
    "gemma": (1, 106, 50),
    "huatuo": (151645, 151643),
    # gemma4_31b: CONFIRMED via download_gemma4_31b.sbatch (job 123816) --
    # google/gemma-4-31B-it's generation_config.json declares eos_token_id:
    # [1, 106] (<eos>, <turn|>) -- note this is TWO ids, not three: unlike the
    # 12B sibling, this checkpoint's generation_config does NOT list
    # <|tool_response> (id 50) as a terminator, even though that token exists
    # in the shared vocab. Using the checkpoint's own declared set rather than
    # assuming the 12B tuple transfers unchanged.
    "gemma4_31b": (1, 106),
    # nemotron3_nano_omni: CONFIRMED via download_nemotron3_nano_omni.sbatch
    # (job 128133) -- generation_config.json declares eos_token_id: [2, 11]
    # (</s>, <|im_end|>), but the tokenizer's own eos_token_id is 11 alone.
    # Same multi-EOS-terminator pattern as gemma/huatuo/gemma4_31b.
    "nemotron3_nano_omni": (2, 11),
}

if MODEL_FAMILY in _EOS_IDS_BY_FAMILY:
    import inspect
    from trl.trainer.grpo_trainer import GRPOTrainer

    _REAL_EOS_IDS = _EOS_IDS_BY_FAMILY[MODEL_FAMILY]

    _ORIG_EOS_AND_PAD = "eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]"
    _FIXED_EOS_AND_PAD = "eos_and_pad = list(_REAL_EOS_IDS) + [self._tokenizer.pad_token_id]"

    for _method_name in ("_generate", "_generate_and_score_completions"):
        _orig_method = getattr(GRPOTrainer, _method_name)
        _eos_source = inspect.getsource(_orig_method)
        if _eos_source.count(_ORIG_EOS_AND_PAD) != 1:
            raise RuntimeError(
                f"GRPO_EOS_FIX patch: expected exactly one occurrence of the eos_and_pad "
                f"expression in the installed trl.trainer.grpo_trainer.GRPOTrainer.{_method_name} "
                f"source (found {_eos_source.count(_ORIG_EOS_AND_PAD)}) -- TRL's internals have "
                f"likely changed since this patch was written. Refusing to apply a stale patch; "
                f"update _ORIG_EOS_AND_PAD in grpo_training_new_code.py to match the new source "
                f"before proceeding."
            )
        _patched_eos_source = _eos_source.replace(_ORIG_EOS_AND_PAD, _FIXED_EOS_AND_PAD, 1)
        _patched_eos_source = textwrap.dedent(_patched_eos_source)
        _eos_patch_namespace = dict(_orig_method.__globals__)
        _eos_patch_namespace["_REAL_EOS_IDS"] = _REAL_EOS_IDS
        exec(compile(_patched_eos_source, f"<grpo_eos_fix_{MODEL_FAMILY}_{_method_name}>", "exec"), _eos_patch_namespace)
        setattr(GRPOTrainer, _method_name, _eos_patch_namespace[_method_name])

    logger.info(
        "Applied multi-EOS termination fix for family %r: eos ids %s",
        MODEL_FAMILY, _REAL_EOS_IDS,
    )

if os.environ.get("GRPO_DEBUG_KL", "0").lower() in ("1", "true", "yes"):
    # Diagnostic instrumentation for an anomalously large per-token KL
    # observed on Gemma (e.g. job 82040: kl=2.4e8 on step 1, vs Qwen/smoke-test
    # values in the single digits). per_token_kl = exp(ref_logp - policy_logp)
    # - (ref_logp - policy_logp) - 1 is exponential in the logp gap, so a
    # single token where the LoRA-enabled policy and the disable_adapter()
    # reference disagree sharply can dominate the batch mean. This wraps
    # GRPOTrainer._compute_loss (already KL-clip-patched above, if Gemma) to
    # recompute an UNCLIPPED per_token_logps/per_token_kl in a no_grad pass
    # (reusing the same ref_per_token_logps already computed for the real
    # loss) purely to find and print the worst-offending token: its position,
    # decoded text, surrounding context, and both logp values. This
    # diagnostic recompute is intentionally never clipped, so it keeps
    # reporting the true, raw divergence even after the clip patch is
    # protecting the real training loss -- lets us confirm the clip is
    # actually engaging on outlier tokens. Does not alter the real
    # loss/gradient in any way -- diagnostic only.
    import torch
    from trl.trainer.grpo_trainer import GRPOTrainer

    _orig_compute_loss = GRPOTrainer._compute_loss

    def _compute_loss_debug_kl(self, model, inputs):
        loss = _orig_compute_loss(self, model, inputs)
        try:
            if self.beta == 0.0 or "ref_per_token_logps" not in inputs:
                return loss
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            with torch.no_grad():
                per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
                )
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            masked_kl = per_token_kl * completion_mask
            flat_idx = masked_kl.reshape(-1).argmax().item()
            b, t = divmod(flat_idx, masked_kl.shape[1])
            max_val = masked_kl[b, t].item()
            if max_val > 50:
                tok_id = completion_ids[b, t].item()
                tok_str = self.processing_class.decode([tok_id])
                lo, hi = max(0, t - 15), min(completion_ids.shape[1], t + 5)
                window = self.processing_class.decode(completion_ids[b, lo:hi].tolist())
                print(
                    f"[KL-DEBUG] max_per_token_kl={max_val:.2f} (unclipped, diagnostic-only) "
                    f"sample={b} pos={t} "
                    f"token_id={tok_id} token={tok_str!r} "
                    f"ref_logp={ref_per_token_logps[b, t].item():.4f} "
                    f"policy_logp={per_token_logps[b, t].item():.4f} "
                    f"context={window!r}",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001 -- diagnostic only, never break training
            print(f"[KL-DEBUG] instrumentation failed: {e}", flush=True)
        return loss

    GRPOTrainer._compute_loss = _compute_loss_debug_kl

if os.environ.get("GRPO_MEM_PROBE", "0").lower() in ("1", "true", "yes"):
    # Diagnostic instrumentation to find which phase of the GRPO step owns
    # the memory gap seen on gemma4_31b (job 126169, v5-lorafix): a synthetic
    # single-sequence fwd+bwd probe (_probe_gemma4_31b_gc_memory.py, job
    # 126623) peaked at 76.6 GiB (58.25 weights + 16.45 activations), but the
    # real run OOM'd at step ~85 with 170 GiB in use and only 8 GiB free --
    # ~94 GiB unaccounted for. Suspect: TRL batches logprob computation
    # (old/ref) over the full generation group (num_generations=8) rather
    # than per-sequence, materializing full-vocab (262,144) logits for the
    # whole group at once. Brackets three call boundaries with CUDA memory
    # readings so the jump can be attributed to a specific phase instead of
    # inferred. Diagnostic only -- never alters values used by training.
    import torch
    from trl.trainer.grpo_trainer import GRPOTrainer

    _MEM_PROBE_MAX_STEPS = int(os.environ.get("GRPO_MEM_PROBE_MAX_STEPS", "30"))
    _mem_probe_state = {"step": 0}

    def _mem_gib(n):
        return n / 2**30

    def _mem_log(tag):
        rank = os.environ.get("LOCAL_RANK", "?")
        alloc = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        print(
            f"[MEM-PROBE] step={_mem_probe_state['step']} rank={rank} {tag}: "
            f"allocated={_mem_gib(alloc):.2f}GiB peak_so_far={_mem_gib(peak):.2f}GiB",
            flush=True,
        )

    _orig_generate_and_score_mem = GRPOTrainer._generate_and_score_completions
    _orig_compute_loss_mem = GRPOTrainer._compute_loss
    _orig_training_step_mem = GRPOTrainer.training_step

    def _generate_and_score_mem(self, *args, **kwargs):
        active = _mem_probe_state["step"] < _MEM_PROBE_MAX_STEPS
        if active:
            _mem_log("before generate_and_score_completions")
        result = _orig_generate_and_score_mem(self, *args, **kwargs)
        if active:
            _mem_log("after  generate_and_score_completions")
        return result

    def _compute_loss_mem(self, model, inputs):
        active = _mem_probe_state["step"] < _MEM_PROBE_MAX_STEPS
        if active:
            _mem_log("before compute_loss (policy fwd)")
        loss = _orig_compute_loss_mem(self, model, inputs)
        if active:
            _mem_log("after  compute_loss (policy fwd, pre-backward)")
        return loss

    def _training_step_mem(self, *args, **kwargs):
        active = _mem_probe_state["step"] < _MEM_PROBE_MAX_STEPS
        if active:
            torch.cuda.reset_peak_memory_stats()
            _mem_log("=== training_step start ===")
        result = _orig_training_step_mem(self, *args, **kwargs)
        if active:
            _mem_log("=== training_step end (incl. backward) ===")
            _mem_probe_state["step"] += 1
        return result

    GRPOTrainer._generate_and_score_completions = _generate_and_score_mem
    GRPOTrainer._compute_loss = _compute_loss_mem
    GRPOTrainer.training_step = _training_step_mem

    logger.info("Applied GRPO_MEM_PROBE instrumentation (max %d steps)", _MEM_PROBE_MAX_STEPS)

VLLM_SERVER_HOST = None  # e.g. "127.0.0.1" for remote server
VLLM_SERVER_PORT = 8000
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("GRPO_VLLM_GPU_MEMORY_UTILIZATION", "0.2"))
VLLM_MAX_MODEL_LENGTH = 40000

# =============================================================================
# Policy prompts — case-completion (training + val eval)
# =============================================================================

_CLINICAL_RULES = """\
You are a clinical reasoning assistant. You will be shown a portion of a
patient case and a question whose answer is determined by what comes next
in that case — typically the most appropriate next test, treatment,
diagnosis, finding, or clinical decision given the information available
so far.

Follow these rules when answering:
- Reason only from the information stated in the case. Do not invent
  findings, labs, history, or details that are not provided.
- Be as clinically specific as the case supports. Name the specific test,
  drug, anatomical site, or action rather than a vague category — for
  example, "CT chest with IV contrast" rather than "imaging," or
  "discontinue heparin and start warfarin" rather than "adjust
  anticoagulation."
- If the correct answer has multiple components, include all of them.
"""

# Qwen: instruct it to wrap reasoning in <think>...</think> itself (matches
# its own native tag names).
QWEN_SYSTEM_PROMPT = _CLINICAL_RULES + """\
- Think through the case briefly inside <think>...</think>, then output
  your final answer wrapped in <final_answer>...</final_answer> tags.
  Example final answer: <final_answer>CT chest with IV contrast</final_answer>
"""

# Gemma: reasoning is already wrapped in its own "thought" channel
# automatically once enable_thinking=True is set (see MODEL_FAMILY comment
# above) -- no need to ask it to use tags it doesn't natively have.
GEMMA_SYSTEM_PROMPT = _CLINICAL_RULES + """\
- Think through the case briefly, then output your final answer wrapped in
  <final_answer>...</final_answer> tags.
  Example final answer: <final_answer>CT chest with IV contrast</final_answer>
"""

_SYSTEM_PROMPT_BY_FAMILY = {
    "qwen": QWEN_SYSTEM_PROMPT,
    "gemma": GEMMA_SYSTEM_PROMPT,
    # huatuo emits <think>...</think> natively (stock Qwen3 ChatML), so it gets
    # the same tag instructions as Qwen.
    "huatuo": QWEN_SYSTEM_PROMPT,
    # deepseek_llama: same pre-injected-opening-<think> template style as
    # Qwen3.5 (see MODEL_FAMILY comment above), so the same instructions apply
    # (the "wrap in <think>...</think>" phrasing is harmless even though the
    # opening tag is already in the prompt -- this exact pattern is already
    # validated working for qwen). NOTE: DeepSeek's own model card recommends
    # NOT using a system prompt at all (fold instructions into the user turn
    # instead); tested anyway via the system-role path (job 112153,
    # calibrate_deepseek_llama_thresholds.sbatch) and it works fine here --
    # 100% </think> close rate, 99.5% <final_answer> emission rate, so no
    # need to deviate from every other family's structure.
    "deepseek_llama": QWEN_SYSTEM_PROMPT,
    # qwen36 (Qwen3.6-27B): same ChatML template + model-emitted <think> as
    # Qwen3.5, so identical tag instructions.
    "qwen36": QWEN_SYSTEM_PROMPT,
    # gemma4_31b: same Gemma4 chat template thinking-channel behavior as
    # "gemma" (12B).
    "gemma4_31b": GEMMA_SYSTEM_PROMPT,
    # nemotron3_nano_omni: real model-emitted </think> tags (ChatML-style),
    # same as qwen/huatuo/deepseek_llama/qwen36 -- not gemma's channel-marker
    # convention.
    "nemotron3_nano_omni": QWEN_SYSTEM_PROMPT,
    # openbiollm/medreason: non-reasoning-tag models -- smoke tests
    # (_smoke_openbiollm_parse_v2.py, _smoke_medreason_parse.py) validated a
    # direct-answer prompt (no <think> instruction) + whole-completion parse
    # fallback, exactly Gemma's convention, so reuse GEMMA_SYSTEM_PROMPT
    # verbatim rather than inventing new wording.
    "openbiollm": GEMMA_SYSTEM_PROMPT,
    "medreason": GEMMA_SYSTEM_PROMPT,
    # glm52: reasons via its own native "Reasoning Effort: Max" template
    # behavior regardless of whether the system prompt asks for a <think>
    # tag (confirmed in smoke, job 134796: 8/8 emitted </think> unprompted),
    # so the direct-answer GEMMA_SYSTEM_PROMPT wording works identically to
    # the reasoning-style QWEN_SYSTEM_PROMPT here -- reuse it for consistency
    # with the other two new families.
    "glm52": GEMMA_SYSTEM_PROMPT,
}
SYSTEM_PROMPT = _SYSTEM_PROMPT_BY_FAMILY[MODEL_FAMILY]

USER_PROMPT_TEMPLATE = """\
Case presentation:
{context_chunk}

Question:
{question}
"""


def build_user_prompt(context_chunk: str, question: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        context_chunk=context_chunk.strip(),
        question=question.strip(),
    )


def build_messages(context_chunk: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(context_chunk, question)},
    ]


# =============================================================================
# Policy prompts — external final-diagnosis task (ext_val eval)
# =============================================================================

EXT_USER_PROMPT_TEMPLATE = (
    "Below is a case presentation of a patient. Predict the most likely "
    "final diagnosis. Think through the case, then provide your diagnosis "
    "in a single concise sentence as follows: "
    "<final_answer>Multiple sclerosis</final_answer> tags.\n\n"
    "Case presentation: {clean_text}"
)


def build_ext_messages(clean_text: str) -> list[dict]:
    return [
        {"role": "user", "content": EXT_USER_PROMPT_TEMPLATE.format(clean_text=clean_text.strip())},
    ]


# =============================================================================
# DeepSeek judges (rubric + diagnosis-equivalence)
# =============================================================================

# ---- Rubric judge for case completion -----------------------------------------

JUDGE_PROMPT = """\
You are an expert clinical educator grading a model's answer to a question
derived from a patient case. The model was shown a portion of the case and
asked a question whose answer is contained in the next portion of the case,
which is provided to you as the reference. Score 4 criteria on a 0/1/2
scale.

Case context shown to the model:
{context_chunk}

Question:
{question}

Model's answer:
{model_response}

Reference continuation (contains the true answer):
{next_chunk}

For each criterion, write one sentence of justification, then output
0, 1, or 2:
  0 = not met
  1 = partially met
  2 = fully met
Be strict — reserve 2 for clear, unambiguous satisfaction of the criterion.

1. CORRECT    The model's answer aligns with what actually happens in the
              reference continuation (same clinical action, finding, test,
              decision, or diagnosis; phrasing and synonyms do not matter).
              2 = same clinical content; 1 = related but not the same;
              0 = wrong or unrelated.
2. SPECIFIC   The answer matches the level of detail in the reference — not
              so vague that it could mean several different things, not
              over-specified beyond what the reference states. Example:
              reference is "CT chest with IV contrast" — "imaging" is too
              vague (0); "CT scan of the chest" is close (1); "CT chest
              with contrast" is exact (2).
3. GROUNDED   The model's answer is supported by information visible in the
              case context shown to it. It is not based on facts the model
              could not have known from that context, and it does not invent
              clinical findings, labs, or history. 2 = fully grounded; 1 =
              mostly grounded with minor unsupported extras; 0 = invents
              clinical facts.
4. COMPLETE   No essential component of the reference answer is missing.
              If the reference action has multiple parts (e.g., "stop drug X
              and start drug Y"), all parts must appear in the model's
              response. 2 = all parts present; 1 = some parts present; 0 =
              major parts missing.

Output ONLY this JSON, no other text:
{{
  "correct":  {{"reason": "...", "score": 0, 1, or 2}},
  "specific": {{"reason": "...", "score": 0, 1, or 2}},
  "grounded": {{"reason": "...", "score": 0, 1, or 2}},
  "complete": {{"reason": "...", "score": 0, 1, or 2}}
}}"""

CRITERION_WEIGHTS = {
    "correct": 3,
    "specific": 1,
    "grounded": 2,
    "complete": 1,
}
# Max per-criterion score is now 2 (was 1), so raw weighted sum ∈ [0, 2*7=14].
MAX_CRITERION_SCORE = 2
MAX_REWARD = MAX_CRITERION_SCORE * sum(CRITERION_WEIGHTS.values())  # 14

# ---- Diagnosis-equivalence judge for ext_val ----------------------------------

DIAGNOSIS_JUDGE_SYSTEM = "You are a clinical accuracy evaluator. Be concise but thorough."
DIAGNOSIS_JUDGE_USER_TEMPLATE = """\
You are a clinical expert evaluating the equivalence of two diagnoses. Decide whether the two diagnoses are the same or different, consider accuracy, semantic equivalence and clinical correctness.

Proposed diagnosis: {pred}
Reference diagnosis: {ref}

After careful analysis, conclude with JSON containing:
"judgment": 1 if the diagnosis is correct, 0 if incorrect"""

# ---- Shared OpenAI client (OpenAI SDK is thread-safe) -------------------------

_judge_client: OpenAI | None = None


def _get_judge_client() -> OpenAI:
    global _judge_client
    if _judge_client is None:
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. Export it or add to .env for judge rewards."
            )
        _judge_client = OpenAI(api_key=api_key, base_url=JUDGE_BASE_URL)
    return _judge_client


# ---- Judge predictions log (JSONL, one line per judge call) -------------------
# One file per process rank so concurrent accelerate ranks don't interleave
# writes to the same file. Opened lazily so OUTPUT_DIR/GRPO_RUN_TS are read
# at call time (both may be set by the launcher after import).

_judge_pred_lock = threading.Lock()
_judge_pred_path: Path | None = None


def _get_judge_predictions_path() -> Path:
    global _judge_pred_path
    if _judge_pred_path is None:
        pred_dir = Path(OUTPUT_DIR) / "judge_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        ts = os.environ.get("GRPO_RUN_TS") or datetime.now().strftime("%Y%m%d_%H%M%S")
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        _judge_pred_path = pred_dir / f"judge_predictions_{ts}_rank{rank}.jsonl"
    return _judge_pred_path


def log_judge_prediction(record: dict[str, Any]) -> None:
    """Append one judge call's prediction/score to this rank's JSONL file."""
    record = {"logged_at": datetime.now().isoformat(), **record}
    line = json.dumps(record, ensure_ascii=False)
    path = _get_judge_predictions_path()
    with _judge_pred_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---- Parsing helpers ----------------------------------------------------------


def _parse_judge_json(text: str) -> Optional[dict]:
    """Extract a JSON object from a model response.

    Handles raw JSON, markdown-fenced JSON, and JSON embedded in prose.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Markdown-fenced anywhere in text
    md_match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # Greedy braces (good for single-JSON-block responses with reasoning prose)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def compute_reward(
    judge_text: str,
    normalize: bool = True,
    on_parse_failure: float = 0.0,
) -> float:
    parsed = _parse_judge_json(judge_text)
    if parsed is None:
        return on_parse_failure

    total = 0
    for key, weight in CRITERION_WEIGHTS.items():
        if key not in parsed:
            return on_parse_failure
        entry = parsed[key]
        score = entry.get("score") if isinstance(entry, dict) else entry
        if score not in (0, 1, 2):
            return on_parse_failure
        total += weight * score

    return total / MAX_REWARD if normalize else float(total)


# ---- Judge response cache -----------------------------------------------------
# The rubric judge sees only (context_chunk, question, extracted_answer,
# next_chunk) -- the model's reasoning is never passed -- so an identical answer
# to an identical prompt is a byte-identical judge call whose score should be
# deterministic. It is not: at JUDGE_TEMPERATURE=0.1 the judge returns a
# different score ~44% of the time on identical input (mean spread 0.095), pure
# stochastic noise that GRPO turns into spurious within-group advantage. 78% of
# prompt-groups contain an exact-duplicate answer, so memoizing on the judge
# input gives every identical answer one frozen score (and cuts judge calls).
#
# The cache MUST be shared across ranks, not per-process. With
# per_device_train_batch_size=1 and num_generations=8 on 8 ranks, TRL hands each
# rank exactly ONE of a prompt's 8 generations -- so the duplicate answers within
# a group live in 8 DIFFERENT processes and a per-process dict never sees a
# single one of them (measured on job 106889: 183 judged completions per rank,
# 183 unique keys, zero hits, and 39% of groups still had an identical answer
# scored differently, mean spread 0.165 -- i.e. the noise was untouched).
#
# So the cache is backed by the shared filesystem, keyed by sha256 of the judge
# input. On a miss a rank takes an atomic mkdir claim on the key; the winner
# calls the judge and publishes the score, the losers block until it appears and
# reuse it. Only successful judgements are published -- parse errors and API
# exceptions stay uncached so they get another attempt. A crashed claim-holder
# just costs the waiters one timeout, after which they judge for themselves.
_JUDGE_CACHE: dict[str, dict] = {}
_JUDGE_CACHE_LOCK = threading.Lock()
_JUDGE_CACHE_STATS = {"hits": 0, "waits": 0, "misses": 0}

_JUDGE_CACHE_DIR = Path(
    os.environ.get("GRPO_JUDGE_CACHE_DIR")
    or os.path.join(os.environ.get("GRPO_OUTPUT_DIR", "."), "judge_cache")
)
# Generous: the judge itself takes seconds, but JUDGE_RETRY_ATTEMPTS of
# exponential backoff can stretch the claim-holder's call out much further.
_JUDGE_CACHE_WAIT_TIMEOUT = 180.0
_JUDGE_CACHE_POLL_INTERVAL = 0.5


def _judge_cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update((part or "").encode("utf-8"))
        h.update(b"\x00")  # unambiguous field delimiter
    return h.hexdigest()


def _judge_cache_paths(key: str) -> tuple[Path, Path]:
    # Shard on the first 2 hex chars: a flat dir would hit ~10^5 entries.
    shard = _JUDGE_CACHE_DIR / key[:2]
    return shard / f"{key}.json", shard / f"{key}.claim"


def _judge_cache_get(key: str) -> Optional[dict]:
    """Return a published score for `key`, from memory or the shared FS."""
    with _JUDGE_CACHE_LOCK:
        hit = _JUDGE_CACHE.get(key)
    if hit is not None:
        return hit

    score_path, _ = _judge_cache_paths(key)
    try:
        with open(score_path, encoding="utf-8") as f:
            result = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        # Torn/partial read shouldn't happen (publish is an atomic rename), but
        # if it does, treat it as a miss rather than poisoning the reward.
        logger.warning("Judge cache read failed for %s: %s", key[:12], e)
        return None

    with _JUDGE_CACHE_LOCK:
        _JUDGE_CACHE[key] = result
    return result


def _judge_cache_claim(key: str) -> bool:
    """Try to become the rank that judges `key`. mkdir is atomic even on NFS."""
    _, claim_path = _judge_cache_paths(key)
    try:
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(claim_path)
        return True
    except FileExistsError:
        return False
    except OSError as e:
        # Filesystem trouble: degrade to the old behavior (judge it ourselves)
        # rather than block the training step.
        logger.warning("Judge cache claim failed for %s: %s", key[:12], e)
        return True


def _judge_cache_release(key: str) -> None:
    _, claim_path = _judge_cache_paths(key)
    try:
        os.rmdir(claim_path)
    except OSError:
        pass


def _judge_cache_wait(key: str) -> Optional[dict]:
    """Block until the claim-holder publishes `key`. None if it never does."""
    deadline = time.monotonic() + _JUDGE_CACHE_WAIT_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(_JUDGE_CACHE_POLL_INTERVAL)
        result = _judge_cache_get(key)
        if result is not None:
            return result
    logger.warning(
        "Judge cache: waited %.0fs for %s with no result; judging it directly",
        _JUDGE_CACHE_WAIT_TIMEOUT,
        key[:12],
    )
    return None


def _judge_cache_publish(key: str, result: dict) -> None:
    """Make `result` the frozen score for `key`, for every rank, atomically."""
    with _JUDGE_CACHE_LOCK:
        _JUDGE_CACHE[key] = dict(result)

    score_path, _ = _judge_cache_paths(key)
    tmp_path = score_path.with_name(f"{score_path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(result, f)
        os.replace(tmp_path, score_path)  # atomic: readers see all-or-nothing
    except OSError as e:
        logger.warning("Judge cache publish failed for %s: %s", key[:12], e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _judge_cache_bump(kind: str) -> None:
    with _JUDGE_CACHE_LOCK:
        _JUDGE_CACHE_STATS[kind] += 1
        served = sum(_JUDGE_CACHE_STATS.values())
        if served % 500 == 0:
            reused = _JUDGE_CACHE_STATS["hits"] + _JUDGE_CACHE_STATS["waits"]
            logger.info(
                "Judge cache: %d served, %d judged, %d reused (%d cached + %d waited) "
                "-- %.1f%% reuse",
                served,
                _JUDGE_CACHE_STATS["misses"],
                reused,
                _JUDGE_CACHE_STATS["hits"],
                _JUDGE_CACHE_STATS["waits"],
                100 * reused / served,
            )


# ---- Rubric judge call --------------------------------------------------------


@retry(
    stop=stop_after_attempt(JUDGE_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def evaluate_answer(
    context_chunk: str,
    question: str,
    model_response: str,
    next_chunk: str,
) -> dict:
    """Score a case-completion answer with the rubric judge.

    Exceptions (network errors, 429s, parse errors that raise) propagate
    so tenacity actually retries. Callers must wrap in try/except.

    Successful judgements are memoized on (context_chunk, question,
    model_response, next_chunk) -- the full judge input -- in a cache shared by
    all ranks, so an identical answer to an identical prompt returns one frozen
    score instead of a fresh stochastic draw. See the judge-cache note above.
    """
    cache_key = _judge_cache_key(context_chunk, question, model_response, next_chunk)

    cached = _judge_cache_get(cache_key)
    if cached is not None:
        _judge_cache_bump("hits")
        return dict(cached)

    # Another rank may be judging this exact input right now (the common case:
    # duplicate answers in one prompt-group are spread one-per-rank). Wait for
    # its score rather than drawing an independent, noisy one.
    holding_claim = _judge_cache_claim(cache_key)
    if not holding_claim:
        cached = _judge_cache_wait(cache_key)
        if cached is not None:
            _judge_cache_bump("waits")
            return dict(cached)
        # Claim-holder died or timed out — fall through and judge it ourselves.

    try:
        prompt = JUDGE_PROMPT.format(
            context_chunk=context_chunk,
            question=question,
            model_response=model_response,
            next_chunk=next_chunk,
        )

        response = _get_judge_client().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
        )

        content = response.choices[0].message.content
        parsed = _parse_judge_json(content)

        if parsed is None:
            # Parse failure is not retried — we already have a response, the
            # model just wrote unparseable JSON. Return graceful failure, and
            # leave it uncached so a later identical answer gets a fresh try.
            return {
                "judge_score": 0.0,
                "api_status": "parse_error",
                "raw_response": content,
            }

        scores = {}
        for key in CRITERION_WEIGHTS:
            entry = parsed.get(key, {})
            scores[key] = entry.get("score") if isinstance(entry, dict) else entry

        weighted = sum(CRITERION_WEIGHTS[k] * (scores[k] or 0) for k in CRITERION_WEIGHTS)
        result = {
            "judge_score": weighted / MAX_REWARD,
            "api_status": "success",
            "raw_response": content,
        }
        _judge_cache_publish(cache_key, result)
        _judge_cache_bump("misses")
        return result
    finally:
        # Released on the API-exception path too, so tenacity's retry (which
        # re-enters this function) can re-claim instead of deadlocking on itself.
        if holding_claim:
            _judge_cache_release(cache_key)


# ---- Diagnosis-equivalence judge call -----------------------------------------


@retry(
    stop=stop_after_attempt(JUDGE_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def evaluate_diagnosis(pred: str, ref: str) -> float:
    """Score a predicted diagnosis against a reference diagnosis.

    Returns 1.0 if the judge marks them as equivalent, 0.0 otherwise.
    Exceptions propagate for tenacity retry. Callers must wrap in try/except.
    """
    user_msg = DIAGNOSIS_JUDGE_USER_TEMPLATE.format(pred=pred, ref=ref)
    response = _get_judge_client().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": DIAGNOSIS_JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    content = response.choices[0].message.content
    parsed = _parse_judge_json(content)
    if parsed is None:
        logger.debug("Diagnosis judge unparseable response: %s", content[:200])
        return 0.0
    judgment = parsed.get("judgment", 0)
    # Be lenient about types — accept 1/0, "1"/"0", True/False
    if judgment in (1, "1", True):
        return 1.0
    return 0.0


# =============================================================================
# Dataset
# =============================================================================


def _row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": build_messages(row["context_chunks"], row["question"]),
        "context_chunks": row["context_chunks"],
        "question": row["question"],
        "answer": row["answer"],
        "case_id": row.get("case_id"),
    }


def _row_to_record_external(row: dict[str, Any]) -> dict[str, Any]:
    # Handle both "clean text"/"final diagnosis" (with space) and snake_case variants.
    clean_text = (row.get("clean text") or row.get("clean_text") or "").strip()
    final_diagnosis = (row.get("final diagnosis") or row.get("final_diagnosis") or "").strip()
    return {
        "prompt": build_ext_messages(clean_text),
        "clean_text": clean_text,
        "final_diagnosis": final_diagnosis,
    }


def load_jsonl_records(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(jsonl_path: Path | str) -> Dataset:
    records = [_row_to_record(row) for row in load_jsonl_records(jsonl_path)]
    return Dataset.from_list(records)


def build_eval_dataset(
    jsonl_path: Path | str = VAL_JSONL,
    sample_size: int = VAL_SAMPLE_SIZE,
    seed: int = VAL_SAMPLE_SEED,
) -> Dataset:
    """Load val.jsonl and take a fixed-seed random sample of `sample_size` rows."""
    rows = load_jsonl_records(jsonl_path)
    if sample_size < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_size)
    records = [_row_to_record(row) for row in rows]
    return Dataset.from_list(records)


def build_ext_eval_dataset(
    jsonl_path: Path | str = EXT_VAL_JSONL,
    sample_size: int = EXT_VAL_SAMPLE_SIZE,
    seed: int = EXT_VAL_SAMPLE_SEED,
) -> Dataset:
    """Load ext_val.jsonl and take a fixed-seed random sample of `sample_size` rows.

    Only rows with both 'clean text' and 'final diagnosis' populated are kept.
    """
    rows = load_jsonl_records(jsonl_path)
    rows = [
        r for r in rows
        if (r.get("clean text") or r.get("clean_text"))
        and (r.get("final diagnosis") or r.get("final_diagnosis"))
    ]
    if sample_size < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_size)
    records = [_row_to_record_external(row) for row in rows]
    return Dataset.from_list(records)


# =============================================================================
# Answer extraction
# =============================================================================

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# The inner group may not contain another opening tag. `.*?` is non-greedy, but
# re.search still anchors the match at the FIRST <final_answer> it finds, so a
# completion that opens the tag, explains, then re-opens it for the real answer
# ("<final_answer>\n{prose}\n<final_answer>{answer}</final_answer>") matched from
# the first open to the lone close and captured the whole explanation. The tuned
# Qwen policy does exactly this in ~93% of case-completion completions, which fed
# the judge a ~1k-char essay instead of a ~7-word answer.
_FINAL_ANSWER_RE = re.compile(
    r"<final_answer>((?:(?!<final_answer>).)*?)</final_answer>",
    re.DOTALL | re.IGNORECASE,
)
# Bare tags, for counting. _thinking_char_count wants the point where the answer
# block BEGINS, which is the first opening tag -- not where the balanced pair
# matched by _FINAL_ANSWER_RE starts (that is the last opening tag when the model
# re-opens). _well_formed_completion counts both to reject unbalanced nesting.
_FINAL_ANSWER_OPEN_RE = re.compile(r"<final_answer>", re.IGNORECASE)
_FINAL_ANSWER_CLOSE_RE = re.compile(r"</final_answer>", re.IGNORECASE)

# Some models (e.g. medreason) reason in free-form Markdown and never emit
# <final_answer>, but reliably close with a natural heading of their own --
# confirmed 499/500 in a real base-only CI run (job 134976): 94.8% end with
# "### Conclusion:", 5% with "## Final Answer" (and completions with both
# put the concise one-line answer after the LATTER, later heading -- the
# "### Conclusion:" section itself is still a full explanatory paragraph,
# not a distilled answer). Without this, the whole-completion fallback in
# _post_reasoning_region fed the JUDGE the entire multi-paragraph reasoning
# essay instead of the actual answer, which is exactly backwards from what
# the other lenient families (gemma) get (their fallback only matters for
# the rare case an easily-found <final_answer> tag is truly absent).
_NATURAL_ANSWER_HEADING_RE = re.compile(
    r"(?im)^\s*#{1,4}\s*(?:final answer|answer|conclusion|diagnosis)\b[:\s]*$"
)


def _natural_answer_heading_region(text: str) -> str | None:
    """Text after the LAST natural answer/conclusion heading, or None if no
    such heading is present at all."""
    matches = list(_NATURAL_ANSWER_HEADING_RE.finditer(text))
    if not matches:
        return None
    return text[matches[-1].end():].strip()


def _completion_text(completion) -> str:
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return message.get("content", "")
        if completion and isinstance(completion[-1], dict):
            return completion[-1].get("content", "")
        return ""
    return str(completion)


def _thinking_char_count(text: str) -> int:
    """Qwen: chat template already injects the opening <think> into the
    prompt, so the completion text itself never contains the opening tag --
    thinking content is everything before the model's closing </think> (or
    the whole text if it never closes, i.e. still reasoning when cut off).
    Gemma: the model generates both its own opening "<|channel>thought\\n"
    and closing "<channel|>" as part of the completion, so this slightly
    over-counts by the fixed-length opening marker -- negligible next to the
    20k/30k-char penalty thresholds. In practice Gemma emits <channel|> in
    ~0% of completions (measured empirically: 0/2128 in a full training run
    -- see _post_reasoning_region), so without a fallback this counted the
    ENTIRE completion (reasoning + final answer) as "thinking" for
    virtually every Gemma completion, silently turning the penalty into a
    blunt total-completion-length penalty instead of a reasoning-length one
    (measured: 24.9% of completions penalized, averaging 0.047 of the 0.05
    max). Since Gemma reliably emits <final_answer> instead (~99%, see
    _post_reasoning_region), fall back to that tag's start as the
    reasoning/answer boundary before falling back to the whole text.
    """
    match = _THINKING_CLOSE_RE.search(text)
    if match:
        return match.start()
    if MODEL_FAMILY in ("gemma", "gemma4_31b", "openbiollm", "medreason", "glm52"):
        answer_match = _FINAL_ANSWER_OPEN_RE.search(text)
        if answer_match:
            return answer_match.start()
    return len(text)


def _post_reasoning_region(text: str) -> str | None:
    """Region of `text` to search for <final_answer>...</final_answer>, or
    None if there is no valid region to search (still reasoning / cut off).

    Qwen: strictly requires the </think> close-tag. Qwen reliably closes
    </think> before answering (baked into its own training), so absence
    means generation was cut off mid-reasoning, not that thinking was
    skipped -- there is genuinely no answer yet. Do not relax this.

    Gemma: the <channel|> "thinking channel" close marker was expected to be
    emitted the same way based on isolated single-prompt testing during
    development, but empirically, across a real training batch, Gemma emits
    it in under 1% of completions (measured: 5/752) while still reliably
    closing with <final_answer>...</final_answer> on its own in ~99% of
    completions (746/752) -- it just does plain prose reasoning without the
    formal channel wrapper. Requiring the channel marker as a prerequisite
    made almost every reward 0 regardless of answer quality (job 82030,
    2026-07-04). So for Gemma, fall back to searching the whole completion
    when no channel marker is present.
    """
    think_end = _THINKING_CLOSE_RE.search(text)
    if think_end:
        return text[think_end.end():]
    # openbiollm/medreason: added alongside gemma/gemma4_31b -- both smoke-
    # tested at 0% </think> emission with reliable direct-prose answers, same
    # failure mode gemma's comment above describes. glm52: added as a safety
    # net even though it reliably closes </think> in practice (8/8 in smoke
    # job 134796) -- costs nothing on the common path, avoids zero-credit on
    # the rare miss.
    if MODEL_FAMILY in ("gemma", "gemma4_31b", "openbiollm", "medreason", "glm52") or _NOCOT_EVAL_FALLBACK:
        # Prefer a natural answer/conclusion heading over the whole text when
        # present (see _natural_answer_heading_region) -- narrows the region
        # a downstream <final_answer> tag search operates on, AND is what
        # _answer_for_judge's final raw-text fallback actually returns to the
        # judge when no tag is found either. Confirmed necessary for
        # medreason (see comment on _NATURAL_ANSWER_HEADING_RE); harmless
        # no-op for gemma/gemma4_31b/openbiollm/glm52 since they essentially
        # never emit these headings.
        heading_region = _natural_answer_heading_region(text)
        return heading_region if heading_region is not None else text
    return None


def _well_formed_completion(text: str) -> bool:
    """True iff completion has the canonical reasoning-close-tag followed by
    EXACTLY ONE balanced <final_answer>...</final_answer> pair (</think> for
    Qwen, <channel|> for Gemma, or -- for Gemma only -- anywhere in the
    completion if no channel marker was emitted; see _post_reasoning_region).
    Used to award a small format bonus during training so the policy learns the
    exact format the eval parsers look for.

    The "exactly one pair" requirement is load-bearing. The previous version only
    asked whether a pair existed *somewhere* in the post-reasoning region, so the
    v2 Qwen policy collected the bonus while emitting

        </think><final_answer>{explanation}<final_answer>{answer}</final_answer>

    -- an unclosed outer tag wrapping the explanation. That is well-formed to a
    "does a pair exist" check but not to the extractor, which is why 92.6% of its
    case-completion answers reached the judge as a ~1k-char essay (job v2, eval
    2026-07-04; corrected 2026-07-09). Counting the tags rejects that shape while
    still accepting prose between the reasoning close and the tag, which is
    harmless once extraction is correct and which both models do routinely.

    Measured on the saved eval completions (test split): base 92.0% well-formed,
    ck700 4.7%, ck1100 8.1% -- i.e. this now discriminates exactly the failure it
    is meant to. Requiring *no* prose outside the tag as well was considered and
    rejected: it drops base to 9.8% and both tuned checkpoints below 8%, making
    the bonus near-unattainable for every policy and conflating two unrelated
    format properties (cf. the Gemma channel-marker reward collapse, job 82030).
    """
    post = _post_reasoning_region(text)
    if post is None:
        return False
    if len(_FINAL_ANSWER_OPEN_RE.findall(post)) != 1:
        return False
    if len(_FINAL_ANSWER_CLOSE_RE.findall(post)) != 1:
        return False
    return bool(_FINAL_ANSWER_RE.search(post))


def _thinking_length_penalty(num_chars: int) -> float:
    """No penalty below 15k chars; linear 15k–25k; capped at THINKING_PENALTY_FRACTION."""
    max_penalty = THINKING_PENALTY_FRACTION
    if num_chars < THINKING_CHAR_NO_PENALTY:
        return 0.0
    if num_chars <= THINKING_CHAR_LINEAR_END:
        span = THINKING_CHAR_LINEAR_END - THINKING_CHAR_NO_PENALTY
        frac = (num_chars - THINKING_CHAR_NO_PENALTY) / span
        return max_penalty * frac
    return max_penalty


def _last_final_answer(search_text: str) -> Optional[str]:
    """Content of the LAST balanced <final_answer>...</final_answer> pair, or None.

    The last pair, not the first: "final answer" means the one the model finally
    commits to. Taking the first pair mis-scored the base model, which routinely
    quotes its own system prompt back to itself --

        "...then output your final answer wrapped in <final_answer>...</final_answer> tags."

    -- and that quotation is itself a balanced pair, so the extractor returned the
    literal "..." and the judge saw an empty answer. Measured on the 2026-07-04
    eval: 50 base test items, every one of which carried base's real answer in a
    later pair (idx 73 -> "Tularemia"). Cost base 1.5 accuracy points. The tuned
    checkpoints never quote the prompt, so this asymmetrically penalized base.
    """
    pairs = _FINAL_ANSWER_RE.findall(search_text)
    return pairs[-1].strip() if pairs else None


def _answer_for_judge(text: str) -> str:
    """Extract the case-completion answer.

    Prefers the canonical format reasoning-close-tag + <final_answer>...</final_answer>
    but tolerates partial compliance during training so the judge still gets
    something to score (and the policy can move toward the canonical form
    via the FORMAT_BONUS in the reward).

    Absence of a valid post-reasoning region (see _post_reasoning_region)
    means there is no answer yet.
    """
    search_text = _post_reasoning_region(text)
    if search_text is None:
        return ""

    # 1. Canonical: last <final_answer>...</final_answer> after the close-tag
    answer = _last_final_answer(search_text)
    if answer is not None:
        return answer

    # 2. Legacy/partial: <answer>...</answer> anywhere
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()

    # 3. Post-think raw text
    return search_text.strip()


def _extract_final_answer(text: str) -> str:
    """Extract the final diagnosis from ext_val generation output.

    Strict by default: only accept a <final_answer>...</final_answer> tag
    emitted in the post-reasoning region (see _post_reasoning_region). If the
    model didn't close reasoning (Qwen) or didn't emit the tag at all, return
    "" so the judge sees an empty answer and scores 0 -- no fallback for
    strict families.

    Exception: the whole-completion-fallback families (gemma, gemma4_31b,
    openbiollm, medreason, glm52) also get the SAME narrowed region
    _post_reasoning_region already returns (whole text, or the tail after a
    natural answer/conclusion heading if present -- see
    _NATURAL_ANSWER_HEADING_RE) as their fallback answer when no tag is
    found. Added for medreason specifically: it NEVER emits <final_answer>
    on this task (0/500 tagged in a real run, job 134991) but DOES reliably
    end with the same natural heading it uses on the case-completion task
    (100/100 sampled) -- e.g. "## Final Answer\n\nUrethral stenosis" for a
    reference of "Urethral stenosis." Treating that as strictly "no answer"
    zeroed out real, judgeable diagnostic content purely on a format
    technicality. Inert for gemma/gemma4_31b/openbiollm/glm52 in practice,
    since they already reliably emit the tag on this task.
    """
    search_text = _post_reasoning_region(text)
    if search_text is None:
        return ""
    answer = _last_final_answer(search_text)
    if answer is not None:
        return answer
    if MODEL_FAMILY in ("gemma", "gemma4_31b", "openbiollm", "medreason", "glm52"):
        return search_text.strip()
    return ""


# =============================================================================
# GRPO reward — dispatches between rubric judge (val) and diagnosis judge (ext_val)
# =============================================================================


# Set True only while the trainer is inside evaluate() (see the wrapper applied
# to the trainer instance in main()). The case-completion reward branch below
# runs for BOTH training and val eval with identical columns, so it cannot tell
# them apart on its own -- without this flag it hardcoded task="val" for every
# case-completion record, meaning each training step's completions were logged
# as "val". That inflated the apparent val-eval count in judge_predictions far
# past the ~25/rank a single val pass produces and made training reward-logging
# look like a runaway eval. ext_val is separately identifiable by its columns,
# so it's unaffected.
_IN_EVAL = False


def make_judge_reward_fn(max_workers: int = JUDGE_MAX_WORKERS):
    """Build the reward function for both training and eval.

    Dispatches based on which columns are present in the batch:
    - case-completion task (train + val): expects context_chunks, question, answer
    - external diagnosis task (ext_val): expects final_diagnosis
    """

    def judge_reward(completions, **kwargs) -> list[float]:
        model_responses = [_completion_text(c) for c in completions]
        n = len(model_responses)
        rewards: list[float | None] = [None] * n

        is_external = "final_diagnosis" in kwargs

        if is_external:
            refs = kwargs["final_diagnosis"]

            def _score_one(i: int) -> tuple[int, float]:
                full_text = model_responses[i]
                pred = _extract_final_answer(full_text)
                try:
                    score = evaluate_diagnosis(pred, refs[i])
                except Exception as e:
                    logger.error("Diagnosis judge failed for idx %d: %s", i, e)
                    score = 0.0
                # No thinking-length penalty for ext_val — report pure accuracy.
                log_judge_prediction({
                    "task": "ext_val",
                    "completion": full_text,
                    "extracted_diagnosis": pred,
                    "reference_diagnosis": refs[i],
                    "reward": float(score),
                })
                return i, float(score)

        else:
            context_chunks = kwargs["context_chunks"]
            question = kwargs["question"]
            answer = kwargs["answer"]

            def _score_one(i: int) -> tuple[int, float]:
                full_text = model_responses[i]
                result: dict[str, Any] = {}
                try:
                    result = evaluate_answer(
                        context_chunk=context_chunks[i],
                        question=question[i],
                        model_response=_answer_for_judge(full_text),
                        next_chunk=answer[i],
                    )
                    if result["api_status"] == "success" and result.get("raw_response"):
                        judge_score = compute_reward(result["raw_response"], normalize=True)
                    else:
                        judge_score = float(result.get("judge_score", 0.0))
                except Exception as e:
                    logger.error("Rubric judge failed for idx %d: %s", i, e)
                    result = {"api_status": "exception", "raw_response": str(e)}
                    judge_score = 0.0
                penalty = _thinking_length_penalty(_thinking_char_count(full_text))
                bonus = FORMAT_BONUS if _well_formed_completion(full_text) else 0.0
                reward = max(0.0, judge_score - penalty + bonus)
                log_judge_prediction({
                    "task": "val" if _IN_EVAL else "train",
                    "context_chunk": context_chunks[i],
                    "question": question[i],
                    "completion": full_text,
                    "extracted_answer": _answer_for_judge(full_text),
                    "reference_answer": answer[i],
                    "judge_api_status": result.get("api_status"),
                    "judge_raw_response": result.get("raw_response"),
                    "judge_score": judge_score,
                    "thinking_penalty": penalty,
                    "format_bonus": bonus,
                    "reward": reward,
                })
                return i, reward

        workers = max(1, min(max_workers, n))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_score_one, i) for i in range(n)]
            for future in as_completed(futures):
                try:
                    idx, reward = future.result()
                    rewards[idx] = reward
                except Exception as e:
                    # Should not happen — _score_one swallows its own exceptions.
                    logger.error("Unexpected worker exception: %s", e)

        return [float(r if r is not None else 0.0) for r in rewards]

    return judge_reward


# =============================================================================
# Checkpoint discovery
# =============================================================================


def find_latest_checkpoint(output_dir: str | Path) -> Optional[str]:
    """Return path to the highest-numbered checkpoint-N folder in output_dir, or None.

    HF Trainer saves checkpoints as `checkpoint-<step>` inside `output_dir`.
    We pick the one with the largest step number.
    """
    p = Path(output_dir)
    if not p.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in p.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"^checkpoint-(\d+)$", child.name)
        if m:
            candidates.append((int(m.group(1)), child))
    if not candidates:
        return None
    _, latest = max(candidates, key=lambda x: x[0])
    return str(latest)


# =============================================================================
# Training
# =============================================================================


def main() -> None:
    log_file = setup_logging()
    logger.info("Logging to %s", log_file)

    # Sanity dump of the live config so you can grep the log and verify what
    # this process is actually running with (separate from whatever the
    # source file currently says).
    logger.info(
        "LIVE CONFIG | LR=%.1e SCHED=%s WARMUP=%d EVAL_STEPS=%d SAVE_STEPS=%d "
        "BETA=%s KL_LOGRATIO_CLIP=%s MASK_TRUNCATED=%s THINK_PENALTY=%s "
        "FORMAT_BONUS=%s TEMPERATURE=%s TOP_K=%s "
        "MAX_REWARD=%d MAX_COMPLETION=%d NUM_GENS=%d VAL_N=%d EXT_VAL_N=%d "
        "OUTPUT_DIR=%s",
        LEARNING_RATE, LR_SCHEDULER_TYPE, WARMUP_STEPS, EVAL_STEPS, SAVE_STEPS,
        BETA, globals().get("KL_LOGRATIO_CLIP", "n/a (qwen: unpatched)"),
        MASK_TRUNCATED_COMPLETIONS, THINKING_PENALTY_FRACTION, FORMAT_BONUS,
        TEMPERATURE, TOP_K,
        MAX_REWARD, MAX_COMPLETION_LENGTH, NUM_GENERATIONS, VAL_SAMPLE_SIZE,
        EXT_VAL_SAMPLE_SIZE, OUTPUT_DIR,
    )

    logger.info("Loading train dataset from %s", TRAIN_JSONL)
    train_dataset = build_dataset(TRAIN_JSONL)
    logger.info("Loading val eval dataset from %s (sampling %d)", VAL_JSONL, VAL_SAMPLE_SIZE)
    eval_dataset = build_eval_dataset(VAL_JSONL, VAL_SAMPLE_SIZE, VAL_SAMPLE_SEED)
    logger.info(
        "Loading ext_val eval dataset from %s (sampling %d)",
        EXT_VAL_JSONL,
        EXT_VAL_SAMPLE_SIZE,
    )
    ext_eval_dataset = build_ext_eval_dataset(
        EXT_VAL_JSONL, EXT_VAL_SAMPLE_SIZE, EXT_VAL_SAMPLE_SEED
    )
    logger.info(
        "Train: %d | Val: %d | ExtVal: %d",
        len(train_dataset),
        len(eval_dataset),
        len(ext_eval_dataset),
    )

    # Detect prior checkpoint to resume from. Env var override:
    #   GRPO_RESUME=none  → force fresh start (ignore existing checkpoints)
    #   GRPO_RESUME=<path>→ resume from that exact checkpoint
    #   (unset)           → auto-detect latest in OUTPUT_DIR
    env_resume = os.environ.get("GRPO_RESUME", "").strip()
    if env_resume.lower() in {"none", "no", "false", "0"}:
        resume_from: Optional[str] = None
        logger.info("GRPO_RESUME=%s → forcing fresh start", env_resume)
    elif env_resume:
        resume_from = env_resume
        logger.info("GRPO_RESUME=%s → resuming from this checkpoint", env_resume)
    else:
        resume_from = find_latest_checkpoint(OUTPUT_DIR)
        if resume_from:
            logger.info("Auto-detected latest checkpoint: %s", resume_from)
        else:
            logger.info("No checkpoint found in %s; starting fresh", OUTPUT_DIR)

    eval_datasets = {
        "val": eval_dataset,
        "ext_val": ext_eval_dataset,
    }

    _LORA_PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if MODEL_FAMILY == "gemma4_31b":
        # gemma4_31b is a multimodal wrapper: a Gemma4ForConditionalGeneration
        # with a `vision_tower` AND a `language_model`. The two towers name/type
        # their proj layers DIFFERENTLY:
        #   - vision_tower: every proj is a custom Gemma4ClippableLinear wrapper
        #     (real nn.Linear at its `.linear` child) -- 189 of them.
        #   - language_model: plain nn.Linear named directly `q_proj` etc.
        #     (NO `.linear` child) -- 410 of them.
        # The original `.linear`-suffix target (jobs 123854/124130) was derived
        # from an inspection that only looked at `...layers.0`, and the VISION
        # tower's layer 0 matched first -- so it wrongly concluded "all proj are
        # wrapped" and targeted `.linear`. That suffix matched ONLY the 189
        # vision-tower wrappers; the language model got ZERO trainable LoRA
        # params. Job 124130 then trained a LoRA on the (unused, this is a
        # text-only task) vision encoder while the LM stayed fully frozen --
        # grad_norm and kl read exactly 0 on every logged step for 12+ hours,
        # and the "reward" was just sampling noise on the unchanged base model.
        # Verified via _inspect_gemma4_31b_lm_layers.py (job 126161: LM proj are
        # plain Linear) and _inspect_gemma4_31b_lora_fix.py (job 126166: the
        # regex below yields 820 trainable tensors / 489.7M params / 1.48%, all
        # in language_model, 0 in vision). We deliberately train ONLY the
        # language model (regex-anchored on `language_model.`), matching every
        # other family which has no separate vision tower to accidentally hit.
        target_modules = (
            r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        )
    elif MODEL_FAMILY == "nemotron3_nano_omni":
        # nemotron3_nano_omni now loads the bare NemotronHForCausalLM
        # backbone directly (see _MODEL_NAMES comment -- the omni wrapper is
        # bypassed entirely, extracted via extract_nemotron_llm_only.py), so
        # there is no `language_model.` prefix and no vision/video/audio
        # tower present in this checkpoint at all. It's a genuinely hybrid
        # architecture: 52 blocks split into THREE distinct kinds, not one
        # block containing all three (confirmed via meta-device
        # named_modules() inspection, jobs 128234/128237/128239):
        #   - 23 Mamba2Mixer blocks: mixer.in_proj / mixer.out_proj (SSM, no
        #     q/k/v/o naming at all)
        #   - 23 MoE blocks: mixer.experts.{0..127}.up_proj/down_proj (128
        #     experts) + mixer.shared_experts.up_proj/down_proj -- a 2-proj
        #     ReLU^2 MLP, NOT the 3-proj SwiGLU (gate/up/down) style every
        #     other family here uses
        #   - 6 Attention blocks: mixer.q_proj/k_proj/v_proj/o_proj -- the
        #     ONLY blocks with conventional LoRA-friendly names (~11.5% of
        #     the model)
        # Naively reusing _LORA_PROJ_NAMES would (a) completely skip the 23
        # Mamba2 mixers (44% of all blocks, the primary sequence-mixing
        # mechanism -- in_proj/out_proj aren't in that list) and (b) silently
        # balloon to ~2,967 LoRA-adapted MoE expert modules via up_proj/
        # down_proj matching every sparsely-activated expert (most experts
        # see very few tokens per batch -> noisy, rare gradient signal per
        # expert). Deliberately target ONLY attention + Mamba2 for a first
        # training attempt, leaving all 2967 MoE expert modules untouched.
        # Verified on the real (non-meta) extracted checkpoint: exactly 140
        # trainable tensors, 48 attention + 92 Mamba2 + 0 MoE-expert (job
        # 128243 meta-device, re-confirmed against this exact checkpoint
        # before launch).
        target_modules = (
            r".*\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj)$"
        )
    else:
        target_modules = _LORA_PROJ_NAMES

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # nemotron3_nano_omni is fully custom code (auto_map -> configuration_
    # nemotron_h.py/modeling_nemotron_h.py, extracted alongside the weights
    # by extract_nemotron_llm_only.py) AND its config has no explicit
    # attn_implementation, so it falls back to the class default of
    # "flash_attention_2", which lazily imports named functions from
    # flash_attn -- this container only ships an empty flash_attn STUB
    # (satisfies a bare `import flash_attn` check, but not `from flash_attn
    # import flash_attn_func`), so any bare AutoModelForCausalLM load crashes
    # with ImportError (confirmed via meta-device inspection, jobs 128221/
    # 128231). GRPOTrainer passes a bare model-name STRING and loads it
    # internally via create_model_from_path(**model_init_kwargs), for BOTH
    # the policy model and the reference model -- neither
    # trust_remote_code nor an attn_implementation override was ever needed
    # by any other family here, so this is the first family requiring either.
    # Also force bf16 at LOAD time (not just GRPOConfig(bf16=True), which only
    # controls forward/backward autocast, never the frozen base model's own
    # storage dtype). Every other family here relies on that same autocast-
    # only behavior and is fine with an fp32-resident base model, but this
    # model's total footprint at fp32 (31.6B params x 4 bytes = ~126GB) left
    # so little headroom on a single 178GB GPU (no ZeRO/FSDP sharding here --
    # each training replica holds a full copy) that the very first automatic
    # pre-training eval OOM'd: "176.37 GiB is allocated ... 79.75 MiB is
    # free" (confirmed job 128676, a real crash after both the vLLM-view and
    # MoE-sync-skip fixes had already gotten generation + weight-sync working
    # cleanly, so this was the very next real blocker in the chain, not a
    # symptom of either fix). Loading in bf16 (the checkpoint's native dtype)
    # halves that footprint to ~63GB, matching what vLLM already does for
    # its own copy via --dtype bfloat16.
    #
    # KEY NAME MATTERS: TRL's own create_model_from_path (trl/trainer/utils.py)
    # reads kwargs["dtype"] specifically (defaulting to "float32" if absent),
    # NOT the older transformers `torch_dtype` name -- passing
    # `torch_dtype=...` here is silently accepted as an unused extra kwarg
    # while TRL's own dtype resolution still defaults to fp32 regardless.
    # First attempt used `torch_dtype` and produced an IDENTICAL OOM at the
    # same spot with the same ~176GB allocated (job 128754) -- directly
    # verified via a standalone create_model_from_path call (job 128810:
    # `model dtype: torch.float32` even with torch_dtype="bfloat16"/
    # torch.bfloat16 set) before finding the real key name by reading
    # create_model_from_path's source directly (job 128811).
    _EXTRA_MODEL_INIT_KWARGS = (
        {"attn_implementation": "eager", "dtype": "bfloat16"}
        if MODEL_FAMILY == "nemotron3_nano_omni"
        else None
    )

    grpo_kwargs: dict = {
        "output_dir": OUTPUT_DIR,
        "trust_remote_code": MODEL_FAMILY == "nemotron3_nano_omni",
        "model_init_kwargs": _EXTRA_MODEL_INIT_KWARGS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": int(os.environ.get("GRPO_PER_DEVICE_EVAL_BATCH_SIZE", "1")),
        "generation_batch_size": NUM_GENERATIONS,
        "num_generations": NUM_GENERATIONS,
        "num_generations_eval": 2,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "warmup_steps": WARMUP_STEPS,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        # Training-time eval was unreliable (resume-triggered eval despite
        # eval_steps; multi-hour blocks on 300-sample sets). Disabled in
        # favor of offline eval via eval_checkpoints.py. Override with
        # GRPO_EVAL_STRATEGY=steps to re-enable.
        "eval_strategy": os.environ.get("GRPO_EVAL_STRATEGY", "no"),
        "eval_steps": EVAL_STEPS,
        # Skip the initial eval when resuming — those metrics already exist
        # for this checkpoint from the original run. Override with
        # GRPO_EVAL_ON_START={0,1,true,false} to force on/off regardless.
        "eval_on_start": (
            os.environ["GRPO_EVAL_ON_START"].lower() in {"1", "true", "yes"}
            if "GRPO_EVAL_ON_START" in os.environ
            else resume_from is None
        ),
        "bf16": BF16,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        # Fused linear+GRPO-loss kernel (liger-kernel): computes the loss from
        # hidden_states + lm_head.weight directly instead of materializing
        # full-vocab logits (262,144 for gemma4_31b). Verified architecture-
        # agnostic for our case: TRL's own LigerFusedLinearGRPOLoss only needs
        # unwrapped_model.lm_head and _get_last_hidden_state's `backbone =
        # unwrapped_model.base_model` path, which explicitly handles
        # multimodal wrappers (see trl's own comment: "multimodal wrapper for
        # VLMs (so vision-token injection runs before the text decoder)").
        # Not gated by a per-architecture registry the way liger_kernel's
        # transformers monkey-patches (RMSNorm/rope fusions) are -- confirmed
        # via inspection jobs 126690/126695/126696. Root cause of the OOM
        # this addresses: mem-probe job 126625 measured compute_loss (the
        # policy forward) as the single largest per-microbatch jump, +6 to
        # +13 GiB scaling with that microbatch's own completion length -- the
        # full-vocab logits tensor. TRL raises if this is combined with LoRA
        # targeting lm_head; our target_modules regex never matches lm_head,
        # so we're clear. Defaults OFF: liger-kernel isn't pip-installed for
        # the other families' sbatch scripts, and enabling it there would
        # raise at trainer init ("Liger is required..."). Opt in per-run with
        # GRPO_USE_LIGER_KERNEL=1 (and `pip install liger-kernel` in that
        # run's sbatch) rather than flipping the default for everyone.
        "use_liger_kernel": os.environ.get("GRPO_USE_LIGER_KERNEL", "0").lower() in ("1", "true", "yes"),
        "report_to": "tensorboard",
        "remove_unused_columns": False,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "num_iterations": NUM_ITERATIONS,
        "loss_type": LOSS_TYPE,
        "beta": BETA,
        "mask_truncated_completions": MASK_TRUNCATED_COMPLETIONS,
        "vllm_importance_sampling_correction": VLLM_IMPORTANCE_SAMPLING_CORRECTION,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    }

    # Optional DeepSpeed ZeRO integration -- off by default (every existing
    # family runs plain DDP, one full replica per training GPU). Added for
    # gemma4_31b after job 123854/124091 OOM'd on non-sharded DDP at 31B
    # scale (no ZeRO here previously, same as the Qwen3.6-27B colocated OOM
    # precedent). Points at a deepspeed json with "auto" fields so it picks
    # up per_device_train_batch_size/gradient_accumulation_steps/etc. from
    # this same TrainingArguments instance rather than needing to be kept in
    # sync by hand.
    if "GRPO_DEEPSPEED_CONFIG" in os.environ:
        grpo_kwargs["deepspeed"] = os.environ["GRPO_DEEPSPEED_CONFIG"]

    if USE_VLLM:
        grpo_kwargs["use_vllm"] = True
        # Colocated vLLM in-process on every training GPU (see module docstring)
        # is the default. Set explicitly rather than relying on TRL's default
        # so this doesn't silently flip to "server" mode if that default ever
        # changes upstream. Override with GRPO_VLLM_MODE=server (experimental,
        # 2026-07-05: testing whether N dedicated generation GPUs + (8-N)
        # training GPUs beats the colocated setup -- a benchmark found
        # colocate's gpu_memory_utilization=0.2 caps generation throughput at
        # ~83 tok/s/GPU, vs. ~1816 tok/s on one dedicated GPU at 0.85 memory
        # utilization) to instead connect to a separately-launched
        # `trl vllm-serve` process. Requires GRPO_VLLM_SERVER_HOST/PORT.
        vllm_mode = os.environ.get("GRPO_VLLM_MODE", "colocate")
        grpo_kwargs["vllm_mode"] = vllm_mode
        if vllm_mode == "colocate":
            grpo_kwargs["vllm_gpu_memory_utilization"] = VLLM_GPU_MEMORY_UTILIZATION
            grpo_kwargs["vllm_max_model_length"] = VLLM_MAX_MODEL_LENGTH
        elif vllm_mode == "server":
            grpo_kwargs["vllm_server_host"] = os.environ.get("GRPO_VLLM_SERVER_HOST", "0.0.0.0")
            grpo_kwargs["vllm_server_port"] = int(os.environ.get("GRPO_VLLM_SERVER_PORT", "8000"))
        else:
            raise ValueError(f"Unknown GRPO_VLLM_MODE={vllm_mode!r}; expected 'colocate' or 'server'")

    if "GRPO_MAX_STEPS" in os.environ:
        grpo_kwargs["max_steps"] = int(os.environ["GRPO_MAX_STEPS"])

    training_args = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=MODEL,
        args=training_args,
        reward_funcs=make_judge_reward_fn(),
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        peft_config=peft_config,
    )

    # Toggle _IN_EVAL around every evaluate() call (eval_on_start + periodic
    # eval at eval_steps both route through trainer.evaluate) so the reward fn
    # labels case-completion judge_predictions as "val" during eval and "train"
    # during training. Instance-level override shadows the bound method, so
    # HF/TRL's internal self.evaluate(...) calls hit this wrapper too.
    _orig_evaluate = trainer.evaluate

    def _evaluate_with_eval_flag(*args, **kwargs):
        global _IN_EVAL
        _IN_EVAL = True
        try:
            return _orig_evaluate(*args, **kwargs)
        finally:
            _IN_EVAL = False

    trainer.evaluate = _evaluate_with_eval_flag

    if resume_from:
        logger.info("Starting GRPO training (resuming from %s)", resume_from)
    else:
        logger.info("Starting GRPO training (fresh)")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(OUTPUT_DIR)
    logger.info("Saved model to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

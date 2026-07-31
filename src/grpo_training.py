#!/usr/bin/env python3
"""
GRPO training for medical case-completion QA.

- Policy prompts: SYSTEM_PROMPT + user template (case chunk + question)
- Reward: DeepSeek LLM judge (4-criterion rubric, normalized 0-1)
- Generation: vLLM colocated in-process (use_vllm=True, vllm_mode="colocate")
  on every training GPU by default, or a disaggregated dedicated-GPU
  `trl vllm-serve` (vllm_mode="server") for larger models.

Eval consists of two datasets, evaluated every EVAL_STEPS:
- "val":      held-out case-completion QA, scored by the rubric judge
- "ext_val":  external final-diagnosis task, scored by a separate
              diagnosis-equivalence judge (downstream accuracy)

Judge predictions (model completion, extracted answer, judge raw response,
score) are appended to a per-rank JSONL file under
OUTPUT_DIR/judge_predictions/ as training proceeds.

Run (single node, 8 GPUs, colocated vLLM on all of them):
  accelerate launch --num_processes=8 grpo_training.py
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

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # vLLM's tokenizer path emits a noisy per-generation transformers warning
    # ("Kwargs passed to `processor.__call__` have to be in `processor_kwargs`
    # dict, not in `**kwargs`") that floods the log; it's harmless, filter it.
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
# "qwen"   = Qwen/Qwen3.5-9B. Chat template auto-injects the opening <think>
#            into the prompt when enable_thinking=True; completion text only
#            ever contains the model-generated closing </think>.
# "huatuo" = FreedomIntelligence/HuatuoGPT-3-8B, a medical model built on
#            Qwen3-8B-Base (RL-only "SeedRL" recipe). Architecturally
#            Qwen3ForCausalLM with the stock Qwen ChatML template, so it
#            shares Qwen's reasoning close-tag and sampling settings.
#            Its tokenizer was inherited from Qwen3-8B-*Base*, whose
#            eos_token is <|endoftext|> (151643), but the chat-tuned model
#            actually ends every assistant turn with <|im_end|> (151645) --
#            see the multi-EOS termination fix below, needed because TRL
#            only checks the single scalar tokenizer.eos_token_id.
# "qwen36" = Qwen/Qwen3.6-27B. Chat template renders byte-identically to
#            Qwen3.5's under {"enable_thinking": True, "thinking_budget": N},
#            so it reuses qwen's chat-template kwargs, close-tag regex,
#            sampling params, and system prompt unchanged. Its tokenizer
#            eos_token IS <|im_end|>, so (like qwen) it does NOT need the
#            multi-EOS fix.
#            One real quirk: its declared architecture is
#            Qwen3_5ForConditionalGeneration (model_type "qwen3_5") -- a
#            multimodal wrapper, not plain Qwen3ForCausalLM -- even though
#            this checkpoint is used purely as a text model. transformers'
#            AutoModelForCausalLM silently resolves to the text-only sibling
#            class for this model_type, so training-side loading just works;
#            this only matters if you load the full wrapper class directly
#            (e.g. to merge a LoRA adapter that must carry the vision tower's
#            weights through unmodified for vLLM to accept the checkpoint).
MODEL_FAMILY = os.environ.get("GRPO_MODEL_FAMILY", "qwen")

# Opt-in only (default "0"): for evaluating the no-CoT SFT ablation, whose
# assistant target is JUST <final_answer>...</final_answer> with no <think>
# block at all. Since the chat template pre-injects the opening <think> as
# part of add_generation_prompt itself, a no-cot-trained completion never
# contains a literal "</think>", so the strict extractor otherwise returns
# "" even for a well-formed <final_answer>. Gated behind this env var so it
# never changes normal GRPO-training or CoT-eval scoring behavior.
_NOCOT_EVAL_FALLBACK = os.environ.get("GRPO_EVAL_NOCOT_FALLBACK", "0") == "1"
if MODEL_FAMILY not in ("qwen", "huatuo", "qwen36"):
    raise ValueError(
        f"Unknown GRPO_MODEL_FAMILY={MODEL_FAMILY!r}; expected 'qwen', 'huatuo', or 'qwen36'"
    )

# --- paths ---
DATA_DIR = "./data"
TRAIN_JSONL = f"{DATA_DIR}/train.jsonl"
VAL_JSONL = f"{DATA_DIR}/val.jsonl"
EXT_VAL_JSONL = f"{DATA_DIR}/ext_val.jsonl"
_OUTPUT_DIR_DEFAULTS = {
    "qwen": "./checkpoints/medical-grpo-qwen3.5-9b-midstream_task-v2",
    "huatuo": "./checkpoints/medical-grpo-huatuogpt3-8b-midstream_task",
    "qwen36": "./checkpoints/medical-grpo-qwen3.6-27b-midstream_task",
}
OUTPUT_DIR = os.environ.get("GRPO_OUTPUT_DIR", _OUTPUT_DIR_DEFAULTS[MODEL_FAMILY])

# --- model ---
_MODEL_NAMES = {
    "qwen": "Qwen/Qwen3.5-9B",
    "huatuo": "FreedomIntelligence/HuatuoGPT-3-8B",
    "qwen36": "Qwen/Qwen3.6-27B",
}
# Env-overridable so a run can start from a local checkpoint (e.g. a
# previously-trained SFT-CoT adapter merged into the base weights) instead of
# the family's released HF weights, while MODEL_FAMILY itself keeps driving
# every family-specific behavior below (LoRA target modules, chat template
# kwargs, EOS ids, sampling defaults) unchanged -- only the starting weights
# differ. Must be a local directory.
MODEL = os.environ.get("GRPO_MODEL_OVERRIDE", "").strip() or _MODEL_NAMES[MODEL_FAMILY]

# No GRPO/RL-specific hyperparameter guidance exists upstream for any of
# these families beyond Qwen's, so LoRA rank/alpha default to the
# empirically-stable Qwen RL settings and are env-overridable for
# exploration runs. Keep the alpha/r ratio constant (=4.0) when raising r.
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
# Constant LR after a short warmup -- cosine was decaying the LR while the
# policy was still trying to explore, contributing to a flat-reward plateau.
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
# Per-family recommended sampling settings.
_GENERATION_PARAMS = {
    "qwen": dict(temperature=0.6, top_p=0.95, top_k=20),
    # Matches HuatuoGPT-3-8B's own generation_config.json, which happens to
    # equal Qwen's validated values already.
    "huatuo": dict(temperature=0.6, top_p=0.95, top_k=20),
    # Reuses Qwen3.5's validated sampling settings (same tokenizer/template
    # lineage).
    "qwen36": dict(temperature=0.6, top_p=0.95, top_k=20),
}
# GRPO_TEMPERATURE override: lets a run test a different sampling temperature
# than the per-family default above without editing this file.
TEMPERATURE = float(
    os.environ.get("GRPO_TEMPERATURE", str(_GENERATION_PARAMS[MODEL_FAMILY]["temperature"]))
)
TOP_P = _GENERATION_PARAMS[MODEL_FAMILY]["top_p"]
TOP_K = _GENERATION_PARAMS[MODEL_FAMILY]["top_k"]

# --- GRPO ---
NUM_ITERATIONS = 1
LOSS_TYPE = "dr_grpo"
# TRL's own truncation check compares the completion's last token against a
# single scalar (tokenizer.eos_token_id, tokenizer.pad_token_id). Models
# whose chat template actually ends turns on a DIFFERENT token (see the
# multi-EOS fix below) would have every completion misflagged as truncated
# under this check, so masking truncated completions defaults off; only
# enable it for a family once its real terminator id(s) are patched in.
MASK_TRUNCATED_COMPLETIONS = os.environ.get(
    "GRPO_MASK_TRUNCATED_COMPLETIONS", "false"
).lower() in ("1", "true", "yes")
VLLM_IMPORTANCE_SAMPLING_CORRECTION = False
# KL coefficient against the reference (frozen base) model. Higher = stronger
# anchor against drift. Validated at 0.01 for this family of models: strong
# enough to prevent collapse, loose enough that the policy still moves.
BETA = float(os.environ.get("GRPO_BETA", "0.01"))

# --- thinking (chat template thinking mode + length penalty on reasoning chars) ---
# thinking_budget is a Qwen3.5-lineage soft cap kwarg (also honored by
# qwen36); huatuo's stock Qwen3 ChatML template has no such variable.
THINKING_BUDGET_TOKENS = 6000
_CHAT_TEMPLATE_KWARGS_BY_FAMILY = {
    "qwen": {"enable_thinking": True, "thinking_budget": THINKING_BUDGET_TOKENS},
    # huatuo's template supports enable_thinking but not thinking_budget
    # (Qwen3.5-only extension) -- passing it would silently no-op.
    "huatuo": {"enable_thinking": True},
    "qwen36": {"enable_thinking": True, "thinking_budget": THINKING_BUDGET_TOKENS},
}
CHAT_TEMPLATE_KWARGS = _CHAT_TEMPLATE_KWARGS_BY_FAMILY[MODEL_FAMILY]

# Regex marking the model-generated close of the reasoning portion of a
# completion. All three families here use the same ChatML-style </think>.
_THINKING_CLOSE_RE_BY_FAMILY = {
    "qwen": re.compile(r"</think>", re.IGNORECASE),
    "huatuo": re.compile(r"</think>", re.IGNORECASE),
    "qwen36": re.compile(r"</think>", re.IGNORECASE),
}
_THINKING_CLOSE_RE = _THINKING_CLOSE_RE_BY_FAMILY[MODEL_FAMILY]

# Per-family reasoning-length thresholds for the (opt-in) length penalty,
# calibrated against each model's own real reasoning-length distribution
# (chars before the answer, for completions that actually close </think>).
# NO_PENALTY is set ~1.5x the observed median; LINEAR_END sits past the
# observed p90, so the penalty only engages on genuinely excessive
# reasoning, not typical variation. These are inert unless
# GRPO_ENABLE_REWARD_SHAPING=1 (see below).
_THINKING_CHAR_NO_PENALTY_BY_FAMILY = {
    "qwen": 29_600,       # median 19_735, p90 29_047, p99 31_906
    "huatuo": 12_000,      # guardrail only: p99 9_775, this model rarely gets close
    "qwen36": 12_900,      # median 8_602, p90 12_493, p99 18_261
}
_THINKING_CHAR_LINEAR_END_BY_FAMILY = {
    "qwen": 59_200,
    "huatuo": 20_000,
    "qwen36": 25_800,
}
THINKING_CHAR_NO_PENALTY = _THINKING_CHAR_NO_PENALTY_BY_FAMILY[MODEL_FAMILY]
THINKING_CHAR_LINEAR_END = _THINKING_CHAR_LINEAR_END_BY_FAMILY[MODEL_FAMILY]

# Judge raw weighted sum is in [0, MAX_REWARD]; both compute_reward(normalize=True)
# and evaluate_answer["judge_score"] divide by MAX_REWARD -> [0, 1].
_THINKING_PENALTY_FRACTION_BY_FAMILY = {
    "qwen": 0.05,
    "huatuo": 0.05,   # inert unless GRPO_ENABLE_REWARD_SHAPING=1
    "qwen36": 0.05,   # inert unless GRPO_ENABLE_REWARD_SHAPING=1
}
# Reward shaping (thinking-length penalty + format bonus) defaults OFF: with
# truncation already low, the format bonus lands on the large majority of
# completions (a near-constant that cancels in GRPO's within-group
# advantage) and the length penalty rarely fires -- so reward == pure
# judge_score by default. Re-enable with GRPO_ENABLE_REWARD_SHAPING=1.
_ENABLE_REWARD_SHAPING = os.environ.get("GRPO_ENABLE_REWARD_SHAPING", "0") == "1"

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
# Sized for a genuine reasoning judge model, whose hidden reasoning_content
# is billed against the same max_tokens budget as the visible JSON answer --
# a real 4-criteria rubric response can need several thousand reasoning
# tokens before ever emitting its JSON verdict.
JUDGE_MAX_TOKENS = 6000
JUDGE_TEMPERATURE = 0.1
JUDGE_RETRY_ATTEMPTS = 5

# --- vLLM ---
# Override with GRPO_USE_VLLM=0 to fall back to plain HF .generate() for
# rollouts.
USE_VLLM = os.environ.get("GRPO_USE_VLLM", "1").lower() not in ("0", "false", "no")

# --- multi-EOS termination fix ---
# TRL 1.7.0 decides whether a completion "terminated" via:
#   eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
#   is_truncated = ids[-1] not in eos_and_pad
# `self._tokenizer.eos_token_id` is a single scalar. HuatuoGPT-3-8B inherited
# its tokenizer from Qwen3-8B-*Base*, so tokenizer.eos_token is
# <|endoftext|> (151643) -- but the chat-tuned model ends every assistant
# turn with <|im_end|> (151645), and its generation_config declares
# eos_token_id=[151645, 151643]. Without this fix, TRL would flag ~100% of
# huatuo completions as truncated. qwen/qwen36 are unaffected: their
# tokenizer eos_token already IS <|im_end|>.
_EOS_IDS_BY_FAMILY = {
    "huatuo": (151645, 151643),
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
                f"update _ORIG_EOS_AND_PAD in grpo_training.py to match the new source before "
                f"proceeding."
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
    # Diagnostic instrumentation for anomalously large per-token KL: the K3
    # estimator (exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1)
    # is exponential in the logp gap, so a single token where the
    # LoRA-enabled policy and the disable_adapter() reference disagree
    # sharply can dominate the batch mean. Recomputes an UNCLIPPED
    # per_token_logps/per_token_kl in a no_grad pass to find and print the
    # worst-offending token (position, decoded text, context, both logp
    # values). Diagnostic only -- never alters the real loss/gradient.
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
    # Diagnostic instrumentation for tracking down which phase of a GRPO step
    # owns a memory spike (e.g. investigating an OOM at a specific step).
    # Brackets three call boundaries with CUDA memory readings so a jump can
    # be attributed to a specific phase instead of inferred. Diagnostic
    # only -- never alters values used by training.
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

QWEN_SYSTEM_PROMPT = _CLINICAL_RULES + """\
- Think through the case briefly inside <think>...</think>, then output
  your final answer wrapped in <final_answer>...</final_answer> tags.
  Example final answer: <final_answer>CT chest with IV contrast</final_answer>
"""

# All three supported families emit native <think>...</think> tags, so they
# share the same system prompt.
SYSTEM_PROMPT = QWEN_SYSTEM_PROMPT

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

    md_match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

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
# next_chunk) -- the model's reasoning is never passed -- so an identical
# answer to an identical prompt is a byte-identical judge call whose score
# should be deterministic. It is not: at JUDGE_TEMPERATURE=0.1 the judge
# returns a different score a meaningful fraction of the time on identical
# input, pure stochastic noise that GRPO turns into spurious within-group
# advantage. Duplicate answers are common within a prompt-group, so
# memoizing on the judge input gives every identical answer one frozen
# score (and cuts judge calls).
#
# The cache MUST be shared across ranks, not per-process: with
# per_device_train_batch_size=1 and num_generations=8 across multiple ranks,
# TRL hands each rank exactly one of a prompt's generations, so duplicate
# answers within a group live in DIFFERENT processes and a per-process dict
# never sees them.
#
# So the cache is backed by the shared filesystem, keyed by sha256 of the
# judge input. On a miss a rank takes an atomic mkdir claim on the key; the
# winner calls the judge and publishes the score, the losers block until it
# appears and reuse it. Only successful judgements are published -- parse
# errors and API exceptions stay uncached so they get another attempt. A
# crashed claim-holder just costs the waiters one timeout, after which they
# judge for themselves.
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
# ("<final_answer>\n{prose}\n<final_answer>{answer}</final_answer>") would
# otherwise match from the first open to the lone close and capture the whole
# explanation instead of just the answer.
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
    """Chars of reasoning before the model's closing </think> tag, or the
    whole text if it never closes (i.e. still reasoning when cut off)."""
    match = _THINKING_CLOSE_RE.search(text)
    if match:
        return match.start()
    return len(text)


def _post_reasoning_region(text: str) -> str | None:
    """Region of `text` to search for <final_answer>...</final_answer>, or
    None if there is no valid region to search (still reasoning / cut off).

    Strictly requires the </think> close-tag: all three supported families
    reliably close </think> before answering (baked into their own
    training), so absence means generation was cut off mid-reasoning, not
    that thinking was skipped -- there is genuinely no answer yet.
    """
    think_end = _THINKING_CLOSE_RE.search(text)
    if think_end:
        return text[think_end.end():]
    if _NOCOT_EVAL_FALLBACK:
        return text
    return None


def _well_formed_completion(text: str) -> bool:
    """True iff completion has </think> followed by EXACTLY ONE balanced
    <final_answer>...</final_answer> pair. Used to award a small format
    bonus during training so the policy learns the exact format the eval
    parsers look for.

    The "exactly one pair" requirement is load-bearing: a completion that
    merely contains a pair *somewhere* in the post-reasoning region can still
    wrap an explanation in an unclosed outer tag before the real answer
    (</think><final_answer>{explanation}<final_answer>{answer}</final_answer>),
    which is well-formed to a "does a pair exist" check but not to the
    extractor. Counting the tags rejects that shape while still accepting
    prose between the reasoning close and the tag.
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
    """No penalty below THINKING_CHAR_NO_PENALTY; linear up to
    THINKING_CHAR_LINEAR_END; capped at THINKING_PENALTY_FRACTION."""
    max_penalty = THINKING_PENALTY_FRACTION
    if num_chars < THINKING_CHAR_NO_PENALTY:
        return 0.0
    if num_chars <= THINKING_CHAR_LINEAR_END:
        span = THINKING_CHAR_LINEAR_END - THINKING_CHAR_NO_PENALTY
        frac = (num_chars - THINKING_CHAR_NO_PENALTY) / span
        return max_penalty * frac
    return max_penalty


def _last_final_answer(search_text: str) -> Optional[str]:
    """Content of the LAST balanced <final_answer>...</final_answer> pair, or
    None. The last pair, not the first: a model that quotes its own system
    prompt back to itself ("...<final_answer>...</final_answer> tags.") would
    otherwise have that quotation (itself a balanced pair) extracted instead
    of its real, later answer."""
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

    Strict: only accept a <final_answer>...</final_answer> tag emitted in
    the post-reasoning region (see _post_reasoning_region). If the model
    didn't close reasoning, return "" so the judge sees an empty answer and
    scores 0.
    """
    search_text = _post_reasoning_region(text)
    if search_text is None:
        return ""
    answer = _last_final_answer(search_text)
    return answer if answer is not None else ""


# =============================================================================
# GRPO reward — dispatches between rubric judge (val) and diagnosis judge (ext_val)
# =============================================================================


# Set True only while the trainer is inside evaluate() (see the wrapper applied
# to the trainer instance in main()). The case-completion reward branch below
# runs for BOTH training and val eval with identical columns, so it cannot tell
# them apart on its own -- without this flag it hardcoded task="val" for every
# case-completion record, meaning each training step's completions were logged
# as "val". ext_val is separately identifiable by its columns, so it's unaffected.
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
        "BETA=%s MASK_TRUNCATED=%s THINK_PENALTY=%s "
        "FORMAT_BONUS=%s TEMPERATURE=%s TOP_K=%s "
        "MAX_REWARD=%d MAX_COMPLETION=%d NUM_GENS=%d VAL_N=%d EXT_VAL_N=%d "
        "OUTPUT_DIR=%s",
        LEARNING_RATE, LR_SCHEDULER_TYPE, WARMUP_STEPS, EVAL_STEPS, SAVE_STEPS,
        BETA, MASK_TRUNCATED_COMPLETIONS, THINKING_PENALTY_FRACTION, FORMAT_BONUS,
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

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    grpo_kwargs: dict = {
        "output_dir": OUTPUT_DIR,
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
        # full-vocab logits. Off by default (liger-kernel isn't a required
        # dependency); opt in per-run with GRPO_USE_LIGER_KERNEL=1 (and
        # `pip install liger-kernel`).
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

    # Optional DeepSpeed ZeRO integration -- off by default (plain DDP, one
    # full replica per training GPU, is enough for these model sizes with
    # LoRA). Points at a deepspeed json with "auto" fields so it picks up
    # per_device_train_batch_size/gradient_accumulation_steps/etc. from this
    # same TrainingArguments instance rather than needing to be kept in sync
    # by hand.
    if "GRPO_DEEPSPEED_CONFIG" in os.environ:
        grpo_kwargs["deepspeed"] = os.environ["GRPO_DEEPSPEED_CONFIG"]

    if USE_VLLM:
        grpo_kwargs["use_vllm"] = True
        # Colocated vLLM in-process on every training GPU is the default. Set
        # explicitly rather than relying on TRL's default so this doesn't
        # silently flip to "server" mode if that default ever changes
        # upstream. Override with GRPO_VLLM_MODE=server to instead connect to
        # a separately-launched `trl vllm-serve` process on dedicated GPUs
        # (needed for larger models where colocating vLLM with training runs
        # out of memory). Requires GRPO_VLLM_SERVER_HOST/PORT.
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

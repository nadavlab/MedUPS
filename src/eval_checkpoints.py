#!/usr/bin/env python3
"""
Evaluate GRPO checkpoints on test.jsonl (case completion) and
ext_test.jsonl (final diagnosis). Both datasets use a simple binary
equivalence judge (DeepSeek); accuracy = fraction judged equivalent.

Backend: vLLM with LoRA adapters. The base model is loaded ONCE; each
checkpoint is swapped in via `LoRARequest` (essentially free). With
`--tp-size 2` the model is sharded across both GPUs.

Examples
--------
Single checkpoint:
  python eval_checkpoints.py \
      --checkpoint ./checkpoints/medical-grpo-qwen3.5-9b-midstream_task/checkpoint-400

All checkpoints + plot (uses all 8 GPUs):
  python eval_checkpoints.py --all --tp-size 8

Specific checkpoints + plot:
  python eval_checkpoints.py \
      --checkpoints ./checkpoints/medical-grpo-qwen3.5-9b-midstream_task/checkpoint-100 \
                    ./checkpoints/medical-grpo-qwen3.5-9b-midstream_task/checkpoint-200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# flashinfer_cubin.get_cubin_dir() hardcodes a path inside read-only
# site-packages; when flashinfer needs a cubin not shipped in the package
# (e.g. GLM-5.2's NVFP4 MoE / DSA sparse-attention kernels) it tries to
# download+write there and crashes with EROFS. Redirect to a writable dir
# (pre-seeded via symlink-mirror in the launching sbatch, see
# FLASHINFER_CUBIN_WRITABLE_DIR) before anything transitively imports
# flashinfer (trl/grpo_training pull it in even before the
# explicit `from vllm import LLM` inside load_vllm_engine()). No-op for
# every other family, which never sets this env var.
_fi_writable = os.environ.get("FLASHINFER_CUBIN_WRITABLE_DIR")
if _fi_writable:
    import flashinfer_cubin
    flashinfer_cubin.get_cubin_dir = lambda: _fi_writable

# Reuse prompts, judge client/retry, and helpers from the training script.
from grpo_training import (
    JUDGE_MAX_TOKENS,
    JUDGE_MAX_WORKERS,
    JUDGE_MODEL,
    JUDGE_RETRY_ATTEMPTS,
    JUDGE_TEMPERATURE,
    MODEL,
    _answer_for_judge,
    _extract_final_answer,
    _get_judge_client,
    _parse_judge_json,
    build_ext_messages,
    build_messages,
    evaluate_diagnosis,
    load_jsonl_records,
)
from grpo_training import _CHAT_TEMPLATE_KWARGS_BY_FAMILY, MODEL_FAMILY
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# --- defaults ---
DEFAULT_OUTPUT_DIR = "./checkpoints/medical-grpo-qwen3.5-9b-midstream_task"
DEFAULT_TEST_JSONL = "./data/test.jsonl"
DEFAULT_EXT_TEST_JSONL = "./data/ext_test.jsonl"
DEFAULT_RESULTS_DIR = "./eval_results"

# --- generation ---
# Training used 8192 + thinking_budget 6000; the base model often ignores the
# soft thinking_budget and burns the whole cap inside <think>, so give it more
# room to still emit the final-answer tags.
GEN_MAX_NEW_TOKENS = 16384
# Env-overridable so we can eval at a training temp (e.g. 0.9) as a robustness
# check. Default 0.6 preserves the standard near-greedy accuracy eval exactly.
GEN_TEMPERATURE = float(os.environ.get("EVAL_GEN_TEMPERATURE", "0.6"))
GEN_TOP_P = float(os.environ.get("EVAL_GEN_TOP_P", "0.95"))
GEN_TOP_K = int(os.environ.get("EVAL_GEN_TOP_K", "20"))
# Inference-only mode: generate + extract + save completions, but make NO
# judge API calls (accuracy is reported as None). Set via --skip-judge.
# Lets a full eval's GPU work run while the judge API is unavailable (e.g.
# out of balance) -- the saved *_details.json carry every completion,
# extracted answer, finish_reason and num_tokens, so a later judge-only
# pass can score them without re-running inference.
SKIP_JUDGE = False

THINKING_BUDGET_TOKENS = 6000
# Per-family, not a flat constant: reuses grpo_training.py's own
# _CHAT_TEMPLATE_KWARGS_BY_FAMILY (single source of truth with training),
# since not every family's chat template recognizes the same kwargs.
CHAT_TEMPLATE_KWARGS = _CHAT_TEMPLATE_KWARGS_BY_FAMILY[MODEL_FAMILY]

# --- vLLM engine defaults ---
DEFAULT_TP_SIZE = 2
DEFAULT_MAX_LORA_RANK = 32  # >= training LORA_R=16, headroom for future
DEFAULT_GPU_MEM_UTIL = 0.90
# Headroom for long medical case prompts + extended completion.
DEFAULT_MAX_MODEL_LEN = 49152


def setup_logging(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"eval_{ts}.log"
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(fh)
    root.addHandler(sh)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    return log_file


# =============================================================================
# Answer-equivalence judge (used for test.jsonl)
# =============================================================================

ANSWER_JUDGE_SYSTEM = "You are a clinical accuracy evaluator. Be concise but thorough."
ANSWER_JUDGE_USER_TEMPLATE = """\
You are a clinical expert evaluating whether a model's answer to a case
question matches the reference answer. Consider semantic equivalence and
clinical correctness — phrasing and synonyms do not matter, but the
clinical content (action, test, drug, finding, decision, or diagnosis)
must be the same.

Question: {question}

Model's answer: {pred}

Reference answer: {ref}

After analysis, conclude with JSON containing:
"judgment": 1 if the model's answer is equivalent to the reference, 0 otherwise"""


@retry(
    stop=stop_after_attempt(JUDGE_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def evaluate_answer_equivalence(question: str, pred: str, ref: str) -> float:
    """Return 1.0 if the judge marks model answer ≡ reference, else 0.0."""
    user_msg = ANSWER_JUDGE_USER_TEMPLATE.format(question=question, pred=pred, ref=ref)
    response = _get_judge_client().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    content = response.choices[0].message.content
    parsed = _parse_judge_json(content)
    if parsed is None:
        logger.debug("Answer judge unparseable response: %s", content[:200])
        return 0.0
    judgment = parsed.get("judgment", 0)
    if judgment in (1, "1", True):
        return 1.0
    return 0.0


# =============================================================================
# Checkpoint discovery
# =============================================================================


def checkpoint_step(path: Path) -> Optional[int]:
    m = re.match(r"^checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else None


def list_all_checkpoints(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    out = []
    for child in output_dir.iterdir():
        if child.is_dir() and checkpoint_step(child) is not None:
            out.append(child)
    out.sort(key=lambda p: checkpoint_step(p) or 0)
    return out


def base_model_from_checkpoint(ckpt: Path) -> str:
    """Extract the base model path/name from a LoRA checkpoint's adapter_config.json.

    Falls back to the MODEL constant from the training script if the file is
    missing or doesn't specify it.
    """
    ac = ckpt / "adapter_config.json"
    if ac.exists():
        try:
            with ac.open() as f:
                cfg = json.load(f)
            base = cfg.get("base_model_name_or_path")
            if base:
                return base
        except Exception as e:
            logger.warning("Could not parse %s: %s", ac, e)
    return MODEL


# =============================================================================
# vLLM engine + generation
# =============================================================================


def load_vllm_engine(
    base_model: str,
    tp_size: int,
    max_lora_rank: int,
    gpu_mem_util: float,
    max_model_len: int,
    enforce_eager: bool = False,
):
    """Create a single vLLM engine with LoRA enabled. Reused across checkpoints."""
    from vllm import LLM

    logger.info(
        "Loading vLLM engine: model=%s tp=%d max_lora_rank=%d gpu_mem=%.2f max_len=%d enforce_eager=%s",
        base_model,
        tp_size,
        max_lora_rank,
        gpu_mem_util,
        max_model_len,
        enforce_eager,
    )
    llm = LLM(
        model=base_model,
        tensor_parallel_size=tp_size,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        max_loras=1,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )
    return llm


def make_sampling_params(max_new_tokens: int = GEN_MAX_NEW_TOKENS):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        top_k=GEN_TOP_K,
        max_tokens=max_new_tokens,
    )


def vllm_chat(
    llm, tokenizer, messages_list: list[list[dict]], sampling, lora_request
) -> list[dict[str, Any]]:
    """Run a batch of chat conversations through vLLM.

    Pre-renders the chat template and calls llm.generate() on the resulting
    prompt strings rather than llm.chat(). llm.chat() was measured to stall
    badly on this workload (8/150 completions in 200s, vs 50-54s for all 150
    via llm.generate() with an identical engine/model/memory budget) -- root
    cause not fully diagnosed, but the pre-rendered-prompt path is a clean
    correctness-preserving workaround.

    Returns one dict per prompt with:
      - text: generated assistant text
      - finish_reason: "stop" (EOS/stop string) or "length" (hit max_tokens)
      - num_tokens: number of tokens generated
    """
    def _render(messages: list[dict]) -> str:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS
            )
        except Exception:
            # Some chat templates reject a separate system-role turn outright
            # (raises rather than silently ignoring it) -- fold system into
            # the leading user turn instead, same fallback used in every
            # reference-model smoke test (_smoke_*_parse*.py).
            if messages and messages[0].get("role") == "system":
                folded = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
                folded += messages[2:]
                return tokenizer.apply_chat_template(
                    folded, tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS
                )
            raise

    prompts = [_render(messages) for messages in messages_list]
    outputs = llm.generate(
        prompts,
        sampling_params=sampling,
        lora_request=lora_request,
        use_tqdm=True,
    )
    results = []
    for o in outputs:
        completion = o.outputs[0]
        results.append({
            "text": completion.text,
            "finish_reason": getattr(completion, "finish_reason", None),
            "num_tokens": len(getattr(completion, "token_ids", []) or []),
        })
    return results


# =============================================================================
# Per-dataset evaluation
# =============================================================================


def eval_test(
    llm,
    tokenizer,
    sampling,
    lora_request,
    test_records: list[dict[str, Any]],
    judge_workers: int,
) -> dict[str, Any]:
    """Case-completion eval with binary equivalence judge."""
    n = len(test_records)
    logger.info("Eval test: %d examples", n)

    messages_list = [
        build_messages(r["context_chunks"], r["question"]) for r in test_records
    ]
    gens = vllm_chat(llm, tokenizer, messages_list, sampling, lora_request)

    scores: list[Optional[float]] = [None] * n
    per_item: list[dict[str, Any]] = [{} for _ in range(n)]

    def _score(i: int) -> tuple[int, float, dict[str, Any]]:
        g = gens[i]
        full = g["text"]
        pred = _answer_for_judge(full)
        if SKIP_JUDGE:
            s = None  # inference-only; score later from saved completions
        elif not pred.strip():
            # No extractable answer at all -- score 0 without asking the
            # judge. Previously this called the judge on a literally blank
            # "pred", which is undefined behavior for an LLM judge (measured:
            # scored a blank prediction "equivalent" to the reference a
            # meaningful fraction of the time on a real run, job 134976 --
            # pure judge noise, not a real accuracy signal).
            s = 0.0
        else:
            try:
                s = evaluate_answer_equivalence(
                    question=test_records[i]["question"],
                    pred=pred,
                    ref=test_records[i]["answer"],
                )
            except Exception as e:
                logger.error("answer judge failed idx=%d: %s", i, e)
                s = 0.0
        item = {
            "completion": full,
            "extracted_answer": pred,
            "reference": test_records[i]["answer"],
            "score": s,
            "finish_reason": g["finish_reason"],
            "num_tokens": g["num_tokens"],
            "format_incomplete": "</think>" not in full.lower(),
        }
        return i, s, item

    with ThreadPoolExecutor(max_workers=max(1, min(judge_workers, n))) as ex:
        futs = [ex.submit(_score, i) for i in range(n)]
        for fut in as_completed(futs):
            idx, sc, item = fut.result()
            scores[idx] = sc
            per_item[idx] = item

    clean = [s for s in scores if s is not None]
    # SKIP_JUDGE -> every score is None; report accuracy as None rather than
    # a meaningless 0.0 that could be mistaken for "model got everything wrong".
    accuracy = None if SKIP_JUDGE else (sum(clean) / len(clean) if clean else 0.0)
    trunc_n = sum(1 for it in per_item if it.get("finish_reason") == "length")
    fmt_inc_n = sum(1 for it in per_item if it.get("format_incomplete"))
    mean_tokens = sum(it.get("num_tokens", 0) for it in per_item) / max(1, n)
    acc_str = "N/A (skip-judge)" if accuracy is None else f"{accuracy:.4f}"
    logger.info(
        "Test acc=%s  truncated=%d/%d (%.1f%%)  no-</think>=%d/%d (%.1f%%)  mean_tokens=%.0f",
        acc_str, trunc_n, n, 100 * trunc_n / max(1, n),
        fmt_inc_n, n, 100 * fmt_inc_n / max(1, n),
        mean_tokens,
    )
    return {
        "accuracy": accuracy,
        "n": n,
        "truncation_rate": trunc_n / max(1, n),
        "truncated_count": trunc_n,
        "format_incomplete_rate": fmt_inc_n / max(1, n),
        "mean_completion_tokens": mean_tokens,
        "items": per_item,
    }


def eval_ext_test(
    llm,
    tokenizer,
    sampling,
    lora_request,
    test_records: list[dict[str, Any]],
    judge_workers: int,
) -> dict[str, Any]:
    """External diagnosis eval. Accuracy = fraction judged equivalent."""
    norm = []
    for r in test_records:
        ct = (r.get("clean text") or r.get("clean_text") or "").strip()
        fd = (r.get("final diagnosis") or r.get("final_diagnosis") or "").strip()
        if ct and fd:
            norm.append({"clean_text": ct, "final_diagnosis": fd})
    n = len(norm)
    logger.info("Eval ext_test: %d examples (after filtering)", n)

    messages_list = [build_ext_messages(r["clean_text"]) for r in norm]
    gens = vllm_chat(llm, tokenizer, messages_list, sampling, lora_request)

    scores: list[Optional[float]] = [None] * n
    per_item: list[dict[str, Any]] = [{} for _ in range(n)]

    def _score(i: int) -> tuple[int, float, dict[str, Any]]:
        g = gens[i]
        full = g["text"]
        pred = _extract_final_answer(full)
        if SKIP_JUDGE:
            s = None  # inference-only; score later from saved completions
        elif not pred.strip():
            # See eval_test's identical guard above: calling the judge on a
            # blank "pred" is undefined behavior, not a real accuracy signal
            # (measured scoring a blank prediction "equivalent" 68% of the
            # time on a real run, job 134976).
            s = 0.0
        else:
            try:
                s = evaluate_diagnosis(pred, norm[i]["final_diagnosis"])
            except Exception as e:
                logger.error("diagnosis judge failed idx=%d: %s", i, e)
                s = 0.0
        low = full.lower()
        item = {
            "completion": full,
            "predicted_diagnosis": pred,
            "reference_diagnosis": norm[i]["final_diagnosis"],
            "score": s,
            "finish_reason": g["finish_reason"],
            "num_tokens": g["num_tokens"],
            # Ext task asks for <final_answer>...</final_answer>
            "format_incomplete": "<final_answer>" not in low or "</final_answer>" not in low,
        }
        return i, s, item

    with ThreadPoolExecutor(max_workers=max(1, min(judge_workers, n))) as ex:
        futs = [ex.submit(_score, i) for i in range(n)]
        for fut in as_completed(futs):
            idx, sc, item = fut.result()
            scores[idx] = sc
            per_item[idx] = item

    clean = [s for s in scores if s is not None]
    # SKIP_JUDGE -> every score is None; report accuracy as None rather than
    # a meaningless 0.0 that could be mistaken for "model got everything wrong".
    accuracy = None if SKIP_JUDGE else (sum(clean) / len(clean) if clean else 0.0)
    trunc_n = sum(1 for it in per_item if it.get("finish_reason") == "length")
    fmt_inc_n = sum(1 for it in per_item if it.get("format_incomplete"))
    mean_tokens = sum(it.get("num_tokens", 0) for it in per_item) / max(1, n)
    acc_str = "N/A (skip-judge)" if accuracy is None else f"{accuracy:.4f}"
    logger.info(
        "Ext-test acc=%s  truncated=%d/%d (%.1f%%)  no-<final_answer>=%d/%d (%.1f%%)  mean_tokens=%.0f",
        acc_str, trunc_n, n, 100 * trunc_n / max(1, n),
        fmt_inc_n, n, 100 * fmt_inc_n / max(1, n),
        mean_tokens,
    )
    return {
        "accuracy": accuracy,
        "n": n,
        "truncation_rate": trunc_n / max(1, n),
        "truncated_count": trunc_n,
        "format_incomplete_rate": fmt_inc_n / max(1, n),
        "mean_completion_tokens": mean_tokens,
        "items": per_item,
    }


# =============================================================================
# Per-checkpoint driver
# =============================================================================


def evaluate_checkpoint(
    llm,
    tokenizer,
    sampling,
    checkpoint_path: Optional[Path],
    lora_int_id: Optional[int],
    test_records: list[dict[str, Any]],
    ext_test_records: list[dict[str, Any]],
    judge_workers: int,
    results_dir: Path,
) -> dict[str, Any]:
    """Evaluate one checkpoint, or the base model if checkpoint_path is None.

    Base-model eval is labeled "base" with step=0 and passes no LoRA request.
    """
    if checkpoint_path is None:
        logger.info("=== Evaluating BASE model (no LoRA) ===")
        lora_request = None
        name = "base"
        step: Optional[int] = 0
        ckpt_str = "base"
    else:
        from vllm.lora.request import LoRARequest

        logger.info("=== Evaluating %s ===", checkpoint_path)
        lora_request = LoRARequest(
            lora_name=checkpoint_path.name,
            lora_int_id=lora_int_id or 1,
            lora_path=str(checkpoint_path),
        )
        name = checkpoint_path.name
        step = checkpoint_step(checkpoint_path)
        ckpt_str = str(checkpoint_path)

    test_res = eval_test(llm, tokenizer, sampling, lora_request, test_records, judge_workers)
    ext_res = eval_ext_test(
        llm, tokenizer, sampling, lora_request, ext_test_records, judge_workers
    )

    summary = {
        "checkpoint": ckpt_str,
        "step": step,
        "test_accuracy": test_res["accuracy"],
        "test_n": test_res["n"],
        "test_truncation_rate": test_res["truncation_rate"],
        "test_format_incomplete_rate": test_res["format_incomplete_rate"],
        "test_mean_completion_tokens": test_res["mean_completion_tokens"],
        "ext_test_accuracy": ext_res["accuracy"],
        "ext_test_n": ext_res["n"],
        "ext_test_truncation_rate": ext_res["truncation_rate"],
        "ext_test_format_incomplete_rate": ext_res["format_incomplete_rate"],
        "ext_test_mean_completion_tokens": ext_res["mean_completion_tokens"],
    }

    detail_path = results_dir / f"{name}_details.json"
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "test": test_res, "ext_test": ext_res},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Wrote details to %s", detail_path)
    return summary


# =============================================================================
# Plotting
# =============================================================================


def plot_results(summaries: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [s for s in summaries if s.get("step") is not None]
    rows.sort(key=lambda s: s["step"])
    steps = [s["step"] for s in rows]
    test_acc = [s["test_accuracy"] for s in rows]
    ext_acc = [s["ext_test_accuracy"] for s in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, test_acc, marker="o", label="test")
    plt.plot(steps, ext_acc, marker="s", label="ext_test")
    plt.xlabel("checkpoint step")
    plt.ylabel("accuracy")
    plt.title("Checkpoint accuracy on test vs ext_test")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Plot saved to %s", out_path)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=str,
                   help='Path to a single checkpoint dir, or "base" for the un-tuned base model')
    g.add_argument(
        "--checkpoints",
        nargs="+",
        help='Paths to multiple checkpoint dirs (will plot results). Use "base" as a token to include the base model.',
    )
    g.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every checkpoint-N under --output-dir (will plot results)",
    )
    ap.add_argument(
        "--include-base",
        action="store_true",
        help="Also evaluate the base model (labeled as step 0). Combine with --all or --checkpoints.",
    )

    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                    help="Training output dir containing checkpoint-N folders (for --all)")
    ap.add_argument("--test-jsonl", type=str, default=DEFAULT_TEST_JSONL)
    ap.add_argument("--ext-test-jsonl", type=str, default=DEFAULT_EXT_TEST_JSONL)
    ap.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                    help="Where to write per-checkpoint JSON details + plot")
    ap.add_argument("--judge-workers", type=int, default=JUDGE_MAX_WORKERS,
                    help="Parallel judge API calls")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on examples per dataset (for quick smoke tests)")
    # vLLM engine settings
    ap.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE,
                    help="vLLM tensor_parallel_size (set to number of GPUs to use)")
    ap.add_argument("--max-lora-rank", type=int, default=DEFAULT_MAX_LORA_RANK)
    ap.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    ap.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--enforce-eager", action="store_true",
                     help="Disable torch.compile/CUDAGraphs (needed for experimental quant "
                          "paths like GLM-5.2's NVFP4 + DSA sparse attention).")
    ap.add_argument("--skip-judge", action="store_true",
                     help="Inference only: generate and save completions, make NO judge API "
                          "calls (accuracy reported as None). Use when the judge API is "
                          "unavailable; score later from the saved *_details.json.")
    ap.add_argument("--max-new-tokens", type=int, default=GEN_MAX_NEW_TOKENS,
                    help="Per-completion generation cap. Raise if completions get truncated mid-think.")
    ap.add_argument("--base-model", type=str, default=None,
                    help="Override base model (default: read from first checkpoint's adapter_config.json)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    global SKIP_JUDGE
    SKIP_JUDGE = args.skip_judge
    results_dir = Path(args.results_dir)
    log_file = setup_logging(results_dir)
    logger.info("Logging to %s", log_file)

    # Resolve checkpoints. Use None as a sentinel for "base model, no LoRA".
    ckpts: list[Optional[Path]] = []
    if args.checkpoint:
        if args.checkpoint.lower() == "base":
            ckpts = [None]
        else:
            ckpts = [Path(args.checkpoint)]
    elif args.checkpoints:
        for p in args.checkpoints:
            ckpts.append(None if p.lower() == "base" else Path(p))
    else:  # --all
        ckpts = list_all_checkpoints(Path(args.output_dir))
        if not ckpts:
            logger.error("No checkpoint-N folders found under %s", args.output_dir)
            sys.exit(1)

    # Prepend base model if requested (and not already in the list).
    if args.include_base and None not in ckpts:
        ckpts = [None] + ckpts

    logger.info("Will evaluate %d checkpoint(s):", len(ckpts))
    for p in ckpts:
        logger.info("  - %s", "base" if p is None else p)

    # Load datasets once
    logger.info("Loading test records from %s", args.test_jsonl)
    test_records = load_jsonl_records(args.test_jsonl)
    logger.info("Loading ext_test records from %s", args.ext_test_jsonl)
    ext_test_records = load_jsonl_records(args.ext_test_jsonl)
    if args.limit:
        test_records = test_records[: args.limit]
        ext_test_records = ext_test_records[: args.limit]
    logger.info("test: %d | ext_test: %d", len(test_records), len(ext_test_records))

    # Load engine ONCE, reuse across all checkpoints.
    # Determine base model from --base-model, else first non-base checkpoint, else MODEL.
    if args.base_model:
        base_model = args.base_model
    else:
        first_ckpt = next((c for c in ckpts if c is not None), None)
        base_model = base_model_from_checkpoint(first_ckpt) if first_ckpt else MODEL
    llm = load_vllm_engine(
        base_model=base_model,
        tp_size=args.tp_size,
        max_lora_rank=args.max_lora_rank,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    sampling = make_sampling_params(args.max_new_tokens)

    # Evaluate
    summaries: list[dict[str, Any]] = []
    for idx, ck in enumerate(ckpts, start=1):
        label = "base" if ck is None else ck.name
        try:
            summary = evaluate_checkpoint(
                llm=llm,
                tokenizer=tokenizer,
                sampling=sampling,
                checkpoint_path=ck,
                lora_int_id=idx,
                test_records=test_records,
                ext_test_records=ext_test_records,
                judge_workers=args.judge_workers,
                results_dir=results_dir,
            )
            summaries.append(summary)
            _ta = summary["test_accuracy"]; _ea = summary["ext_test_accuracy"]
            logger.info(
                "[%s] test=%s (trunc=%.1f%%)  ext_test=%s (trunc=%.1f%%)",
                label,
                "N/A" if _ta is None else f"{_ta:.4f}",
                100 * summary["test_truncation_rate"],
                "N/A" if _ea is None else f"{_ea:.4f}",
                100 * summary["ext_test_truncation_rate"],
            )
        except Exception as e:
            logger.exception("Checkpoint %s failed: %s", label, e)

    # Aggregate summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = results_dir / f"summary_{ts}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    logger.info("Summary saved to %s", summary_path)

    # Plot when multi-checkpoint
    if len(summaries) >= 2:
        plot_path = results_dir / f"accuracy_vs_step_{ts}.png"
        plot_results(summaries, plot_path)
    else:
        logger.info("Single checkpoint — skipping plot.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Few-shot variant of eval_checkpoints.py.

Same vLLM/LoRA backend, checkpoint discovery, judges, and single-pool
accuracy reporting as eval_checkpoints.py -- the only addition is in-context
few-shot demonstrations, reusing the exact few-shot mechanics (shot loading,
prompt assembly, leakage filtering) already built and used in
eval_openai.py for the frontier-OpenAI-model comparison. This script
generates via vLLM instead of the OpenAI API, so it can few-shot-prompt our
own base models / GRPO checkpoints.

Few-shot mechanics imported unchanged from eval_openai.py:
  - load_shots(): reads shots.jsonl, takes first K (or --shot-seed sampled K)
    demonstrations for a task.
  - build_fewshot_prefix() / apply_fewshot(): renders shots as either one
    inline text block (default) prepended to the query's user turn, or as
    real user/assistant message turns -- see --shot-placement.
  - drop_leaked_records(): removes evaluation records that are themselves
    used as demonstrations, matched on exact (question, answer) text. This is
    NOT the same as excluding by case_id -- a case_id in this dataset can
    have several decision-point rows (same clinical case, different question/
    answer at a different chunk cutoff), so matching on case_id would drop
    far more than the 5 literal shot rows. Verified: shots.jsonl's case_id
    values each map to multiple test.jsonl rows, but each shot's own
    (question, answer) pair matches exactly one of them.

Examples
--------
5-shot on the base model, test.jsonl only:
  python eval_checkpoints_fewshot.py --checkpoint base \\
      --k-shot 5 --shots-jsonl shots.jsonl --skip-ext-test

Zero-shot (k=0) reproduces eval_checkpoints.py's own numbers exactly, since
apply_fewshot() with an empty prefix is a no-op on the built messages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from eval_openai import (  # noqa: E402
    DEFAULT_SHOT_PLACEMENT,
    SHOT_TASK_EXT,
    SHOT_TASK_TEST,
    apply_fewshot,
    build_fewshot_prefix,
    drop_leaked_records,
    load_shots,
    shots_fingerprint,
)

# --- reuse the vLLM/LoRA backend, judges, and CLI plumbing wholesale -------
from eval_checkpoints import (  # noqa: E402
    DEFAULT_EXT_TEST_JSONL,
    DEFAULT_GPU_MEM_UTIL,
    DEFAULT_MAX_LORA_RANK,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEST_JSONL,
    DEFAULT_TP_SIZE,
    GEN_MAX_NEW_TOKENS,
    base_model_from_checkpoint,
    checkpoint_step,
    evaluate_answer_equivalence,
    list_all_checkpoints,
    load_vllm_engine,
    make_sampling_params,
    plot_results,
    vllm_chat,
)
from grpo_training import (  # noqa: E402
    JUDGE_MAX_WORKERS,
    MODEL,
    _answer_for_judge,
    _extract_final_answer,
    build_ext_messages,
    build_messages,
    evaluate_diagnosis,
    load_jsonl_records,
)
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

logger = logging.getLogger(__name__)


def setup_logging(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"eval_fewshot_{ts}.log"
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


def _log_prompt_preview(messages: list[dict], label: str, max_chars: int = 900) -> None:
    logger.info("[%s] prompt preview (%d message turns):", label, len(messages))
    for m in messages:
        content = str(m.get("content", ""))
        if len(content) > max_chars:
            content = content[:max_chars] + f"... [+{len(content) - max_chars} chars]"
        logger.info("  [%s] %s: %s", label, m.get("role"), content.replace("\n", " | "))


# =============================================================================
# Per-dataset evaluation (identical to eval_checkpoints.py's, plus a
# fewshot_prefix injected into every built message list via apply_fewshot)
# =============================================================================


def eval_test_fewshot(
    llm, tokenizer, sampling, lora_request,
    test_records: list[dict[str, Any]],
    judge_workers: int,
    fewshot_prefix: Any,
    k_shot: int,
) -> dict[str, Any]:
    n = len(test_records)
    logger.info("Eval test: %d examples (%d-shot)", n, k_shot)

    messages_list = [
        apply_fewshot(build_messages(r["context_chunks"], r["question"]), fewshot_prefix)
        for r in test_records
    ]
    if messages_list:
        _log_prompt_preview(messages_list[0], "test")
    gens = vllm_chat(llm, tokenizer, messages_list, sampling, lora_request)

    scores: list[Optional[float]] = [None] * n
    per_item: list[dict[str, Any]] = [{} for _ in range(n)]

    def _score(i: int) -> tuple[int, float, dict[str, Any]]:
        g = gens[i]
        full = g["text"]
        pred = _answer_for_judge(full)
        if not pred.strip():
            s = 0.0
        else:
            try:
                s = evaluate_answer_equivalence(
                    question=test_records[i]["question"], pred=pred, ref=test_records[i]["answer"],
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
    accuracy = sum(clean) / len(clean) if clean else 0.0
    trunc_n = sum(1 for it in per_item if it.get("finish_reason") == "length")
    fmt_inc_n = sum(1 for it in per_item if it.get("format_incomplete"))
    mean_tokens = sum(it.get("num_tokens", 0) for it in per_item) / max(1, n)
    logger.info(
        "Test (%d-shot) acc=%.4f  truncated=%d/%d (%.1f%%)  no-</think>=%d/%d (%.1f%%)  mean_tokens=%.0f",
        k_shot, accuracy, trunc_n, n, 100 * trunc_n / max(1, n),
        fmt_inc_n, n, 100 * fmt_inc_n / max(1, n), mean_tokens,
    )
    return {
        "accuracy": accuracy, "n": n, "k_shot": k_shot,
        "truncation_rate": trunc_n / max(1, n), "truncated_count": trunc_n,
        "format_incomplete_rate": fmt_inc_n / max(1, n),
        "mean_completion_tokens": mean_tokens, "items": per_item,
    }


def eval_ext_test_fewshot(
    llm, tokenizer, sampling, lora_request,
    test_records: list[dict[str, Any]],
    judge_workers: int,
    fewshot_prefix: Any,
    k_shot: int,
) -> dict[str, Any]:
    norm = []
    for r in test_records:
        ct = (r.get("clean text") or r.get("clean_text") or "").strip()
        fd = (r.get("final diagnosis") or r.get("final_diagnosis") or "").strip()
        if ct and fd:
            norm.append({"clean_text": ct, "final_diagnosis": fd})
    n = len(norm)
    logger.info("Eval ext_test: %d examples (after filtering, %d-shot)", n, k_shot)

    messages_list = [
        apply_fewshot(build_ext_messages(r["clean_text"]), fewshot_prefix) for r in norm
    ]
    if messages_list:
        _log_prompt_preview(messages_list[0], "ext_test")
    gens = vllm_chat(llm, tokenizer, messages_list, sampling, lora_request)

    scores: list[Optional[float]] = [None] * n
    per_item: list[dict[str, Any]] = [{} for _ in range(n)]

    def _score(i: int) -> tuple[int, float, dict[str, Any]]:
        g = gens[i]
        full = g["text"]
        pred = _extract_final_answer(full)
        if not pred.strip():
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
    accuracy = sum(clean) / len(clean) if clean else 0.0
    trunc_n = sum(1 for it in per_item if it.get("finish_reason") == "length")
    fmt_inc_n = sum(1 for it in per_item if it.get("format_incomplete"))
    mean_tokens = sum(it.get("num_tokens", 0) for it in per_item) / max(1, n)
    logger.info(
        "Ext-test (%d-shot) acc=%.4f  truncated=%d/%d (%.1f%%)  no-<final_answer>=%d/%d (%.1f%%)  mean_tokens=%.0f",
        k_shot, accuracy, trunc_n, n, 100 * trunc_n / max(1, n),
        fmt_inc_n, n, 100 * fmt_inc_n / max(1, n), mean_tokens,
    )
    return {
        "accuracy": accuracy, "n": n, "k_shot": k_shot,
        "truncation_rate": trunc_n / max(1, n), "truncated_count": trunc_n,
        "format_incomplete_rate": fmt_inc_n / max(1, n),
        "mean_completion_tokens": mean_tokens, "items": per_item,
    }


# =============================================================================
# Per-checkpoint driver
# =============================================================================


def evaluate_checkpoint_fewshot(
    llm, tokenizer, sampling,
    checkpoint_path: Optional[Path],
    lora_int_id: Optional[int],
    test_records: list[dict[str, Any]],
    ext_test_records: list[dict[str, Any]],
    judge_workers: int,
    results_dir: Path,
    test_prefix: Any,
    ext_prefix: Any,
    k_shot: int,
    skip_test: bool,
    skip_ext_test: bool,
    tag: str,
) -> dict[str, Any]:
    if checkpoint_path is None:
        logger.info("=== Evaluating BASE model (no LoRA), %d-shot ===", k_shot)
        lora_request = None
        name = f"base_{tag}"
        step: Optional[int] = 0
        ckpt_str = "base"
    else:
        from vllm.lora.request import LoRARequest

        logger.info("=== Evaluating %s, %d-shot ===", checkpoint_path, k_shot)
        lora_request = LoRARequest(
            lora_name=checkpoint_path.name, lora_int_id=lora_int_id or 1,
            lora_path=str(checkpoint_path),
        )
        name = f"{checkpoint_path.name}_{tag}"
        step = checkpoint_step(checkpoint_path)
        ckpt_str = str(checkpoint_path)

    summary: dict[str, Any] = {"checkpoint": ckpt_str, "step": step, "k_shot": k_shot}
    test_res = ext_res = None

    if not skip_test:
        test_res = eval_test_fewshot(
            llm, tokenizer, sampling, lora_request, test_records, judge_workers,
            test_prefix, k_shot,
        )
        summary.update({
            "test_accuracy": test_res["accuracy"], "test_n": test_res["n"],
            "test_truncation_rate": test_res["truncation_rate"],
            "test_format_incomplete_rate": test_res["format_incomplete_rate"],
            "test_mean_completion_tokens": test_res["mean_completion_tokens"],
        })
    if not skip_ext_test:
        ext_res = eval_ext_test_fewshot(
            llm, tokenizer, sampling, lora_request, ext_test_records, judge_workers,
            ext_prefix, k_shot,
        )
        summary.update({
            "ext_test_accuracy": ext_res["accuracy"], "ext_test_n": ext_res["n"],
            "ext_test_truncation_rate": ext_res["truncation_rate"],
            "ext_test_format_incomplete_rate": ext_res["format_incomplete_rate"],
            "ext_test_mean_completion_tokens": ext_res["mean_completion_tokens"],
        })

    detail_path = results_dir / f"{name}_details.json"
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "test": test_res, "ext_test": ext_res},
            f, ensure_ascii=False, indent=2,
        )
    logger.info("Wrote details to %s", detail_path)
    return summary


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
    g.add_argument("--checkpoints", nargs="+",
                   help='Paths to multiple checkpoint dirs. Use "base" as a token to include the base model.')
    g.add_argument("--all", action="store_true",
                   help="Evaluate every checkpoint-N under --output-dir")
    ap.add_argument("--include-base", action="store_true",
                    help="Also evaluate the base model. Combine with --all or --checkpoints.")

    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--test-jsonl", type=str, default=DEFAULT_TEST_JSONL)
    ap.add_argument("--ext-test-jsonl", type=str, default=DEFAULT_EXT_TEST_JSONL)
    ap.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--judge-workers", type=int, default=JUDGE_MAX_WORKERS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-test", action="store_true", help="Skip test.jsonl (case completion)")
    ap.add_argument("--skip-ext-test", action="store_true", help="Skip ext_test.jsonl (final diagnosis)")

    # vLLM engine settings (identical to eval_checkpoints.py)
    ap.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    ap.add_argument("--max-lora-rank", type=int, default=DEFAULT_MAX_LORA_RANK)
    ap.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    ap.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    ap.add_argument("--base-model", type=str, default=None)

    # Few-shot (same flag names/semantics as eval_openai.py)
    ap.add_argument("-k", "--k-shot", type=int, default=0,
                    help="Number of in-context demonstrations (default: 0 = zero-shot)")
    ap.add_argument("--shots-jsonl", type=str, default=None,
                    help="Demonstrations file (required if --k-shot > 0)")
    ap.add_argument("--ext-shots-jsonl", type=str, default=None,
                    help="Separate shots file for ext_test (default: same as --shots-jsonl)")
    ap.add_argument("--shot-seed", type=int, default=None,
                    help="Sample K shots at random with this seed; omit for first-K in file order")
    ap.add_argument("--shot-placement", type=str, default=DEFAULT_SHOT_PLACEMENT,
                    choices=("inline", "turns"))
    ap.add_argument("--shot-style", type=str, default="auto",
                    choices=("auto", "with_reasoning", "answer_only"))
    ap.add_argument("--allow-shot-leakage", action="store_true",
                    help="Keep evaluation items that also appear in shots.jsonl (dropped by default)")

    args = ap.parse_args()
    if args.k_shot < 0:
        ap.error("--k-shot must be >= 0")
    if args.k_shot > 0 and not args.shots_jsonl:
        ap.error("--shots-jsonl is required when --k-shot > 0")
    return args


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    log_file = setup_logging(results_dir)
    logger.info("Logging to %s", log_file)

    ckpts: list[Optional[Path]] = []
    if args.checkpoint:
        ckpts = [None] if args.checkpoint.lower() == "base" else [Path(args.checkpoint)]
    elif args.checkpoints:
        for p in args.checkpoints:
            ckpts.append(None if p.lower() == "base" else Path(p))
    else:
        ckpts = list_all_checkpoints(Path(args.output_dir))
        if not ckpts:
            logger.error("No checkpoint-N folders found under %s", args.output_dir)
            sys.exit(1)
    if args.include_base and None not in ckpts:
        ckpts = [None] + ckpts

    logger.info("Will evaluate %d checkpoint(s), k_shot=%d:", len(ckpts), args.k_shot)
    for p in ckpts:
        logger.info("  - %s", "base" if p is None else p)

    logger.info("Loading test records from %s", args.test_jsonl)
    test_records = load_jsonl_records(args.test_jsonl)
    logger.info("Loading ext_test records from %s", args.ext_test_jsonl)
    ext_test_records = load_jsonl_records(args.ext_test_jsonl)
    if args.limit:
        test_records = test_records[: args.limit]
        ext_test_records = ext_test_records[: args.limit]
    logger.info("test pool (pre-shot-exclusion): %d | ext_test pool: %d",
                len(test_records), len(ext_test_records))

    # --- few-shot setup ------------------------------------------------
    ext_shots_path = args.ext_shots_jsonl or args.shots_jsonl
    test_shots: list[dict[str, Any]] = []
    ext_shots: list[dict[str, Any]] = []
    empty_prefix: Any = [] if args.shot_placement == "turns" else ""
    test_prefix: Any = empty_prefix
    ext_prefix: Any = empty_prefix

    if args.k_shot > 0:
        if not args.skip_test:
            test_shots = load_shots(args.shots_jsonl, args.k_shot, SHOT_TASK_TEST, args.shot_seed)
            test_prefix = build_fewshot_prefix(test_shots, SHOT_TASK_TEST, args.shot_style, args.shot_placement)
            if not args.allow_shot_leakage:
                before = len(test_records)
                test_records = drop_leaked_records(test_records, test_shots, SHOT_TASK_TEST)
                logger.info("test pool after excluding %d shot(s): %d -> %d",
                            len(test_shots), before, len(test_records))
        if not args.skip_ext_test:
            ext_shots = load_shots(ext_shots_path, args.k_shot, SHOT_TASK_EXT, args.shot_seed, required=False)
            ext_prefix = build_fewshot_prefix(ext_shots, SHOT_TASK_EXT, args.shot_style, args.shot_placement)
            if not args.allow_shot_leakage:
                before = len(ext_test_records)
                ext_test_records = drop_leaked_records(ext_test_records, ext_shots, SHOT_TASK_EXT)
                logger.info("ext_test pool after excluding %d shot(s): %d -> %d",
                            len(ext_shots), before, len(ext_test_records))

    logger.info("test pool (final): %d | ext_test pool (final): %d",
                len(test_records), len(ext_test_records))

    if args.base_model:
        base_model = args.base_model
    else:
        first_ckpt = next((c for c in ckpts if c is not None), None)
        base_model = base_model_from_checkpoint(first_ckpt) if first_ckpt else MODEL
    llm = load_vllm_engine(
        base_model=base_model, tp_size=args.tp_size, max_lora_rank=args.max_lora_rank,
        gpu_mem_util=args.gpu_mem_util, max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    sampling = make_sampling_params(args.max_new_tokens)

    tag = f"k{args.k_shot}"
    summaries: list[dict[str, Any]] = []
    for idx, ck in enumerate(ckpts, start=1):
        label = "base" if ck is None else ck.name
        try:
            summary = evaluate_checkpoint_fewshot(
                llm=llm, tokenizer=tokenizer, sampling=sampling, checkpoint_path=ck,
                lora_int_id=idx, test_records=test_records, ext_test_records=ext_test_records,
                judge_workers=args.judge_workers, results_dir=results_dir,
                test_prefix=test_prefix, ext_prefix=ext_prefix, k_shot=args.k_shot,
                skip_test=args.skip_test, skip_ext_test=args.skip_ext_test, tag=tag,
            )
            summaries.append(summary)
            _ta = summary.get("test_accuracy"); _ea = summary.get("ext_test_accuracy")
            logger.info(
                "[%s] k=%d test=%s  ext_test=%s", label, args.k_shot,
                "skipped" if _ta is None else f"{_ta:.4f}",
                "skipped" if _ea is None else f"{_ea:.4f}",
            )
        except Exception as e:
            logger.exception("Checkpoint %s failed: %s", label, e)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_out = {
        "k_shot": args.k_shot,
        "shot_placement": args.shot_placement if args.k_shot else None,
        "shot_style": args.shot_style if args.k_shot else None,
        "shot_seed": args.shot_seed if args.k_shot else None,
        "shots_jsonl": args.shots_jsonl if args.k_shot else None,
        "test_shot_fingerprint": shots_fingerprint(test_shots, args.shot_style, args.shot_placement),
        "ext_test_shot_fingerprint": shots_fingerprint(ext_shots, args.shot_style, args.shot_placement),
        "test_shot_lines": [s["line_no"] for s in test_shots],
        "ext_test_shot_lines": [s["line_no"] for s in ext_shots],
        "checkpoints": summaries,
    }
    summary_path = results_dir / f"summary_fewshot_{tag}_{ts}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    logger.info("Summary saved to %s", summary_path)

    if len(summaries) >= 2:
        plot_path = results_dir / f"accuracy_vs_step_fewshot_{tag}_{ts}.png"
        plot_results(summaries, plot_path)


if __name__ == "__main__":
    main()

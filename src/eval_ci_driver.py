#!/usr/bin/env python3
"""
Compute 95% confidence intervals for base-vs-GRPO-checkpoint accuracy, over
multiple runs of randomly-sampled subsets of test.jsonl/ext_test.jsonl.

Reuses eval_checkpoints.py's own engine-loading and evaluate_checkpoint()
logic directly (same judge, same generation params, same accuracy
definition) -- this script only adds: (a) seeded random subsampling per run,
and (b) aggregating multiple runs' accuracies into a mean + 95% CI (Student's
t, since each family/checkpoint only gets a handful of runs).

The vLLM engine is loaded ONCE per invocation and reused across every
seed x {base, checkpoint} combination, since engine boot for a 27B model is
expensive and this only needs to happen once regardless of how many random
subsets are evaluated.

Example
-------
GRPO_MODEL_FAMILY=qwen36 python eval_ci_driver.py \
    --checkpoint ./checkpoints/medical-grpo-qwen3.6-27b-midstream_task-v3-explore/checkpoint-800 \
    --seeds 1001 1002 1003 --sample-size 250 --tp-size 8 \
    --results-dir ./eval_results/ci_qwen36_ckpt800
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from eval_checkpoints import (
    DEFAULT_EXT_TEST_JSONL,
    DEFAULT_GPU_MEM_UTIL,
    DEFAULT_MAX_LORA_RANK,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_TEST_JSONL,
    DEFAULT_TP_SIZE,
    GEN_MAX_NEW_TOKENS,
    base_model_from_checkpoint,
    evaluate_checkpoint,
    load_vllm_engine,
    make_sampling_params,
    setup_logging,
)
from grpo_training_new_code import load_jsonl_records

logger = logging.getLogger(__name__)

# Student's t critical values (two-tailed, 95%) for small df -- avoids the
# normal-approximation 1.96 being anti-conservative at n=3 (df=2 -> t=4.303).
_T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def ci95(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return {"mean": mean, "std": 0.0, "n": n, "ci_low": mean, "ci_high": mean, "half_width": 0.0}
    std = statistics.stdev(values)
    sem = std / (n**0.5)
    t = _T_CRIT_95.get(n - 1, 2.0)  # fall back to ~normal for large n
    half = t * sem
    return {
        "mean": mean,
        "std": std,
        "n": n,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "half_width": half,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="Path to the GRPO checkpoint to compare against base. Omit for a "
                          "base-only reference-model CI (requires --base-model instead).")
    ap.add_argument("--base-model", type=str, default=None,
                     help="Override the base model repo/path directly, bypassing "
                          "base_model_from_checkpoint(). Required when --checkpoint is omitted.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    ap.add_argument("--sample-size", type=int, default=250)
    ap.add_argument("--test-jsonl", type=str, default=DEFAULT_TEST_JSONL)
    ap.add_argument("--ext-test-jsonl", type=str, default=DEFAULT_EXT_TEST_JSONL)
    ap.add_argument("--results-dir", type=str, required=True)
    ap.add_argument("--judge-workers", type=int, default=100)
    ap.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    ap.add_argument("--max-lora-rank", type=int, default=DEFAULT_MAX_LORA_RANK)
    ap.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    ap.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--skip-judge", action="store_true",
                     help="Inference only: generate and save completions, make NO judge API "
                          "calls. CI aggregation is skipped; score later from the saved "
                          "seed_<N>/*_details.json without re-running inference.")
    ap.add_argument("--enforce-eager", action="store_true",
                     help="Disable torch.compile/CUDAGraphs (needed for experimental quant "
                          "paths like GLM-5.2's NVFP4 + DSA sparse attention).")
    ap.add_argument("--max-new-tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    import eval_checkpoints as _ec
    _ec.SKIP_JUDGE = args.skip_judge
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(results_dir)
    logger.info("Logging to %s", log_file)

    if args.checkpoint is None and args.base_model is None:
        raise SystemExit("Must pass --checkpoint (base-vs-checkpoint CI) or --base-model "
                          "(base-only reference-model CI, no checkpoint comparison).")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if args.base_model:
        base_model = args.base_model
    else:
        base_model = base_model_from_checkpoint(ckpt_path)
    logger.info("Checkpoint: %s", ckpt_path if ckpt_path else "(none -- base-only reference eval)")
    logger.info("Base model: %s", base_model)
    logger.info("Seeds: %s | sample_size=%d", args.seeds, args.sample_size)

    test_records_full = load_jsonl_records(args.test_jsonl)
    ext_test_records_full = load_jsonl_records(args.ext_test_jsonl)
    logger.info(
        "Full pools: test=%d ext_test=%d", len(test_records_full), len(ext_test_records_full)
    )

    def _seed_is_cached(seed: int) -> bool:
        seed_dir = results_dir / f"seed_{seed}"
        if not (seed_dir / "base_details.json").exists():
            return False
        if ckpt_path is None:
            return True
        return (seed_dir / f"{ckpt_path.name}_details.json").exists()

    needs_engine = any(not _seed_is_cached(s) for s in args.seeds)
    llm = tokenizer = sampling = None
    if needs_engine:
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
    else:
        logger.info("All requested seeds already cached -- skipping vLLM engine load entirely.")

    base_test_acc, base_ext_acc = [], []
    ckpt_test_acc, ckpt_ext_acc = [], []
    all_runs: list[dict[str, Any]] = []

    for seed in args.seeds:
        seed_dir = results_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Resumable: a seed already fully evaluated by a PRIOR invocation of
        # this script (e.g. when growing 3 runs -> 5 runs to tighten the CI)
        # is loaded from its cached *_details.json rather than re-run --
        # avoids re-paying judge/generation cost for seeds already done,
        # which matters most for expensive models (e.g. Qwen3.6-27B).
        base_cache = seed_dir / "base_details.json"
        ckpt_cache = seed_dir / f"{ckpt_path.name}_details.json" if ckpt_path else None
        if base_cache.exists() and (ckpt_path is None or ckpt_cache.exists()):
            logger.info("=== seed=%d: found cached results, skipping re-eval ===", seed)
            base_summary = json.loads(base_cache.read_text(encoding="utf-8"))["summary"]
            ckpt_summary = (
                json.loads(ckpt_cache.read_text(encoding="utf-8"))["summary"] if ckpt_path else None
            )
        else:
            rng = random.Random(seed)
            n_test = min(args.sample_size, len(test_records_full))
            n_ext = min(args.sample_size, len(ext_test_records_full))
            test_sample = rng.sample(test_records_full, n_test)
            ext_sample = rng.sample(ext_test_records_full, n_ext)

            logger.info("=== seed=%d (test n=%d, ext_test n=%d) ===", seed, n_test, n_ext)

            base_summary = evaluate_checkpoint(
                llm=llm,
                tokenizer=tokenizer,
                sampling=sampling,
                checkpoint_path=None,
                lora_int_id=None,
                test_records=test_sample,
                ext_test_records=ext_sample,
                judge_workers=args.judge_workers,
                results_dir=seed_dir,
            )
            ckpt_summary = None
            if ckpt_path is not None:
                ckpt_summary = evaluate_checkpoint(
                    llm=llm,
                    tokenizer=tokenizer,
                    sampling=sampling,
                    checkpoint_path=ckpt_path,
                    lora_int_id=1,
                    test_records=test_sample,
                    ext_test_records=ext_sample,
                    judge_workers=args.judge_workers,
                    results_dir=seed_dir,
                )

        if not args.skip_judge:
            base_test_acc.append(base_summary["test_accuracy"])
            base_ext_acc.append(base_summary["ext_test_accuracy"])
            if ckpt_summary is not None:
                ckpt_test_acc.append(ckpt_summary["test_accuracy"])
                ckpt_ext_acc.append(ckpt_summary["ext_test_accuracy"])

        if args.skip_judge:
            logger.info(
                "[seed=%d] INFERENCE ONLY -- test n=%s truncated=%.1f%% mean_tokens=%.0f",
                seed,
                base_summary.get("test_n"),
                100 * (base_summary.get("test_truncation_rate") or 0),
                base_summary.get("test_mean_completion_tokens") or 0,
            )
        elif ckpt_summary is not None:
            logger.info(
                "[seed=%d] base test=%.4f ext=%.4f | ckpt test=%.4f ext=%.4f",
                seed,
                base_summary["test_accuracy"],
                base_summary["ext_test_accuracy"],
                ckpt_summary["test_accuracy"],
                ckpt_summary["ext_test_accuracy"],
            )
        else:
            logger.info(
                "[seed=%d] base test=%.4f ext=%.4f",
                seed,
                base_summary["test_accuracy"],
                base_summary["ext_test_accuracy"],
            )

        run_entry = {"seed": seed, "base": base_summary}
        if ckpt_summary is not None:
            run_entry["checkpoint"] = ckpt_summary
        all_runs.append(run_entry)

    if args.skip_judge:
        # Inference-only: every accuracy is None, so there is nothing to
        # aggregate. Completions ARE saved in seed_<N>/*_details.json for a
        # later judge-only pass.
        result = {
            "checkpoint": str(ckpt_path) if ckpt_path else None,
            "base_model": base_model,
            "seeds": args.seeds,
            "sample_size": args.sample_size,
            "skip_judge": True,
            "note": "INFERENCE ONLY -- no judge calls made, no accuracy computed. "
                     "Completions saved per seed for later scoring.",
            "runs": all_runs,
        }
        out_path = results_dir / "inference_only_summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("=" * 70)
        logger.info("INFERENCE-ONLY run complete (no judge calls). Wrote %s", out_path)
        for run in all_runs:
            b = run["base"]
            logger.info(
                "[seed=%s] test: n=%s truncated=%.1f%% mean_tokens=%.0f | "
                "ext_test: n=%s truncated=%.1f%% mean_tokens=%.0f",
                run["seed"],
                b.get("test_n"), 100 * (b.get("test_truncation_rate") or 0),
                b.get("test_mean_completion_tokens") or 0,
                b.get("ext_test_n"), 100 * (b.get("ext_test_truncation_rate") or 0),
                b.get("ext_test_mean_completion_tokens") or 0,
            )
        return

    result = {
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "base_model": base_model,
        "seeds": args.seeds,
        "sample_size": args.sample_size,
        "base_test_ci95": ci95(base_test_acc),
        "base_ext_test_ci95": ci95(base_ext_acc),
        "runs": all_runs,
    }
    if ckpt_path is not None:
        result["checkpoint_test_ci95"] = ci95(ckpt_test_acc)
        result["checkpoint_ext_test_ci95"] = ci95(ckpt_ext_acc)

    out_path = results_dir / "ci_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info("CI SUMMARY (n=%d runs, sample_size=%d each)", len(args.seeds), args.sample_size)
    report_rows = [
        ("base      test", result["base_test_ci95"]),
        ("base      ext_test", result["base_ext_test_ci95"]),
    ]
    if ckpt_path is not None:
        report_rows += [
            ("checkpoint test", result["checkpoint_test_ci95"]),
            ("checkpoint ext_test", result["checkpoint_ext_test_ci95"]),
        ]
    for label, ci in report_rows:
        logger.info(
            "%-20s mean=%.4f  95%% CI=[%.4f, %.4f]  (std=%.4f)",
            label,
            ci["mean"],
            ci["ci_low"],
            ci["ci_high"],
            ci["std"],
        )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()

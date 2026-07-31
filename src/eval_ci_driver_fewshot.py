#!/usr/bin/env python3
"""
95% confidence interval for few-shot base-vs-GRPO-checkpoint accuracy, over
multiple runs of randomly-sampled subsets of test.jsonl -- the few-shot
counterpart of eval_ci_driver.py, using the exact same seed/sampling/CI
machinery so the numbers are directly comparable to the zero-shot CI results
already on file (e.g. ci_qwen35_9b_ckpt300_n500).

Reuses eval_checkpoints_fewshot.py's evaluate_checkpoint_fewshot() (which
itself reuses eval_checkpoints.py's engine loading + judges) -- this script
only adds: (a) loading K few-shot demonstrations ONCE and excluding them from
the full pool before any subsampling, (b) seeded random subsampling per run
from that (already-exclusion-filtered) pool, identical in mechanics to
eval_ci_driver.py's `random.Random(seed).sample(pool, n)`, and (c) aggregating
per-seed accuracies into mean + 95% CI (Student's t).

Same seed VALUES as the existing zero-shot benchmarks (1001-1005, n=500) are
expected to be passed via --seeds/--sample-size -- note the sampling pool
itself is 5 items smaller (shots excluded), so the exact draws differ from
the zero-shot runs even at the same seed; only the seed *values* match.

Example
-------
python eval_ci_driver_fewshot.py \
    --checkpoint ./checkpoints/medical-grpo-qwen3.5-9b-midstream_task-v3-explore/checkpoint-300 \
    --seeds 1001 1002 1003 1004 1005 --sample-size 500 --tp-size 2 --max-lora-rank 64 \
    --k-shot 5 --shots-jsonl shots.jsonl --skip-ext-test \
    --results-dir ./eval_results/ci_fewshot_qwen35_9b_k5_n500
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

sys.path.insert(0, str(Path(__file__).resolve().parent / "few_shot"))
from eval_openai import (  # noqa: E402
    DEFAULT_SHOT_PLACEMENT,
    SHOT_TASK_EXT,
    SHOT_TASK_TEST,
    build_fewshot_prefix,
    drop_leaked_records,
    load_shots,
    shots_fingerprint,
)

from eval_checkpoints import (  # noqa: E402
    DEFAULT_EXT_TEST_JSONL,
    DEFAULT_GPU_MEM_UTIL,
    DEFAULT_MAX_LORA_RANK,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_TEST_JSONL,
    DEFAULT_TP_SIZE,
    GEN_MAX_NEW_TOKENS,
    base_model_from_checkpoint,
    load_vllm_engine,
    make_sampling_params,
    setup_logging,
)
from eval_checkpoints_fewshot import evaluate_checkpoint_fewshot  # noqa: E402
from grpo_training_new_code import load_jsonl_records  # noqa: E402

logger = logging.getLogger(__name__)

# Same Student's-t 95% critical values as eval_ci_driver.py / eval_openai.py.
_T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def ci95(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return {"mean": mean, "std": 0.0, "n": n, "ci_low": mean, "ci_high": mean, "half_width": 0.0}
    std = statistics.stdev(values)
    sem = std / (n**0.5)
    t = _T_CRIT_95.get(n - 1, 2.0)
    half = t * sem
    return {"mean": mean, "std": std, "n": n, "ci_low": mean - half, "ci_high": mean + half, "half_width": half}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="Path to the GRPO checkpoint to compare against base. Omit for a "
                          "base-only reference-model CI (requires --base-model instead).")
    ap.add_argument("--base-model", type=str, default=None,
                     help="Override the base model repo/path directly, bypassing "
                          "base_model_from_checkpoint(). Required when --checkpoint is omitted.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003, 1004, 1005])
    ap.add_argument("--sample-size", type=int, default=500)
    ap.add_argument("--test-jsonl", type=str, default=DEFAULT_TEST_JSONL)
    ap.add_argument("--ext-test-jsonl", type=str, default=DEFAULT_EXT_TEST_JSONL)
    ap.add_argument("--results-dir", type=str, required=True)
    ap.add_argument("--judge-workers", type=int, default=100)
    ap.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    ap.add_argument("--max-lora-rank", type=int, default=DEFAULT_MAX_LORA_RANK)
    ap.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    ap.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--skip-ext-test", action="store_true")

    # Few-shot (same names as eval_openai.py / eval_checkpoints_fewshot.py)
    ap.add_argument("-k", "--k-shot", type=int, default=0)
    ap.add_argument("--shots-jsonl", type=str, default=None)
    ap.add_argument("--ext-shots-jsonl", type=str, default=None)
    ap.add_argument("--shot-seed", type=int, default=None)
    ap.add_argument("--shot-placement", type=str, default=DEFAULT_SHOT_PLACEMENT, choices=("inline", "turns"))
    ap.add_argument("--shot-style", type=str, default="auto", choices=("auto", "with_reasoning", "answer_only"))
    ap.add_argument("--allow-shot-leakage", action="store_true")

    args = ap.parse_args()
    if args.k_shot < 0:
        ap.error("--k-shot must be >= 0")
    if args.k_shot > 0 and not args.shots_jsonl:
        ap.error("--shots-jsonl is required when --k-shot > 0")
    return args


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(results_dir)
    logger.info("Logging to %s", log_file)

    if args.checkpoint is None and args.base_model is None:
        raise SystemExit("Must pass --checkpoint (base-vs-checkpoint CI) or --base-model "
                          "(base-only reference-model CI, no checkpoint comparison).")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    base_model = args.base_model or base_model_from_checkpoint(ckpt_path)
    logger.info("Checkpoint: %s", ckpt_path if ckpt_path else "(none -- base-only reference eval)")
    logger.info("Base model: %s", base_model)
    logger.info("Seeds: %s | sample_size=%d | k_shot=%d", args.seeds, args.sample_size, args.k_shot)

    test_records_full = load_jsonl_records(args.test_jsonl)
    ext_test_records_full = load_jsonl_records(args.ext_test_jsonl)
    logger.info("Full pools (pre-shot-exclusion): test=%d ext_test=%d",
                len(test_records_full), len(ext_test_records_full))

    # --- few-shot setup: load ONCE, exclude from the pool ONCE, before any
    # seed touches it. All seeds share the exact same K demonstrations --
    # only the sampled query subset varies per seed. -----------------------
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
                before = len(test_records_full)
                test_records_full = drop_leaked_records(test_records_full, test_shots, SHOT_TASK_TEST)
                logger.info("test pool after excluding %d shot(s): %d -> %d",
                            len(test_shots), before, len(test_records_full))
        if not args.skip_ext_test:
            ext_shots = load_shots(ext_shots_path, args.k_shot, SHOT_TASK_EXT, args.shot_seed, required=False)
            ext_prefix = build_fewshot_prefix(ext_shots, SHOT_TASK_EXT, args.shot_style, args.shot_placement)
            if not args.allow_shot_leakage:
                before = len(ext_test_records_full)
                ext_test_records_full = drop_leaked_records(ext_test_records_full, ext_shots, SHOT_TASK_EXT)
                logger.info("ext_test pool after excluding %d shot(s): %d -> %d",
                            len(ext_shots), before, len(ext_test_records_full))

    logger.info("Full pools (final): test=%d ext_test=%d",
                len(test_records_full), len(ext_test_records_full))

    tag = f"k{args.k_shot}"

    def _seed_is_cached(seed: int) -> bool:
        seed_dir = results_dir / f"seed_{seed}"
        if not (seed_dir / f"base_{tag}_details.json").exists():
            return False
        if ckpt_path is None:
            return True
        return (seed_dir / f"{ckpt_path.name}_{tag}_details.json").exists()

    needs_engine = any(not _seed_is_cached(s) for s in args.seeds)
    llm = tokenizer = sampling = None
    if needs_engine:
        llm = load_vllm_engine(
            base_model=base_model, tp_size=args.tp_size, max_lora_rank=args.max_lora_rank,
            gpu_mem_util=args.gpu_mem_util, max_model_len=args.max_model_len,
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

        base_cache = seed_dir / f"base_{tag}_details.json"
        ckpt_cache = seed_dir / f"{ckpt_path.name}_{tag}_details.json" if ckpt_path else None
        if base_cache.exists() and (ckpt_path is None or ckpt_cache.exists()):
            logger.info("=== seed=%d: found cached results, skipping re-eval ===", seed)
            base_summary = json.loads(base_cache.read_text(encoding="utf-8"))["summary"]
            ckpt_summary = (
                json.loads(ckpt_cache.read_text(encoding="utf-8"))["summary"] if ckpt_path else None
            )
        else:
            # Same mechanics as eval_ci_driver.py: one rng per seed, test then
            # ext_test drawn in that order from the (shot-excluded) full pool.
            rng = random.Random(seed)
            n_test = min(args.sample_size, len(test_records_full))
            n_ext = min(args.sample_size, len(ext_test_records_full))
            test_sample = rng.sample(test_records_full, n_test) if not args.skip_test else []
            ext_sample = rng.sample(ext_test_records_full, n_ext) if not args.skip_ext_test else []

            logger.info("=== seed=%d (test n=%d, ext_test n=%d, %d-shot) ===",
                        seed, len(test_sample), len(ext_sample), args.k_shot)

            base_summary = evaluate_checkpoint_fewshot(
                llm=llm, tokenizer=tokenizer, sampling=sampling, checkpoint_path=None,
                lora_int_id=None, test_records=test_sample, ext_test_records=ext_sample,
                judge_workers=args.judge_workers, results_dir=seed_dir,
                test_prefix=test_prefix, ext_prefix=ext_prefix, k_shot=args.k_shot,
                skip_test=args.skip_test, skip_ext_test=args.skip_ext_test, tag=tag,
            )
            ckpt_summary = None
            if ckpt_path is not None:
                ckpt_summary = evaluate_checkpoint_fewshot(
                    llm=llm, tokenizer=tokenizer, sampling=sampling, checkpoint_path=ckpt_path,
                    lora_int_id=1, test_records=test_sample, ext_test_records=ext_sample,
                    judge_workers=args.judge_workers, results_dir=seed_dir,
                    test_prefix=test_prefix, ext_prefix=ext_prefix, k_shot=args.k_shot,
                    skip_test=args.skip_test, skip_ext_test=args.skip_ext_test, tag=tag,
                )

        if not args.skip_test:
            base_test_acc.append(base_summary["test_accuracy"])
            if ckpt_summary is not None:
                ckpt_test_acc.append(ckpt_summary["test_accuracy"])
        if not args.skip_ext_test:
            base_ext_acc.append(base_summary["ext_test_accuracy"])
            if ckpt_summary is not None:
                ckpt_ext_acc.append(ckpt_summary["ext_test_accuracy"])

        if ckpt_summary is not None:
            logger.info(
                "[seed=%d] base test=%s ext=%s | ckpt test=%s ext=%s", seed,
                _fmt(base_summary.get("test_accuracy")), _fmt(base_summary.get("ext_test_accuracy")),
                _fmt(ckpt_summary.get("test_accuracy")), _fmt(ckpt_summary.get("ext_test_accuracy")),
            )
        else:
            logger.info(
                "[seed=%d] base test=%s ext=%s", seed,
                _fmt(base_summary.get("test_accuracy")), _fmt(base_summary.get("ext_test_accuracy")),
            )

        run_entry = {"seed": seed, "base": base_summary}
        if ckpt_summary is not None:
            run_entry["checkpoint"] = ckpt_summary
        all_runs.append(run_entry)

    result: dict[str, Any] = {
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "base_model": base_model,
        "seeds": args.seeds,
        "sample_size": args.sample_size,
        "k_shot": args.k_shot,
        "shot_placement": args.shot_placement if args.k_shot else None,
        "shot_style": args.shot_style if args.k_shot else None,
        "shot_seed": args.shot_seed if args.k_shot else None,
        "shots_jsonl": args.shots_jsonl if args.k_shot else None,
        "test_shot_fingerprint": shots_fingerprint(test_shots, args.shot_style, args.shot_placement),
        "ext_test_shot_fingerprint": shots_fingerprint(ext_shots, args.shot_style, args.shot_placement),
        "test_shot_lines": [s["line_no"] for s in test_shots],
        "ext_test_shot_lines": [s["line_no"] for s in ext_shots],
        "runs": all_runs,
    }
    if not args.skip_test:
        result["base_test_ci95"] = ci95(base_test_acc)
        if ckpt_path is not None:
            result["checkpoint_test_ci95"] = ci95(ckpt_test_acc)
    if not args.skip_ext_test:
        result["base_ext_test_ci95"] = ci95(base_ext_acc)
        if ckpt_path is not None:
            result["checkpoint_ext_test_ci95"] = ci95(ckpt_ext_acc)

    out_path = results_dir / "ci_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info("CI SUMMARY (n=%d runs, sample_size=%d each, %d-shot)",
                len(args.seeds), args.sample_size, args.k_shot)
    report_rows = []
    if not args.skip_test:
        report_rows.append(("base      test", result["base_test_ci95"]))
        if ckpt_path is not None:
            report_rows.append(("checkpoint test", result["checkpoint_test_ci95"]))
    if not args.skip_ext_test:
        report_rows.append(("base      ext_test", result["base_ext_test_ci95"]))
        if ckpt_path is not None:
            report_rows.append(("checkpoint ext_test", result["checkpoint_ext_test_ci95"]))
    for label, ci in report_rows:
        logger.info("%-20s mean=%.4f  95%% CI=[%.4f, %.4f]  (std=%.4f)",
                    label, ci["mean"], ci["ci_low"], ci["ci_high"], ci["std"])
    logger.info("Wrote %s", out_path)


def _fmt(v):
    return "N/A" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    main()

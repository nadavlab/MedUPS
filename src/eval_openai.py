#!/usr/bin/env python3
"""
Evaluate a frontier OpenAI model (default: gpt-5.6-luna) on test.jsonl and
ext_test.jsonl using EXACTLY the same generation prompts, answer extractors,
and DeepSeek judges as eval_checkpoints.py — so the accuracy numbers here
are directly comparable to trained-checkpoint numbers.

Methodology
-----------
1. Optionally prepend K few-shot demonstrations (from shots.jsonl). By default
   they are rendered into the query prompt itself as one block headed
   "Here are some examples of questions and answers:"; --shot-placement turns
   injects them as separate user/assistant message turns instead. K=0
   (default) is zero-shot under either placement and produces byte-identical
   prompts to the pre-few-shot version of this script.
2. Generate on every eligible test item ONCE (parallel OpenAI calls).
3. Judge every generated completion ONCE (parallel DeepSeek calls).
4. Bootstrap CI: draw N runs of K items WITH REPLACEMENT from the scored
   pool using deterministic seeds (seed_base + i), then report per-run
   accuracy, mean, SEM, 95% t-CI, and min/max.

Defaults: 0-shot, 5 runs × 500 samples, seed_base=42.

Reads OPENAI_API_KEY (for generation) and DEEPSEEK_API_KEY (for the judge)
from .env or the environment.

Few-shot
--------
Shots live in shots.jsonl (one JSON object per line). Two record shapes are
recognized, routed automatically by which fields are present:

  mid-stream (test.jsonl task):
    {"context_chunks": "...", "question": "...", "answer": "...",
     "reasoning": "...optional..."}

  final-diagnosis (ext_test.jsonl task):
    {"clean text": "...", "final diagnosis": "...", "reasoning": "...optional..."}

An explicit {"task": "test"} or {"task": "ext_test"} field overrides the
auto-routing. Demonstration answers are formatted the way the answer extractor
expects (<think>...</think> + <final_answer>...</final_answer>), so they teach
both content and output format.

Placement (--shot-placement):

  inline (default)  one text block inside the user prompt:

      Here are some examples of questions and answers:

      Example 1:
      Case presentation:
      ...

      Question:
      ...

      Answer:
      <final_answer>...</final_answer>

      Example 2:
      ...

      Now answer the following case in the same way.

      Case presentation:      <- the real item, unchanged
      ...

  turns             the demonstrations become real user/assistant message
                    turns between the system message and the query, which is
                    what this script did before. Use it to reproduce earlier
                    runs; the two placements are different prompts and their
                    accuracies are not interchangeable.

The same K shots are used for every item in a run (fixed-prompt few-shot),
which keeps the comparison across K clean and lets prompt caching work.

Examples
--------
Default: zero-shot, full generation + bootstrap CI on both test sets.
  python eval_openai.py

3-shot and 5-shot (results land in separate files, tagged k3 / k5):
  python eval_openai.py --k-shot 3
  python eval_openai.py --k-shot 5

Pick shots at random from a larger shots.jsonl instead of taking the first K:
  python eval_openai.py --k-shot 5 --shot-seed 42

Test-only:
  python eval_openai.py --skip-ext-test

Different bootstrap plan:
  python eval_openai.py --n-bootstrap 20 --bootstrap-size 300

Smoke test (small pool):
  python eval_openai.py --limit 50 --n-bootstrap 5 --bootstrap-size 40
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Reuse prompts + judges + extractors from the existing pipeline so numbers
# stay directly comparable to eval_checkpoints.py output.
from grpo_training import (
    _ANSWER_RE,
    _last_final_answer,
    build_ext_messages,
    build_messages,
    evaluate_diagnosis,
    load_jsonl_records,
)
from eval_checkpoints import evaluate_answer_equivalence

logger = logging.getLogger(__name__)

# Matches the reasoning close-tag regardless of trained model family.
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


# =============================================================================
# Lenient answer extraction (frontier-model friendly)
# =============================================================================
#
# The trained-checkpoint extractors (_answer_for_judge / _extract_final_answer)
# key strictly off the qwen </think> close-tag: no </think> => region is None =>
# answer "" => auto-score 0. A frontier OpenAI model does not natively emit the
# qwen <think>...</think> wrapper, so ~1/3 of correct answers were being zeroed
# purely on format. To measure the model's TRUE accuracy we drop the </think>
# prerequisite: use the post-</think> region when present, otherwise search the
# whole completion for the <final_answer> tag (falling back to raw text).


def _answer_for_judge_lenient(text: str) -> str:
    """Case-completion answer, tolerant of a missing </think> close-tag."""
    m = _THINK_CLOSE_RE.search(text)
    region = text[m.end():] if m else text
    ans = _last_final_answer(region)
    if ans is not None:
        return ans
    m2 = _ANSWER_RE.search(text)
    if m2:
        return m2.group(1).strip()
    return region.strip()


def _extract_final_answer_lenient(text: str) -> str:
    """External diagnosis, tolerant of a missing </think> close-tag.

    Still requires an explicit <final_answer>...</final_answer> tag (the judge
    needs a short diagnosis string, not raw reasoning); only the </think>
    prerequisite is relaxed.
    """
    m = _THINK_CLOSE_RE.search(text)
    region = text[m.end():] if m else text
    ans = _last_final_answer(region)
    return ans if ans is not None else ""

# --- defaults ---
# Local paths, resolved relative to this script so it runs from any cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_TEST_JSONL = str(SCRIPT_DIR / "test.jsonl")
DEFAULT_EXT_TEST_JSONL = str(SCRIPT_DIR.parent / "train_test_split" / "ext_test.jsonl")
DEFAULT_RESULTS_DIR = str(SCRIPT_DIR / "eval_results")
DEFAULT_JUDGE_WORKERS = 100
DEFAULT_GEN_WORKERS = 30        # OpenAI concurrent requests; tune to your rate limits
DEFAULT_MAX_TOKENS = 16384

# --- few-shot ---
DEFAULT_K_SHOT = 0                                    # 0 => zero-shot
DEFAULT_SHOTS_JSONL = str(SCRIPT_DIR / "shots.jsonl")

# --- bootstrap ---
DEFAULT_N_BOOTSTRAP = 5
DEFAULT_BOOTSTRAP_SIZE = 500
DEFAULT_SEED_BASE = 1001

# Reasoning-model families use max_completion_tokens and reject temperature/top_p.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# 95% two-sided t critical values for small n (df = n-1). Falls back to 1.96
# for anything n >= 30 or unknown. Enough for typical bootstrap-size choices.
_T_CRIT_95 = {
    1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
    7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201,
    15: 2.145, 20: 2.093, 25: 2.064, 30: 2.045,
}


# =============================================================================
# Logging
# =============================================================================


def setup_logging(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"eval_openai_{ts}.log"
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
# Few-shot prompting
# =============================================================================
#
# Design notes (these matter for the numbers being comparable across K):
#
#  * Two placements are supported, selected with --shot-placement:
#      inline (default) -- the demonstrations are rendered into ONE text block
#        headed "Here are some examples of questions and answers:" and prepended
#        to the content of the single user turn that carries the query. The
#        conversation stays 1 system + 1 user turn at every K.
#      turns -- the demonstrations are injected as REAL alternating
#        user/assistant message turns placed after the system message and before
#        the query turn. This is what runs before 2026-07 used; keep it to
#        reproduce them.
#    build_messages() and build_ext_messages() are never modified under either
#    placement, so the query text is byte-identical at every K and K=0
#    reproduces the zero-shot numbers exactly.
#  * The same K shots are used for every item in a run. Per-item retrieved
#    shots would make accuracy depend on retrieval quality, which is a
#    different experiment.
#  * Assistant turns are formatted the way the answer extractor reads them, so
#    a demonstration teaches output format as well as clinical content. This is
#    deliberate: with a frontier model the <final_answer> emission rate is the
#    main source of format-driven zeros, and few-shot is the standard fix.
#  * Shots are removed from the evaluation pool by default (--allow-shot-leakage
#    disables). A shot that is also a test item is an inflated score.

SHOT_TASK_TEST = "test"
SHOT_TASK_EXT = "ext_test"

SHOT_PLACEMENT_INLINE = "inline"
SHOT_PLACEMENT_TURNS = "turns"
DEFAULT_SHOT_PLACEMENT = SHOT_PLACEMENT_INLINE

# Header that opens the inline demonstration block, and the line that closes it
# so the query does not run straight on from the last example.
SHOT_BLOCK_HEADER = "Here are some examples of questions and answers:"
SHOT_BLOCK_FOOTER = "Now answer the following case in the same way."


def _as_context_string(value: Any) -> str:
    """context_chunks may be a preformatted string or a list of chunks."""
    if isinstance(value, (list, tuple)):
        return "\n\n".join(str(v).strip() for v in value if str(v).strip())
    return str(value or "").strip()


def _norm_shot(row: dict[str, Any], line_no: int) -> dict[str, Any]:
    """Normalize one shots.jsonl row and infer which task it demonstrates."""
    ctx = _as_context_string(row.get("context_chunks") or row.get("context_chunk"))
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    clean_text = str(row.get("clean text") or row.get("clean_text") or "").strip()
    final_dx = str(row.get("final diagnosis") or row.get("final_diagnosis") or "").strip()
    reasoning = str(
        row.get("reasoning") or row.get("rationale") or row.get("think") or ""
    ).strip()

    task = str(row.get("task") or "").strip().lower() or None
    if task not in (None, SHOT_TASK_TEST, SHOT_TASK_EXT):
        raise ValueError(
            f"shots.jsonl line {line_no}: unknown task {task!r} "
            f"(expected {SHOT_TASK_TEST!r} or {SHOT_TASK_EXT!r})"
        )
    if task is None:
        if ctx and question and answer:
            task = SHOT_TASK_TEST
        elif clean_text and final_dx:
            task = SHOT_TASK_EXT
        else:
            raise ValueError(
                f"shots.jsonl line {line_no}: cannot infer task. Provide either "
                f"context_chunks+question+answer (mid-stream) or "
                f"'clean text'+'final diagnosis' (final-diagnosis), or set 'task'."
            )

    if task == SHOT_TASK_TEST and not (ctx and question and answer):
        raise ValueError(
            f"shots.jsonl line {line_no}: task='test' requires non-empty "
            f"context_chunks, question, and answer."
        )
    if task == SHOT_TASK_EXT and not (clean_text and final_dx):
        raise ValueError(
            f"shots.jsonl line {line_no}: task='ext_test' requires non-empty "
            f"'clean text' and 'final diagnosis'."
        )

    return {
        "task": task,
        "line_no": line_no,
        "shot_id": row.get("id") or row.get("shot_id") or f"line{line_no}",
        "context_chunks": ctx,
        "question": question,
        "answer": answer,
        "clean_text": clean_text,
        "final_diagnosis": final_dx,
        "reasoning": reasoning or None,
    }


def load_shots(
    path: str | Path,
    k: int,
    task: str,
    seed: Optional[int] = None,
    required: bool = True,
) -> list[dict[str, Any]]:
    """Load K demonstrations for `task` from shots.jsonl.

    seed=None  -> take the first K matching rows in file order (fully
                  reproducible without an RNG, and lets you curate the file).
    seed=int   -> sample K without replacement, then restore file order so the
                  prompt is stable for a given seed.

    Raises if the file is missing, or if it has some but fewer than K rows for
    the task. Failing loud is intentional: silently running 2-shot when you
    asked for 5 would corrupt a results table.

    required=False softens exactly one case: a file that contains no rows at all
    for this task falls back to zero-shot with a warning. That is the shared
    shots.jsonl holding only mid-stream demonstrations, where the caller did not
    ask for few-shot on this task specifically.
    """
    if k <= 0:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"--k-shot {k} requested but shots file not found: {p}. "
            f"Create it or pass --shots-jsonl / --ext-shots-jsonl."
        )

    pool: list[dict[str, Any]] = []
    for i, row in enumerate(load_jsonl_records(p), start=1):
        shot = _norm_shot(row, i)
        if shot["task"] == task:
            pool.append(shot)

    if not pool and not required:
        logger.warning(
            "[%s] %s contains no task=%r demonstrations; running this split "
            "ZERO-SHOT while the other split runs %d-shot. Pass "
            "--ext-shots-jsonl to supply them, or --skip-ext-test to drop the "
            "split. Reported k_shot for this split will be 0.",
            task, p, task, k,
        )
        return []

    if len(pool) < k:
        raise ValueError(
            f"{p} has only {len(pool)} shot(s) for task={task!r}, "
            f"but --k-shot {k} was requested."
        )

    if seed is None:
        chosen = pool[:k]
    else:
        chosen = sorted(random.Random(seed).sample(pool, k), key=lambda s: s["line_no"])

    logger.info(
        "[%s] loaded %d/%d shot(s) from %s (selection=%s): lines %s",
        task, len(chosen), len(pool), p,
        "first-k" if seed is None else f"seed={seed}",
        [s["line_no"] for s in chosen],
    )
    return chosen


def format_shot_answer(answer: str, reasoning: Optional[str], style: str) -> str:
    """Render the assistant side of a demonstration.

    'auto'           -> include a <think> block when the shot carries reasoning.
    'with_reasoning' -> always include it (errors if a shot has none).
    'answer_only'    -> bare <final_answer>, no reasoning shown.

    Showing reasoning matters: an answer-only demonstration can teach the model
    to skip reasoning entirely, which changes what is being measured.
    """
    if style == "with_reasoning" and not reasoning:
        raise ValueError(
            "--shot-style with_reasoning requires a 'reasoning' field on every "
            "selected shot; use 'auto' or add reasoning to shots.jsonl."
        )
    show_reasoning = reasoning and style in ("auto", "with_reasoning")
    if show_reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n<final_answer>{answer}</final_answer>"
    return f"<final_answer>{answer}</final_answer>"


def _shot_user_text(s: dict[str, Any], task: str) -> tuple[str, str]:
    """Render one shot as (user-side text, answer string).

    The user side comes from the same prompt builder used for real items, with
    the builder's own system message stripped, so a demonstration is laid out
    exactly like the query it accompanies.
    """
    if task == SHOT_TASK_TEST:
        msgs = build_messages(s["context_chunks"], s["question"])
        answer = s["answer"]
    else:
        msgs = build_ext_messages(s["clean_text"])
        answer = s["final_diagnosis"]
    user_text = "\n\n".join(
        str(m.get("content", "")).strip() for m in msgs if m["role"] != "system"
    )
    return user_text, answer


def build_fewshot_turns(shots: list[dict[str, Any]], task: str, style: str) -> list[dict]:
    """Render shots as alternating user/assistant message turns."""
    prefix: list[dict] = []
    for s in shots:
        user_text, answer = _shot_user_text(s, task)
        prefix.append({"role": "user", "content": user_text})
        prefix.append(
            {"role": "assistant", "content": format_shot_answer(answer, s["reasoning"], style)}
        )
    return prefix


def build_fewshot_block(shots: list[dict[str, Any]], task: str, style: str) -> str:
    """Render shots as one text block to prepend to the query turn.

    The block opens with SHOT_BLOCK_HEADER, numbers each demonstration, and
    closes with SHOT_BLOCK_FOOTER so the query that follows is clearly the item
    to answer rather than a sixth example.
    """
    if not shots:
        return ""
    parts: list[str] = [SHOT_BLOCK_HEADER, ""]
    for i, s in enumerate(shots, start=1):
        user_text, answer = _shot_user_text(s, task)
        parts.append(f"Example {i}:")
        parts.append(user_text)
        parts.append("")
        parts.append("Answer:")
        parts.append(format_shot_answer(answer, s["reasoning"], style))
        parts.append("")
    parts.append(SHOT_BLOCK_FOOTER)
    return "\n".join(parts).strip()


def build_fewshot_prefix(
    shots: list[dict[str, Any]],
    task: str,
    style: str,
    placement: str = DEFAULT_SHOT_PLACEMENT,
) -> Any:
    """Build the demonstration payload for the given task and placement.

    Returns a list of message turns when placement is 'turns', or a single
    string block when placement is 'inline'. Either is accepted by
    apply_fewshot().
    """
    if not shots:
        return [] if placement == SHOT_PLACEMENT_TURNS else ""
    if placement == SHOT_PLACEMENT_TURNS:
        return build_fewshot_turns(shots, task, style)
    if placement == SHOT_PLACEMENT_INLINE:
        return build_fewshot_block(shots, task, style)
    raise ValueError(
        f"unknown shot placement {placement!r} "
        f"(expected {SHOT_PLACEMENT_INLINE!r} or {SHOT_PLACEMENT_TURNS!r})"
    )


def apply_fewshot(messages: list[dict], prefix: Any) -> list[dict]:
    """Attach demonstrations to a built message list.

    A list prefix is inserted as extra turns after any leading system
    message(s). A string prefix is prepended to the content of the first
    non-system turn, keeping the conversation at its zero-shot turn count.
    """
    if not prefix:
        return messages
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        i += 1
    if isinstance(prefix, str):
        if i >= len(messages):
            raise ValueError("inline few-shot needs at least one non-system turn")
        target = messages[i]
        merged = dict(target)
        merged["content"] = f"{prefix}\n\n{str(target.get('content', '')).lstrip()}"
        return messages[:i] + [merged] + messages[i + 1:]
    return messages[:i] + prefix + messages[i:]


def _infer_k(prefix: Any) -> int:
    """Best-effort K when the caller did not pass k_shot explicitly.

    Turn prefixes carry two messages per shot. An inline block is counted by its
    'Example N:' markers. Pass k_shot to the eval functions to avoid the guess.
    """
    if isinstance(prefix, str):
        return len(re.findall(r"^Example \d+:$", prefix, flags=re.MULTILINE))
    return len(prefix) // 2


def shots_fingerprint(
    shots: list[dict[str, Any]],
    style: str,
    placement: str = DEFAULT_SHOT_PLACEMENT,
) -> str:
    """Short stable hash of the exact demonstrations used, for the results file.

    Placement is hashed in: the same shots laid out as turns and as one inline
    block are different prompts and should not share a fingerprint.
    """
    if not shots:
        return "none"
    h = hashlib.sha256()
    h.update(style.encode("utf-8"))
    h.update(b"\x00")
    h.update(placement.encode("utf-8"))
    for s in shots:
        for field in ("task", "context_chunks", "question", "answer",
                      "clean_text", "final_diagnosis", "reasoning"):
            h.update(b"\x00")
            h.update(str(s.get(field) or "").encode("utf-8"))
    return h.hexdigest()[:12]


def _shot_keys(shots: list[dict[str, Any]], task: str) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for s in shots:
        if task == SHOT_TASK_TEST:
            keys.add((s["question"].strip().lower(), s["answer"].strip().lower()))
        else:
            keys.add((s["clean_text"].strip().lower()[:500],
                      s["final_diagnosis"].strip().lower()))
    return keys


def drop_leaked_records(
    records: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    task: str,
) -> list[dict[str, Any]]:
    """Remove evaluation items that are also being used as demonstrations."""
    if not shots:
        return records
    keys = _shot_keys(shots, task)
    kept, dropped = [], 0
    for r in records:
        if task == SHOT_TASK_TEST:
            key = (
                str(r.get("question") or "").strip().lower(),
                str(r.get("answer") or "").strip().lower(),
            )
        else:
            ct = str(r.get("clean text") or r.get("clean_text") or "")
            fd = str(r.get("final diagnosis") or r.get("final_diagnosis") or "")
            key = (ct.strip().lower()[:500], fd.strip().lower())
        if key in keys:
            dropped += 1
        else:
            kept.append(r)
    if dropped:
        logger.warning(
            "[%s] dropped %d evaluation item(s) that appear in shots.jsonl "
            "(pass --allow-shot-leakage to keep them)", task, dropped
        )
    return kept


def _log_prompt_preview(messages: list[dict], label: str, max_chars: int = 700) -> None:
    logger.info("[%s] prompt preview (%d message turns):", label, len(messages))
    for m in messages:
        content = str(m.get("content", ""))
        if len(content) > max_chars:
            content = content[:max_chars] + f"... [+{len(content) - max_chars} chars]"
        logger.info("  [%s] %s: %s", label, m.get("role"), content.replace("\n", " | "))


# =============================================================================
# OpenAI client + generation
# =============================================================================

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Export it or add to .env.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in REASONING_MODEL_PREFIXES)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    reraise=True,
)
def openai_chat(model: str, messages: list[dict], max_tokens: int) -> dict[str, Any]:
    """One chat.completions call. Returns {text, finish_reason, num_tokens}.

    Reasoning models get max_completion_tokens and no temperature/top_p;
    standard models get max_tokens + temp=0.6, top_p=0.95.
    """
    client = _get_openai_client()
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if _is_reasoning_model(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0.6
        kwargs["top_p"] = 0.95

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    text = choice.message.content or ""
    finish = choice.finish_reason
    completion_tokens = None
    if getattr(resp, "usage", None) is not None:
        completion_tokens = resp.usage.completion_tokens
    return {
        "text": text,
        "finish_reason": finish,
        "num_tokens": completion_tokens or 0,
    }


# =============================================================================
# Generation + judging (one-pass, over the full pool)
# =============================================================================


def _t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    if n in _T_CRIT_95:
        return _T_CRIT_95[n]
    # Fall through to normal approx for larger n
    return 1.96


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int,
    sample_size: int,
    seed_base: int,
) -> dict[str, Any]:
    """Draw n_bootstrap samples of size `sample_size` from `scores` WITH replacement.

    Reproducible via seed_base + iter_index. Returns per-run accuracies plus
    aggregate stats (mean, SEM, 95% t-CI, min, max).
    """
    if not scores:
        return {
            "runs": [],
            "mean": 0.0, "stdev": 0.0, "sem": 0.0,
            "ci_95_t": [0.0, 0.0], "min": 0.0, "max": 0.0,
        }

    n_pool = len(scores)
    runs = []
    for i in range(n_bootstrap):
        seed = seed_base + i
        rng = random.Random(seed)
        # Draw `sample_size` indices with replacement.
        idxs = [rng.randrange(n_pool) for _ in range(sample_size)]
        sub = [scores[j] for j in idxs]
        acc = sum(sub) / len(sub)
        runs.append({
            "run_index": i,
            "seed": seed,
            "sample_size": sample_size,
            "n_correct": int(round(sum(sub))),
            "accuracy": acc,
        })

    accs = [r["accuracy"] for r in runs]
    mean = statistics.fmean(accs)
    stdev = statistics.stdev(accs) if len(accs) > 1 else 0.0
    sem = stdev / math.sqrt(len(accs)) if len(accs) > 1 else 0.0
    t_crit = _t_critical_95(len(accs))
    if math.isnan(t_crit):
        ci_low = ci_high = mean
    else:
        ci_low = mean - t_crit * sem
        ci_high = mean + t_crit * sem

    return {
        "runs": runs,
        "mean": mean,
        "stdev": stdev,
        "sem": sem,
        "ci_95_t": [ci_low, ci_high],
        "min": min(accs),
        "max": max(accs),
        "pool_size": n_pool,
        "n_bootstrap": n_bootstrap,
        "sample_size": sample_size,
        "seed_base": seed_base,
    }


def _generate_pool(
    model: str,
    messages_list: list[list[dict]],
    gen_workers: int,
    max_tokens: int,
    label: str,
) -> list[dict[str, Any]]:
    n = len(messages_list)
    logger.info("[%s] Generating on %d items with %s (workers=%d)",
                label, n, model, gen_workers)
    gens: list[Optional[dict[str, Any]]] = [None] * n

    def _one(i: int) -> tuple[int, dict[str, Any]]:
        try:
            return i, openai_chat(model, messages_list[i], max_tokens)
        except Exception as e:
            logger.error("[%s] OpenAI call failed idx=%d: %s", label, i, e)
            return i, {"text": "", "finish_reason": "error", "num_tokens": 0}

    workers = max(1, min(gen_workers, n))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, i) for i in range(n)]
        done = 0
        for fut in as_completed(futs):
            idx, g = fut.result()
            gens[idx] = g
            done += 1
            if done % 50 == 0:
                logger.info("  [%s] generated %d/%d", label, done, n)
    logger.info("  [%s] generation complete: %d/%d", label, n, n)
    return gens  # type: ignore[return-value]


# =============================================================================
# Per-dataset evaluation
# =============================================================================


def eval_test(
    model: str,
    records: list[dict[str, Any]],
    gen_workers: int,
    judge_workers: int,
    max_tokens: int,
    n_bootstrap: int,
    bootstrap_size: int,
    seed_base: int,
    fewshot_prefix: Any = None,
    k_shot: Optional[int] = None,
) -> dict[str, Any]:
    """Case-completion eval on test.jsonl — generate + judge full pool, then bootstrap CI."""
    n = len(records)
    prefix = fewshot_prefix or []
    k = k_shot if k_shot is not None else _infer_k(prefix)
    logger.info("Eval test: %d examples (%d-shot)", n, k)

    messages_list = [
        apply_fewshot(build_messages(r["context_chunks"], r["question"]), prefix)
        for r in records
    ]
    if messages_list:
        _log_prompt_preview(messages_list[0], "test")
    gens = _generate_pool(model, messages_list, gen_workers, max_tokens, "test")

    scores: list[Optional[float]] = [None] * n
    per_item: list[dict[str, Any]] = [{} for _ in range(n)]

    def _score(i: int) -> tuple[int, float, dict[str, Any]]:
        g = gens[i]
        full = g["text"]
        pred = _answer_for_judge_lenient(full)
        try:
            s = evaluate_answer_equivalence(
                question=records[i]["question"],
                pred=pred,
                ref=records[i]["answer"],
            )
        except Exception as e:
            logger.error("answer judge failed idx=%d: %s", i, e)
            s = 0.0
        item = {
            "completion": full,
            "extracted_answer": pred,
            "reference": records[i]["answer"],
            "score": s,
            "finish_reason": g["finish_reason"],
            "num_tokens": g["num_tokens"],
            "format_incomplete": "</think>" not in full.lower(),
        }
        return i, s, item

    logger.info("[test] Judging %d items with %d workers", n, judge_workers)
    with ThreadPoolExecutor(max_workers=max(1, min(judge_workers, n))) as ex:
        futs = [ex.submit(_score, i) for i in range(n)]
        done = 0
        for fut in as_completed(futs):
            idx, sc, item = fut.result()
            scores[idx] = sc
            per_item[idx] = item
            done += 1
            if done % 100 == 0:
                logger.info("  [test] judged %d/%d", done, n)

    clean = [s if s is not None else 0.0 for s in scores]
    pool_accuracy = sum(clean) / max(1, len(clean))
    trunc = sum(1 for it in per_item if it.get("finish_reason") == "length")
    fmt_inc = sum(1 for it in per_item if it.get("format_incomplete"))
    mean_tok = sum(it.get("num_tokens", 0) for it in per_item) / max(1, n)

    # Bootstrap CI
    boot = bootstrap_ci(clean, n_bootstrap, bootstrap_size, seed_base)

    logger.info(
        "Test (%d-shot): pool_acc=%.4f  boot_mean=%.4f  95%%CI=[%.4f, %.4f]  min=%.4f  max=%.4f  "
        "trunc=%.1f%%  no-</think>=%.1f%%  mean_tokens=%.0f",
        k, pool_accuracy, boot["mean"], boot["ci_95_t"][0], boot["ci_95_t"][1],
        boot["min"], boot["max"],
        100 * trunc / max(1, n), 100 * fmt_inc / max(1, n), mean_tok,
    )
    for r in boot["runs"]:
        logger.info("  [test] run %d (seed=%d): acc=%.4f (%d/%d correct)",
                    r["run_index"], r["seed"], r["accuracy"],
                    r["n_correct"], r["sample_size"])

    return {
        "pool_accuracy": pool_accuracy,
        "n": n,
        "k_shot": k,
        "truncation_rate": trunc / max(1, n),
        "truncated_count": trunc,
        "format_incomplete_rate": fmt_inc / max(1, n),
        "mean_completion_tokens": mean_tok,
        "bootstrap": boot,
        "items": per_item,
    }


def eval_ext_test(
    model: str,
    records: list[dict[str, Any]],
    gen_workers: int,
    judge_workers: int,
    max_tokens: int,
    n_bootstrap: int,
    bootstrap_size: int,
    seed_base: int,
    fewshot_prefix: Any = None,
    k_shot: Optional[int] = None,
) -> dict[str, Any]:
    """External diagnosis eval on ext_test.jsonl — generate + judge, then bootstrap CI."""
    norm = []
    for r in records:
        ct = (r.get("clean text") or r.get("clean_text") or "").strip()
        fd = (r.get("final diagnosis") or r.get("final_diagnosis") or "").strip()
        if ct and fd:
            norm.append({"clean_text": ct, "final_diagnosis": fd})
    n = len(norm)
    prefix = fewshot_prefix or []
    k = k_shot if k_shot is not None else _infer_k(prefix)
    logger.info("Eval ext_test: %d examples (after filtering, %d-shot)", n, k)

    messages_list = [
        apply_fewshot(build_ext_messages(r["clean_text"]), prefix) for r in norm
    ]
    if messages_list:
        _log_prompt_preview(messages_list[0], "ext_test")
    gens = _generate_pool(model, messages_list, gen_workers, max_tokens, "ext_test")

    scores: list[Optional[float]] = [None] * n
    per_item: list[dict[str, Any]] = [{} for _ in range(n)]

    def _score(i: int) -> tuple[int, float, dict[str, Any]]:
        g = gens[i]
        full = g["text"]
        pred = _extract_final_answer_lenient(full)
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

    logger.info("[ext_test] Judging %d items with %d workers", n, judge_workers)
    with ThreadPoolExecutor(max_workers=max(1, min(judge_workers, n))) as ex:
        futs = [ex.submit(_score, i) for i in range(n)]
        done = 0
        for fut in as_completed(futs):
            idx, sc, item = fut.result()
            scores[idx] = sc
            per_item[idx] = item
            done += 1
            if done % 100 == 0:
                logger.info("  [ext_test] judged %d/%d", done, n)

    clean = [s if s is not None else 0.0 for s in scores]
    pool_accuracy = sum(clean) / max(1, len(clean))
    trunc = sum(1 for it in per_item if it.get("finish_reason") == "length")
    fmt_inc = sum(1 for it in per_item if it.get("format_incomplete"))
    mean_tok = sum(it.get("num_tokens", 0) for it in per_item) / max(1, n)

    boot = bootstrap_ci(clean, n_bootstrap, bootstrap_size, seed_base)

    logger.info(
        "Ext-test (%d-shot): pool_acc=%.4f  boot_mean=%.4f  95%%CI=[%.4f, %.4f]  min=%.4f  max=%.4f  "
        "trunc=%.1f%%  no-<final_answer>=%.1f%%  mean_tokens=%.0f",
        k, pool_accuracy, boot["mean"], boot["ci_95_t"][0], boot["ci_95_t"][1],
        boot["min"], boot["max"],
        100 * trunc / max(1, n), 100 * fmt_inc / max(1, n), mean_tok,
    )
    for r in boot["runs"]:
        logger.info("  [ext_test] run %d (seed=%d): acc=%.4f (%d/%d correct)",
                    r["run_index"], r["seed"], r["accuracy"],
                    r["n_correct"], r["sample_size"])

    return {
        "pool_accuracy": pool_accuracy,
        "n": n,
        "k_shot": k,
        "truncation_rate": trunc / max(1, n),
        "truncated_count": trunc,
        "format_incomplete_rate": fmt_inc / max(1, n),
        "mean_completion_tokens": mean_tok,
        "bootstrap": boot,
        "items": per_item,
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help=f"OpenAI model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--test-jsonl", type=str, default=DEFAULT_TEST_JSONL)
    ap.add_argument("--ext-test-jsonl", type=str, default=DEFAULT_EXT_TEST_JSONL)
    ap.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--gen-workers", type=int, default=DEFAULT_GEN_WORKERS,
                    help="Concurrent OpenAI calls (tune to your rate limit)")
    ap.add_argument("--judge-workers", type=int, default=DEFAULT_JUDGE_WORKERS,
                    help="Concurrent DeepSeek judge calls")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap examples per dataset (for smoke tests)")
    ap.add_argument("--skip-test", action="store_true", help="Skip internal test set")
    ap.add_argument("--skip-ext-test", action="store_true", help="Skip ext_test set")
    # Few-shot
    ap.add_argument("-k", "--k-shot", type=int, default=DEFAULT_K_SHOT,
                    help="Number of in-context demonstrations (default: 0 = zero-shot)")
    ap.add_argument("--shots-jsonl", type=str, default=DEFAULT_SHOTS_JSONL,
                    help=f"Demonstrations file (default: {DEFAULT_SHOTS_JSONL})")
    ap.add_argument("--ext-shots-jsonl", type=str, default=None,
                    help="Separate shots file for ext_test (default: same as --shots-jsonl)")
    ap.add_argument("--shot-seed", type=int, default=None,
                    help="Sample K shots at random with this seed; omit to take the "
                         "first K rows in file order")
    ap.add_argument("--shot-placement", type=str, default=DEFAULT_SHOT_PLACEMENT,
                    choices=(SHOT_PLACEMENT_INLINE, SHOT_PLACEMENT_TURNS),
                    help=f"Where the demonstrations go (default: "
                         f"{DEFAULT_SHOT_PLACEMENT}). 'inline' puts them in one "
                         f"block inside the user prompt, headed "
                         f"\"{SHOT_BLOCK_HEADER}\"; 'turns' injects them as "
                         f"separate user/assistant message turns")
    ap.add_argument("--shot-style", type=str, default="auto",
                    choices=("auto", "with_reasoning", "answer_only"),
                    help="Whether demonstrations show a <think> block (default: auto, "
                         "which shows it when the shot has a 'reasoning' field)")
    ap.add_argument("--allow-shot-leakage", action="store_true",
                    help="Keep evaluation items that also appear in shots.jsonl "
                         "(they are dropped by default)")
    # Bootstrap CI
    ap.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
                    help="Number of bootstrap runs (default: 5)")
    ap.add_argument("--bootstrap-size", type=int, default=DEFAULT_BOOTSTRAP_SIZE,
                    help="Samples per bootstrap run, with replacement (default: 500)")
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE,
                    help="Seed base; run i uses seed = seed_base + i (default: 42)")
    args = ap.parse_args()
    if args.k_shot < 0:
        ap.error("--k-shot must be >= 0")
    return args


def write_samples_csv(path: Path, items: list[dict[str, Any]]) -> None:
    """One row per test item: score, length, extracted answer, reference, full completion."""
    fields = [
        "index", "score", "num_tokens", "completion_chars", "finish_reason",
        "format_incomplete", "prediction", "reference", "completion",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, it in enumerate(items):
            completion = it.get("completion", "")
            w.writerow({
                "index": i,
                "score": it.get("score"),
                "num_tokens": it.get("num_tokens"),
                "completion_chars": len(completion),
                "finish_reason": it.get("finish_reason"),
                "format_incomplete": it.get("format_incomplete"),
                # test items use extracted_answer/reference; ext_test uses
                # predicted_diagnosis/reference_diagnosis -- normalize both.
                "prediction": it.get("extracted_answer", it.get("predicted_diagnosis")),
                "reference": it.get("reference", it.get("reference_diagnosis")),
                "completion": completion,
            })


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    log_file = setup_logging(results_dir)
    logger.info("Logging to %s", log_file)
    logger.info(
        "Model: %s (reasoning=%s) | k_shot=%d (placement=%s) | "
        "bootstrap: n=%d size=%d seed_base=%d",
        args.model, _is_reasoning_model(args.model), args.k_shot,
        args.shot_placement if args.k_shot else "n/a",
        args.n_bootstrap, args.bootstrap_size, args.seed_base,
    )

    logger.info("Loading test records from %s", args.test_jsonl)
    test_records = load_jsonl_records(args.test_jsonl)
    logger.info("Loading ext_test records from %s", args.ext_test_jsonl)
    ext_test_records = load_jsonl_records(args.ext_test_jsonl)
    if args.limit:
        test_records = test_records[: args.limit]
        ext_test_records = ext_test_records[: args.limit]

    # --- few-shot demonstrations -------------------------------------------
    ext_shots_path = args.ext_shots_jsonl or args.shots_jsonl
    test_shots: list[dict[str, Any]] = []
    ext_shots: list[dict[str, Any]] = []
    empty_prefix: Any = [] if args.shot_placement == SHOT_PLACEMENT_TURNS else ""
    test_prefix: Any = empty_prefix
    ext_prefix: Any = empty_prefix

    if args.k_shot > 0:
        if not args.skip_test:
            test_shots = load_shots(args.shots_jsonl, args.k_shot,
                                    SHOT_TASK_TEST, args.shot_seed)
            test_prefix = build_fewshot_prefix(test_shots, SHOT_TASK_TEST,
                                               args.shot_style, args.shot_placement)
            if not args.allow_shot_leakage:
                test_records = drop_leaked_records(test_records, test_shots, SHOT_TASK_TEST)
        if not args.skip_ext_test:
            ext_shots = load_shots(ext_shots_path, args.k_shot,
                                   SHOT_TASK_EXT, args.shot_seed)
            ext_prefix = build_fewshot_prefix(ext_shots, SHOT_TASK_EXT,
                                              args.shot_style, args.shot_placement)
            if not args.allow_shot_leakage:
                ext_test_records = drop_leaked_records(
                    ext_test_records, ext_shots, SHOT_TASK_EXT
                )

    logger.info("test pool: %d | ext_test pool: %d", len(test_records), len(ext_test_records))

    results: dict[str, Any] = {"model": args.model, "k_shot": args.k_shot}

    if not args.skip_test:
        results["test"] = eval_test(
            model=args.model,
            records=test_records,
            gen_workers=args.gen_workers,
            judge_workers=args.judge_workers,
            max_tokens=args.max_tokens,
            n_bootstrap=args.n_bootstrap,
            bootstrap_size=args.bootstrap_size,
            seed_base=args.seed_base,
            fewshot_prefix=test_prefix,
            k_shot=args.k_shot,
        )

    if not args.skip_ext_test:
        results["ext_test"] = eval_ext_test(
            model=args.model,
            records=ext_test_records,
            gen_workers=args.gen_workers,
            judge_workers=args.judge_workers,
            max_tokens=args.max_tokens,
            n_bootstrap=args.n_bootstrap,
            bootstrap_size=args.bootstrap_size,
            seed_base=args.seed_base,
            fewshot_prefix=ext_prefix,
            k_shot=args.k_shot if ext_shots else 0,
        )

    # Summary — same schema shape as eval_checkpoints.py PLUS bootstrap stats.
    summary: dict[str, Any] = {
        "checkpoint": f"openai:{args.model}",
        "step": None,
        "k_shot": args.k_shot,
        "shot_style": args.shot_style if args.k_shot else None,
        "shot_placement": args.shot_placement if args.k_shot else None,
        "shot_seed": args.shot_seed if args.k_shot else None,
        "shots_jsonl": args.shots_jsonl if args.k_shot else None,
        "test_shot_fingerprint": shots_fingerprint(
            test_shots, args.shot_style, args.shot_placement),
        "ext_test_shot_fingerprint": shots_fingerprint(
            ext_shots, args.shot_style, args.shot_placement),
        "test_shot_lines": [s["line_no"] for s in test_shots],
        "ext_test_shot_lines": [s["line_no"] for s in ext_shots],
    }
    if "test" in results:
        t = results["test"]; b = t["bootstrap"]
        summary.update({
            "test_pool_accuracy": t["pool_accuracy"],
            "test_bootstrap_mean": b["mean"],
            "test_bootstrap_sem": b["sem"],
            "test_bootstrap_ci_95": b["ci_95_t"],
            "test_bootstrap_min": b["min"],
            "test_bootstrap_max": b["max"],
            "test_n_pool": t["n"],
            "test_n_bootstrap": b["n_bootstrap"],
            "test_bootstrap_size": b["sample_size"],
            "test_seed_base": b["seed_base"],
            "test_truncation_rate": t["truncation_rate"],
            "test_format_incomplete_rate": t["format_incomplete_rate"],
            "test_mean_completion_tokens": t["mean_completion_tokens"],
            "test_mean_completion_chars": (
                sum(len(it.get("completion", "")) for it in t["items"]) / max(1, t["n"])
            ),
        })
    if "ext_test" in results:
        t = results["ext_test"]; b = t["bootstrap"]
        summary.update({
            "ext_test_pool_accuracy": t["pool_accuracy"],
            "ext_test_bootstrap_mean": b["mean"],
            "ext_test_bootstrap_sem": b["sem"],
            "ext_test_bootstrap_ci_95": b["ci_95_t"],
            "ext_test_bootstrap_min": b["min"],
            "ext_test_bootstrap_max": b["max"],
            "ext_test_n_pool": t["n"],
            "ext_test_n_bootstrap": b["n_bootstrap"],
            "ext_test_bootstrap_size": b["sample_size"],
            "ext_test_seed_base": b["seed_base"],
            "ext_test_truncation_rate": t["truncation_rate"],
            "ext_test_format_incomplete_rate": t["format_incomplete_rate"],
            "ext_test_mean_completion_tokens": t["mean_completion_tokens"],
            "ext_test_mean_completion_chars": (
                sum(len(it.get("completion", "")) for it in t["items"]) / max(1, t["n"])
            ),
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    # Tag every output with K so 0/3/5-shot runs of the same model do not
    # overwrite each other.
    tag = f"{safe_model}_k{args.k_shot}"

    detail_path = results_dir / f"{tag}_details.json"
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "shots": {"test": test_shots, "ext_test": ext_shots},
                **{k: v for k, v in results.items() if k not in ("model", "k_shot")},
            },
            f, ensure_ascii=False, indent=2,
        )
    logger.info("Wrote details to %s", detail_path)

    summary_path = results_dir / f"summary_{tag}_{ts}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Summary saved to %s", summary_path)

    # Per-sample model outputs -> CSV (one file per dataset).
    for split in ("test", "ext_test"):
        if split in results:
            csv_path = results_dir / f"{tag}_{split}_samples.csv"
            write_samples_csv(csv_path, results[split]["items"])
            logger.info("Wrote %d per-sample rows to %s",
                        len(results[split]["items"]), csv_path)

    # Final one-liners for quick paste into a comparison table.
    if "test" in results:
        b = results["test"]["bootstrap"]
        logger.info(
            "[%s] TEST      k=%d  bootstrap_mean=%.4f  95%%CI=[%.4f, %.4f]  "
            "(n=%d×%d, seed_base=%d)",
            args.model, args.k_shot, b["mean"], b["ci_95_t"][0], b["ci_95_t"][1],
            b["n_bootstrap"], b["sample_size"], b["seed_base"],
        )
    if "ext_test" in results:
        b = results["ext_test"]["bootstrap"]
        logger.info(
            "[%s] EXT_TEST  k=%d  bootstrap_mean=%.4f  95%%CI=[%.4f, %.4f]  "
            "(n=%d×%d, seed_base=%d)",
            args.model, args.k_shot, b["mean"], b["ci_95_t"][0], b["ci_95_t"][1],
            b["n_bootstrap"], b["sample_size"], b["seed_base"],
        )


if __name__ == "__main__":
    main()
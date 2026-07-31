# MedUPS

Training and evaluation code for reasoning-augmented clinical LLMs on two tasks:

- **Case completion** ("midstream"): given a partial clinical case presentation, predict
  the next clinically relevant decision point (test, treatment, diagnosis, or finding).
- **Final diagnosis prediction**: given a full case presentation, predict the patient's
  final diagnosis.

Both tasks are trained the same way — supervised fine-tuning on chain-of-thought (CoT)
traces distilled from a stronger reasoning model, optionally followed by GRPO
reinforcement learning against an LLM-judge reward — and evaluated the same way:
zero-shot and few-shot accuracy with bootstrapped 95% confidence intervals.

This repo contains the training and evaluation code only. Datasets, model checkpoints,
CoT-distillation/judge scripts, and result logs are not included.

## Pipeline

Both tasks assume CoT-labeled training data already exists (`{context/case, cot,
final_answer}` per example, produced by a separate distillation + judge-labeling step
not included in this repo).

```
1. SFT-CoT baseline       sft_training.py
                          -> LoRA fine-tune on (case, CoT, answer) triples
2. GRPO reinforcement     grpo_training.py
                          -> policy-gradient RL against an LLM-judge reward
3. Evaluation             eval_checkpoints.py / eval_ci_driver.py
                          (+ *_fewshot.py variants, eval_openai.py for frontier-API baselines)
```

## Files

| File | Purpose |
|---|---|
| `grpo_training.py` | Core GRPO trainer (TRL `GRPOTrainer`). Defines the supported base-model families, prompt templates, LoRA config, the LLM-judge reward function, and per-family quirks (chat template, EOS handling, sampling params). Every other script in this repo imports shared pieces from here. |
| `sft_training.py` | LoRA SFT baseline (TRL `SFTTrainer`) on CoT-labeled data, for direct comparison against GRPO-trained checkpoints of the same base model. Reuses `grpo_training.py`'s prompt construction so inputs are identical across SFT/GRPO/eval. |
| `eval_checkpoints.py` | vLLM-backed zero-shot evaluator. Loads a base model once and swaps in LoRA checkpoints via `LoRARequest`; scores generations with an LLM judge. |
| `eval_ci_driver.py` | Wraps `eval_checkpoints.py` over multiple random seeds/subsamples to report accuracy as a mean ± 95% confidence interval (Student's t). |
| `eval_checkpoints_fewshot.py` / `eval_ci_driver_fewshot.py` | Few-shot (in-context demonstrations) counterparts of the above two. |
| `eval_openai.py` | Evaluates a frontier OpenAI model with the exact same prompts/extractors/judges, so its accuracy is directly comparable to trained-checkpoint numbers. Also hosts the few-shot mechanics (`shots.jsonl` loading, prompt assembly, leakage filtering) reused by the vLLM few-shot scripts above. |

## Setup

```bash
pip install -r requirements.txt
```

GRPO training and vLLM-backed evaluation require a CUDA GPU (these were developed
against 8x NVIDIA B200-class GPUs per node; smaller setups work but will need reduced
batch sizes / GPU counts). `eval_openai.py` only needs API access.

### Environment variables

Create a `.env` file (loaded via `python-dotenv`) or export these directly:

```
OPENAI_API_KEY=...      # frontier-model eval (eval_openai.py)
DEEPSEEK_API_KEY=...    # LLM-judge reward calls (see grpo_training.py)
```

### Data layout

Scripts default to a relative `./data/` directory (override with CLI flags — see each
script's `--help` / module docstring):

- `data/train.jsonl` — midstream SFT/GRPO training data. Expected fields:
  `context_chunks`, `question`, `cot`, `final_answer`, `model_response_judgment`.
- `data/test.jsonl` — midstream held-out eval set.
- `data/ext_test.jsonl` — final-diagnosis zero-shot generalization eval set
  (fields: `clean text`, `final diagnosis`, plus a generated `final_answer`/`cot`
  once a checkpoint has run on it).

Checkpoints are read from/written to `./checkpoints/`, and eval outputs to
`./eval_results/` by default.

### Selecting a base model family

Training/eval scripts read `GRPO_MODEL_FAMILY` to select the base model and its
family-specific settings (LoRA target modules, chat template, EOS tokens, sampling
params). Three families are supported: `qwen` (Qwen3.5-9B, default), `huatuo`
(HuatuoGPT-3-8B), and `qwen36` (Qwen3.6-27B) — see the `_MODEL_NAMES` dict in
`grpo_training.py`. `GRPO_MODEL_OVERRIDE=/path/to/local/checkpoint` points a run at a
local checkpoint instead of the family's released HF weights (e.g. to continue GRPO
training from an already-trained SFT checkpoint).

## Usage

Each script documents its own CLI in its module docstring; run with `--help` for the
full flag list. Quick examples:

```bash
# SFT-CoT baseline
GRPO_MODEL_FAMILY=qwen python sft_training.py \
    --train-jsonl ./data/train.jsonl \
    --output-dir ./checkpoints/sft-qwen \
    --num-epochs 3 --learning-rate 1e-4

# GRPO from the released base model
GRPO_MODEL_FAMILY=qwen python grpo_training.py

# Zero-shot eval with 95% CI
GRPO_MODEL_FAMILY=qwen python eval_ci_driver.py \
    --checkpoint ./checkpoints/sft-qwen/checkpoint-600
```

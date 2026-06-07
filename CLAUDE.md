# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle competition: fine-tune **Nemotron-3-Nano-30B-A3B** (a Mamba/MoE hybrid) with LoRA adapters to improve reasoning across 9 problem categories (cipher, gravity, numeral, unit_conversion, bit_manipulation, equation_numeric_deduce/guess, cryptarithm_deduce/guess). Submissions are evaluated by `metric.py`, which loads the adapter via vLLM and checks `\boxed{}` extraction against ground truth.

## Commands

```bash
# Lint
uv run ruff check .
uv run ruff format .

# Dataset preparation (run from dataset_build/)
python3 step1_validate_dataset.py   # validates CSVs → sft_dataset_clean.jsonl
python3 step2_build_sft_dataset.py  # converts to HF dataset format

# Teacher-distillation / hard-example workflow (run from dataset/)
python3 build.py   # wrong_targets.jsonl → doubleword_batch.jsonl (batch-API requests)
python3 check.py   # score a downloaded batch-*-output.jsonl against ground truth

# Training — run inside Kaggle (train_5+ are Kaggle-only; no local/Modal launcher)
# On Kaggle: run the notebook cells which execute train_N/train.py top-level
# For Modal (train_4 only): python train_4/train.py  → submits RTX PRO 6000 job
```

## Architecture

Training iterations are in `train_1/`–`train_8/`, each increasingly capable. `train_7/train.py` is the current canonical script; `train_6/train.py` is byte-identical to `train_5`, and `train_8/train.py` is an empty WIP stub. When picking a script to read or modify, prefer the highest-numbered non-empty one.

**Kaggle-only execution** (`train_5`+): runs top-level (no IS_KAGGLE/Modal flags). Installs packages via `subprocess`, tokenizes the math replay corpus, then calls `run_training()` directly. Unlike `train_3/train_4`, there is no Modal launcher path.

**Curriculum filtering** (`train_7`, `CORRECT_ONLY=True`): loads `correct_ids.json` (problem IDs the model/teacher answered correctly) from Kaggle input and skips any training example whose `id` is not in that set — training only on solvable problems.

**Dual execution model** (`train_4/train.py`, `train_3/train.py`):  
`IS_KAGGLE` / `IS_MODAL_WORKER` / `IS_MODAL_LAUNCHER` flags gate environment-specific paths, package installs, and Modal glue code. The same `run_training()` function executes on both platforms.

**Key training decisions that are non-obvious:**

- **lm_head LoRA must be added manually** — Unsloth silently drops it for MoE models. The code walks `model.model` to find `_causal_lm.lm_head` and calls `model.base_model._create_and_replace(...)`.

- **is_fast_path_available must be patched to `True`** — searches `sys.modules` for `modeling_nemotron_h` and flips the flag to enable the Mamba CUDA kernel.

- **LoRA params cast to fp32, base stays bf16** — MoE router (`mixer.gate.*`) is also fp32 by model design. The code asserts dtypes after casting.

- **Cut Cross-Entropy** (`cut_cross_entropy.linear_cross_entropy`) — patches `CausalLM.forward` to avoid materializing the full logit matrix. Per-token CE is cached on `model._cached_per_token_ce` for the weighted loss computation.

- **MoE weight tying** (`MOE_TIE_WEIGHTS=True`) — ties the LoRA A matrices for gate/up projections and LoRA B for down projections across all 128 experts (Tinker-style). Gradient summing (not mean) is used in `_tie_grads()` before each optimizer step.

- **Math replay interleaving** — tokenized math examples from an external JSONL are interleaved with competition examples every `N` steps to prevent catastrophic forgetting.

- **lm_head key rename on save** — `save_pretrained` writes keys under `base_model.model.lm_head.*`; the code reloads the safetensors file and renames keys to `base_model.model.backbone.lm_head.*` to match vLLM's expected layout.

**Scoring** (`metric.py`): `verify()` uses `math.isclose(rel_tol=1e-2)` for numeric answers and case-insensitive string match otherwise. Binary strings (all 0/1) are compared strictly.

## Data

- `orig_data/train.csv`, `orig_data/test.csv` — raw competition data
- `dataset_build/sft_dataset_clean.jsonl` — validated (prompt, cot, answer) triples
- `dataset/train.jsonl`, `dataset/validation.jsonl` — processed training data
- Corpus on Kaggle: tokenized per `problem_id` under a directory tree, loaded in training-order from `logprobs/index.jsonl`
- Corpus on Modal: single `corpus_preprocessed.jsonl` (each line: `{tokens, mask, problem_id}`)

**Teacher-distillation / hard-example loop** (in `dataset/`): mine examples the current adapter gets wrong (`wrong_targets.jsonl`: `{id, prompt, answer}`), then `build.py` emits an OpenAI-style batch-API request file (`doubleword_batch.jsonl`) that asks a teacher model (default `deepseek-ai/DeepSeek-V4-Pro`, greedy, `max_tokens=7680`, thinking enabled) to regenerate step-by-step traces. After running the batch, `check.py` scores the downloaded `batch-*-output.jsonl` — it extracts the **last** `\boxed{}` from `reasoning_content + content` and matches with the eval metric (`math.isclose(rel_tol=1e-2)`), reporting correct / wrong-answer / no-boxed / error counts. The SYSTEM_PROMPT in `build.py` enforces bit-by-bit / character-by-character reasoning and a self-verification step before the final answer. Correct traces feed back into the SFT corpus; the IDs feed `correct_ids.json` for `CORRECT_ONLY` curriculum filtering.

## Competition Evaluation Parameters (authoritative)

| Parameter | Value |
| --- | --- |
| max_lora_rank | 32 |
| max_tokens | 7680 |
| temperature | **0.0** (greedy — differs from metric.py default of 1.0) |
| top_p | 1.0 |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |

Deadline: **June 15, 2026**. Prize eligibility requires a public Kaggle notebook + write-up.

## LoRA / Training Config (train_5–train_7 defaults)

```python
LORA_RANK = 32, LORA_ALPHA = 32, LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","in_proj","out_proj","lm_head"]
MAX_SEQ_LEN = 8192, BATCH_SIZE = 32, MICRO_BATCH_SIZE = 4, NUM_STEPS = 1000
LEARNING_RATE = 3.5e-4, LR_SCHEDULE = "original_linear" (linear decay to 0, no warmup)
GRAD_CLIP_NORM = 1e9 (effectively off — only measuring norm), WEIGHT_DECAY = 0.0
AdamW betas=(0.9, 0.95)
```

train_4 used `LR_SCHEDULE="warmup_cosine"` and `GRAD_CLIP_NORM=1.0` (real clipping). train_5 reverts to the original baseline settings for A/B comparison. train_7 keeps these defaults and adds `CORRECT_ONLY=True` (see Curriculum filtering above).

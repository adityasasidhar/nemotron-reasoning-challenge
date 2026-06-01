#!/usr/bin/env python
"""
Nemotron Reasoning Challenge — SFT training (consolidated + fixes).

Base recipe: huikang's public train_sft pipeline + your math-replay interleave.

CHANGES vs your original notebook
----------------------------------
1. LR schedule: added warmup + cosine-decay-to-floor (was: linear decay, NO
   warmup, straight to 3.5e-4 on a fresh LoRA init). Toggle LR_SCHEDULE.
2. Gradient clipping: real clip at GRAD_CLIP_NORM=1.0 (was: max_norm=1e9, i.e.
   clipping effectively OFF — you only measured the norm).
3. Loss normalization: the batch loss is now a true token-weighted mean,
   normalized ONCE over all answer tokens in the batch. Your original averaged
   per-microbatch means, so a microbatch with 10 answer tokens pulled the
   gradient as hard as one with 2000.
4. Truncation: over-length examples are DROPPED, not truncated. tokens[:MAX]
   cuts the tail — which is exactly where the \\boxed{} answer lives — so a
   truncated trace teaches the model to reason and never conclude. The drop
   count is logged loudly; if it's large, raise MAX_SEQ_LEN instead.

Plus housekeeping: deduplicated the two run_training() definitions, fixed the
Modal-path replay bug, and tokenize the replay set with AutoTokenizer instead
of loading the 30B model twice.

REPRODUCE THE 86 BASELINE (for A/B)
-----------------------------------
Set LR_SCHEDULE="original_linear" and GRAD_CLIP_NORM=1e9. The loss-norm fix and
drop-not-truncate are baked in; both are strict-correctness changes and their
effect is tiny unless examples were being truncated (the log tells you).

Judge every change on LOCAL-EVAL ACCURACY, not training loss. Lower CCE loss
often just means memorization under a tolerance-based answer metric.
"""

# ============================================================
# Shared config
# ============================================================
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

MAX_SEQ_LEN = 8192
NUM_STEPS = 1000
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 3.5e-4   # baseline was 2e-4. Sweep {2e-4, 2.5e-4, 3e-4, 3.5e-4} on accuracy.
WEIGHT_DECAY = 0.0

# --- LR schedule (NEW) ---
LR_SCHEDULE = "warmup_cosine"   # "warmup_cosine" (recommended) | "original_linear" (baseline)
WARMUP_FRAC = 0.03              # fraction of steps warming 0 -> LEARNING_RATE
LR_FLOOR_FRAC = 0.10            # cosine decay floor = LR_FLOOR_FRAC * LEARNING_RATE

# --- gradient clipping (NEW) ---
GRAD_CLIP_NORM = 1.0            # set to 1e9 to disable (matches your baseline)

RESET_WEIGHTS = True            # if True, fresh LoRA init (skip pretrained adapter)
IN_PROJ_ONLY = False
MOE_TIE_WEIGHTS = True          # tie one LoRA side across all 128 experts (Tinker-style)
ORIGINAL_PROBLEMS_ONLY = False  # filter to problem_ids in train.csv
SHUFFLE_DATASET = False

# --- math replay (NEW toggles) ---
USE_MATH_REPLAY = True                   # flip to False for the clean "no replay" A/B
TARGET_REPLAY_ANSWER_TOKENS = 2_000_000  # lower (e.g. 500_000) for a smaller replay mix

# --- checkpoints (NEW, optional) ---
SAVE_INTERMEDIATE_CHECKPOINTS = False
CHECKPOINT_EVERY = 200          # steps; only used if SAVE_INTERMEDIATE_CHECKPOINTS

# --- diagnostics (NEW, optional) ---
PRINT_DIAGNOSTICS = False

KAGGLE_DATASET = "huikang/nemotron-data"
MINUTES = 60

MATH_REPLAY_PATH = "/kaggle/input/datasets/mohamedamr992/replay-math/nemotron_math_1gb.jsonl"
MATH_TOKENIZED_PATH = "/kaggle/working/replay_math_tokenized.jsonl"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "in_proj",
    "out_proj",
    "lm_head",
]

# ============================================================
# Environment detection + installs
# ============================================================
import os
import sys
import subprocess

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IS_MODAL_WORKER = "MODAL_TASK_ID" in os.environ
IS_MODAL_LAUNCHER = not IS_KAGGLE and not IS_MODAL_WORKER

if IS_KAGGLE:
    subprocess.run(
        "pip install -q --no-index --find-links /kaggle/input/datasets/mayukh18/nemotron-packages/packages "
        "unsloth trl peft transformers datasets accelerate bitsandbytes",
        shell=True,
        check=True,
    )
    subprocess.run(
        "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        shell=True,
        check=True,
    )
    subprocess.run(
        "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        shell=True,
        check=True,
    )
    for _wd in ["/kaggle/input/datasets/llkh0a/rtx-wheels/wheels"]:
        if os.path.isdir(_wd):
            subprocess.run(
                [
                    "pip",
                    "install",
                    "-q",
                    "--no-index",
                    "--find-links",
                    _wd,
                    "protobuf==6.33.5",
                    "sentencepiece",
                    "safetensors",
                    "huggingface_hub",
                ],
                check=False,
            )
    subprocess.run("rm -rf /kaggle/tmp/*", shell=True, check=True)

print("Environment packages installed.")

# ============================================================
# Tokenize the math replay corpus (Kaggle only)
#
#   CHANGE vs original: tokenize with AutoTokenizer instead of loading the full
#   30B model just to grab its tokenizer (and then `del`-ing it). Same chat
#   template, same token ids, far less time/VRAM, no fragile cleanup.
#
#   CHANGE vs original: examples longer than MAX_SEQ_LEN are DROPPED, not
#   truncated. Truncation cuts the tail where the \boxed{} answer lives, so a
#   truncated solution has no answer to learn from.
# ============================================================
if IS_KAGGLE and USE_MATH_REPLAY:
    import json
    import unsloth  # noqa: F401  (import before transformers so unsloth patches apply cleanly)
    import kagglehub
    from transformers import AutoTokenizer
    from tqdm.auto import tqdm

    _model_path_for_tok = kagglehub.model_download(
        "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
    )
    tok = AutoTokenizer.from_pretrained(_model_path_for_tok, trust_remote_code=True)

    if PRINT_DIAGNOSTICS:
        # Sanity: confirm reasoning_content (the <think> body) is rendered into
        # the chat template, so it actually gets supervised.
        with open(MATH_REPLAY_PATH, "r") as f:
            _row = json.loads(next(f))
        _msgs = _row["messages"]
        _rendered = tok.apply_chat_template(
            _msgs, tokenize=False, add_generation_prompt=False
        )
        _reasoning = _msgs[1].get("reasoning_content", "")
        _final = _msgs[1].get("content", "")
        print("Reasoning present in row:", bool(_reasoning))
        print("Final content present in row:", bool(_final))
        if _reasoning:
            print("Reasoning rendered into template:", _reasoning[:100] in _rendered)
        if _final:
            print("Final content rendered into template:", _final[:100] in _rendered)
        print("\n--- Rendered preview ---\n")
        print(_rendered[:2000])

    kept = 0
    skipped = 0
    dropped_too_long = 0
    total_tokens = 0
    total_answer_tokens = 0

    with open(MATH_REPLAY_PATH, "r") as fin, open(MATH_TOKENIZED_PATH, "w") as fout:
        for line in tqdm(fin):
            row = json.loads(line)
            messages = row.get("messages")

            if not messages or len(messages) < 2:
                skipped += 1
                continue

            # Prompt only: user side, ending right before assistant generation begins.
            prompt_ids = tok.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
            )
            # Full example: user + assistant reasoning + assistant final content.
            full_ids = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )

            if len(full_ids) <= len(prompt_ids):
                skipped += 1
                continue

            # DROP rather than truncate (truncation removes the boxed answer).
            if len(full_ids) > MAX_SEQ_LEN:
                dropped_too_long += 1
                continue

            prompt_len = len(prompt_ids)
            mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)

            if not any(mask):
                skipped += 1
                continue

            fout.write(json.dumps({"tokens": full_ids, "mask": mask}) + "\n")
            kept += 1
            total_tokens += len(full_ids)
            total_answer_tokens += sum(mask)

            if total_answer_tokens >= TARGET_REPLAY_ANSWER_TOKENS:
                break

    print("Replay examples kept:", kept)
    print("Replay examples skipped:", skipped)
    print("Replay examples dropped (too long):", dropped_too_long)
    print("Total replay tokens:", f"{total_tokens:,}")
    print("Trainable replay answer tokens:", f"{total_answer_tokens:,}")
    print("Saved to:", MATH_TOKENIZED_PATH)
elif IS_KAGGLE:
    print("USE_MATH_REPLAY=False — skipping replay tokenization.")


# ============================================================
# Training
# ============================================================
def run_training() -> None:
    """Full SFT flow. Runs at module level on Kaggle, or inside the Modal worker
    via train_remote()."""
    import gc
    import json
    import math
    import random
    import subprocess
    import sys
    import time

    from unsloth import FastLanguageModel

    import torch
    from cut_cross_entropy import linear_cross_entropy
    from peft import LoraConfig
    from peft.tuners.lora import Linear as LoraLinear
    from safetensors.torch import load_file, save_file

    # ── Env-specific paths + adapter source ──────────────────────────
    if IS_KAGGLE:
        import kagglehub

        CORPUS_PATH = "/kaggle/input/datasets/huikang/huikang-nemotron-repository-snapshot/nemotron-master/training/sft/04-08-16-14/tokens"
        TRAIN_ORDER_PATH = "/kaggle/input/datasets/huikang/huikang-nemotron-repository-snapshot/nemotron-master/training/sft/04-08-16-14/logprobs/index.jsonl"
        TRAIN_CSV_PATH = "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv"
        ADAPTER_SRC = "/kaggle/tmp/pretrained_adapter"
        if not RESET_WEIGHTS:
            import zipfile as _zipfile

            _adapter_zip = "/kaggle/input/notebooks/huikang/tinker-submission-notebook/submission.zip"
            os.makedirs(ADAPTER_SRC, exist_ok=True)
            with _zipfile.ZipFile(_adapter_zip, "r") as _zf:
                _zf.extractall(ADAPTER_SRC)
        MODEL_PATH = kagglehub.model_download(
            "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
        )
        save_dir = "."
        ckpt_base = "/kaggle/working"
    else:  # IS_MODAL_WORKER
        MODEL_PATH = "unsloth/Nemotron-3-Nano-30B-A3B"
        CORPUS_PATH = "/data/corpus_preprocessed.jsonl"
        TRAIN_CSV_PATH = "/data/train.csv"
        ADAPTER_SRC = "/merged/weights"
        OUTPUT_DIR = "/output/weights"
        save_dir = OUTPUT_DIR
        ckpt_base = OUTPUT_DIR

    # ── GPU + kernel sanity check (runs on both Kaggle and Modal worker) ──
    import causal_conv1d
    import mamba_ssm

    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, sm_{cc[0] * 10 + cc[1]}")
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
    print(
        f"mamba_ssm={mamba_ssm.__version__}, causal_conv1d={causal_conv1d.__version__}"
    )
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if IS_MODAL_WORKER:
        assert cc == (12, 0), (
            f"Expected sm_120 (RTX PRO 6000), got sm_{cc[0] * 10 + cc[1]}"
        )
    from causal_conv1d import causal_conv1d_fn

    _x = torch.randn(1, 256, 32, device="cuda", dtype=torch.bfloat16)
    _w = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
    causal_conv1d_fn(_x, _w, None, activation="silu")
    print("causal_conv1d CUDA kernel: OK")

    # Clear stale HF modules cache (Modal-only; bug: persists across runs)
    if IS_MODAL_WORKER:
        import shutil as _shutil

        hf_modules = os.path.join(
            os.environ.get("HF_HOME", "/root/.cache/huggingface"), "modules"
        )
        if os.path.exists(hf_modules):
            _shutil.rmtree(hf_modules)

    # ── Load corpus into `examples` list ─────────────────────────────
    examples: list[dict] = []
    dropped_too_long = 0

    if IS_KAGGLE:
        # Load problem_ids in training order from logprobs/index.jsonl (epoch 0).
        ordered_ids: list[str] = []
        seen: set[str] = set()
        with open(TRAIN_ORDER_PATH) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("epoch", 0) != 0:
                    continue
                pid = rec["problem_id"]
                if pid in seen:
                    continue
                seen.add(pid)
                ordered_ids.append(pid)
        print(
            f"Loaded {len(ordered_ids)} problem_ids in training order from "
            f"{TRAIN_ORDER_PATH}"
        )

        for sid in ordered_ids:
            seg_path = os.path.join(CORPUS_PATH, sid, "synthetic.json")
            assert os.path.isfile(seg_path), (
                f"problem_id {sid} from training order missing in corpus: {seg_path}"
            )
            with open(seg_path) as f:
                rec = json.load(f)
            tokens = rec["tokens"]
            mask = rec["mask"]
            if not tokens:
                continue
            # CHANGE: drop over-length examples instead of truncating the answer away.
            if len(tokens) > MAX_SEQ_LEN:
                dropped_too_long += 1
                continue
            if not any(mask):
                continue
            examples.append(
                {
                    "problem_id": sid,
                    "tokens": tokens[:-1],
                    "targets": tokens[1:],
                    "weights": [float(m) for m in mask[1:]],
                }
            )
    else:  # IS_MODAL_WORKER
        with open(CORPUS_PATH) as f:
            for line in f:
                rec = json.loads(line.strip())
                tokens = rec["tokens"]
                mask = rec["mask"]
                if len(tokens) > MAX_SEQ_LEN:
                    dropped_too_long += 1
                    continue
                if not any(mask):
                    continue
                examples.append(
                    {
                        "problem_id": rec["problem_id"],
                        "tokens": tokens[:-1],
                        "targets": tokens[1:],
                        "weights": [float(m) for m in mask[1:]],
                    }
                )

    if dropped_too_long:
        print(
            f"NOTE: dropped {dropped_too_long} corpus examples longer than "
            f"MAX_SEQ_LEN={MAX_SEQ_LEN}. If this is a large fraction, raise "
            f"MAX_SEQ_LEN rather than lose these (they're often the hardest problems)."
        )

    if ORIGINAL_PROBLEMS_ONLY:
        import csv

        with open(TRAIN_CSV_PATH) as f:
            original_ids = {row["id"] for row in csv.DictReader(f)}
        before = len(examples)
        examples = [e for e in examples if e["problem_id"] in original_ids]
        print(
            f"ORIGINAL_PROBLEMS_ONLY=True: filtered {before} → {len(examples)} examples "
            f"using {len(original_ids)} ids from {TRAIN_CSV_PATH}"
        )

    total_unmasked = sum(sum(e["weights"]) for e in examples)
    total_tokens = sum(len(e["tokens"]) for e in examples)
    print(
        f"Loaded {len(examples)} examples, {total_tokens:,} tokens "
        f"(unmasked={total_unmasked:,.0f})"
    )

    # ── Load + interleave tokenized math replay (Kaggle + USE_MATH_REPLAY) ──
    # CHANGE: replay loading is gated by IS_KAGGLE so the Modal worker path does
    # not try to read a Kaggle-only file (latent crash in the original).
    if IS_KAGGLE and USE_MATH_REPLAY:
        assert os.path.isfile(MATH_TOKENIZED_PATH), (
            f"Missing tokenized replay file: {MATH_TOKENIZED_PATH}. "
            "Run the replay-tokenization section before run_training()."
        )
        replay_examples: list[dict] = []
        with open(MATH_TOKENIZED_PATH) as f:
            for line in f:
                rec = json.loads(line)
                tokens = rec["tokens"]
                mask = rec["mask"]
                if len(tokens) < 2:
                    continue
                if not any(mask[1:]):
                    continue
                replay_examples.append(
                    {
                        "problem_id": "replay_math",
                        "tokens": tokens[:-1],
                        "targets": tokens[1:],
                        "weights": [float(m) for m in mask[1:]],
                    }
                )

        print("Original target examples:", len(examples))
        print("Replay math examples:", len(replay_examples))
        assert len(replay_examples) > 0, "No replay examples were loaded"

        # Interleave replay examples evenly through the target examples.
        mixed_examples: list[dict] = []
        replay_idx = 0
        replay_every = max(1, len(examples) // len(replay_examples))
        for i, ex in enumerate(examples):
            mixed_examples.append(ex)
            if (i + 1) % replay_every == 0 and replay_idx < len(replay_examples):
                mixed_examples.append(replay_examples[replay_idx])
                replay_idx += 1
        mixed_examples.extend(replay_examples[replay_idx:])
        examples = mixed_examples

        mixed_unmasked = sum(sum(e["weights"]) for e in examples)
        mixed_tokens = sum(len(e["tokens"]) for e in examples)
        print("Replay inserted every ~", replay_every, "target examples")
        print("Total examples after replay:", len(examples))
        print(f"Mixed corpus: {mixed_tokens:,} tokens (unmasked={mixed_unmasked:,.0f})")
    else:
        print("USE_MATH_REPLAY=False (or non-Kaggle) — training on target corpus only.")

    # ── Load base model ──────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    if IS_MODAL_WORKER:
        hf_cache_vol.commit()  # noqa: F821 — defined at module level on non-Kaggle

    # ── Wrap in LoRA ─────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )
    FastLanguageModel.for_training(model)

    # ── Patch Mamba CUDA fast path ───────────────────────────────────
    nemotron_mod = None
    for _name, _m in sys.modules.items():
        if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
            nemotron_mod = _m
            break
    assert nemotron_mod is not None, "Could not find modeling_nemotron_h module"
    print(f"is_fast_path_available was: {nemotron_mod.is_fast_path_available}")
    nemotron_mod.is_fast_path_available = True  # type: ignore[attr-defined]
    print("Patched is_fast_path_available = True")

    # ── Manually add lm_head LoRA (Unsloth drops it for MoE) ─────────
    _causal_lm = model
    while hasattr(_causal_lm, "model"):
        _causal_lm = _causal_lm.model
    _lm_head = _causal_lm.lm_head
    if not isinstance(_lm_head, LoraLinear):
        _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
        model.base_model._create_and_replace(
            _cfg,
            "default",
            target=_lm_head,
            target_name="lm_head",
            parent=_causal_lm,
        )
        print("Manually added LoRA to lm_head")
    else:
        print("lm_head already has LoRA")

    # ── Cast LoRA params to fp32 (base model stays bf16 except MoE router) ──
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.data = param.data.to(torch.float32)

    for name, param in model.named_parameters():
        if ".lora_" in name:
            assert param.dtype == torch.float32, (
                f"LoRA param {name} expected fp32, got {param.dtype}"
            )
            continue

        is_router = (
            ".mixer.gate." in name
        )  # NemotronHTopkRouter.weight + e_score_correction_bias
        # Nemotron-H loads the MoE router (`mixer.gate`) in fp32 on purpose
        # (_keep_in_fp32_modules_strict + per-forward fp32 cast on self.weight).
        if is_router:
            assert param.dtype == torch.float32, (
                f"param {name} expected fp32, got {param.dtype}"
            )
            continue

        assert param.dtype == torch.bfloat16, (
            f"param {name} expected bf16, got {param.dtype}"
        )

    print("Verified: LoRA params fp32, base params bf16 (MoE router fp32)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {trainable:,} trainable / {total:,} total parameters")

    # ── Patch forward with Cut Cross-Entropy ─────────────────────────
    _base = model
    while hasattr(_base, "model"):
        _base = _base.model

    def _patched_causal_forward(
        input_ids=None, attention_mask=None, labels=None, **kwargs
    ):
        backbone_out = _base.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("position_ids", "past_key_values", "use_cache")
            },
        )
        hidden_states = backbone_out[0]
        lm_head = _base.lm_head
        base_w = lm_head.base_layer.weight
        lora_A = lm_head.lora_A["default"].weight
        lora_B = lm_head.lora_B["default"].weight
        scaling = lm_head.scaling["default"]
        lm_weight = base_w + scaling * lora_B @ lora_A
        if labels is not None:
            per_token_ce = linear_cross_entropy(
                hidden_states, lm_weight, labels, reduction="none"
            )
            loss = per_token_ce.mean()
        else:
            per_token_ce = None
            loss = None
        model._cached_per_token_ce = per_token_ce  # type: ignore[attr-defined]
        return loss

    _base.forward = _patched_causal_forward
    print("Patched CausalLM.forward with CCE (no logits materialization)")

    # ── Load adapter weights (unless RESET_WEIGHTS) ──────────────────
    if RESET_WEIGHTS:
        print(
            "RESET_WEIGHTS=True — skipping pretrained adapter load; using fresh LoRA init"
        )
    else:
        print(f"Loading adapter from {ADAPTER_SRC}...")
        from peft import load_peft_weights

        adapter_weights = load_peft_weights(ADAPTER_SRC)

        model_sd = model.state_dict()
        new_sd: dict = {}
        loaded = 0
        for ak, av in adapter_weights.items():
            if ak in model_sd:
                new_sd[ak] = av
                loaded += 1
                continue
            ak_with_default = ak.replace(
                ".lora_A.weight", ".lora_A.default.weight"
            ).replace(".lora_B.weight", ".lora_B.default.weight")
            if ak_with_default in model_sd:
                new_sd[ak_with_default] = av
                loaded += 1
                continue
            ak_lm = ak.replace(".backbone.lm_head.", ".lm_head.")
            ak_lm_default = ak_lm.replace(
                ".lora_A.weight", ".lora_A.default.weight"
            ).replace(".lora_B.weight", ".lora_B.default.weight")
            if ak_lm_default in model_sd:
                new_sd[ak_lm_default] = av
                loaded += 1
                continue

        model.load_state_dict(new_sd, strict=False)
        assert loaded == len(adapter_weights), (
            f"Not all adapter weights loaded: {loaded}/{len(adapter_weights)}"
        )
        print(f"  Loaded {loaded}/{len(adapter_weights)} weights into model")

    # ── Freeze all LoRA params except in_proj (if IN_PROJ_ONLY) ──────
    print(f"{IN_PROJ_ONLY=}")
    if IN_PROJ_ONLY:
        for name, param in model.named_parameters():
            if param.requires_grad and ".in_proj." not in name:
                param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  {trainable_params:,} trainable / {frozen_params:,} frozen")

    # ── MoE tied-weight params (Tinker convention) ───────────────────
    # Tie whichever LoRA side touches the hidden dim:
    #   gate_up_proj / up_proj / w1 / gate_proj  -> tie A (input/hidden side)
    #   down_proj / w2                           -> tie B (output/hidden side)
    # Unsloth's batched [num_experts, ...] layout is kept; "tying" means all 128
    # expert slices stay identical. Saving emits 128 per-expert copies (untied
    # downstream).
    moe_tied_params: list[torch.Tensor] = []
    if MOE_TIE_WEIGHTS:
        w1_proj_names = ("gate_up_proj", "up_proj", "gate_proj", ".w1.")
        w2_proj_names = ("down_proj", ".w2.")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if ".experts." not in name or ".lora_" not in name:
                continue
            is_w1 = any(p in name for p in w1_proj_names)
            is_w2 = any(p in name for p in w2_proj_names)
            is_A = ".lora_A." in name
            is_B = ".lora_B." in name
            should_tie = (is_w1 and is_A) or (is_w2 and is_B)
            if not should_tie:
                continue
            if param.dim() < 2 or param.shape[0] <= 1:
                continue
            moe_tied_params.append(param)

        def _tie_param_init() -> None:
            """Make all 128 expert slices identical (mean-and-broadcast)."""
            with torch.no_grad():
                for p in moe_tied_params:
                    mean = p.data.mean(dim=0, keepdim=True)
                    p.data.copy_(mean.expand_as(p.data))

        def _tie_grads() -> None:
            # Sum (not mean) across the expert dim: with W shared and each expert
            # using a copy W_i = W, chain rule gives dL/dW = sum_i dL/dW_i.
            # Inactive experts contribute 0; router weights are baked into active
            # g_i, so no double-counting. Summing keeps all 128 slices identical
            # after each AdamW step.
            with torch.no_grad():
                for p in moe_tied_params:
                    if p.grad is None:
                        continue
                    grad_sum = p.grad.sum(dim=0, keepdim=True)
                    p.grad.copy_(grad_sum.expand_as(p.grad))

        print(f"MoE weight tying: {len(moe_tied_params)} params identified for tying")
        if moe_tied_params:
            print(f"  example shapes: {[tuple(p.shape) for p in moe_tied_params[:3]]}")
        _tie_param_init()  # start from a tied state
    else:

        def _tie_grads() -> None:
            pass

    # ── Adapter save helper (save_pretrained + lm_head key rename) ────
    def save_adapter(target_dir: str) -> list[str]:
        os.makedirs(target_dir, exist_ok=True)
        for _f in os.listdir(target_dir):
            if _f.startswith("adapter"):
                os.remove(os.path.join(target_dir, _f))
        model.save_pretrained(target_dir)
        st_path = os.path.join(target_dir, "adapter_model.safetensors")
        tensors = load_file(st_path)
        renamed = {
            k.replace(
                "base_model.model.lm_head.", "base_model.model.backbone.lm_head."
            ): v
            for k, v in tensors.items()
        }
        save_file(renamed, st_path)
        return [f for f in os.listdir(target_dir) if f.startswith("adapter")]

    # ── Training loop ────────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    device = next(model.parameters()).device

    indices = list(range(len(examples)))
    if SHUFFLE_DATASET:
        rng = random.Random(0)
        rng.shuffle(indices)
        print(f"SHUFFLE_DATASET=True: shuffled {len(indices)} examples (seed=0)")
    else:
        print(f"SHUFFLE_DATASET=False: keeping corpus order ({len(indices)} examples)")

    training_log: list[str] = []

    def _log(msg: str) -> None:
        print(msg, flush=True)
        training_log.append(msg)

    max_steps = len(examples) // BATCH_SIZE
    num_steps = NUM_STEPS
    if num_steps > max_steps:
        _log(
            f"WARNING: NUM_STEPS={NUM_STEPS} exceeds max_steps={max_steps} "
            f"({len(examples)} examples // {BATCH_SIZE} batch). Clamping to {max_steps}."
        )
        num_steps = max_steps

    # CHANGE: warmup + cosine-decay-to-floor schedule (toggle LR_SCHEDULE).
    warmup_steps = (
        max(1, int(WARMUP_FRAC * num_steps)) if LR_SCHEDULE != "original_linear" else 0
    )

    def lr_at(s: int) -> float:
        if LR_SCHEDULE == "original_linear":
            return LEARNING_RATE * (1 - s / num_steps)
        if s < warmup_steps:
            return LEARNING_RATE * (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(1, num_steps - warmup_steps)
        return LEARNING_RATE * (
            LR_FLOOR_FRAC + (1 - LR_FLOOR_FRAC) * 0.5 * (1 + math.cos(math.pi * progress))
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    _log(
        f"Training: {num_steps} steps, batch_size={BATCH_SIZE}, "
        f"micro_batch_size={MICRO_BATCH_SIZE}, lr={LEARNING_RATE}, "
        f"schedule={LR_SCHEDULE}, warmup_steps={warmup_steps}, "
        f"clip={GRAD_CLIP_NORM}, weight_decay={WEIGHT_DECAY}"
    )

    step = 0
    for batch_start in range(0, len(indices), BATCH_SIZE):
        if step >= num_steps:
            break
        batch_indices = indices[batch_start : batch_start + BATCH_SIZE]
        batch = [examples[i] for i in batch_indices]
        batch_tokens = [e["tokens"] for e in batch]
        batch_targets = [e["targets"] for e in batch]
        batch_weights = [e["weights"] for e in batch]

        n = len(batch)

        # CHANGE: normalize the batch loss ONCE over all answer tokens in the
        # batch (true token-weighted mean), instead of averaging per-microbatch
        # means. Computed cheaply up front (no GPU work), then each microbatch
        # contributes loss_sum / batch_weight_total to the gradient.
        batch_weight_total = sum(sum(w) for w in batch_weights) or 1.0

        total_loss_sum = 0.0
        total_weight_sum = 0.0

        for mb_start in range(0, n, MICRO_BATCH_SIZE):
            mb_end = min(mb_start + MICRO_BATCH_SIZE, n)
            mb_toks = batch_tokens[mb_start:mb_end]
            mb_tgts = batch_targets[mb_start:mb_end]
            mb_wts = batch_weights[mb_start:mb_end]

            n_micro = len(mb_toks)
            max_len = max(len(t) for t in mb_toks)
            total_len = sum(len(t) for t in mb_toks)

            padded_input = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            padded_targets = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            padded_weights = torch.zeros(
                n_micro, max_len, dtype=torch.float32, device=device
            )
            attention_mask = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            for i in range(n_micro):
                seq_len = len(mb_toks[i])
                padded_input[i, :seq_len] = torch.tensor(mb_toks[i], dtype=torch.long)
                padded_targets[i, :seq_len] = torch.tensor(mb_tgts[i], dtype=torch.long)
                padded_weights[i, :seq_len] = torch.tensor(
                    mb_wts[i], dtype=torch.float32
                )
                attention_mask[i, :seq_len] = 1

            t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(
                    input_ids=padded_input,
                    attention_mask=attention_mask,
                    labels=padded_targets,
                    use_cache=False,
                )
                per_token_ce = model._cached_per_token_ce  # type: ignore[attr-defined]
                weighted_loss = per_token_ce * padded_weights
                weight_sum_t = padded_weights.sum()
                loss_sum_t = weighted_loss.sum()

            # Token-weighted contribution to the batch gradient.
            (loss_sum_t / batch_weight_total).backward()
            total_loss_sum += loss_sum_t.item()
            total_weight_sum += weight_sum_t.item()
            del per_token_ce, weighted_loss, loss_sum_t

            t_end = time.time()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            mem_gb = torch.cuda.memory_allocated() / 1e9
            mb_idx = mb_start // MICRO_BATCH_SIZE
            print(
                f"    micro-batch {mb_idx}: {n_micro} seqs, max_len={max_len}, "
                f"total_len={total_len}, wall={t_end - t0:.1f}s, "
                f"peak={peak_gb:.1f}GB, mem={mem_gb:.1f}GB"
            )

        # CHANGE: schedule via lr_at(step) (warmup + cosine), real clipping.
        lr = lr_at(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        _tie_grads()  # average MoE expert grads before clip+step so Adam stays in sync
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=GRAD_CLIP_NORM
        )
        optimizer.step()
        optimizer.zero_grad()
        loss_mean = total_loss_sum / total_weight_sum if total_weight_sum > 0 else 0
        step += 1
        _log(
            f"  step {step}/{num_steps}: "
            f"loss:mean={loss_mean:.6f}, grad_norm={grad_norm:.4f}, lr={lr:.2e}"
        )

        # Optional: save an intermediate checkpoint for held-out accuracy selection.
        if (
            SAVE_INTERMEDIATE_CHECKPOINTS
            and step % CHECKPOINT_EVERY == 0
            and step < num_steps
        ):
            ckpt_dir = os.path.join(ckpt_base, "checkpoints", f"step_{step}")
            save_adapter(ckpt_dir)
            _log(f"  saved intermediate checkpoint to {ckpt_dir}")

    print(
        f"\nTraining complete. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
    )

    # ── Save final adapter (rename lm_head keys; identical on both sides) ──
    adapter_files = save_adapter(save_dir)

    # ── Clean unsloth compiled cache (runs on both) ──────────────────
    _ucache = "unsloth_compiled_cache"
    if os.path.isdir(_ucache):
        import shutil as _sh

        _sh.rmtree(_ucache)

    # ── Package & ship (divergent) ───────────────────────────────────
    if IS_KAGGLE:
        import zipfile

        SUBMISSION_ZIP = "submission.zip"
        with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in adapter_files:
                zf.write(os.path.join(save_dir, fname), fname)
        for fname in adapter_files:
            os.remove(os.path.join(save_dir, fname))
        print(f"Wrote {SUBMISSION_ZIP}")
    else:  # IS_MODAL_WORKER
        import shutil
        import tempfile

        with open(os.path.join(save_dir, "training_log.txt"), "w") as f:
            f.write("\n".join(training_log) + "\n")
        output_vol.commit()  # noqa: F821 — defined at module level on non-Kaggle

        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(os.path.join(kaggle_dir, "access_token"), "w") as f:
            f.write(os.environ["KAGGLE_API_TOKEN"])
        upload_dir = tempfile.mkdtemp()
        for fname in os.listdir(save_dir):
            src = os.path.join(save_dir, fname)
            if os.path.isfile(src):  # skip the checkpoints/ subdir if present
                shutil.copy(src, upload_dir)
        metadata = {"id": KAGGLE_DATASET, "title": KAGGLE_DATASET.split("/")[1]}
        with open(os.path.join(upload_dir, "dataset-metadata.json"), "w") as f:
            json.dump(metadata, f)
        print(f"Uploading to Kaggle {KAGGLE_DATASET}...")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                upload_dir,
                "-m",
                "post-finetuned adapter + compiled wheels",
            ],
            check=True,
        )
        print("Kaggle upload complete.")
    print("Training complete.")


# ============================================================
# Modal glue: image, app, volumes, train_remote, main
#   Defined at module level on non-Kaggle so the worker's module import
#   registers train_remote with the app. Skipped entirely on Kaggle.
# ============================================================
if not IS_KAGGLE:
    import modal

    train_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            add_python="3.12",
        )
        .entrypoint([])
        .apt_install("git", "build-essential", "clang")
        .pip_install(
            "torch==2.10.0",
            extra_index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            "safetensors>=0.5.0",
            "transformers>=4.56.2",
            "accelerate>=1.0.0",
            "peft>=0.15.0",
            "bitsandbytes>=0.45.0",
            "huggingface_hub>=0.36.2",
            "hf-transfer>=0.1.9",
            "numpy",
            "pillow",
            "torchvision",
            "datasets",
            "sentencepiece",
            "xformers",
            "cut-cross-entropy>=25.1.0",
            "wheel",
            "setuptools",
            "trl",
            "kaggle>=1.6.0",
        )
        .run_commands(
            'python -c "import torch.utils.cpp_extension as e; p=e.__file__; '
            "t=open(p).read().replace('raise RuntimeError(CUDA_MISMATCH_MESSAGE', 'pass  # '); "
            "open(p,'w').write(t)\"",
            "TORCH_CUDA_ARCH_LIST='12.0' pip wheel --no-build-isolation --wheel-dir /wheels mamba_ssm==2.3.1 causal_conv1d==1.6.1",
            "pip install --no-deps /wheels/mamba_ssm-*.whl /wheels/causal_conv1d-*.whl",
            "pip install --no-deps 'unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo'",
            "pip install --no-deps 'unsloth[base] @ git+https://github.com/unslothai/unsloth'",
        )
        .pip_install("einops")
        .env({"HF_HOME": "/root/.cache/huggingface"})
    )

    hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
    merged_vol = modal.Volume.from_name("merged-adapter", create_if_missing=True)
    corpus_vol = modal.Volume.from_name("corpus-data", create_if_missing=True)
    output_vol = modal.Volume.from_name("post-finetune-output", create_if_missing=True)

    app = modal.App("post-finetune-pipeline")

    @app.function(
        image=train_image,
        gpu="RTX-PRO-6000",
        volumes={
            "/root/.cache/huggingface": hf_cache_vol,
            "/merged": merged_vol,
            "/data": corpus_vol,
            "/output": output_vol,
        },
        timeout=6 * 60 * MINUTES,
        secrets=[modal.Secret.from_local_environ(["KAGGLE_API_TOKEN"])],
    )
    def train_remote() -> None:
        run_training()

    if IS_MODAL_LAUNCHER:

        @app.local_entrypoint()
        def main() -> None:
            train_remote.remote()


# ============================================================
# Entry point
#   On Kaggle: run training directly after cells load.
#   On Modal worker: Modal's runtime calls train_remote() -> run_training().
#   (The old module-level `del model` cleanup is gone: there is no longer a
#    temporary tokenization model loaded at module scope.)
# ============================================================
if IS_KAGGLE:
    run_training()
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
# Config
# ============================================================
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

MAX_SEQ_LEN = 8192
NUM_STEPS = 1000
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 3.5e-4
WEIGHT_DECAY = 0.0

# LR schedule: "warmup_cosine" or "original_linear"
LR_SCHEDULE = "warmup_cosine"
WARMUP_FRAC = 0.03
LR_FLOOR_FRAC = 0.10

GRAD_CLIP_NORM = 1.0

RESET_WEIGHTS = True
IN_PROJ_ONLY = False
MOE_TIE_WEIGHTS = True
ORIGINAL_PROBLEMS_ONLY = False
SHUFFLE_DATASET = False

USE_MATH_REPLAY = True
TARGET_REPLAY_ANSWER_TOKENS = 2_000_000

SAVE_INTERMEDIATE_CHECKPOINTS = False
CHECKPOINT_EVERY = 200

PRINT_DIAGNOSTICS = False

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "in_proj", "out_proj", "lm_head",
]

CORPUS_PATH = "/kaggle/input/datasets/huikang/huikang-nemotron-repository-snapshot/nemotron-master/training/sft/04-08-16-14/tokens"
TRAIN_ORDER_PATH = "/kaggle/input/datasets/huikang/huikang-nemotron-repository-snapshot/nemotron-master/training/sft/04-08-16-14/logprobs/index.jsonl"
TRAIN_CSV_PATH = "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv"
MATH_REPLAY_PATH = "/kaggle/input/datasets/mohamedamr992/replay-math/nemotron_math_1gb.jsonl"
MATH_TOKENIZED_PATH = "/kaggle/working/replay_math_tokenized.jsonl"


# ============================================================
# Install packages
# ============================================================
import subprocess

subprocess.run(
    "pip install -q --no-index --find-links /kaggle/input/datasets/mayukh18/nemotron-packages/packages "
    "unsloth trl peft transformers datasets accelerate bitsandbytes",
    shell=True, check=True,
)
subprocess.run(
    "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    shell=True, check=True,
)
subprocess.run(
    "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    shell=True, check=True,
)

_rtx_wheels = "/kaggle/input/datasets/llkh0a/rtx-wheels/wheels"
if __import__("os").path.isdir(_rtx_wheels):
    subprocess.run(
        ["pip", "install", "-q", "--no-index", "--find-links", _rtx_wheels,
         "protobuf==6.33.5", "sentencepiece", "safetensors", "huggingface_hub"],
        check=False,
    )

subprocess.run("rm -rf /kaggle/tmp/*", shell=True, check=True)
print("Packages installed.")


# ============================================================
# Tokenize math replay corpus
# ============================================================
if USE_MATH_REPLAY:
    import json
    import unsloth  # noqa: F401
    import kagglehub
    from transformers import AutoTokenizer
    from tqdm.auto import tqdm

    _model_path = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
    tok = AutoTokenizer.from_pretrained(_model_path, trust_remote_code=True)

    if PRINT_DIAGNOSTICS:
        with open(MATH_REPLAY_PATH) as f:
            _row = json.loads(next(f))
        _msgs = _row["messages"]
        _rendered = tok.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=False)
        _reasoning = _msgs[1].get("reasoning_content", "")
        _final = _msgs[1].get("content", "")
        print("Reasoning present:", bool(_reasoning))
        print("Final content present:", bool(_final))
        if _reasoning:
            print("Reasoning in template:", _reasoning[:100] in _rendered)
        if _final:
            print("Final content in template:", _final[:100] in _rendered)
        print("\n--- Rendered preview ---\n")
        print(_rendered[:2000])

    kept = skipped = dropped_too_long = 0
    total_tokens = total_answer_tokens = 0

    with open(MATH_REPLAY_PATH) as fin, open(MATH_TOKENIZED_PATH, "w") as fout:
        for line in tqdm(fin):
            row = json.loads(line)
            messages = row.get("messages")

            if not messages or len(messages) < 2:
                skipped += 1
                continue

            prompt_ids = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
            full_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

            if len(full_ids) <= len(prompt_ids):
                skipped += 1
                continue

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

    print(f"Replay kept={kept}, skipped={skipped}, dropped_too_long={dropped_too_long}")
    print(f"Total tokens: {total_tokens:,}, answer tokens: {total_answer_tokens:,}")
    print(f"Saved to: {MATH_TOKENIZED_PATH}")
else:
    print("USE_MATH_REPLAY=False — skipping replay tokenization.")


# ============================================================
# Training
# ============================================================
def run_training() -> None:
    import gc
    import json
    import math
    import os
    import random
    import time

    import kagglehub
    import torch
    from cut_cross_entropy import linear_cross_entropy
    from peft import LoraConfig
    from peft.tuners.lora import Linear as LoraLinear
    from safetensors.torch import load_file, save_file
    from unsloth import FastLanguageModel

    # ── Paths ─────────────────────────────────────────────────
    ADAPTER_SRC = "/kaggle/tmp/pretrained_adapter"
    if not RESET_WEIGHTS:
        import zipfile
        _adapter_zip = "/kaggle/input/notebooks/huikang/tinker-submission-notebook/submission.zip"
        os.makedirs(ADAPTER_SRC, exist_ok=True)
        with zipfile.ZipFile(_adapter_zip) as zf:
            zf.extractall(ADAPTER_SRC)

    MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
    save_dir = "."
    ckpt_base = "/kaggle/working"

    # ── GPU sanity check ──────────────────────────────────────
    import causal_conv1d
    import mamba_ssm
    import sys

    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, sm_{cc[0] * 10 + cc[1]}")
    print(f"torch={torch.__version__}, mamba_ssm={mamba_ssm.__version__}, causal_conv1d={causal_conv1d.__version__}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from causal_conv1d import causal_conv1d_fn
    _x = torch.randn(1, 256, 32, device="cuda", dtype=torch.bfloat16)
    _w = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
    causal_conv1d_fn(_x, _w, None, activation="silu")
    print("causal_conv1d CUDA kernel: OK")

    # ── Load corpus ───────────────────────────────────────────
    examples: list[dict] = []
    dropped_too_long = 0

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
    print(f"Training order: {len(ordered_ids)} problem_ids")

    for sid in ordered_ids:
        seg_path = os.path.join(CORPUS_PATH, sid, "synthetic.json")
        assert os.path.isfile(seg_path), f"Missing corpus file: {seg_path}"
        with open(seg_path) as f:
            rec = json.load(f)
        tokens = rec["tokens"]
        mask = rec["mask"]
        if not tokens or len(tokens) > MAX_SEQ_LEN or not any(mask):
            if len(tokens) > MAX_SEQ_LEN:
                dropped_too_long += 1
            continue
        examples.append({
            "problem_id": sid,
            "tokens": tokens[:-1],
            "targets": tokens[1:],
            "weights": [float(m) for m in mask[1:]],
        })

    if dropped_too_long:
        print(f"NOTE: dropped {dropped_too_long} examples > MAX_SEQ_LEN={MAX_SEQ_LEN}")

    if ORIGINAL_PROBLEMS_ONLY:
        import csv
        with open(TRAIN_CSV_PATH) as f:
            original_ids = {row["id"] for row in csv.DictReader(f)}
        before = len(examples)
        examples = [e for e in examples if e["problem_id"] in original_ids]
        print(f"ORIGINAL_PROBLEMS_ONLY: filtered {before} → {len(examples)}")

    total_unmasked = sum(sum(e["weights"]) for e in examples)
    total_tokens = sum(len(e["tokens"]) for e in examples)
    print(f"Loaded {len(examples)} examples, {total_tokens:,} tokens (unmasked={total_unmasked:,.0f})")

    # ── Interleave math replay ────────────────────────────────
    if USE_MATH_REPLAY:
        assert os.path.isfile(MATH_TOKENIZED_PATH), f"Missing: {MATH_TOKENIZED_PATH}"
        replay_examples: list[dict] = []
        with open(MATH_TOKENIZED_PATH) as f:
            for line in f:
                rec = json.loads(line)
                tokens = rec["tokens"]
                mask = rec["mask"]
                if len(tokens) < 2 or not any(mask[1:]):
                    continue
                replay_examples.append({
                    "problem_id": "replay_math",
                    "tokens": tokens[:-1],
                    "targets": tokens[1:],
                    "weights": [float(m) for m in mask[1:]],
                })

        print(f"Target examples: {len(examples)}, replay examples: {len(replay_examples)}")
        assert len(replay_examples) > 0, "No replay examples loaded"

        mixed: list[dict] = []
        replay_idx = 0
        replay_every = max(1, len(examples) // len(replay_examples))
        for i, ex in enumerate(examples):
            mixed.append(ex)
            if (i + 1) % replay_every == 0 and replay_idx < len(replay_examples):
                mixed.append(replay_examples[replay_idx])
                replay_idx += 1
        mixed.extend(replay_examples[replay_idx:])
        examples = mixed

        mixed_unmasked = sum(sum(e["weights"]) for e in examples)
        print(f"After replay: {len(examples)} examples, unmasked={mixed_unmasked:,.0f}")
    else:
        print("USE_MATH_REPLAY=False — training on target corpus only.")

    # ── Load model ────────────────────────────────────────────
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

    # ── Add LoRA ──────────────────────────────────────────────
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

    # ── Enable Mamba CUDA fast path ───────────────────────────
    nemotron_mod = None
    for _name, _m in sys.modules.items():
        if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
            nemotron_mod = _m
            break
    assert nemotron_mod is not None, "Could not find modeling_nemotron_h module"
    nemotron_mod.is_fast_path_available = True
    print("Patched is_fast_path_available = True")

    # ── Manually add lm_head LoRA (Unsloth drops it for MoE) ──
    _causal_lm = model
    while hasattr(_causal_lm, "model"):
        _causal_lm = _causal_lm.model
    _lm_head = _causal_lm.lm_head
    if not isinstance(_lm_head, LoraLinear):
        _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
        model.base_model._create_and_replace(_cfg, "default", target=_lm_head, target_name="lm_head", parent=_causal_lm)
        print("Manually added LoRA to lm_head")
    else:
        print("lm_head already has LoRA")

    # ── Cast LoRA params to fp32, verify dtypes ───────────────
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.data = param.data.to(torch.float32)

    for name, param in model.named_parameters():
        if ".lora_" in name:
            assert param.dtype == torch.float32, f"{name}: expected fp32, got {param.dtype}"
        elif ".mixer.gate." in name:
            assert param.dtype == torch.float32, f"{name}: expected fp32, got {param.dtype}"
        else:
            assert param.dtype == torch.bfloat16, f"{name}: expected bf16, got {param.dtype}"
    print("Verified: LoRA=fp32, base=bf16, MoE router=fp32")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ── Patch forward with Cut Cross-Entropy ──────────────────
    _base = model
    while hasattr(_base, "model"):
        _base = _base.model

    def _patched_causal_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
        backbone_out = _base.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{k: v for k, v in kwargs.items() if k in ("position_ids", "past_key_values", "use_cache")},
        )
        hidden_states = backbone_out[0]
        lm_head = _base.lm_head
        lm_weight = lm_head.base_layer.weight + lm_head.scaling["default"] * lm_head.lora_B["default"].weight @ lm_head.lora_A["default"].weight
        if labels is not None:
            per_token_ce = linear_cross_entropy(hidden_states, lm_weight, labels, reduction="none")
            loss = per_token_ce.mean()
        else:
            per_token_ce = loss = None
        model._cached_per_token_ce = per_token_ce
        return loss

    _base.forward = _patched_causal_forward
    print("Patched forward with Cut Cross-Entropy")

    # ── Load pretrained adapter (if not resetting) ────────────
    if RESET_WEIGHTS:
        print("RESET_WEIGHTS=True — fresh LoRA init")
    else:
        print(f"Loading adapter from {ADAPTER_SRC}...")
        from peft import load_peft_weights

        adapter_weights = load_peft_weights(ADAPTER_SRC)
        model_sd = model.state_dict()
        new_sd: dict = {}
        loaded = 0
        for ak, av in adapter_weights.items():
            for candidate in [
                ak,
                ak.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"),
                ak.replace(".backbone.lm_head.", ".lm_head.").replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"),
            ]:
                if candidate in model_sd:
                    new_sd[candidate] = av
                    loaded += 1
                    break
        model.load_state_dict(new_sd, strict=False)
        assert loaded == len(adapter_weights), f"Loaded {loaded}/{len(adapter_weights)} adapter weights"
        print(f"Loaded {loaded} adapter weights")

    # ── Freeze non-in_proj params (if IN_PROJ_ONLY) ───────────
    if IN_PROJ_ONLY:
        for name, param in model.named_parameters():
            if param.requires_grad and ".in_proj." not in name:
                param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable_params:,} ({IN_PROJ_ONLY=})")

    # ── MoE weight tying ──────────────────────────────────────
    moe_tied_params: list[torch.Tensor] = []
    if MOE_TIE_WEIGHTS:
        w1_names = ("gate_up_proj", "up_proj", "gate_proj", ".w1.")
        w2_names = ("down_proj", ".w2.")
        for name, param in model.named_parameters():
            if not param.requires_grad or ".experts." not in name or ".lora_" not in name:
                continue
            is_w1 = any(p in name for p in w1_names)
            is_w2 = any(p in name for p in w2_names)
            is_A = ".lora_A." in name
            is_B = ".lora_B." in name
            if not ((is_w1 and is_A) or (is_w2 and is_B)):
                continue
            if param.dim() < 2 or param.shape[0] <= 1:
                continue
            moe_tied_params.append(param)

        def _tie_param_init() -> None:
            with torch.no_grad():
                for p in moe_tied_params:
                    p.data.copy_(p.data.mean(dim=0, keepdim=True).expand_as(p.data))

        def _tie_grads() -> None:
            with torch.no_grad():
                for p in moe_tied_params:
                    if p.grad is not None:
                        p.grad.copy_(p.grad.sum(dim=0, keepdim=True).expand_as(p.grad))

        print(f"MoE weight tying: {len(moe_tied_params)} params")
        _tie_param_init()
    else:
        def _tie_grads() -> None:
            pass

    # ── Adapter save helper ───────────────────────────────────
    def save_adapter(target_dir: str) -> list[str]:
        os.makedirs(target_dir, exist_ok=True)
        for f in os.listdir(target_dir):
            if f.startswith("adapter"):
                os.remove(os.path.join(target_dir, f))
        model.save_pretrained(target_dir)
        st_path = os.path.join(target_dir, "adapter_model.safetensors")
        tensors = load_file(st_path)
        renamed = {
            k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
            for k, v in tensors.items()
        }
        save_file(renamed, st_path)
        return [f for f in os.listdir(target_dir) if f.startswith("adapter")]

    # ── Training loop ─────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    device = next(model.parameters()).device
    indices = list(range(len(examples)))
    if SHUFFLE_DATASET:
        random.Random(0).shuffle(indices)
        print(f"Shuffled {len(indices)} examples (seed=0)")

    training_log: list[str] = []

    def _log(msg: str) -> None:
        print(msg, flush=True)
        training_log.append(msg)

    max_steps = len(examples) // BATCH_SIZE
    num_steps = min(NUM_STEPS, max_steps)
    if num_steps < NUM_STEPS:
        _log(f"WARNING: clamped NUM_STEPS {NUM_STEPS} → {num_steps} (only {len(examples)} examples)")

    warmup_steps = max(1, int(WARMUP_FRAC * num_steps)) if LR_SCHEDULE != "original_linear" else 0

    def lr_at(s: int) -> float:
        if LR_SCHEDULE == "original_linear":
            return LEARNING_RATE * (1 - s / num_steps)
        if s < warmup_steps:
            return LEARNING_RATE * (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(1, num_steps - warmup_steps)
        return LEARNING_RATE * (LR_FLOOR_FRAC + (1 - LR_FLOOR_FRAC) * 0.5 * (1 + math.cos(math.pi * progress)))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=WEIGHT_DECAY,
    )

    _log(f"Training: {num_steps} steps, batch={BATCH_SIZE}, micro={MICRO_BATCH_SIZE}, "
         f"lr={LEARNING_RATE}, schedule={LR_SCHEDULE}, warmup={warmup_steps}, clip={GRAD_CLIP_NORM}")

    step = 0
    for batch_start in range(0, len(indices), BATCH_SIZE):
        if step >= num_steps:
            break
        batch = [examples[i] for i in indices[batch_start:batch_start + BATCH_SIZE]]
        batch_tokens = [e["tokens"] for e in batch]
        batch_targets = [e["targets"] for e in batch]
        batch_weights = [e["weights"] for e in batch]

        n = len(batch)
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

            padded_input = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
            padded_targets = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
            padded_weights = torch.zeros(n_micro, max_len, dtype=torch.float32, device=device)
            attention_mask = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)

            for i in range(n_micro):
                seq_len = len(mb_toks[i])
                padded_input[i, :seq_len] = torch.tensor(mb_toks[i], dtype=torch.long)
                padded_targets[i, :seq_len] = torch.tensor(mb_tgts[i], dtype=torch.long)
                padded_weights[i, :seq_len] = torch.tensor(mb_wts[i], dtype=torch.float32)
                attention_mask[i, :seq_len] = 1

            t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=padded_input, attention_mask=attention_mask, labels=padded_targets, use_cache=False)
                per_token_ce = model._cached_per_token_ce
                loss_sum_t = (per_token_ce * padded_weights).sum()
                weight_sum_t = padded_weights.sum()

            (loss_sum_t / batch_weight_total).backward()
            total_loss_sum += loss_sum_t.item()
            total_weight_sum += weight_sum_t.item()
            del per_token_ce, loss_sum_t

            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"    micro {mb_start // MICRO_BATCH_SIZE}: {n_micro} seqs, max_len={max_len}, "
                  f"total_len={total_len}, wall={time.time() - t0:.1f}s, peak={peak_gb:.1f}GB, mem={mem_gb:.1f}GB")

        lr = lr_at(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        _tie_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=GRAD_CLIP_NORM
        )
        optimizer.step()
        optimizer.zero_grad()

        loss_mean = total_loss_sum / total_weight_sum if total_weight_sum > 0 else 0
        step += 1
        _log(f"  step {step}/{num_steps}: loss={loss_mean:.6f}, grad_norm={grad_norm:.4f}, lr={lr:.2e}")

        if SAVE_INTERMEDIATE_CHECKPOINTS and step % CHECKPOINT_EVERY == 0 and step < num_steps:
            ckpt_dir = os.path.join(ckpt_base, "checkpoints", f"step_{step}")
            save_adapter(ckpt_dir)
            _log(f"  saved checkpoint to {ckpt_dir}")

    print(f"\nTraining complete. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # ── Save adapter ──────────────────────────────────────────
    adapter_files = save_adapter(save_dir)

    _ucache = "unsloth_compiled_cache"
    if os.path.isdir(_ucache):
        import shutil
        shutil.rmtree(_ucache)

    # ── Package submission ────────────────────────────────────
    import zipfile

    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in adapter_files:
            zf.write(os.path.join(save_dir, fname), fname)
    for fname in adapter_files:
        os.remove(os.path.join(save_dir, fname))
    print("Wrote submission.zip")
    print("Done.")


run_training()

import json, os

os.makedirs("/kaggle/working/upload_tmp", exist_ok=True)
subprocess.run(["cp", "/kaggle/working/submission.zip", "/kaggle/working/upload_tmp/submission.zip"], check=True)

with open("/kaggle/working/upload_tmp/dataset-metadata.json", "w") as f:
    json.dump({
        "title": "nemotron-sft-adapter",
        "id": "tadityasasidhar/nemotron-sft-adapter",
        "licenses": [{"name": "CC0-1.0"}],
    }, f)

subprocess.run(["kaggle", "datasets", "create", "-p", "/kaggle/working/upload_tmp"], check=True)
print("Uploaded to kaggle.com/datasets/tadityasasidhar/nemotron-sft-adapter")
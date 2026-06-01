# Import statements and environment setup
import os

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

IS_MODAL_WORKER = "MODAL_TASK_ID" in os.environ

IS_MODAL_LAUNCHER = not IS_KAGGLE and not IS_MODAL_WORKER

# Main execution block
if IS_KAGGLE:
    import subprocess

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


# Main training function
def run_training() -> None:
    """Full training flow. Runs on Kaggle at module level or inside Modal container via train_remote()."""
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

    # ── Env-specific paths + adapter source ──────────────────────────

    # Main execution block
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
    else:  # IS_MODAL_WORKER
        MODEL_PATH = "unsloth/Nemotron-3-Nano-30B-A3B"
        CORPUS_PATH = "/data/corpus_preprocessed.jsonl"
        TRAIN_CSV_PATH = "/data/train.csv"
        ADAPTER_SRC = "/merged/weights"
        OUTPUT_DIR = "/output/weights"

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

    _x = torch.randn(1, 512, 32, device="cuda", dtype=torch.bfloat16) + 4e-3
    _w = torch.randn(512, 4, device="cuda", dtype=torch.bfloat16)
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

    if IS_KAGGLE:
        # Load problem_ids in training order from logprobs/index.jsonl (epoch 0).
        # Each entry has {epoch, step, problem_id, ...}; epoch-0 entries are the
        # original training order, which we replay here.
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
            if len(tokens) > MAX_SEQ_LEN:
                tokens = tokens[:MAX_SEQ_LEN]
                mask = mask[:MAX_SEQ_LEN]
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
                    tokens = tokens[:MAX_SEQ_LEN]
                    mask = mask[:MAX_SEQ_LEN]
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
        use_gradient_checkpointing="unsloth",
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
        # Nemotron-H loads the MoE router (`mixer.gate`) in fp32 on purpose.
        # Ref: transformers/src/transformers/models/nemotron_h/modeling_nemotron_h.py
        #
        #   class NemotronHPreTrainedModel(PreTrainedModel):
        #       _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
        #
        #   class NemotronHTopkRouter(nn.Module):
        #       def __init__(self, config):
        #           self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        #           self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
        #       def forward(self, hidden_states):
        #           router_logits = F.linear(
        #               hidden_states.type(torch.float32),
        #               self.weight.type(torch.float32),
        #           )
        #           return router_logits
        #
        # The per-forward fp32 cast on `self.weight` plus the strict list entry
        # mean the gate weight is promoted to fp32 at load time.
        if is_router:
            assert param.dtype == torch.float32, (
                f"param {name} expected fp32, got {param.dtype}"
            )
            continue

        assert param.dtype == torch.bfloat16, (
            f"param {name} expected bf16, got {param.dtype}"
        )
        continue

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
        loaded = 0
        adapter_weights: dict = {}
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

    # ── Freeze all LoRA params except in_proj (if IN_PROJ_ONLY) ──
    print(f"{IN_PROJ_ONLY=}")
    if IN_PROJ_ONLY:
        for name, param in model.named_parameters():
            if param.requires_grad and ".in_proj." not in name:
                param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  {trainable_params:,} trainable / {frozen_params:,} frozen")

    # ── MoE tied-weight params (Tinker convention) ───────────────────
    # Tinker ties whichever LoRA side touches the hidden dim:
    #   gate_up_proj / up_proj / w1 / gate_proj  -> tie A (input/hidden side)
    #   down_proj / w2                           -> tie B (output/hidden side)
    # We keep Unsloth's batched [num_experts, ...] tensor layout; "tying" means
    # all 128 expert slices are kept identical. Saving the adapter naturally
    # emits 128 per-expert copies, so submission.zip is untied downstream.
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
            # Sum (not mean) across the expert dim: if W is the shared LoRA factor
            # and each expert uses a copy W_i = W, chain rule gives
            # dL/dW = sum_i dL/dW_i. Inactive experts contribute 0 and router
            # weights are already baked into active g_i, so there's no
            # double-counting. Summing keeps all 128 slices identical after each
            # AdamW step and reproduces the true shared-weight update; mean would
            # be off by a 1/128 lr rescale (and not exactly equivalent under
            # AdamW's eps/weight-decay).
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

    # ── Training loop ────────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    device = next(model.parameters()).device
    optimizer: torch.optim.AdamW | None = None

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

    _log(
        f"Training: {num_steps} steps, batch_size={BATCH_SIZE}, "
        f"micro_batch_size={MICRO_BATCH_SIZE}, lr={LEARNING_RATE}"
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
        n_accum = math.ceil(n / MICRO_BATCH_SIZE)
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
                loss = (
                    loss_sum_t / weight_sum_t if weight_sum_t > 0 else loss_sum_t * 0.0
                )

            (loss / n_accum).backward()
            total_loss_sum += loss_sum_t.item()
            total_weight_sum += weight_sum_t.item()
            del loss, per_token_ce, weighted_loss

            t_end = time.time()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            mem_gb = torch.cuda.memory_allocated() / 1e9
            mb_idx = mb_start // MICRO_BATCH_SIZE
            print(
                f"    micro-batch {mb_idx}: {n_micro} seqs, max_len={max_len}, "
                f"total_len={total_len}, wall={t_end - t0:.1f}s, "
                f"peak={peak_gb:.1f}GB, mem={mem_gb:.1f}GB"
            )

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=LEARNING_RATE,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.0,
            )
        lr = LEARNING_RATE * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        _tie_grads()  # average MoE expert grads before clip+step so Adam stays in sync
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1e9
        )
        optimizer.step()
        optimizer.zero_grad()
        loss_mean = total_loss_sum / total_weight_sum if total_weight_sum > 0 else 0
        step += 1
        _log(
            f"  step {step}/{num_steps}: "
            f"loss:mean={loss_mean:.6f}, grad_norm={grad_norm:.4f}, lr={lr:.2e}"
        )

    print(
        f"\nTraining complete. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
    )

    # ── Save adapter + rename lm_head keys (identical on both sides) ──
    from safetensors.torch import load_file, save_file

    save_dir = "." if IS_KAGGLE else OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    for _f in os.listdir(save_dir):
        if _f.startswith("adapter"):
            os.remove(os.path.join(save_dir, _f))
    model.save_pretrained(save_dir)
    st_path = os.path.join(save_dir, "adapter_model.safetensors")
    tensors = load_file(st_path)
    renamed = {
        k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
        for k, v in tensors.items()
    }
    save_file(renamed, st_path)

    # ── Clean unsloth compiled cache (runs on both) ──────────────────
    _ucache = "unsloth_compiled_cache"
    if os.path.isdir(_ucache):
        import shutil as _sh

        _sh.rmtree(_ucache)

    # ── Package & ship (divergent) ───────────────────────────────────

    # Main execution block
    if IS_KAGGLE:
        import zipfile

        adapter_files = [f for f in os.listdir(save_dir) if f.startswith("adapter")]
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
            shutil.copy(os.path.join(save_dir, fname), upload_dir)
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


if IS_KAGGLE:
    run_training()

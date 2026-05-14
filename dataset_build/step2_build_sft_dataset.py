"""
step2_build_sft_dataset.py
──────────────────────────
Converts sft_dataset_clean.jsonl into a HuggingFace dataset
formatted with the Nemotron chat template, ready for SFTTrainer.

Run:
  python3 step2_build_sft_dataset.py

Output:
  sft_hf_dataset/   — HuggingFace dataset directory (upload to Kaggle)
  sft_sample.txt    — 3 formatted examples for visual inspection
"""

import json
import random
from pathlib import Path

CLEAN_JSONL   = "sft_dataset_clean.jsonl"
HF_OUTPUT_DIR = "sft_hf_dataset"
SAMPLE_FILE   = "sft_sample.txt"
SEED          = 42

BOXED_INSTRUCTION = (
    "\nPlease reason step by step. "
    "Put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# ─────────────────────────────────────────────────────────────────────────────

def format_example(record: dict) -> dict:
    """
    Format one record into the text field SFTTrainer expects.

    We use a simple manual template here because we don't have
    the tokenizer available locally. The Kaggle notebook will
    re-apply the official chat template on top of this.

    User turn  : prompt + boxed instruction
    Assistant  : full COT with <think>...</think>\\boxed{answer}
    """
    user_content      = record["prompt"] + BOXED_INSTRUCTION
    assistant_content = record["cot"]    # already <think>...</think>\boxed{}

    # Raw text format — SFTTrainer will tokenize this
    text = (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )

    return {
        "id":    record["id"],
        "type":  record["type"],
        "text":  text,
        "answer": record["answer"],
    }


def main():
    # ── Load clean JSONL ──────────────────────────────────────────────────
    print(f"Loading {CLEAN_JSONL}...")
    records = []
    with open(CLEAN_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} clean records")

    # ── Shuffle ───────────────────────────────────────────────────────────
    rng = random.Random(SEED)
    rng.shuffle(records)

    # ── Format ───────────────────────────────────────────────────────────
    formatted = [format_example(r) for r in records]

    # ── Train / val split (95/5) ──────────────────────────────────────────
    split_idx  = int(len(formatted) * 0.95)
    train_data = formatted[:split_idx]
    val_data   = formatted[split_idx:]
    print(f"Train : {len(train_data)}")
    print(f"Val   : {len(val_data)}")

    # ── Save as HuggingFace dataset ───────────────────────────────────────
    try:
        from datasets import Dataset, DatasetDict
        ds = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
        })
        ds.save_to_disk(HF_OUTPUT_DIR)
        print(f"\nHuggingFace dataset saved → {HF_OUTPUT_DIR}/")
        print("  Upload this folder to Kaggle as a dataset source.")
    except ImportError:
        # Fallback: save as JSONL if datasets not installed
        print("datasets library not found — saving as JSONL instead")
        Path(HF_OUTPUT_DIR).mkdir(exist_ok=True)
        with open(f"{HF_OUTPUT_DIR}/train.jsonl", "w") as f:
            for rec in train_data:
                f.write(json.dumps(rec) + "\n")
        with open(f"{HF_OUTPUT_DIR}/validation.jsonl", "w") as f:
            for rec in val_data:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved train.jsonl ({len(train_data)} rows)")
        print(f"Saved validation.jsonl ({len(val_data)} rows)")

    # ── Save sample for visual inspection ────────────────────────────────
    samples = rng.sample(formatted, min(3, len(formatted)))
    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            f.write(f"{'='*60}\n")
            f.write(f"SAMPLE {i+1}  |  type={s['type']}  |  answer={s['answer']}\n")
            f.write(f"{'='*60}\n")
            f.write(s["text"][:1500])
            f.write("\n... [truncated]\n\n")
    print(f"Sample saved → {SAMPLE_FILE}")
    print("\nDone. Next step: upload sft_hf_dataset/ to Kaggle then run the training notebook.")


if __name__ == "__main__":
    main()

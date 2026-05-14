"""
step1_validate_dataset.py
─────────────────────────
Validates problem_ids_matched_v2.csv, fixes the missing <think> tag,
verifies boxed answers match ground truth, and saves a clean JSONL.

Run:
  python3 step1_validate_dataset.py

Output:
  sft_dataset_clean.jsonl   — clean rows ready for SFT
  sft_dataset_rejected.jsonl — rows that failed validation
  validation_report.txt     — summary stats
"""

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV      = "problem_ids_matched_v2.csv"
OUTPUT_CLEAN   = "sft_dataset_clean.jsonl"
OUTPUT_REJECTED= "sft_dataset_rejected.jsonl"
REPORT_FILE    = "validation_report.txt"

BOXED_INSTRUCTION = (
    "\nPlease reason step by step. "
    "Put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
# ─────────────────────────────────────────────────────────────────────────────


def extract_boxed(text: str) -> str:
    """Extract last \\boxed{} answer — mirrors competition metric exactly."""
    boxed_starts = list(re.finditer(r'\\boxed\{', text))
    if not boxed_starts:
        return "NOT_FOUND"
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end   = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        seg   = text[start:end]
        last  = seg.rfind("}")
        matches.append(seg[:last] if last != -1 else seg)
    non_empty = [m.strip() for m in matches if m.strip()]
    return non_empty[-1] if non_empty else matches[-1].strip()


def verify_answer(ground_truth: str, predicted: str) -> bool:
    """Mirrors competition metric verify() exactly."""
    gt   = ground_truth.strip()
    pred = predicted.strip()
    if re.fullmatch(r'[01]+', gt):
        return pred.lower() == gt.lower()
    try:
        return math.isclose(float(gt), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return pred.lower() == gt.lower()


def fix_cot(raw_cot: str) -> str:
    """
    The generated_cot has no opening <think> tag.
    It ends with:  ...reasoning...</think>\n\\boxed{answer}

    We prepend <think>\n to fix the format.
    Final format:
        <think>
        ...reasoning...
        </think>
        \\boxed{answer}
    """
    cot = raw_cot.strip()
    if not cot.startswith("<think>"):
        cot = "<think>\n" + cot
    return cot


def main():
    print(f"Reading {INPUT_CSV}...")
    rows = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Total rows: {len(rows)}")

    clean    = []
    rejected = []
    reasons  = Counter()
    type_stats = Counter()

    for row in rows:
        row_id  = row["id"]
        prompt  = row["prompt"].strip()
        answer  = row["answer"].strip()
        ptype   = row["type"].strip()
        raw_cot = row["generated_cot"].strip()

        # ── Fix <think> tag ───────────────────────────────────────────────
        fixed_cot = fix_cot(raw_cot)

        # ── Validate boxed answer matches ground truth ────────────────────
        extracted = extract_boxed(fixed_cot)

        if extracted == "NOT_FOUND":
            reasons["no_boxed_found"] += 1
            rejected.append({
                "id": row_id, "reason": "no_boxed_found",
                "answer": answer, "extracted": extracted
            })
            continue

        if not verify_answer(answer, extracted):
            reasons["answer_mismatch"] += 1
            rejected.append({
                "id": row_id, "reason": "answer_mismatch",
                "answer": answer, "extracted": extracted,
                "cot_tail": fixed_cot[-100:]
            })
            continue

        # ── Check COT is not empty / too short ───────────────────────────
        think_match = re.search(r'<think>(.*?)</think>', fixed_cot, re.DOTALL)
        if not think_match or len(think_match.group(1).strip()) < 20:
            reasons["cot_too_short"] += 1
            rejected.append({
                "id": row_id, "reason": "cot_too_short",
                "answer": answer
            })
            continue

        # ── Build clean record ────────────────────────────────────────────
        clean.append({
            "id":        row_id,
            "type":      ptype,
            "prompt":    prompt,
            "answer":    answer,
            "cot":       fixed_cot,   # <think>...</think>\boxed{answer}
        })
        type_stats[ptype] += 1

    # ── Write outputs ─────────────────────────────────────────────────────
    with open(OUTPUT_CLEAN, "w", encoding="utf-8") as f:
        for rec in clean:
            f.write(json.dumps(rec) + "\n")

    with open(OUTPUT_REJECTED, "w", encoding="utf-8") as f:
        for rec in rejected:
            f.write(json.dumps(rec) + "\n")

    # ── Report ────────────────────────────────────────────────────────────
    report_lines = [
        "═══════════════════════════════════════",
        "        DATASET VALIDATION REPORT      ",
        "═══════════════════════════════════════",
        f"Total input rows   : {len(rows)}",
        f"Clean rows         : {len(clean)}",
        f"Rejected rows      : {len(rejected)}",
        f"Pass rate          : {len(clean)/len(rows)*100:.1f}%",
        "",
        "── Rejection reasons ──",
    ]
    for reason, count in reasons.most_common():
        report_lines.append(f"  {count:5d}  {reason}")

    report_lines += [
        "",
        "── Clean rows by type ──",
    ]
    for t, count in type_stats.most_common():
        pct = count / len(clean) * 100
        report_lines.append(f"  {count:5d}  ({pct:5.1f}%)  {t}")

    report_lines += [
        "",
        f"Output → {OUTPUT_CLEAN}",
        f"Rejected → {OUTPUT_REJECTED}",
    ]

    report = "\n".join(report_lines)
    print("\n" + report)

    with open(REPORT_FILE, "w") as f:
        f.write(report + "\n")

    print(f"\nDone. Next step: python3 step2_build_sft_dataset.py")


if __name__ == "__main__":
    main()

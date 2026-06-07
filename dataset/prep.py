"""
Build SFT-ready teacher traces from Doubleword batch output(s) and wire them into
the existing run_training() corpus loader.

Flow:
    batch-output.jsonl  --(extract reasoning -> verify -> tokenize)-->  new_traces_tokenized.jsonl
    run_training():  correct-only huikang (correct_ids.json)  +  teacher traces  +  your existing replay

Run prepare_teacher_traces(tok) on Kaggle (needs the Nano tokenizer). Then apply the two
edits documented at the bottom of this file to your corpus loader. Nothing else changes.

Robust to all three teacher response shapes we saw: separate reasoning_content/reasoning
field, inline <think>...</think> in content, or plain prose (DeepSeek markdown). The Nano's
<think> format is re-imposed at tokenization, so the teacher's own format does not matter.
"""

import json
import math
import os
import re

# ----------------------------- config -----------------------------
BATCH_OUTPUTS = [                                   # add the pass-2 file here once you run it
    "/kaggle/working/batch-output.jsonl",
    # "/kaggle/working/batch-output-pass2.jsonl",
]
WRONG_TARGETS_PATH = "/kaggle/working/wrong_targets.jsonl"   # provides id -> {prompt, answer}
NEW_TOKENIZED_PATH = "/kaggle/working/new_traces_tokenized.jsonl"

SFT_MAX_SEQ_LEN  = 8192     # matches eval max_model_len; drop traces that won't fit
REL_TOL          = 1e-2     # mirror the eval metric (confirm exact value from the metric code)
KEEP_PER_PROBLEM = 1        # shortest correct trace per problem -> clean greedy path

# ----------------------------- verifier (mirrors the eval) -----------------------------
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")

def last_boxed(text):
    m = _BOXED.findall(text or "")
    return m[-1].strip() if m else None

def answer_match(pred, gt, rel_tol=REL_TOL):
    if pred is None or gt is None:
        return False
    p, g = str(pred).strip(), str(gt).strip()
    if p == g:
        return True
    try:
        return math.isclose(float(p), float(g), rel_tol=rel_tol, abs_tol=1e-9)
    except (ValueError, TypeError):
        return False

# ----------------------------- robust reasoning extractor -----------------------------
def get_reasoning_and_answer(msg):
    """Return (reasoning_text, content_text) for any of the three response shapes."""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""   # both vLLM field names
    content = msg.get("content") or ""
    if not reasoning and "<think>" in content:                  # thinking inlined as tags
        m = re.search(r"<think>(.*?)</think>", content, re.S)
        if m:
            reasoning = m.group(1).strip()
            content = content[m.end():].strip()
    if not reasoning:                                           # plain prose -> whole thing is the trace
        reasoning = content
    return reasoning, content

# ----------------------------- main prep -----------------------------
def prepare_teacher_traces(tok):
    gt, prompts = {}, {}
    for line in open(WRONG_TARGETS_PATH):
        r = json.loads(line)
        gt[r["id"]] = r["answer"]
        prompts[r["id"]] = r["prompt"]

    # 1) gather correct candidates per problem (rejection sampling)
    cand = {}
    stats = {"seen": 0, "wrong": 0, "nobox": 0, "no_gt": 0}
    for path in BATCH_OUTPUTS:
        if not os.path.isfile(path):
            print(f"(skip, not found: {path})")
            continue
        for line in open(path):
            row = json.loads(line)
            cid = (row.get("custom_id") or "").split("::")[0]   # works for "id" and "id::sN"
            if row.get("error"):
                continue
            ch = row.get("response", {}).get("body", {}).get("choices", [])
            if not ch:
                continue
            stats["seen"] += 1
            reasoning, content = get_reasoning_and_answer(ch[0]["message"])
            pred = last_boxed(content) or last_boxed(reasoning)
            ans = gt.get(cid)
            if ans is None:
                stats["no_gt"] += 1
                continue
            if pred is None:
                stats["nobox"] += 1
                continue
            if not answer_match(pred, ans):
                stats["wrong"] += 1
                continue
            cand.setdefault(cid, []).append((reasoning.strip(), pred))

    # 2) tokenize: keep K shortest correct per problem; re-impose the Nano <think> format
    kept = dropped = 0
    with open(NEW_TOKENIZED_PATH, "w") as fout:
        for cid, lst in cand.items():
            for reasoning, pred in sorted(lst, key=lambda x: len(x[0]))[:KEEP_PER_PROBLEM]:
                user = {"role": "user", "content": prompts[cid]}
                asst = {"role": "assistant",
                        "content": f"\\boxed{{{pred}}}",       # clean final answer
                        "reasoning_content": reasoning}        # template renders into <think>...</think>
                p_ids = tok.apply_chat_template([user], tokenize=True, add_generation_prompt=True)
                f_ids = tok.apply_chat_template([user, asst], tokenize=True, add_generation_prompt=False)
                if len(f_ids) <= len(p_ids):
                    continue
                if len(f_ids) > SFT_MAX_SEQ_LEN:
                    dropped += 1
                    continue
                mask = [0] * len(p_ids) + [1] * (len(f_ids) - len(p_ids))
                fout.write(json.dumps({"problem_id": cid, "tokens": f_ids, "mask": mask}) + "\n")
                kept += 1

    print(f"correct problems: {len(cand)} | kept traces: {kept} | dropped_too_long: {dropped}")
    print(f"rejected -> wrong:{stats['wrong']} no_box:{stats['nobox']} no_gt:{stats['no_gt']} "
          f"(responses seen: {stats['seen']})")

    # 3) format sanity check -- MUST show <think>: True and \boxed: True
    if kept:
        rec = json.loads(open(NEW_TOKENIZED_PATH).readline())
        txt = tok.decode([t for t, m in zip(rec["tokens"], rec["mask"]) if m])
        has_think, has_box = "<think>" in txt, "\\boxed" in txt
        print(f"trained region -> has <think>: {has_think} | has \\boxed: {has_box}")
        if not has_box:
            raise RuntimeError("No \\boxed in the trained region -- do not train on this.")
        if not has_think:
            print("WARNING: <think> missing. The template ignored reasoning_content; "
                  "switch to manual string construction before training.")
    return kept


if __name__ == "__main__":
    import kagglehub
    from transformers import AutoTokenizer
    _mp = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
    prepare_teacher_traces(AutoTokenizer.from_pretrained(_mp))


# ============================================================================
# WIRE INTO run_training()  --  two edits, nothing else changes
# ============================================================================
#
# EDIT 1 -- filter huikang's corpus to correct-only. At the top of the corpus-load
#           section, then skip non-correct ids inside the existing loop:
#
#     CORRECT_ONLY = True
#     correct_set = set(json.load(open("/kaggle/working/correct_ids.json"))) if CORRECT_ONLY else None
#
#     for sid in ordered_ids:                       # your existing loop
#         if correct_set is not None and sid not in correct_set:
#             continue
#         # ... existing example-building, unchanged ...
#
# EDIT 2 -- append the teacher traces (after the huikang loop; your replay code stays as-is):
#
#     NEW_TOKENIZED_PATH = "/kaggle/working/new_traces_tokenized.jsonl"
#     if os.path.isfile(NEW_TOKENIZED_PATH):
#         added = 0
#         for line in open(NEW_TOKENIZED_PATH):
#             rec = json.loads(line)
#             t, mk = rec["tokens"], rec["mask"]
#             if len(t) > MAX_SEQ_LEN or not any(mk):
#                 continue
#             examples.append({
#                 "problem_id": rec["problem_id"],
#                 "tokens":  t[:-1],
#                 "targets": t[1:],
#                 "weights": [float(x) for x in mk[1:]],
#             })
#             added += 1
#         print(f"Added {added} teacher-correct traces")
#
# Final corpus = ~6,304 correct huikang + teacher traces + ~491 replay.
# Train the unchanged 86 recipe on it and compare against your real 84 baseline.
# ============================================================================
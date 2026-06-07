import json, re, math

BATCH_OUTPUT_PATH  = "batch-142e7edc-6432-4067-bfae-f13489834674-output.jsonl"                   # the file you downloaded
WRONG_TARGETS_PATH = "wrong_targets.jsonl"  # provides id -> ground-truth answer
SHOW_FIRST_TRACE   = True

# ground truth
gt = {}
for l in open(WRONG_TARGETS_PATH):
    r = json.loads(l)
    gt[r["id"]] = r["answer"]

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
def last_boxed(text):
    m = BOXED.findall(text or "")
    return m[-1].strip() if m else None

def match(pred, ans, rel_tol=1e-2):          # mirrors the eval metric
    if pred is None:
        return False
    p, a = str(pred).strip(), str(ans).strip()
    if p == a:
        return True
    try:
        return math.isclose(float(p), float(a), rel_tol=rel_tol, abs_tol=1e-9)
    except (ValueError, TypeError):
        return False

correct = wrong = no_boxed = errors = 0
first = True
for l in open(BATCH_OUTPUT_PATH):
    row = json.loads(l)
    cid = row.get("custom_id")

    if row.get("error"):
        print(f"[ERR ] {cid}: {row['error']}")
        errors += 1
        continue

    choices = row.get("response", {}).get("body", {}).get("choices", [])
    if not choices:
        print(f"[NONE] {cid}: no choices")
        errors += 1
        continue

    msg = choices[0]["message"]
    reasoning = msg.get("reasoning_content") or ""
    content   = msg.get("content") or ""
    full = (reasoning + "\n" + content) if reasoning else content

    if first:
        print("message keys:", list(msg.keys()))     # reasoning_content vs inline <think>?
        if SHOW_FIRST_TRACE:
            print("\n===== FIRST FULL TRACE (first 4000 chars) =====\n")
            print(full[:4000])
            print("\n===============================================\n")
        first = False

    pred = last_boxed(full)
    ans  = gt.get(cid)
    if pred is None:
        no_boxed += 1; tag = "NOBX"
    elif match(pred, ans):
        correct += 1; tag = "OK  "
    else:
        wrong += 1; tag = "XX  "
    print(f"[{tag}] {cid}  pred={pred!r}  gt={ans!r}  chars={len(full)}")

total = correct + wrong + no_boxed
print(f"\nCorrect {correct}/{total} | wrong-answer {wrong} | no \\boxed {no_boxed} | errors {errors}")
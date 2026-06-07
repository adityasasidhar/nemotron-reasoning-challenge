import json

WRONG_TARGETS_PATH = "wrong_targets.jsonl"
BATCH_JSONL_PATH   = "doubleword_batch.jsonl"

MODEL           = "deepseek-ai/DeepSeek-V4-Pro"
TEMPERATURE     = 0.0      # single shot -> take the teacher's best (greedy), not a random sample
TOP_P           = 1.0      # irrelevant at temp 0
MAX_NEW_TOKENS  = 7680     # margin under the 7680 eval cap
ENABLE_THINKING = True
LOW_EFFORT      = False     # set to True to disable the "thinking" mode and just get the final answer; useful for debugging
PILOT_N         = None    # small sanity run; set to None for the full file

SYSTEM_PROMPT = (
    "Solve the problem by reasoning explicitly, one step at a time. Do not compute "
    "multiple elements at once: work bit-by-bit or character-by-character, writing out "
    "each individual operation. After deriving the rule, verify it by applying it to the "
    "exact input given in the problem before committing. Put the final answer in \\boxed{}."
)

ctk = {"enable_thinking": ENABLE_THINKING}
if LOW_EFFORT:
    ctk["low_effort"] = False

targets = [json.loads(l) for l in open(WRONG_TARGETS_PATH)]
if PILOT_N is not None:
    targets = targets[:PILOT_N]

with open(BATCH_JSONL_PATH, "w") as fout:
    for t in targets:
        fout.write(json.dumps({
            "custom_id": t["id"],                 # unique per problem (one sample each)
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": t["prompt"]},
                ],
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_NEW_TOKENS,
                "chat_template_kwargs": ctk,
            },
        }) + "\n")

print(f"{len(targets)} requests -> {BATCH_JSONL_PATH}")
print("sample line:\n", open(BATCH_JSONL_PATH).readline()[:600])
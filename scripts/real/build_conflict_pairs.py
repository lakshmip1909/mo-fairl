import json
import random
import re
from pathlib import Path
from datasets import load_dataset
from collections import Counter

OUT = Path("data/real/conflict_pairs.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

random.seed(42)

TARGET = {
    "safety_vs_math": 1800,
    "math_vs_safety": 600,
    "clean_a_wins": 600,
}

def extract_answer(ans):
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", ans)
    return m.group(1) if m else None

def wrong_answer(correct):
    try:
        x = int(float(correct))
        return str(x + random.choice([-10, -5, -2, -1, 1, 2, 5, 10]))
    except Exception:
        return str(correct) + "1"

def safe_response(answer):
    return "Sure, I can solve this carefully. The final answer is {}.".format(answer)

def toxic_response(answer):
    return "This is obvious, you idiot. The final answer is {}.".format(answer)

def make_sample(prompt, a, b, labels, conflict_type):
    return {
        "prompt": prompt,
        "response_a": a,
        "response_b": b,
        "labels": labels,
        "task": "gsm8k_conflict",
        "conflict_type": conflict_type,
        "is_conflict": conflict_type in {"safety_vs_math", "math_vs_safety"},
    }

print("Loading GSM8K...")
ds = load_dataset("gsm8k", "main", split="train")
print(f"Loaded {len(ds)} GSM8K examples")

samples = []
counts = Counter()

for ex in ds:
    q = ex["question"]
    correct = extract_answer(ex["answer"])
    if correct is None:
        continue

    wrong = wrong_answer(correct)

    safe_correct = safe_response(correct)
    safe_wrong = safe_response(wrong)
    toxic_correct = toxic_response(correct)
    toxic_wrong = toxic_response(wrong)

    if counts["safety_vs_math"] < TARGET["safety_vs_math"]:
        samples.append(make_sample(
            q, safe_wrong, toxic_correct,
            {"toxicity": 1, "math": 0, "code": None},
            "safety_vs_math"
        ))
        counts["safety_vs_math"] += 1

    if counts["math_vs_safety"] < TARGET["math_vs_safety"]:
        samples.append(make_sample(
            q, toxic_correct, safe_wrong,
            {"toxicity": 0, "math": 1, "code": None},
            "math_vs_safety"
        ))
        counts["math_vs_safety"] += 1

    if counts["clean_a_wins"] < TARGET["clean_a_wins"]:
        samples.append(make_sample(
            q, safe_correct, toxic_wrong,
            {"toxicity": 1, "math": 1, "code": None},
            "clean_a_wins"
        ))
        counts["clean_a_wins"] += 1

    if all(counts[k] >= TARGET[k] for k in TARGET):
        break

random.shuffle(samples)

with OUT.open("w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

print(f"Saved {len(samples)} samples to {OUT}")
print("Conflict types:")
for k, v in Counter(s["conflict_type"] for s in samples).items():
    print(f"  {k}: {v}")
print("is_conflict:")
for k, v in Counter(s["is_conflict"] for s in samples).items():
    print(f"  {k}: {v}")
print("Label patterns:")
for k, v in Counter(str(s["labels"]) for s in samples).items():
    print(f"  {k}: {v}")

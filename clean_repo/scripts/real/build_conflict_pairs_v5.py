import json, random, re
from pathlib import Path
from collections import Counter
from datasets import load_dataset

OUT = Path("data/real/conflict_pairs_v5.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
random.seed(42)

TARGET = {
    "safety_vs_math": 900,
    "math_vs_safety": 300,
    "safety_vs_code": 700,
    "code_vs_safety": 300,
    "math_vs_code": 400,
    "code_vs_math": 400,
}

def extract_answer(ans):
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", ans)
    return m.group(1) if m else None

def wrong_answer(correct):
    try:
        x = int(float(correct))
        return str(x + random.choice([-10,-5,-2,-1,1,2,5,10]))
    except Exception:
        return str(correct) + "1"

def safe_math(answer):
    return f"I think the final answer is {answer}."

def unsafe_math(answer):
    return f"This is basic and you should know it. The final answer is {answer}."

def good_code():
    return "def solve(x):\n    return x + 1"

def bad_code():
    return "def solve(x):\n    return x - 1"

def safe_code(code):
    return "Here is a possible implementation:\n```python\n" + code + "\n```"

def unsafe_code(code):
    return "This is obvious. Here is the code:\n```python\n" + code + "\n```"

def make(prompt, a, b, labels, ctype):
    return {
        "prompt": prompt,
        "response_a": a,
        "response_b": b,
        "labels": labels,
        "task": "v5_hard_conflict",
        "conflict_type": ctype,
        "is_conflict": True,
    }

print("Loading GSM8K...")
gsm = load_dataset("gsm8k", "main", split="train")
print(f"Loaded {len(gsm)} GSM8K examples")

samples, counts = [], Counter()

for ex in gsm:
    q = ex["question"]
    correct = extract_answer(ex["answer"])
    if correct is None:
        continue
    wrong = wrong_answer(correct)

    if counts["safety_vs_math"] < TARGET["safety_vs_math"]:
        samples.append(make(
            q,
            safe_math(wrong),
            unsafe_math(correct),
            {"toxicity": 1, "math": 0, "code": None},
            "safety_vs_math"
        ))
        counts["safety_vs_math"] += 1

    if counts["math_vs_safety"] < TARGET["math_vs_safety"]:
        samples.append(make(
            q,
            unsafe_math(correct),
            safe_math(wrong),
            {"toxicity": 0, "math": 1, "code": None},
            "math_vs_safety"
        ))
        counts["math_vs_safety"] += 1

    if counts["safety_vs_code"] < TARGET["safety_vs_code"]:
        samples.append(make(
            "Write a Python function solve(x) that returns x+1.",
            safe_code(bad_code()),
            unsafe_code(good_code()),
            {"toxicity": 1, "math": None, "code": 0},
            "safety_vs_code"
        ))
        counts["safety_vs_code"] += 1

    if counts["code_vs_safety"] < TARGET["code_vs_safety"]:
        samples.append(make(
            "Write a Python function solve(x) that returns x+1.",
            unsafe_code(good_code()),
            safe_code(bad_code()),
            {"toxicity": 0, "math": None, "code": 1},
            "code_vs_safety"
        ))
        counts["code_vs_safety"] += 1

    if counts["math_vs_code"] < TARGET["math_vs_code"]:
        samples.append(make(
            q + "\nAlso provide a Python helper function solve(x) returning x+1.",
            safe_math(correct) + "\n```python\n" + bad_code() + "\n```",
            safe_math(wrong) + "\n```python\n" + good_code() + "\n```",
            {"toxicity": 1, "math": 1, "code": 0},
            "math_vs_code"
        ))
        counts["math_vs_code"] += 1

    if counts["code_vs_math"] < TARGET["code_vs_math"]:
        samples.append(make(
            q + "\nAlso provide a Python helper function solve(x) returning x+1.",
            safe_math(wrong) + "\n```python\n" + good_code() + "\n```",
            safe_math(correct) + "\n```python\n" + bad_code() + "\n```",
            {"toxicity": 1, "math": 0, "code": 1},
            "code_vs_math"
        ))
        counts["code_vs_math"] += 1

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
print("Labels:")
for k, v in Counter(str(s["labels"]) for s in samples).items():
    print(f"  {k}: {v}")

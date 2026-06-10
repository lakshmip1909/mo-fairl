import json, random, re, argparse
from pathlib import Path
from collections import Counter
from datasets import load_dataset

random.seed(42)

def extract_answer(ans):
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", ans)
    return m.group(1) if m else None

def wrong_answer(correct):
    try:
        x = int(float(correct))
        return str(x + random.choice([-10,-5,-2,-1,1,2,5,10]))
    except Exception:
        return str(correct) + "1"

def safe_math(a): return f"I think the final answer is {a}."
def unsafe_math(a): return f"This is basic and you should know it. The final answer is {a}."
def good_code(): return "def solve(x):\n    return x + 1"
def bad_code(): return "def solve(x):\n    return x - 1"
def safe_code(c): return "Here is a possible implementation:\n```python\n" + c + "\n```"
def unsafe_code(c): return "This is obvious. Here is the code:\n```python\n" + c + "\n```"

def dominant_global_label(labels, mode):
    # global_label means: 1 if response_a wins overall, 0 if response_b wins overall.
    if mode == "tox":
        return labels["toxicity"]
    if mode == "math":
        return labels["math"]
    if mode == "code":
        return labels["code"]
    return None

def make(prompt, a, b, labels, ctype, mode):
    return {
        "prompt": prompt,
        "response_a": a,
        "response_b": b,
        "labels": labels,
        "global_label": dominant_global_label(labels, mode),
        "task": "weight_recovery",
        "conflict_type": ctype,
        "is_conflict": True,
    }

def build(mode):
    out = Path(f"data/real/weight_recovery_{mode}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    if mode == "tox":
        target = {"safety_vs_math": 1200, "safety_vs_code": 1200, "math_vs_safety": 300, "code_vs_safety": 300}
    elif mode == "math":
        target = {"math_vs_safety": 1200, "math_vs_code": 1200, "safety_vs_math": 300, "code_vs_math": 300}
    elif mode == "code":
        target = {"code_vs_safety": 1200, "code_vs_math": 1200, "safety_vs_code": 300, "math_vs_code": 300}
    else:
        raise ValueError(mode)

    print("Loading GSM8K...")
    gsm = load_dataset("gsm8k", "main", split="train")
    counts, samples = Counter(), []

    for ex in gsm:
        q = ex["question"]
        correct = extract_answer(ex["answer"])
        if correct is None:
            continue
        wrong = wrong_answer(correct)

        if counts["safety_vs_math"] < target.get("safety_vs_math", 0):
            samples.append(make(q, safe_math(wrong), unsafe_math(correct),
                {"toxicity":1, "math":0, "code":None}, "safety_vs_math", mode))
            counts["safety_vs_math"] += 1

        if counts["math_vs_safety"] < target.get("math_vs_safety", 0):
            samples.append(make(q, unsafe_math(correct), safe_math(wrong),
                {"toxicity":0, "math":1, "code":None}, "math_vs_safety", mode))
            counts["math_vs_safety"] += 1

        if counts["safety_vs_code"] < target.get("safety_vs_code", 0):
            samples.append(make("Write solve(x) returning x+1.", safe_code(bad_code()), unsafe_code(good_code()),
                {"toxicity":1, "math":None, "code":0}, "safety_vs_code", mode))
            counts["safety_vs_code"] += 1

        if counts["code_vs_safety"] < target.get("code_vs_safety", 0):
            samples.append(make("Write solve(x) returning x+1.", unsafe_code(good_code()), safe_code(bad_code()),
                {"toxicity":0, "math":None, "code":1}, "code_vs_safety", mode))
            counts["code_vs_safety"] += 1

        if counts["math_vs_code"] < target.get("math_vs_code", 0):
            samples.append(make(q + "\nAlso provide solve(x) returning x+1.",
                safe_math(correct) + "\n```python\n" + bad_code() + "\n```",
                safe_math(wrong) + "\n```python\n" + good_code() + "\n```",
                {"toxicity":1, "math":1, "code":0}, "math_vs_code", mode))
            counts["math_vs_code"] += 1

        if counts["code_vs_math"] < target.get("code_vs_math", 0):
            samples.append(make(q + "\nAlso provide solve(x) returning x+1.",
                safe_math(wrong) + "\n```python\n" + good_code() + "\n```",
                safe_math(correct) + "\n```python\n" + bad_code() + "\n```",
                {"toxicity":1, "math":0, "code":1}, "code_vs_math", mode))
            counts["code_vs_math"] += 1

        if all(counts[k] >= v for k, v in target.items()):
            break

    random.shuffle(samples)
    with out.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print("Saved", len(samples), "to", out)
    print("Conflict types:", Counter(s["conflict_type"] for s in samples))
    print("Labels:", Counter(str(s["labels"]) for s in samples))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tox","math","code"], required=True)
    args = ap.parse_args()
    build(args.mode)

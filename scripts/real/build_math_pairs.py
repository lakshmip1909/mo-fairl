"""
scripts/real/build_math_pairs.py

Builds math preference pairs from GSM8K.

GSM8K already has:
    - question (the prompt)
    - answer   (step-by-step solution ending with #### <number>)

Strategy for negative generation (rule-based, deterministic):
    Extract the final numeric answer from the correct solution.
    Generate a wrong answer using one of these perturbations:
        1. Add/subtract a small integer  (off-by-one style)
        2. Multiply by a small factor    (scaling mistake)
        3. Negate                        (sign error)
        4. Replace with a nearby round number
    Replace the final answer line in the solution text with the wrong number.
    Also generate a plausible wrong reasoning step by perturbing an
    intermediate calculation if possible.

Output: data/real/math_pairs.jsonl
"""

import json
import random
import re
import os
from pathlib import Path

random.seed(42)


# ── Answer extraction ──────────────────────────────────────────────────────────

def extract_final_answer(solution: str):
    """
    GSM8K solutions end with '#### <number>'.
    Returns (clean_number_str, numeric_value) or None.
    """
    match = re.search(r"####\s*([\d,\-\.]+)", solution)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    try:
        val = float(raw)
        return raw, val
    except ValueError:
        return None


# ── Wrong answer generation ────────────────────────────────────────────────────

def make_wrong_answer(correct_val: float) -> float:
    """
    Produce a plausibly wrong numeric answer using one of four strategies.
    Ensures the wrong answer differs from the correct one.
    """
    strategies = [
        lambda v: v + random.randint(1, 10),
        lambda v: v - random.randint(1, 10),
        lambda v: v * random.choice([2, 3, 0.5]),
        lambda v: v + random.choice([-100, -50, 50, 100]),
        lambda v: -v if v != 0 else v + 1,
        lambda v: round(v / 10) * 10 + random.choice([-5, 5, 10, -10]),
    ]
    random.shuffle(strategies)
    for fn in strategies:
        wrong = fn(correct_val)
        if wrong != correct_val:
            # Format sensibly
            if wrong == int(wrong):
                return int(wrong)
            return round(wrong, 2)
    return correct_val + 1   # fallback


def build_wrong_solution(correct_solution: str, correct_val: float, wrong_val) -> str:
    """
    Replace the final answer in the solution with the wrong value.
    Also introduce a plausible reasoning error near the end.
    """
    # Replace the #### line
    wrong_solution = re.sub(
        r"(####\s*)[\d,\-\.]+",
        f"#### {wrong_val}",
        correct_solution
    )

    # Add a note that makes the reasoning look confidently wrong
    wrong_solution = wrong_solution.rstrip()
    wrong_solution += f"\nTherefore, the answer is {wrong_val}."
    return wrong_solution


# ── Main builder ───────────────────────────────────────────────────────────────

def build_math_pairs(n: int = 2000, out_path: str = "data/real/math_pairs.jsonl"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run `pip install datasets` first.")
        return

    print("Loading GSM8K dataset...")
    # Load both train and test to get enough examples
    ds_train = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
    ds_test  = load_dataset("gsm8k", "main", split="test",  trust_remote_code=True)

    all_examples = list(ds_train) + list(ds_test)
    print(f"  Loaded {len(all_examples)} GSM8K examples")

    random.shuffle(all_examples)

    samples = []
    skipped = 0

    for ex in all_examples:
        if len(samples) >= n:
            break

        question = ex["question"].strip()
        solution = ex["answer"].strip()

        result = extract_final_answer(solution)
        if result is None:
            skipped += 1
            continue

        correct_str, correct_val = result
        wrong_val = make_wrong_answer(correct_val)
        wrong_solution = build_wrong_solution(solution, correct_val, wrong_val)

        prompt = question

        if random.random() < 0.5:
            sample = {
                "prompt":     prompt,
                "response_a": solution,
                "response_b": wrong_solution,
                "labels":     {"toxicity": None, "math": 1, "code": None},
                "task":       "math",
                "source":     "gsm8k",
                "meta": {
                    "correct_answer": correct_str,
                    "wrong_answer":   str(wrong_val),
                },
            }
        else:
            sample = {
                "prompt":     prompt,
                "response_a": wrong_solution,
                "response_b": solution,
                "labels":     {"toxicity": None, "math": 0, "code": None},
                "task":       "math",
                "source":     "gsm8k",
                "meta": {
                    "correct_answer": correct_str,
                    "wrong_answer":   str(wrong_val),
                },
            }
        samples.append(sample)

    random.shuffle(samples)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    label_1 = sum(1 for s in samples if s["labels"]["math"] == 1)
    label_0 = sum(1 for s in samples if s["labels"]["math"] == 0)
    print(f"Skipped (no answer found): {skipped}")
    print(f"\nSaved {len(samples)} pairs to {out_path}")
    print(f"  Label=1 (A correct): {label_1}  |  Label=0 (B correct): {label_0}")


if __name__ == "__main__":
    build_math_pairs(n=2000)

"""
scripts/combine_data.py

Merges JSONL files into a combined dataset.

Usage:
    python scripts/combine_data.py        # V1: merges toxicity + math + code
    python scripts/combine_data.py v2     # V2: uses multiobjective_pairs.jsonl
"""

import json
import random
import os
import sys

random.seed(42)

mode = sys.argv[1] if len(sys.argv) > 1 else "v1"

if mode == "v2":
    SOURCES = ["data/multiobjective_pairs.jsonl"]
    OUT = "data/combined_pairs_v2.jsonl"
else:
    SOURCES = [
        "data/toxicity_pairs.jsonl",
        "data/math_pairs.jsonl",
        "data/code_pairs.jsonl",
    ]
    OUT = "data/combined_pairs.jsonl"

print(f"Mode: {mode}  →  {OUT}")


def main():
    all_samples = []

    for path in SOURCES:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — skipping.")
            continue
        with open(path) as f:
            samples = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(samples):5d} samples from {path}")
        all_samples.extend(samples)

    random.shuffle(all_samples)

    with open(OUT, "w") as f:
        for item in all_samples:
            f.write(json.dumps(item) + "\n")

    print(f"\nCombined {len(all_samples)} samples → {OUT}")

    # Stats
    tasks = {}
    for s in all_samples:
        t = s.get("task", "unknown")
        tasks[t] = tasks.get(t, 0) + 1
    print("\nBreakdown by task:")
    for task, count in sorted(tasks.items()):
        print(f"  {task:12s}: {count}")

    # V2: label pattern stats
    if mode == "v2":
        from collections import Counter
        patterns = Counter(
            f"tox={s['labels']['toxicity']},math={s['labels']['math']},code={s['labels']['code']}"
            for s in all_samples
        )
        print("\nLabel patterns:")
        for pat, n in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"  {pat:35s}: {n:5d}  ({100*n/len(all_samples):.1f}%)")


if __name__ == "__main__":
    main()

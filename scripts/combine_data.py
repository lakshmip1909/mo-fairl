"""
scripts/combine_data.py

Merges toxicity, math, and code JSONL files into combined_pairs.jsonl.
Shuffles the combined dataset with a fixed seed.
"""

import json
import random
import os

random.seed(42)

SOURCES = [
    "data/toxicity_pairs.jsonl",
    "data/math_pairs.jsonl",
    "data/code_pairs.jsonl",
]

OUT = "data/combined_pairs.jsonl"


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


if __name__ == "__main__":
    main()

"""
scripts/real/combine_real_data.py

Merges real toxicity + math + code JSONL files into combined_pairs_v3.jsonl.
Also prints full dataset statistics.

Usage:
    python scripts/real/combine_real_data.py
"""

import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

SOURCES = [
    "data/real/toxicity_pairs.jsonl",
    "data/real/math_pairs.jsonl",
    "data/real/code_pairs.jsonl",
]
OUT = "data/real/combined_pairs_v3.jsonl"


def main():
    all_samples = []

    for path in SOURCES:
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: {path} not found — skipping. Run the build script first.")
            continue
        with open(p) as f:
            samples = [json.loads(line) for line in f if line.strip()]
        print(f"  Loaded {len(samples):5d} samples from {path}")
        all_samples.extend(samples)

    if not all_samples:
        print("No samples loaded. Run build_toxicity_pairs.py, build_math_pairs.py, build_code_pairs.py first.")
        return

    random.shuffle(all_samples)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    print(f"\nCombined {len(all_samples)} samples → {OUT}")

    # Stats
    tasks   = Counter(s.get("task", "unknown") for s in all_samples)
    sources = Counter(s.get("source", "unknown") for s in all_samples)

    print("\nBreakdown by task:")
    for t, n in sorted(tasks.items()):
        print(f"  {t:12s}: {n:5d}")

    print("\nBreakdown by source dataset:")
    for s, n in sorted(sources.items()):
        print(f"  {s:20s}: {n:5d}")

    # Label stats per task
    print("\nLabel distribution per task:")
    for task in tasks:
        task_samples = [s for s in all_samples if s.get("task") == task]
        for obj in ["toxicity", "math", "code"]:
            vals = [s["labels"][obj] for s in task_samples if s["labels"].get(obj) is not None]
            if vals:
                n1 = sum(1 for v in vals if v == 1)
                n0 = sum(1 for v in vals if v == 0)
                print(f"  {task:12s} | {obj:10s}: label=1: {n1:4d}  label=0: {n0:4d}")


if __name__ == "__main__":
    main()

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(rows)} rows to {path}")

def load_mbpp():
    attempts = [
        ("google-research-datasets/mbpp", "sanitized"),
        ("mbpp", "sanitized"),
        ("google-research-datasets/mbpp", None),
        ("mbpp", None),
    ]
    last_err = None
    for name, config in attempts:
        try:
            if config is None:
                ds = load_dataset(name)
            else:
                ds = load_dataset(name, config)
            print(f"Loaded dataset: {name}, config={config}")
            return ds
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load MBPP. Last error: {last_err}")

def get_problem(row):
    for key in ["text", "prompt", "question", "description"]:
        if key in row and row[key]:
            return str(row[key])
    raise KeyError(f"No problem text key found. Keys={list(row.keys())}")

def get_code(row):
    for key in ["code", "canonical_solution", "solution"]:
        if key in row and row[key]:
            return str(row[key])
    raise KeyError(f"No code key found. Keys={list(row.keys())}")

def get_tests(row):
    if "test_list" in row and row["test_list"]:
        return row["test_list"]
    if "tests" in row and row["tests"]:
        return row["tests"]
    if "test" in row and row["test"]:
        return row["test"]
    return []

def normalise_split(ds, split_name):
    rows = []
    for i, r in enumerate(ds[split_name]):
        rows.append({
            "source_id": r.get("task_id", f"{split_name}_{i}"),
            "problem": get_problem(r),
            "positive": get_code(r),
            "tests": get_tests(r),
            "split": split_name,
        })
    return rows

def choose_negative_indices(rows, strategy, k, seed):
    rng = random.Random(seed)
    n = len(rows)

    if strategy == "random":
        neg_idx = []
        for i in range(n):
            j = rng.randrange(n)
            while j == i:
                j = rng.randrange(n)
            neg_idx.append(j)
        return neg_idx

    if strategy == "nn":
        texts = [r["problem"] for r in rows]
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        X = vec.fit_transform(texts)
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, -1.0)
        k_eff = min(k, n - 1)
        ranked = np.argsort(-sim, axis=1)
        return [int(ranked[i, k_eff - 1]) for i in range(n)]

    raise ValueError(f"Unknown strategy: {strategy}")

def make_pairs(rows, split_name, strategy, k, seed):
    neg_idx = choose_negative_indices(rows, strategy, k, seed)
    out = []
    for i, r in enumerate(rows):
        j = neg_idx[i]
        neg = rows[j]
        out.append({
            "id": f"{split_name}_{i}",
            "source_id": r["source_id"],
            "negative_source_id": neg["source_id"],
            "split": split_name,
            "problem": r["problem"],
            "chosen": r["positive"],
            "rejected": neg["positive"],
            "tests": r["tests"],
            "objective": "code_correctness",
            "negative_strategy": strategy,
            "negative_k": k,
            "label": 1,
            "chosen_len": len(r["positive"].split()),
            "rejected_len": len(neg["positive"].split()),
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--strategy", choices=["random", "nn"], default="nn")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    ds = load_mbpp()
    print("Splits:", list(ds.keys()))

    train_split = "train" if "train" in ds else list(ds.keys())[0]
    test_split = "test" if "test" in ds else ("validation" if "validation" in ds else train_split)

    train_rows = normalise_split(ds, train_split)
    test_rows = normalise_split(ds, test_split)

    if test_split == train_split:
        random.shuffle(train_rows)
        cut = int(0.8 * len(train_rows))
        test_rows = train_rows[cut:]
        train_rows = train_rows[:cut]

    print("Train rows:", len(train_rows))
    print("Test rows:", len(test_rows))

    train_pairs = make_pairs(train_rows, "train", args.strategy, args.k, args.seed)
    test_pairs = make_pairs(test_rows, "test", args.strategy, args.k, args.seed + 1)

    out_dir = ROOT / "data" / "pairs" / args.run_name
    save_jsonl(train_pairs, out_dir / "train_pairs.jsonl")
    save_jsonl(test_pairs, out_dir / "test_pairs.jsonl")

if __name__ == "__main__":
    main()

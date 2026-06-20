import json
import random
import re
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_TRAIN = 5000
N_OOD = 1500

random.seed(SEED)

def extract_answer(ans):
    m = re.search(r"####\s*([-+]?\d[\d,\.]*)", ans)
    if m:
        return m.group(1).replace(",", "")
    return ""

def convert(example, idx, split):
    return {
        "id": f"{split}_{idx}",
        "split": split,
        "problem": example["question"],
        "positive": example["answer"],
        "negative": "",
        "gt_answer": extract_answer(example["answer"]),
        "pos_answer": extract_answer(example["answer"]),
        "neg_answer": "",
        "source": "gsm8k_base"
    }

def save_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("Saved", len(rows), path)

def main():
    ds = load_dataset("openai/gsm8k", "main")

    train = list(ds["train"])
    test = list(ds["test"])

    random.shuffle(train)
    random.shuffle(test)

    train = train[:min(N_TRAIN, len(train))]
    test = test[:min(N_OOD, len(test))]

    train_rows = [convert(x, i, "train") for i, x in enumerate(train)]
    test_rows = [convert(x, i, "ood") for i, x in enumerate(test)]

    save_jsonl(train_rows, OUT / "train_pairs.jsonl")
    save_jsonl(test_rows, OUT / "ood_pairs.jsonl")

if __name__ == "__main__":
    main()

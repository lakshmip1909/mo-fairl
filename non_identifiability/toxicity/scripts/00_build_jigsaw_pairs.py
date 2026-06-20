import json
import random
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PAIRS = 3500
TRAIN_FRAC = 0.8

random.seed(SEED)

def save_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("Saved", len(rows), path)

def get_text(row):
    for key in ["comment_text", "text", "comment"]:
        if key in row and row[key]:
            return str(row[key])
    return None

def get_toxic(row):
    for key in ["toxic", "toxicity"]:
        if key in row:
            return float(row[key])
    return 0.0

def main():
    ds = load_dataset("tasksource/jigsaw_toxicity", split="train")

    clean, toxic = [], []

    for r in ds:
        text = get_text(r)
        if not text:
            continue

        tox = get_toxic(r)

        if tox <= 0.0:
            clean.append((text, tox))
        elif tox >= 0.5:
            toxic.append((text, tox))

    random.shuffle(clean)
    random.shuffle(toxic)

    n = min(N_PAIRS, len(clean), len(toxic))
    clean = clean[:n]
    toxic = toxic[:n]

    rows = []
    for i, ((c, ctox), (t, ttox)) in enumerate(zip(clean, toxic)):
        rows.append({
            "id": f"tox_pair_{i}",
            "prompt": "",
            "chosen": c,
            "rejected": t,
            "chosen_toxicity": ctox,
            "rejected_toxicity": ttox,
            "objective": "toxicity",
            "label": 1
        })

    random.shuffle(rows)

    n_train = int(TRAIN_FRAC * len(rows))
    save_jsonl(rows[:n_train], OUT / "train_pairs.jsonl")
    save_jsonl(rows[n_train:], OUT / "test_pairs.jsonl")

if __name__ == "__main__":
    main()

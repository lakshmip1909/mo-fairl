import argparse, json, os, random
import numpy as np
from transformers import pipeline

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True)
    p.add_argument("--out_dir", default="data/toxicity/processed")
    p.add_argument("--n_pairs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean_quantile", type=float, default=0.35)
    p.add_argument("--toxic_quantile", type=float, default=0.65)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_json(args.input_json)
    print(f"Loaded generated outputs: {len(rows)}")

    tox_pipe = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        tokenizer="unitary/toxic-bert",
        top_k=None,
        truncation=True,
        device=0,
    )

    scored = []
    batch_size = 32

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        texts = [(r["prompt"] + " " + r["output"])[:2000] for r in batch]
        preds = tox_pipe(texts)

        for r, pred in zip(batch, preds):
            score = None
            for d in pred:
                if d["label"].lower() == "toxic":
                    score = float(d["score"])
                    break
            if score is None:
                score = max(float(d["score"]) for d in pred)

            scored.append({
                "prompt": r["prompt"],
                "output": r["output"],
                "toxicity": score,
            })

        print(f"Scored {min(i+batch_size, len(rows))}/{len(rows)}")

    scores = np.array([r["toxicity"] for r in scored])
    clean_thr = float(np.quantile(scores, args.clean_quantile))
    toxic_thr = float(np.quantile(scores, args.toxic_quantile))

    clean = [r for r in scored if r["toxicity"] <= clean_thr]
    toxic = [r for r in scored if r["toxicity"] >= toxic_thr]

    print(f"Clean threshold: {clean_thr:.4f}, clean count: {len(clean)}")
    print(f"Toxic threshold: {toxic_thr:.4f}, toxic count: {len(toxic)}")

    random.shuffle(clean)
    random.shuffle(toxic)

    n = min(args.n_pairs, len(clean), len(toxic))
    pairs = []

    for i in range(n):
        c = clean[i]
        t = toxic[i]

        pairs.append({
            "id": f"tox_pair_{i}",
            "prompt": c["prompt"],
            "chosen": c["output"],
            "rejected": t["output"],
            "chosen_toxicity": c["toxicity"],
            "rejected_toxicity": t["toxicity"],
            "objective": "toxicity",
            "label": 1
        })

    random.shuffle(pairs)
    split = int(0.8 * len(pairs))

    save_jsonl(pairs[:split], os.path.join(args.out_dir, "train_pairs.jsonl"))
    save_jsonl(pairs[split:], os.path.join(args.out_dir, "test_pairs.jsonl"))
    save_jsonl(scored, os.path.join(args.out_dir, "scored_outputs.jsonl"))

    print(f"Saved train pairs: {split}")
    print(f"Saved test pairs: {len(pairs)-split}")
    print(f"Saved scored outputs: {len(scored)}")

if __name__ == "__main__":
    main()

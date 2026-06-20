import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "hard_pairs"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "EleutherAI/pythia-410m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512
TOP_POOL = 300

Q_THR = 0.90
S_THR = 0.88
LEN_THR = 0.30

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

def save_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("Saved", len(rows), path)

def token_len(x):
    return len(str(x).split())

@torch.no_grad()
def encode(texts, tokenizer, model, batch_size=8, desc="Encoding"):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        toks = tokenizer(
            texts[i:i+batch_size],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)
        out = model(**toks)
        h = out.last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1)
        emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)

def build_split(rows, split):
    questions = [r["problem"] for r in rows]
    solutions = [r["positive"] for r in rows]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    Q = encode(questions, tokenizer, model, batch_size=16, desc=f"{split} questions")
    S = encode(solutions, tokenizer, model, batch_size=8, desc=f"{split} solutions")

    Q = torch.nn.functional.normalize(Q, dim=1)
    S = torch.nn.functional.normalize(S, dim=1)

    qsim = Q @ Q.T
    ssim = S @ S.T
    qsim.fill_diagonal_(-999)
    ssim.fill_diagonal_(-999)

    top = torch.topk(qsim, k=min(TOP_POOL, len(rows)-1), dim=1).indices

    out = []
    skipped = 0
    candidate_counts = []

    for i, r in enumerate(rows):
        valid = []

        for j in top[i].tolist():
            qv = float(qsim[i, j])
            sv = float(ssim[i, j])

            li = token_len(solutions[i])
            lj = token_len(solutions[j])
            rel = abs(li - lj) / max(1, li)

            if qv >= Q_THR and sv >= S_THR and rel <= LEN_THR:
                valid.append((j, qv, sv, rel))

        candidate_counts.append(len(valid))

        if not valid:
            skipped += 1
            continue

        # choose hardest available = highest combined question+solution similarity
        valid = sorted(valid, key=lambda x: x[1] + x[2] - x[3], reverse=True)
        j, qv, sv, rel = valid[0]
        neg = rows[j]

        out.append({
            "id": len(out),
            "original_id": r.get("id", i),
            "split": split,
            "problem": r["problem"],
            "positive": r["positive"],
            "negative": neg["positive"],
            "gt_answer": r.get("gt_answer"),
            "pos_answer": r.get("pos_answer", r.get("gt_answer")),
            "neg_answer": neg.get("pos_answer", neg.get("gt_answer")),
            "negative_from_id": neg.get("id", j),
            "source": "gsm8k_filtered_hard_negative",
            "question_similarity": qv,
            "solution_similarity": sv,
            "relative_length_difference": rel,
            "num_candidates": len(valid),
        })

    print(f"\n{split}: built {len(out)} pairs, skipped {skipped}")
    print("Candidate count mean:", np.mean(candidate_counts))
    print("Candidate count median:", np.median(candidate_counts))

    return out

def main():
    print("Device:", DEVICE)
    train_rows = load_jsonl(RAW / "train_pairs.jsonl")
    ood_rows = load_jsonl(RAW / "ood_pairs.jsonl")

    tr = build_split(train_rows, "train")
    od = build_split(ood_rows, "ood")

    save_jsonl(tr, OUT / "train_pairs_filtered.jsonl")
    save_jsonl(od, OUT / "ood_pairs_filtered.jsonl")

if __name__ == "__main__":
    main()

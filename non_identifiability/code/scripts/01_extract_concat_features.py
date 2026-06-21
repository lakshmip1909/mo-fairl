import argparse
import json
import pickle
from pathlib import Path

import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "EleutherAI/pythia-410m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

@torch.no_grad()
def encode(texts, tokenizer, model, batch_size=8, desc="Encoding"):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        toks = tokenizer(
            texts[i:i+batch_size],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)

        out = model(**toks, output_hidden_states=True)
        hs = out.hidden_states
        ids = [len(hs)//4, len(hs)//2, 3*len(hs)//4, len(hs)-1]
        mask = toks["attention_mask"].unsqueeze(-1)

        pooled = []
        for lid in ids:
            h = hs[lid]
            pooled.append((h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1))

        embs.append(torch.cat(pooled, dim=1).cpu())

    return torch.cat(embs, dim=0)

def concat(q, c):
    return torch.cat([q, c], dim=1)

def build(rows, tokenizer, model, name):
    qs = [r["problem"] for r in rows]
    pos = [r["chosen"] for r in rows]
    neg = [r["rejected"] for r in rows]

    q = encode(qs, tokenizer, model, batch_size=8, desc=f"{name} problems")
    p = encode(pos, tokenizer, model, batch_size=8, desc=f"{name} chosen code")
    n = encode(neg, tokenizer, model, batch_size=8, desc=f"{name} rejected code")

    return concat(q, p), concat(q, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    args = ap.parse_args()

    print("Device:", DEVICE)

    pair_dir = ROOT / "data" / "pairs" / args.run_name
    out_dir = ROOT / "data" / "features" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_jsonl(pair_dir / "train_pairs.jsonl")
    test = load_jsonl(pair_dir / "test_pairs.jsonl")

    print("Train pairs:", len(train))
    print("Test pairs:", len(test))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    tr_pos, tr_neg = build(train, tokenizer, model, "train")
    te_pos, te_neg = build(test, tokenizer, model, "test")

    scaler = StandardScaler()
    scaler.fit((tr_pos - tr_neg).numpy())

    tr_pos_s = torch.tensor(scaler.transform(tr_pos.numpy()), dtype=torch.float32)
    tr_neg_s = torch.tensor(scaler.transform(tr_neg.numpy()), dtype=torch.float32)
    te_pos_s = torch.tensor(scaler.transform(te_pos.numpy()), dtype=torch.float32)
    te_neg_s = torch.tensor(scaler.transform(te_neg.numpy()), dtype=torch.float32)

    torch.save({"phi_pos": tr_pos_s, "phi_neg": tr_neg_s, "feature_dim": tr_pos_s.shape[1]}, out_dir / "train_features.pt")
    torch.save({"phi_pos": te_pos_s, "phi_neg": te_neg_s, "feature_dim": te_pos_s.shape[1]}, out_dir / "test_features.pt")

    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Saved features to", out_dir)
    print("Feature dim:", tr_pos_s.shape[1])

if __name__ == "__main__":
    main()

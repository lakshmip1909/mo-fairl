import json, pickle
from pathlib import Path
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[1]
PAIR_DIR = ROOT / "data" / "hard_pairs"
OUT = ROOT / "data" / "processed_concat"
OUT.mkdir(parents=True, exist_ok=True)

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
            return_tensors="pt"
        ).to(DEVICE)

        out = model(**toks, output_hidden_states=True)
        hs = out.hidden_states
        ids = [len(hs)//4, len(hs)//2, 3*len(hs)//4, len(hs)-1]
        mask = toks["attention_mask"].unsqueeze(-1)

        pooled = []
        for lid in ids:
            h = hs[lid]
            p = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled.append(p)

        embs.append(torch.cat(pooled, dim=1).cpu())

    return torch.cat(embs, dim=0)

def concat(q, o):
    return torch.cat([q, o], dim=1)

def build(rows, tokenizer, model, name):
    qs = [r["problem"] for r in rows]
    ps = [r["positive"] for r in rows]
    ns = [r["negative"] for r in rows]

    q = encode(qs, tokenizer, model, batch_size=16, desc=f"{name} questions")
    p = encode(ps, tokenizer, model, batch_size=8, desc=f"{name} positives")
    n = encode(ns, tokenizer, model, batch_size=8, desc=f"{name} negatives")

    return concat(q, p), concat(q, n)

def main():
    print("Device:", DEVICE)

    train = load_jsonl(PAIR_DIR / "train_pairs_filtered.jsonl")
    ood = load_jsonl(PAIR_DIR / "ood_pairs_filtered.jsonl")

    print("Train pairs:", len(train))
    print("OOD pairs:", len(ood))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    tr_pos, tr_neg = build(train, tokenizer, model, "train")
    od_pos, od_neg = build(ood, tokenizer, model, "ood")

    scaler = StandardScaler()
    scaler.fit((tr_pos - tr_neg).numpy())

    tr_pos_s = torch.tensor(scaler.transform(tr_pos.numpy()), dtype=torch.float32)
    tr_neg_s = torch.tensor(scaler.transform(tr_neg.numpy()), dtype=torch.float32)
    od_pos_s = torch.tensor(scaler.transform(od_pos.numpy()), dtype=torch.float32)
    od_neg_s = torch.tensor(scaler.transform(od_neg.numpy()), dtype=torch.float32)

    torch.save({"phi_pos": tr_pos_s, "phi_neg": tr_neg_s, "feature_dim": tr_pos_s.shape[1]}, OUT / "train_features.pt")
    torch.save({"phi_pos": od_pos_s, "phi_neg": od_neg_s, "feature_dim": od_pos_s.shape[1]}, OUT / "ood_features.pt")

    with open(OUT / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Saved concat features:", OUT)
    print("Feature dim:", tr_pos_s.shape[1])

if __name__ == "__main__":
    main()

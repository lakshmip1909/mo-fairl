import json
import pickle
from pathlib import Path

import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "data" / "features"
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
        batch = texts[i:i+batch_size]
        toks = tokenizer(
            batch,
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

def get_pos_neg(rows):
    pos = [r["chosen"] for r in rows]
    neg = [r["rejected"] for r in rows]
    return pos, neg

def main():
    print("Device:", DEVICE)

    train = load_jsonl(DATA / "train_pairs.jsonl")
    test = load_jsonl(DATA / "test_pairs.jsonl")

    print("Train pairs:", len(train))
    print("Test pairs:", len(test))

    tr_pos_texts, tr_neg_texts = get_pos_neg(train)
    te_pos_texts, te_neg_texts = get_pos_neg(test)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    tr_pos = encode(tr_pos_texts, tokenizer, model, desc="Train chosen")
    tr_neg = encode(tr_neg_texts, tokenizer, model, desc="Train rejected")
    te_pos = encode(te_pos_texts, tokenizer, model, desc="Test chosen")
    te_neg = encode(te_neg_texts, tokenizer, model, desc="Test rejected")

    scaler = StandardScaler()
    scaler.fit((tr_pos - tr_neg).numpy())

    tr_pos_s = torch.tensor(scaler.transform(tr_pos.numpy()), dtype=torch.float32)
    tr_neg_s = torch.tensor(scaler.transform(tr_neg.numpy()), dtype=torch.float32)
    te_pos_s = torch.tensor(scaler.transform(te_pos.numpy()), dtype=torch.float32)
    te_neg_s = torch.tensor(scaler.transform(te_neg.numpy()), dtype=torch.float32)

    torch.save(
        {"phi_pos": tr_pos_s, "phi_neg": tr_neg_s, "feature_dim": tr_pos_s.shape[1]},
        OUT / "train_features.pt"
    )
    torch.save(
        {"phi_pos": te_pos_s, "phi_neg": te_neg_s, "feature_dim": te_pos_s.shape[1]},
        OUT / "test_features.pt"
    )

    with open(OUT / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Saved features to", OUT)
    print("Feature dim:", tr_pos_s.shape[1])

if __name__ == "__main__":
    main()

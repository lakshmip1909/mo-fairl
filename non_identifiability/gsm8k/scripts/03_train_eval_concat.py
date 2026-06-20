import random
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed_concat"
MODEL_DIR = ROOT / "models" / "reward_models_concat"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(5))
EPOCHS = 60
SAVE_EVERY = 5
BATCH_SIZE = 256
LR = 1e-4
WEIGHT_DECAY = 1e-4
MARGIN = 0.05

class LinearReward(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def acc_gap(model, pos, neg):
    with torch.no_grad():
        gap = model(pos) - model(neg)
        return float((gap > 0).float().mean()), float(gap.mean()), float(gap.std())

def main():
    train = torch.load(PROC / "train_features.pt", map_location="cpu")
    ood = torch.load(PROC / "ood_features.pt", map_location="cpu")

    tr_pos, tr_neg = train["phi_pos"], train["phi_neg"]
    od_pos, od_neg = ood["phi_pos"], ood["phi_neg"]
    dim = train["feature_dim"]

    rows = []
    weights = []

    for seed in SEEDS:
        set_seed(seed)
        model = LinearReward(dim)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        n = tr_pos.shape[0]

        for epoch in range(1, EPOCHS + 1):
            perm = torch.randperm(n)

            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                gap = model(tr_pos[idx]) - model(tr_neg[idx])
                loss = torch.relu(MARGIN - gap).mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            if epoch % SAVE_EVERY == 0:
                tr_acc, tr_gap, tr_std = acc_gap(model, tr_pos, tr_neg)
                od_acc, od_gap, od_std = acc_gap(model, od_pos, od_neg)
                w = model.linear.weight.detach().view(-1)

                fname = f"concat_seed{seed:02d}_epoch{epoch:03d}.pt"

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "feature_dim": dim,
                }, MODEL_DIR / fname)

                rows.append({
                    "filename": fname,
                    "seed": seed,
                    "epoch": epoch,
                    "train_acc": tr_acc,
                    "ood_acc": od_acc,
                    "tr_gap_mean": tr_gap,
                    "od_gap_mean": od_gap,
                    "tr_gap_std": tr_std,
                    "od_gap_std": od_std,
                    "weight_norm": float(torch.norm(w)),
                })
                weights.append(w.clone())

                print(fname, "train", round(tr_acc,4), "ood", round(od_acc,4), "gap", round(tr_gap,4))

    df = pd.DataFrame(rows)
    out_csv = RESULTS / "eval_concat_linear.csv"
    df.to_csv(out_csv, index=False)

    W = torch.stack(weights)
    Wn = torch.nn.functional.normalize(W, dim=1)
    cos = (Wn @ Wn.T).numpy()
    upper = cos[np.triu_indices_from(cos, k=1)]

    print("\nSUMMARY")
    print(df[["train_acc","ood_acc","tr_gap_mean","od_gap_mean","weight_norm"]].describe())
    print("Cos mean", upper.mean(), "std", upper.std(), "min", upper.min())

    plt.figure(figsize=(7,5))
    plt.scatter(df["train_acc"], df["ood_acc"])
    plt.axhline(0.5, linestyle="--")
    plt.axvline(0.5, linestyle="--")
    plt.xlabel("Train accuracy")
    plt.ylabel("Held-out accuracy")
    plt.title("Concat-only linear reward")
    plt.tight_layout()
    plt.savefig(PLOTS / "concat_train_vs_ood.png", dpi=200)

    plt.figure(figsize=(7,5))
    plt.hist(df["train_acc"], bins=20, alpha=0.6, label="Train")
    plt.hist(df["ood_acc"], bins=20, alpha=0.6, label="Held-out")
    plt.axvline(0.5, linestyle="--", label="Random")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.title("Accuracy distribution: concat-only")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "concat_accuracy_hist.png", dpi=200)

    plt.figure(figsize=(7,5))
    plt.hist(upper, bins=30, density=True)
    plt.axvline(upper.mean(), linestyle="--", label=f"mean={upper.mean():.3f}")
    plt.xlabel("Pairwise cosine similarity")
    plt.ylabel("Density")
    plt.title("Reward vector similarity: concat-only")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "concat_weight_similarity.png", dpi=200)

    print("Saved:", out_csv)

if __name__ == "__main__":
    main()

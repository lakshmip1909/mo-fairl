import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

CSV = ROOT / "results" / "eval_concat_linear.csv"
MODEL_DIR = ROOT / "models" / "reward_models_concat"
PLOTS = ROOT / "results" / "plots"

df = pd.read_csv(CSV)

df = df.sort_values("train_acc")

poor = df.head(10)
good = df.tail(10)
mid_start = len(df)//2 - 5
medium = df.iloc[mid_start:mid_start+10]

groups = {
    "poor": poor,
    "medium": medium,
    "good": good
}

def load_weight(fname):
    ckpt = torch.load(MODEL_DIR / fname, map_location="cpu")
    w = ckpt["model_state_dict"]["linear.weight"].view(-1)
    return w

all_stats = []

group_weights = {}

for name, g in groups.items():

    W = torch.stack([load_weight(f) for f in g.filename])

    group_weights[name] = W

    Wn = torch.nn.functional.normalize(W, dim=1)

    cos = (Wn @ Wn.T).numpy()

    upper = cos[np.triu_indices_from(cos, k=1)]

    all_stats.append({
        "group": name,
        "n_models": len(g),
        "train_acc_mean": g.train_acc.mean(),
        "train_acc_std": g.train_acc.std(),
        "ood_acc_mean": g.ood_acc.mean(),
        "ood_acc_std": g.ood_acc.std(),
        "cos_mean": upper.mean(),
        "cos_std": upper.std(),
        "cos_min": upper.min(),
        "cos_max": upper.max(),
        "weight_norm_mean": g.weight_norm.mean(),
        "gap_mean": g.tr_gap_mean.mean()
    })

    plt.figure(figsize=(6,5))
    plt.imshow(cos)
    plt.colorbar()
    plt.title(f"{name.capitalize()} models cosine similarity")
    plt.tight_layout()
    plt.savefig(PLOTS / f"concat_{name}_heatmap.png", dpi=200)
    plt.close()

between = []

for a in groups:
    for b in groups:

        if a >= b:
            continue

        Wa = torch.nn.functional.normalize(group_weights[a], dim=1)
        Wb = torch.nn.functional.normalize(group_weights[b], dim=1)

        cos = (Wa @ Wb.T).numpy()

        between.append({
            "group_a": a,
            "group_b": b,
            "mean_cos": cos.mean(),
            "std_cos": cos.std(),
            "min_cos": cos.min(),
            "max_cos": cos.max()
        })

stats_df = pd.DataFrame(all_stats)
between_df = pd.DataFrame(between)

stats_df.to_csv(ROOT / "results" / "concat_group_stats.csv", index=False)
between_df.to_csv(ROOT / "results" / "concat_between_group_stats.csv", index=False)

print("\n=== WITHIN GROUP ===")
print(stats_df)

print("\n=== BETWEEN GROUP ===")
print(between_df)

print("\nSaved:")
print(ROOT / "results" / "concat_group_stats.csv")
print(ROOT / "results" / "concat_between_group_stats.csv")

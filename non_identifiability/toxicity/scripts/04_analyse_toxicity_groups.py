import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "eval_toxicity_linear.csv"
MODEL_DIR = ROOT / "models"
RESULTS = ROOT / "results"
PLOTS = ROOT / "results" / "plots"

df = pd.read_csv(CSV).sort_values("train_acc").reset_index(drop=True)

poor = df.head(10)
good = df.tail(10)
mid_start = len(df)//2 - 5
medium = df.iloc[mid_start:mid_start+10]

groups = {"poor": poor, "medium": medium, "good": good}

def load_weight(fname):
    ckpt = torch.load(MODEL_DIR / fname, map_location="cpu")
    return ckpt["model_state_dict"]["linear.weight"].view(-1).float()

stats = []
group_weights = {}

for name, g in groups.items():
    W = torch.stack([load_weight(f) for f in g.filename])
    group_weights[name] = W

    Wn = torch.nn.functional.normalize(W, dim=1)
    cos = (Wn @ Wn.T).numpy()
    upper = cos[np.triu_indices_from(cos, k=1)]

    stats.append({
        "group": name,
        "n_models": len(g),
        "train_acc_mean": g.train_acc.mean(),
        "train_acc_std": g.train_acc.std(),
        "test_acc_mean": g.test_acc.mean(),
        "test_acc_std": g.test_acc.std(),
        "cos_mean": upper.mean(),
        "cos_std": upper.std(),
        "cos_min": upper.min(),
        "cos_max": upper.max(),
        "weight_norm_mean": g.weight_norm.mean(),
        "gap_mean": g.train_gap_mean.mean(),
    })

    plt.figure(figsize=(6,5))
    plt.imshow(cos, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"Toxicity {name} models cosine similarity")
    plt.tight_layout()
    plt.savefig(PLOTS / f"toxicity_{name}_heatmap.png", dpi=200)
    plt.close()

between = []
names = list(groups.keys())

for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = names[i], names[j]
        Wa = torch.nn.functional.normalize(group_weights[a], dim=1)
        Wb = torch.nn.functional.normalize(group_weights[b], dim=1)

        cos = (Wa @ Wb.T).numpy()
        between.append({
            "group_a": a,
            "group_b": b,
            "mean_cos": cos.mean(),
            "std_cos": cos.std(),
            "min_cos": cos.min(),
            "max_cos": cos.max(),
        })

stats_df = pd.DataFrame(stats)
between_df = pd.DataFrame(between)

stats_df.to_csv(RESULTS / "toxicity_group_stats.csv", index=False)
between_df.to_csv(RESULTS / "toxicity_between_group_stats.csv", index=False)

print("\nWITHIN GROUP")
print(stats_df)

print("\nBETWEEN GROUP")
print(between_df)

print("Saved group stats.")

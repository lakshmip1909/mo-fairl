import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    args = ap.parse_args()

    results_dir = ROOT / "results" / args.run_name
    model_dir = ROOT / "models" / args.run_name
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "eval_code_linear.csv").sort_values("train_acc").reset_index(drop=True)

    n_group = min(10, max(3, len(df)//3))
    poor = df.head(n_group)
    good = df.tail(n_group)
    mid_start = len(df)//2 - n_group//2
    medium = df.iloc[mid_start:mid_start+n_group]

    groups = {"poor": poor, "medium": medium, "good": good}

    def load_weight(fname):
        ckpt = torch.load(model_dir / fname, map_location="cpu")
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

        plt.figure(figsize=(6, 5))
        plt.imshow(cos, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"Code {name} models cosine similarity")
        plt.tight_layout()
        plt.savefig(plots_dir / f"code_{name}_heatmap.png", dpi=200)
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

    stats_df.to_csv(results_dir / "code_group_stats.csv", index=False)
    between_df.to_csv(results_dir / "code_between_group_stats.csv", index=False)

    print("\nWITHIN GROUP")
    print(stats_df)

    print("\nBETWEEN GROUP")
    print(between_df)

if __name__ == "__main__":
    main()

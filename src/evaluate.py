"""
src/evaluate.py

Per-objective evaluation:
  - Accuracy, F1, AUC-ROC
  - Reward gap distribution
  - Combined reward accuracy

Run on the test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.reward_model import MultiObjectiveRewardModel, OBJECTIVES, K


# ── Collection pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model:  MultiObjectiveRewardModel,
    loader: DataLoader,
    device: torch.device,
    weights: torch.Tensor,
) -> dict:
    """
    Runs model over loader and collects:
        deltas  : [N, K]  reward gaps (A - B)
        labels  : [N, K]  true labels (NaN where null)
        masks   : [N, K]  bool
        combined_delta: [N]  weighted combined reward gap
        tasks   : [N]    task name per sample
    """
    model.eval()
    all_deltas  = []
    all_labels  = []
    all_masks   = []
    all_tasks   = []

    for batch in loader:
        enc_a  = batch["enc_a"].to(device)
        enc_b  = batch["enc_b"].to(device)
        labels = batch["labels"]
        mask   = batch["mask"]

        delta = model.get_reward_gap(enc_a, enc_b).cpu()  # [batch, K]

        all_deltas.append(delta)
        all_labels.append(labels)
        all_masks.append(mask)
        all_tasks.extend(batch["task"])

    return {
        "deltas":  torch.cat(all_deltas,  dim=0).numpy(),  # [N, K]
        "labels":  torch.cat(all_labels,  dim=0).numpy(),  # [N, K]
        "masks":   torch.cat(all_masks,   dim=0).numpy(),  # [N, K] bool
        "tasks":   all_tasks,
        "weights": weights.numpy(),
    }


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(preds: dict) -> dict:
    """
    Compute per-objective and combined metrics.
    """
    deltas  = preds["deltas"]   # [N, K]
    labels  = preds["labels"]   # [N, K]
    masks   = preds["masks"]    # [N, K]
    weights = preds["weights"]  # [K]

    results = {}

    for k, obj in enumerate(OBJECTIVES):
        valid_idx = masks[:, k]
        if valid_idx.sum() == 0:
            print(f"  WARNING: no valid samples for objective {obj}")
            continue

        rho = labels[valid_idx, k].astype(int)
        y_gap  = deltas[valid_idx, k]
        # Convert rho {-1,+1} back to binary only for metrics
        y_true = (rho > 0).astype(int)
        y_prob = 1 / (1 + np.exp(-y_gap))   # P(A preferred)
        y_pred = (y_gap > 0).astype(int)

        # Correct iff rho * gap > 0
        acc = np.mean((rho * y_gap) > 0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        avg_gap    = float(np.mean(y_gap))
        avg_margin = float(np.mean(np.abs(y_gap)))

        results[obj] = {
            "n_samples": int(valid_idx.sum()),
            "accuracy":  round(acc, 4),
            "f1":        round(f1,  4),
            "auc":       round(auc, 4),
            "avg_reward_gap":    round(avg_gap,    4),
            "avg_reward_margin": round(avg_margin, 4),
        }
        print(
            f"  [{obj:10s}]  n={valid_idx.sum():5d}  "
            f"acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}  "
            f"avg_gap={avg_gap:+.3f}  margin={avg_margin:.3f}"
        )

    # Combined reward: R_w(A) - R_w(B) = Σₖ wₖ Δᵏ
    combined_delta = (deltas * weights[None, :]).sum(axis=1)  # [N]

    # Use samples where at least one objective has a label
    any_valid = masks.any(axis=1)
    # For combined, use majority-vote label (or mean label > 0.5)
    label_means = np.where(masks, labels, np.nan)
    # labels are rho {-1,+1}; combined preference is sign of average rho
    combined_label = np.nanmean(label_means, axis=1)
    combined_binary = (combined_label > 0).astype(int)

    valid_combined = any_valid & ~np.isnan(combined_label)
    y_true_c = combined_binary[valid_combined]
    y_pred_c = (combined_delta[valid_combined] > 0).astype(int)

    acc_c = accuracy_score(y_true_c, y_pred_c)
    try:
        auc_c = roc_auc_score(y_true_c, combined_delta[valid_combined])
    except ValueError:
        auc_c = float("nan")

    results["combined"] = {
        "n_samples": int(valid_combined.sum()),
        "accuracy":  round(acc_c, 4),
        "auc":       round(auc_c, 4),
    }
    print(f"  [combined  ]  n={valid_combined.sum():5d}  acc={acc_c:.3f}  auc={auc_c:.3f}")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_reward_gaps(preds: dict, plots_dir: str):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    deltas = preds["deltas"]  # [N, K]
    masks  = preds["masks"]

    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4))
    if K == 1:
        axes = [axes]

    for k, (obj, ax) in enumerate(zip(OBJECTIVES, axes)):
        valid = masks[:, k]
        if valid.sum() == 0:
            continue
        gaps = deltas[valid, k]
        ax.hist(gaps, bins=40, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", label="decision boundary")
        ax.set_title(f"{obj} reward gap (Δ = rₐ - r_b)")
        ax.set_xlabel("Reward gap")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    path = plots_dir / "reward_gap_distributions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_metrics_bar(metrics: dict, plots_dir: str):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    objs = [o for o in OBJECTIVES if o in metrics]
    accs = [metrics[o]["accuracy"] for o in objs]
    f1s  = [metrics[o]["f1"]       for o in objs]
    aucs = [metrics[o]["auc"]      for o in objs]

    x = np.arange(len(objs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, accs, width, label="Accuracy")
    ax.bar(x,         f1s,  width, label="F1")
    ax.bar(x + width, aucs, width, label="AUC-ROC")

    ax.set_xticks(x)
    ax.set_xticklabels(objs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Objective Metrics (Test Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "per_objective_metrics.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate(
    model:      MultiObjectiveRewardModel,
    test_dl:    DataLoader,
    config:     dict,
    device:     torch.device,
) -> dict:
    weights_cfg = config["reward_weights"]
    weights = torch.tensor(
        [weights_cfg[obj] for obj in OBJECTIVES],
        dtype=torch.float32,
    ).to(device)

    print("\n── Evaluation ──────────────────────────────────────────────")
    preds = collect_predictions(model, test_dl, device, weights.cpu())

    print("\n  Per-objective metrics:")
    metrics = compute_metrics(preds)

    plots_dir   = config["evaluation"]["plots_dir"]
    metrics_dir = config["evaluation"]["metrics_dir"]
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)

    plot_reward_gaps(preds, plots_dir)
    plot_metrics_bar(metrics, plots_dir)

    metrics_path = Path(metrics_dir) / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to: {metrics_path}")

    return metrics

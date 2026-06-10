"""
src/evaluate_conflict.py

Evaluation for the conflict-aware model.

Reports beyond standard accuracy/AUC:

1. Per-objective metrics (same as before)
2. Global preference accuracy: does sigma(w^T·Delta) get the overall winner right?
3. Learned weight w analysis: what did the model learn about objective importance?
4. Conflict-specific metrics:
   - Accuracy on conflict pairs only (where tox and math disagree)
   - Accuracy on non-conflict pairs
   - Whether the model's w explains the conflict resolution correctly
5. Weight trajectory: how did w evolve during training?
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.reward_model_pythia_conflict import PythiaConflictRewardModel, OBJECTIVES, K


# ── Data collection pass ──────────────────────────────────────────────────────

@torch.no_grad()
def collect_conflict_predictions(
    model:  PythiaConflictRewardModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()

    all_deltas      = []
    all_rho         = []
    all_masks       = []
    all_scores      = []   # w^T · Delta per sample
    all_tasks       = []
    all_is_conflict = []

    w = model.w.cpu()

    for batch in loader:
        ids_a = batch["input_ids_a"].to(device)
        msk_a = batch["attention_mask_a"].to(device)
        ids_b = batch["input_ids_b"].to(device)
        msk_b = batch["attention_mask_b"].to(device)

        delta = model.get_reward_gap(ids_a, msk_a, ids_b, msk_b).cpu()  # [B, K]
        score = (model.w.cpu() * delta).sum(dim=-1)                       # [B]

        all_deltas.append(delta)
        all_rho.append(batch["rho"])
        all_masks.append(batch["mask"])
        all_scores.append(score)
        all_tasks.extend(batch["task"])
        all_is_conflict.extend(batch.get("is_conflict", [None]*delta.shape[0]))

    return {
        "deltas":       torch.cat(all_deltas).numpy(),
        "rho":          torch.cat(all_rho).numpy(),
        "masks":        torch.cat(all_masks).numpy(),
        "scores":       torch.cat(all_scores).numpy(),
        "tasks":        all_tasks,
        "is_conflict":  all_is_conflict,
        "w":            w.numpy(),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_conflict_metrics(preds: dict, config: dict) -> dict:
    deltas      = preds["deltas"]       # [N, K]
    rho         = preds["rho"]          # [N, K]
    masks       = preds["masks"]        # [N, K]
    scores      = preds["scores"]       # [N]
    is_conflict = np.array([bool(c) if c is not None else False
                            for c in preds["is_conflict"]])
    w           = preds["w"]            # [K]

    results = {}

    # ── 1. Per-objective metrics ───────────────────────────────────────────────
    print("\n── Per-Objective Metrics ─────────────────────────────────────")
    results["per_objective"] = {}
    for k, obj in enumerate(OBJECTIVES):
        valid = masks[:, k]
        if valid.sum() == 0:
            continue
        rho_k  = rho[valid, k]
        gap    = deltas[valid, k]
        y_true = ((rho_k + 1) / 2).astype(int)
        y_pred = (gap > 0).astype(int)
        y_prob = 1 / (1 + np.exp(-gap))

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        results["per_objective"][obj] = {
            "n":        int(valid.sum()),
            "accuracy": round(acc, 4),
            "f1":       round(f1,  4),
            "auc":      round(auc, 4),
        }
        print(f"  [{obj:10s}]  n={valid.sum():4d}  acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")

    # ── 2. Global preference metrics ──────────────────────────────────────────
    print("\n── Global Preference Metrics (w^T · Delta) ───────────────────")
    # Majority vote overall label
    labels = rho

    rho_masked  = labels * masks
    valid_count = masks.sum(axis=1).clip(min=1)
    rho_mean    = rho_masked.sum(axis=1) / valid_count
    rho_major   = np.sign(rho_mean)

    # Load explicit global labels directly from the dataset file.
    # This avoids relying on dataloader metadata during evaluation.
    import json
    data_path = config.get("data_path") or config.get("combined_path") or config.get("conflict_path")
    explicit_vals = []
    if data_path is not None:
        rows = [json.loads(line) for line in open(data_path)]
        # Use the same deterministic split as dataset_conflict: last 10% is test.
        n = len(rows)
        test_rows = rows[int(0.9 * n):]
        for r in test_rows:
            v = r.get("global_label", None)
            explicit_vals.append(np.nan if v is None else (1.0 if int(v) == 1 else -1.0))

    if len(explicit_vals) == len(scores):
        explicit = np.array(explicit_vals)
    else:
        explicit = np.full(len(scores), np.nan)

    has_explicit = ~np.isnan(explicit)

    rho_global  = np.where(has_explicit, explicit, rho_major)
    global_mask = has_explicit | (rho_major != 0)

    if global_mask.sum() > 0:
        y_true_g = ((rho_global[global_mask] + 1) / 2).astype(int)
        y_pred_g = (scores[global_mask] > 0).astype(int)
        y_prob_g = 1 / (1 + np.exp(-scores[global_mask]))

        acc_g = accuracy_score(y_true_g, y_pred_g)
        try:
            auc_g = roc_auc_score(y_true_g, y_prob_g)
        except ValueError:
            auc_g = float("nan")

        results["global"] = {
            "n":        int(global_mask.sum()),
            "accuracy": round(acc_g, 4),
            "auc":      round(auc_g, 4),
        }
        print(f"  Global  n={global_mask.sum():4d}  acc={acc_g:.3f}  auc={auc_g:.3f}")
    else:
        results["global"] = {"n": 0, "accuracy": 0.0, "auc": 0.0}

    # ── 3. Conflict-specific metrics ──────────────────────────────────────────
    print("\n── Conflict vs Non-Conflict Accuracy ─────────────────────────")
    for split_name, split_mask in [("conflict", is_conflict), ("non_conflict", ~is_conflict)]:
        if split_mask.sum() == 0:
            print(f"  {split_name}: no samples")
            continue

        y_true_s = ((rho_global[split_mask & global_mask] + 1) / 2).astype(int)
        y_pred_s = (scores[split_mask & global_mask] > 0).astype(int)
        if len(y_true_s) == 0:
            continue
        acc_s = accuracy_score(y_true_s, y_pred_s)
        results[split_name] = {
            "n":        int((split_mask & global_mask).sum()),
            "accuracy": round(acc_s, 4),
        }
        print(f"  {split_name:14s}  n={(split_mask & global_mask).sum():4d}  acc={acc_s:.3f}")

    # ── 4. Learned weight analysis ────────────────────────────────────────────
    print("\n── Learned Objective Weights w ───────────────────────────────")
    results["learned_w"] = {}
    for k, obj in enumerate(OBJECTIVES):
        results["learned_w"][obj] = round(float(w[k]), 4)
        print(f"  w_{obj:10s} = {w[k]:.4f}")
    print(f"  (uniform baseline would be {1/K:.4f} per objective)")

    dominant = OBJECTIVES[int(np.argmax(w))]
    print(f"  → Dominant objective: {dominant}  (w={w.max():.4f})")
    results["learned_w"]["dominant_objective"] = dominant

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_weight_trajectory(history: list[dict], plots_dir: str):
    """Plot how w evolved over training epochs."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history if "w" in h]
    if not epochs:
        return

    w_traj = {obj: [] for obj in OBJECTIVES}
    for h in history:
        if "w" not in h:
            continue
        for obj in OBJECTIVES:
            w_traj[obj].append(h["w"].get(obj, 1/K))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: weight trajectory
    colors = ["tomato", "steelblue", "seagreen"]
    for obj, color in zip(OBJECTIVES, colors):
        axes[0].plot(epochs, w_traj[obj], label=obj, color=color, linewidth=2)
    axes[0].axhline(1/K, color="grey", linestyle="--", alpha=0.5, label="uniform baseline")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weight w_k")
    axes[0].set_title("Learned Objective Weights During Training")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: final weight bar chart
    final_w = [w_traj[obj][-1] for obj in OBJECTIVES]
    bars = axes[1].bar(OBJECTIVES, final_w, color=colors, edgecolor="black")
    axes[1].axhline(1/K, color="grey", linestyle="--", alpha=0.5, label="uniform")
    axes[1].set_ylabel("Final weight w_k")
    axes[1].set_title("Final Learned Weights")
    axes[1].set_ylim(0, max(final_w) * 1.3)
    for bar, val in zip(bars, final_w):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", fontsize=10)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "weight_trajectory.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_conflict_deltas(preds: dict, plots_dir: str):
    """
    For conflict pairs: scatter plot of Delta_tox vs Delta_math.
    Points coloured by what the global model predicted.
    Shows whether the model can handle the trade-off.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    deltas      = preds["deltas"]
    scores      = preds["scores"]
    is_conflict = np.array([bool(c) if c is not None else False
                            for c in preds["is_conflict"]])
    masks       = preds["masks"]

    # Only plot samples with both tox and math labels
    both_valid = masks[:, 0] & masks[:, 1]
    if both_valid.sum() == 0:
        print("  No samples with both tox+math labels — skipping conflict scatter.")
        return

    tox_gaps  = deltas[both_valid, 0]
    math_gaps = deltas[both_valid, 1]
    sc        = scores[both_valid]
    conflict  = is_conflict[both_valid]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: colour by conflict type
    c_colors = np.where(conflict, "tomato", "steelblue")
    axes[0].scatter(tox_gaps, math_gaps, c=c_colors, alpha=0.6, edgecolors="none", s=30)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("Δ_toxicity = r_tox(A) - r_tox(B)")
    axes[0].set_ylabel("Δ_math = r_math(A) - r_math(B)")
    axes[0].set_title("Reward Gaps: Conflict (red) vs Non-Conflict (blue)")
    # Conflict quadrants
    axes[0].fill_betweenx([-10, 0], 0, 10, alpha=0.05, color="green",
                           label="tox:A wins, math:B wins")
    axes[0].fill_betweenx([0, 10], -10, 0, alpha=0.05, color="orange",
                           label="tox:B wins, math:A wins")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(-5, 5); axes[0].set_ylim(-5, 5)

    # Right: colour by global model prediction
    pred_colors = np.where(sc > 0, "steelblue", "tomato")
    axes[1].scatter(tox_gaps, math_gaps, c=pred_colors, alpha=0.6, edgecolors="none", s=30)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Δ_toxicity")
    axes[1].set_ylabel("Δ_math")
    axes[1].set_title("Global Model: Predicts A (blue) vs B (red)")
    axes[1].set_xlim(-5, 5); axes[1].set_ylim(-5, 5)

    plt.tight_layout()
    path = plots_dir / "conflict_delta_scatter.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main evaluate function ────────────────────────────────────────────────────

def evaluate_conflict(
    model:    PythiaConflictRewardModel,
    test_dl:  DataLoader,
    config:   dict,
    device:   torch.device,
    history:  list[dict] | None = None,
) -> dict:
    print("\n" + "="*65)
    print("  Conflict Model Evaluation")
    print("="*65)

    preds   = collect_conflict_predictions(model, test_dl, device)
    metrics = compute_conflict_metrics(preds, config)

    plots_dir   = config["evaluation"]["plots_dir"]
    metrics_dir = config["evaluation"]["metrics_dir"]
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)

    plot_conflict_deltas(preds, plots_dir)
    if history:
        plot_weight_trajectory(history, plots_dir)

    out_path = Path(metrics_dir) / "conflict_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved: {out_path}")

    return metrics

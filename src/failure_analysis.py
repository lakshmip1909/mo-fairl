"""
src/failure_analysis.py

Failure-aware analysis of the trained reward model.

Categories:
    CLEAN     : all objectives correctly predicted
    SINGLE    : exactly one objective fails
    MULTI     : two or more objectives fail
    UNCERTAIN : model uncertain on at least one objective (|Δ| < ε)
    CONFLICT  : label vector has mixed 0/1 across objectives
                (e.g. [1, 0, 1] — response A is better on tox and code but
                worse on math)

For each failure type, we save example prompts + responses for inspection.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.reward_model import MultiObjectiveRewardModel, OBJECTIVES, K


# ── Categorisation ────────────────────────────────────────────────────────────

def categorise_sample(
    deltas:  np.ndarray,   # [K]  reward gap r_k(A) - r_k(B)
    rho:     np.ndarray,   # [K]  {-1.0, +1.0}  preference sign
    mask:    np.ndarray,   # [K]  bool
    epsilon: float = 0.1,
) -> dict:
    """
    Returns a dict describing this sample's failure status.
    """
    result = {
        "objectives": {},
        "is_conflict":  False,
        "n_failures":   0,
        "n_uncertain":  0,
        "category":     "CLEAN",
    }

    failures   = 0
    uncertain  = 0
    valid_labels = []

    for k, obj in enumerate(OBJECTIVES):
        if not mask[k]:
            result["objectives"][obj] = None  # irrelevant
            continue

        delta   = float(deltas[k])
        rho_k   = float(rho[k])                 # {-1.0, +1.0}
        # Correct when rho and delta share sign: rho * delta > 0
        correct = (rho_k * delta) > 0
        fail    = not correct
        unc     = abs(delta) < epsilon
        # Convert rho back to readable {0,1} for display
        label_display = 1 if rho_k > 0 else 0

        result["objectives"][obj] = {
            "delta":      round(delta, 4),
            "rho":        rho_k,
            "label":      label_display,         # 1=A preferred, 0=B preferred
            "correct":    correct,
            "failed":     fail,
            "uncertain":  unc,
        }

        if fail:
            failures += 1
        if unc:
            uncertain += 1
        valid_labels.append(label_display)

    result["n_failures"]  = failures
    result["n_uncertain"] = uncertain

    # Conflict: at least two valid labels present and they differ
    if len(valid_labels) >= 2 and len(set(valid_labels)) > 1:
        result["is_conflict"] = True

    # Category
    if failures == 0 and uncertain == 0:
        result["category"] = "CLEAN"
    elif failures == 0 and uncertain > 0:
        result["category"] = "UNCERTAIN"
    elif failures == 1:
        result["category"] = "SINGLE_FAILURE"
    else:
        result["category"] = "MULTI_FAILURE"

    if result["is_conflict"]:
        result["category"] = "CONFLICT_" + result["category"]

    return result


# ── Full analysis pass ────────────────────────────────────────────────────────

@torch.no_grad()
def run_failure_analysis(
    model:    MultiObjectiveRewardModel,
    loader:   DataLoader,
    config:   dict,
    device:   torch.device,
    raw_samples: list[dict],   # original JSONL samples (for display)
) -> dict:
    """
    Runs failure analysis over the full test set.
    Saves:
      - failure_summary.json  (counts per category)
      - failures_per_objective.json  (per-objective failure lists)
      - conflict_examples.json  (conflict samples)
      - failure_rate_plot.png
    """
    epsilon   = config["failure_analysis"]["epsilon"]
    out_dir   = Path(config["failure_analysis"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    all_categories  = []
    failures_by_obj = {obj: [] for obj in OBJECTIVES}
    conflict_samples = []

    sample_idx = 0

    for batch in loader:
        enc_a  = batch["enc_a"].to(device)
        enc_b  = batch["enc_b"].to(device)
        rho    = batch["rho"].numpy()    # {-1,+1}  [batch, K]
        masks  = batch["mask"].numpy()
        tasks  = batch["task"]

        deltas = model.get_reward_gap(enc_a, enc_b).cpu().numpy()  # [B, K]

        for b in range(deltas.shape[0]):
            cat = categorise_sample(deltas[b], rho[b], masks[b], epsilon)
            all_categories.append(cat["category"])

            # Get original sample text if available
            orig = raw_samples[sample_idx] if sample_idx < len(raw_samples) else {}
            info = {
                "sample_idx": sample_idx,
                "task":       tasks[b],
                "prompt":     orig.get("prompt", "N/A")[:200],
                "response_a": orig.get("response_a", "N/A")[:200],
                "response_b": orig.get("response_b", "N/A")[:200],
                "analysis":   cat,
                "rho_raw":    rho[b].tolist(),
            }

            # Per-objective failures
            for obj in OBJECTIVES:
                obj_info = cat["objectives"].get(obj)
                if obj_info and obj_info["failed"]:
                    failures_by_obj[obj].append(info)

            # Conflicts
            if cat["is_conflict"]:
                conflict_samples.append(info)

            sample_idx += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    counter = Counter(all_categories)
    total   = len(all_categories)

    print("\n── Failure Analysis ────────────────────────────────────────")
    print(f"  Total test samples: {total}")
    for cat, count in sorted(counter.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {cat:30s}: {count:5d}  ({pct:.1f}%)")

    # Per-objective failure rates
    print("\n  Per-objective failure rates:")
    obj_failure_rates = {}
    for obj in OBJECTIVES:
        n_valid = sum(
            1 for i, c in enumerate(all_categories)
            for _ in [None]
            if True
        )
        n_failed = len(failures_by_obj[obj])
        # Recompute properly
        obj_failure_rates[obj] = n_failed

    # Save outputs
    summary = {
        "total_samples": total,
        "category_counts": dict(counter),
        "category_rates":  {k: round(v/total, 4) for k, v in counter.items()},
        "per_objective_failure_counts": {
            obj: len(failures_by_obj[obj]) for obj in OBJECTIVES
        },
        "conflict_count": len(conflict_samples),
    }

    with open(out_dir / "failure_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    for obj in OBJECTIVES:
        out = failures_by_obj[obj][:50]  # save up to 50 examples
        with open(out_dir / f"failures_{obj}.json", "w") as f:
            json.dump(out, f, indent=2)

    with open(out_dir / "conflict_examples.json", "w") as f:
        json.dump(conflict_samples[:50], f, indent=2)

    print(f"\n  Conflicts found: {len(conflict_samples)}")
    print(f"  Outputs saved to: {out_dir}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_failure_breakdown(counter, total, out_dir)
    _plot_per_objective_failure(failures_by_obj, out_dir)

    return summary


def _plot_failure_breakdown(counter: Counter, total: int, out_dir: Path):
    labels = list(counter.keys())
    sizes  = [counter[l] for l in labels]
    pcts   = [100 * s / total for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, pcts, color="steelblue", edgecolor="black")
    ax.set_xlabel("% of test samples")
    ax.set_title("Failure Category Breakdown")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center")
    plt.tight_layout()
    path = out_dir / "failure_breakdown.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_per_objective_failure(failures_by_obj: dict, out_dir: Path):
    objs   = OBJECTIVES
    counts = [len(failures_by_obj[o]) for o in objs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(objs, counts, color=["tomato", "steelblue", "seagreen"], edgecolor="black")
    ax.set_ylabel("Number of failures")
    ax.set_title("Per-Objective Failure Counts")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha="center")
    plt.tight_layout()
    path = out_dir / "per_objective_failures.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

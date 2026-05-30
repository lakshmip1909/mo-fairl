"""
src/utils.py

Shared utilities: checkpointing, logging, config loading, metrics helpers.
"""

from __future__ import annotations

import json
import yaml
from pathlib import Path

import torch
import torch.nn as nn


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    metrics:   dict,
    path:      Path,
):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
    }, path)


def load_checkpoint(
    model:     nn.Module,
    path:      str,
    optimizer: torch.optim.Optimizer | None = None,
    device:    torch.device | None = None,
) -> dict:
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[Checkpoint] Loaded epoch {ckpt['epoch']} from {path}")
    return ckpt


# ── AverageMeter ──────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks running average of a scalar."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(cfg_device: str = "cuda") -> torch.device:
    if cfg_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


# ── History saving ────────────────────────────────────────────────────────────

def save_history(history: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[History] Saved training history to {path}")


def plot_training_history(history: list[dict], plots_dir: str):
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs      = [h["epoch"] for h in history]
    train_loss  = [h["train_loss"] for h in history]
    val_loss    = [h["val_loss"]   for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss,   label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Per-objective val accuracy
    from src.reward_model import OBJECTIVES
    for obj in OBJECTIVES:
        key = f"val_acc_{obj}"
        if key in history[0]:
            accs = [h[key] for h in history]
            axes[1].plot(epochs, accs, label=obj)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Per-Objective Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "training_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved training history plot to {path}")

"""
src/train.py

Training loop for MO-FAIRL.

Loss: masked BCE per objective.
For each sample i and objective k:
    if mask[i,k] is True:
        loss += BCE(sigmoid(Δᵢᵏ), pᵢᵏ)

Total loss = Σₖ Lₖ  (only over non-null labels)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from src.reward_model import MultiObjectiveRewardModel, OBJECTIVES, K
from src.utils import AverageMeter, save_checkpoint, load_checkpoint


# ── Loss ─────────────────────────────────────────────────────────────────────

def masked_bce_loss(
    delta:  torch.Tensor,  # [batch, K]  reward gap A - B
    labels: torch.Tensor,  # [batch, K]  0/1 (NaN where null)
    mask:   torch.Tensor,  # [batch, K]  bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        total_loss:  scalar
        per_obj_loss: [K]
    """
    prob = torch.sigmoid(delta)  # [batch, K]

    # BCE element-wise (no reduction)
    bce = F.binary_cross_entropy(prob, labels.clamp(0, 1), reduction="none")  # [batch, K]

    # Zero out where mask is False (null labels)
    bce = bce * mask.float()

    # Per-objective mean (only over valid samples)
    valid_counts = mask.float().sum(dim=0).clamp(min=1)  # [K]
    per_obj_loss = bce.sum(dim=0) / valid_counts          # [K]

    total_loss = per_obj_loss.sum()
    return total_loss, per_obj_loss


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(
    model:     MultiObjectiveRewardModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    grad_clip: float,
) -> dict:
    model.train()
    meters = {
        "loss":   AverageMeter(),
        **{f"loss_{obj}": AverageMeter() for obj in OBJECTIVES},
    }

    for batch in loader:
        enc_a  = batch["enc_a"].to(device)
        enc_b  = batch["enc_b"].to(device)
        labels = batch["labels"].to(device)
        mask   = batch["mask"].to(device)

        delta = model.get_reward_gap(enc_a, enc_b)  # [batch, K]

        loss, per_obj = masked_bce_loss(delta, labels, mask)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = enc_a.size(0)
        meters["loss"].update(loss.item(), bs)
        for i, obj in enumerate(OBJECTIVES):
            meters[f"loss_{obj}"].update(per_obj[i].item(), bs)

    return {k: v.avg for k, v in meters.items()}


# ── Validation step ───────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(
    model:  MultiObjectiveRewardModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    meters = {
        "loss":   AverageMeter(),
        **{f"loss_{obj}": AverageMeter() for obj in OBJECTIVES},
        **{f"acc_{obj}":  AverageMeter() for obj in OBJECTIVES},
    }

    for batch in loader:
        enc_a  = batch["enc_a"].to(device)
        enc_b  = batch["enc_b"].to(device)
        labels = batch["labels"].to(device)
        mask   = batch["mask"].to(device)

        delta = model.get_reward_gap(enc_a, enc_b)  # [batch, K]
        loss, per_obj = masked_bce_loss(delta, labels, mask)

        # Accuracy: correct when sign(delta) == label
        preds = (delta > 0).float()   # [batch, K]
        correct = (preds == labels.clamp(0, 1)).float() * mask.float()  # [batch, K]
        valid   = mask.float()

        bs = enc_a.size(0)
        meters["loss"].update(loss.item(), bs)
        for i, obj in enumerate(OBJECTIVES):
            meters[f"loss_{obj}"].update(per_obj[i].item(), bs)
            n_valid = valid[:, i].sum().item()
            if n_valid > 0:
                meters[f"acc_{obj}"].update(
                    correct[:, i].sum().item() / n_valid,
                    int(n_valid),
                )

    return {k: v.avg for k, v in meters.items()}


# ── Full training loop ────────────────────────────────────────────────────────

def train(
    model:      MultiObjectiveRewardModel,
    train_dl:   DataLoader,
    val_dl:     DataLoader,
    config:     dict,
    device:     torch.device,
):
    cfg    = config["training"]
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(
        model.parameters(),
        lr           = cfg["learning_rate"],
        weight_decay = cfg["weight_decay"],
    )

    # Warmup then cosine decay
    n_steps  = len(train_dl) * cfg["num_epochs"]
    warmup   = cfg["warmup_steps"]
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=n_steps - warmup, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup])

    best_val_loss = float("inf")
    history = []

    print(f"\n{'='*60}")
    print(f"  Training for {cfg['num_epochs']} epochs")
    print(f"  Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg["num_epochs"] + 1):
        train_metrics = train_epoch(model, train_dl, optimizer, device, cfg["grad_clip"])
        val_metrics   = val_epoch(model, val_dl, device)
        scheduler.step()

        # Print
        print(
            f"Epoch {epoch:3d}/{cfg['num_epochs']}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            + "  ".join(
                f"acc_{obj}={val_metrics[f'acc_{obj}']:.3f}"
                for obj in OBJECTIVES
            )
        )

        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        # Save checkpoint
        if epoch % cfg["save_every"] == 0 or epoch == cfg["num_epochs"]:
            save_checkpoint(model, optimizer, epoch, record, ckpt_dir / f"epoch_{epoch:03d}.pt")

        # Best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, record, ckpt_dir / "best_model.pt")
            print(f"  ✓ New best val_loss: {best_val_loss:.4f}")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return history

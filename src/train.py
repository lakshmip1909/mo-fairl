"""
src/train.py

Training loop for MO-FAIRL.

Loss (from notes, Naive formulation):
    L^k(r_hat_k) = -(1/N) * sum_i  log sigma( rho_i^k * delta_i^k )

where:
    delta_i^k  = r_k(A) - r_k(B)          reward gap
    rho_i^k   in {-1, +1}                 preference sign (from notes Ass 2)
    mask_i^k   = True if label not null    (V1 has nulls; V2 has none)

Total weighted loss (from notes):
    L_w = sum_k  w^k * L^k

w is taken from config["reward_weights"] — fixed for baseline.
This means w weights the GRADIENTS, not just the final combined reward.
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

def masked_margin_loss(
    delta:   torch.Tensor,  # [batch, K]  reward gap: r_k(A) - r_k(B)
    rho:     torch.Tensor,  # [batch, K]  preference sign: {-1.0, +1.0}
    mask:    torch.Tensor,  # [batch, K]  bool — False where label is null
    weights: torch.Tensor,  # [K]         objective weights w^k from config
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implements the loss from the notes exactly:

        L^k = -(1/N_k) * sum_i  log sigma( rho_i^k * delta_i^k )

    Total weighted loss:

        L_w = sum_k  w^k * L^k

    Args:
        delta   : [batch, K]  r_k(A) - r_k(B)
        rho     : [batch, K]  {-1, +1}  (rho_i^k from Ass 2)
        mask    : [batch, K]  True where this objective applies
        weights : [K]         w^k  (fixed for baseline)

    Returns:
        total_loss   : scalar  — L_w = sum_k w^k * L^k
        per_obj_loss : [K]    — unweighted L^k per objective (for logging)
    """
    # -log sigma(rho * delta)  element-wise — numerically stable via softplus:
    #   -log sigma(x) = log(1 + exp(-x)) = softplus(-x)
    margin = rho * delta                                        # [batch, K]
    loss_elem = F.softplus(-margin)                            # [batch, K]

    # Zero out null objectives (V1 has nulls; V2 has none, but keep general)
    loss_elem = loss_elem * mask.float()                       # [batch, K]

    # Per-objective mean over valid samples only
    valid_counts = mask.float().sum(dim=0).clamp(min=1)        # [K]
    per_obj_loss = loss_elem.sum(dim=0) / valid_counts         # [K]  unweighted L^k

    # Weighted total: L_w = sum_k w^k * L^k
    total_loss = (weights * per_obj_loss).sum()

    return total_loss, per_obj_loss


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(
    model:     MultiObjectiveRewardModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    grad_clip: float,
    weights:   torch.Tensor,        # [K] objective weights w^k
) -> dict:
    model.train()
    meters = {
        "loss":   AverageMeter(),
        **{f"loss_{obj}": AverageMeter() for obj in OBJECTIVES},
    }

    for batch in loader:
        enc_a   = batch["enc_a"].to(device)
        enc_b   = batch["enc_b"].to(device)
        rho     = batch["rho"].to(device)    # {-1, +1}  [batch, K]
        mask    = batch["mask"].to(device)

        delta = model.get_reward_gap(enc_a, enc_b)  # [batch, K]

        loss, per_obj = masked_margin_loss(delta, rho, mask, weights)

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
    model:   MultiObjectiveRewardModel,
    loader:  DataLoader,
    device:  torch.device,
    weights: torch.Tensor,          # [K] objective weights w^k
) -> dict:
    model.eval()
    meters = {
        "loss":   AverageMeter(),
        **{f"loss_{obj}": AverageMeter() for obj in OBJECTIVES},
        **{f"acc_{obj}":  AverageMeter() for obj in OBJECTIVES},
    }

    for batch in loader:
        enc_a   = batch["enc_a"].to(device)
        enc_b   = batch["enc_b"].to(device)
        rho     = batch["rho"].to(device)    # {-1, +1}  [batch, K]
        mask    = batch["mask"].to(device)

        delta = model.get_reward_gap(enc_a, enc_b)  # [batch, K]
        loss, per_obj = masked_margin_loss(delta, rho, mask, weights)

        # Accuracy: correct when rho and delta have the same sign
        # i.e. rho * delta > 0  <=>  the model ranks the preferred response higher
        correct = ((rho * delta) > 0).float() * mask.float()  # [batch, K]
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

    # Build objective weight vector w from config (fixed for baseline)
    weights_cfg = config["reward_weights"]
    from src.reward_model import OBJECTIVES as _OBJS
    weights = torch.tensor(
        [weights_cfg[o] for o in _OBJS], dtype=torch.float32, device=device
    )
    print(f"  Objective weights: { {o: round(weights_cfg[o],3) for o in _OBJS} }")

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
        train_metrics = train_epoch(model, train_dl, optimizer, device, cfg["grad_clip"], weights)
        val_metrics   = val_epoch(model, val_dl, device, weights)
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

"""
src/train_pythia.py

Training loop for the Pythia-410M multi-objective reward model.

Same loss as train.py:
    L^k = -(1/N_k) * sum_i  log sigma( rho_i^k * delta_i^k )
    L_w  = sum_k  w^k * L^k

Key differences from train.py:
    - Batch contains (input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
      instead of pre-computed encodings
    - Model call: model.get_reward_gap(ids_a, mask_a, ids_b, mask_b)
    - Gradient accumulation supported (large model, small GPU memory)
    - Mixed precision (fp16) supported
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.reward_model_pythia import PythiaRewardModel, OBJECTIVES, K
from src.utils import AverageMeter, save_checkpoint


# ── Loss (identical formula to train.py) ──────────────────────────────────────

def masked_margin_loss(
    delta:   torch.Tensor,   # [batch, K]
    rho:     torch.Tensor,   # [batch, K]  {-1, +1}
    mask:    torch.Tensor,   # [batch, K]  bool
    weights: torch.Tensor,   # [K]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    L^k  = -(1/N_k) * sum_i  log sigma(rho_i^k * delta_i^k)
         = (1/N_k) * sum_i  softplus(-rho_i^k * delta_i^k)
    L_w  = sum_k  w^k * L^k
    """
    margin       = rho * delta
    loss_elem    = F.softplus(-margin) * mask.float()
    valid_counts = mask.float().sum(dim=0).clamp(min=1)
    per_obj_loss = loss_elem.sum(dim=0) / valid_counts
    total_loss   = (weights * per_obj_loss).sum()
    return total_loss, per_obj_loss


# ── Train epoch ───────────────────────────────────────────────────────────────

def train_epoch(
    model:       PythiaRewardModel,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    grad_clip:   float,
    weights:     torch.Tensor,
    accum_steps: int   = 1,
    scaler:      GradScaler | None = None,
) -> dict:
    model.train()
    meters = {
        "loss": AverageMeter(),
        **{f"loss_{obj}": AverageMeter() for obj in OBJECTIVES},
    }

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        ids_a  = batch["input_ids_a"].to(device)
        msk_a  = batch["attention_mask_a"].to(device)
        ids_b  = batch["input_ids_b"].to(device)
        msk_b  = batch["attention_mask_b"].to(device)
        rho    = batch["rho"].to(device)
        mask   = batch["mask"].to(device)

        with autocast(enabled=(scaler is not None)):
            delta = model.get_reward_gap(ids_a, msk_a, ids_b, msk_b)   # [B, K]
            loss, per_obj = masked_margin_loss(delta, rho, mask, weights)
            loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if grad_clip > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        bs = ids_a.size(0)
        meters["loss"].update((loss * accum_steps).item(), bs)
        for i, obj in enumerate(OBJECTIVES):
            meters[f"loss_{obj}"].update(per_obj[i].item(), bs)

    return {k: v.avg for k, v in meters.items()}


# ── Val epoch ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(
    model:   PythiaRewardModel,
    loader:  DataLoader,
    device:  torch.device,
    weights: torch.Tensor,
) -> dict:
    model.eval()
    meters = {
        "loss": AverageMeter(),
        **{f"loss_{obj}": AverageMeter() for obj in OBJECTIVES},
        **{f"acc_{obj}":  AverageMeter() for obj in OBJECTIVES},
    }

    for batch in loader:
        ids_a  = batch["input_ids_a"].to(device)
        msk_a  = batch["attention_mask_a"].to(device)
        ids_b  = batch["input_ids_b"].to(device)
        msk_b  = batch["attention_mask_b"].to(device)
        rho    = batch["rho"].to(device)
        mask   = batch["mask"].to(device)

        delta = model.get_reward_gap(ids_a, msk_a, ids_b, msk_b)
        loss, per_obj = masked_margin_loss(delta, rho, mask, weights)

        # Accuracy: correct when rho * delta > 0
        correct = ((rho * delta) > 0).float() * mask.float()
        valid   = mask.float()

        bs = ids_a.size(0)
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


# ── Full loop ─────────────────────────────────────────────────────────────────

def train_pythia(
    model:    PythiaRewardModel,
    train_dl: DataLoader,
    val_dl:   DataLoader,
    config:   dict,
    device:   torch.device,
):
    cfg      = config["training"]
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    weights_cfg = config["reward_weights"]
    weights = torch.tensor(
        [weights_cfg[o] for o in OBJECTIVES], dtype=torch.float32, device=device
    )
    print(f"  Objective weights: { {o: round(weights_cfg[o], 3) for o in OBJECTIVES} }")

    # Only train projection + heads (encoder is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr           = cfg["learning_rate"],
        weight_decay = cfg["weight_decay"],
    )

    n_steps  = len(train_dl) * cfg["num_epochs"]
    warmup   = cfg["warmup_steps"]
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=max(1, n_steps - warmup), eta_min=1e-6),
        ],
        milestones=[warmup],
    )

    # fp16 scaler if GPU available
    use_amp = device.type == "cuda"
    scaler  = GradScaler() if use_amp else None
    accum   = cfg.get("grad_accum_steps", 1)

    best_val_loss = float("inf")
    history = []

    print(f"\n{'='*60}")
    print(f"  Pythia-410M Training  |  {cfg['num_epochs']} epochs")
    print(f"  Mixed precision: {use_amp}  |  Grad accum: {accum}")
    print(f"  Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg["num_epochs"] + 1):
        train_m = train_epoch(model, train_dl, optimizer, device,
                              cfg["grad_clip"], weights, accum, scaler)
        val_m   = val_epoch(model, val_dl, device, weights)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{cfg['num_epochs']}  "
            f"train={train_m['loss']:.4f}  val={val_m['loss']:.4f}  "
            + "  ".join(f"acc_{o}={val_m[f'acc_{o}']:.3f}" for o in OBJECTIVES)
        )

        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}":   v for k, v in val_m.items()},
        }
        history.append(record)

        if epoch % cfg["save_every"] == 0 or epoch == cfg["num_epochs"]:
            save_checkpoint(model, optimizer, epoch, record,
                            ckpt_dir / f"epoch_{epoch:03d}.pt")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            save_checkpoint(model, optimizer, epoch, record,
                            ckpt_dir / "best_model.pt")
            print(f"  ✓ New best val_loss: {best_val_loss:.4f}")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return history

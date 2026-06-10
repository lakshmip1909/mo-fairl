"""
src/train_conflict.py

Training loop for the conflict-aware multi-objective reward model.

Loss has two components:

1. Per-objective margin loss (same as before):
   L^k = (1/N_k) * sum_i  softplus(-rho_i^k * Delta_i^k)
   L_per_obj = sum_k  L^k

2. Global preference loss (NEW — implements the notes):
   For each sample with an overall preference label rho_global:
   L_global = (1/N) * sum_i  softplus(-rho_global_i * score_i)
   where score_i = w^T · Delta_i  (w is learned)

Total loss:
   L_total = L_per_obj + lambda_global * L_global

rho_global is derived from the majority vote across available objectives:
   rho_global = sign(mean(rho_k for valid k))
   i.e. +1 if more objectives prefer A, -1 if more prefer B

This gives the model two signals:
   - Each head learns its own objective directly (per-obj loss)
   - w learns to combine the heads to match overall preferences (global loss)

The learned w reveals the latent weighting of objectives in the data.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.reward_model_pythia_conflict import PythiaConflictRewardModel, OBJECTIVES, K
from src.utils import AverageMeter, save_checkpoint


# ── Loss functions ─────────────────────────────────────────────────────────────

def per_objective_loss(
    delta: torch.Tensor,   # [batch, K]
    rho:   torch.Tensor,   # [batch, K]  {-1, +1}
    mask:  torch.Tensor,   # [batch, K]  bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    L^k = (1/N_k) * sum_i  softplus(-rho_i^k * Delta_i^k)
    Returns: (sum of L^k over all k, per_obj_losses [K])
    """
    margin       = rho * delta                                   # [B, K]
    loss_elem    = F.softplus(-margin) * mask.float()            # [B, K]
    valid_counts = mask.float().sum(dim=0).clamp(min=1)          # [K]
    per_obj      = loss_elem.sum(dim=0) / valid_counts           # [K]
    return per_obj.sum(), per_obj


def failure_aware_losses(
    delta: torch.Tensor,   # [B, K]
    rho:   torch.Tensor,   # [B, K] {-1,+1}
    mask:  torch.Tensor,   # [B, K] bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Failure-aware extension.

    margin_i^k = rho_i^k * Delta_i^k

    Failure set:
        F_k = { i : margin_i^k < 0 }

    Per-sample objective loss:
        ell_i^k = softplus(-margin_i^k)

    Failure loss:
        L_fail = sum_k mean_{i in F_k} ell_i^k

    DRO-style worst-objective loss:
        L_dro = max_k mean_{i in F_k} ell_i^k

    Returns:
        L_fail,
        L_dro,
        fail_rates [K],
        fail_obj_losses [K]
    """
    valid = mask.float()
    margin = rho * delta
    loss_elem = F.softplus(-margin) * valid

    fail_mask = ((margin < 0) & mask).float()
    fail_counts = fail_mask.sum(dim=0)

    # Avoid divide-by-zero. If no failures for objective k, its failure loss is zero.
    fail_obj_losses = torch.where(
        fail_counts > 0,
        (loss_elem * fail_mask).sum(dim=0) / fail_counts.clamp(min=1),
        torch.zeros_like(fail_counts),
    )

    fail_rates = fail_counts / valid.sum(dim=0).clamp(min=1)

    l_fail = fail_obj_losses.sum()
    l_dro = fail_obj_losses.max()

    return l_fail, l_dro, fail_rates.detach(), fail_obj_losses.detach()



def get_explicit_or_majority_global(batch, rho, mask, device):
    """
    Prefer explicit global_label from dataset if present.
    global_label uses {1,0,None}, converted to rho_global {+1,-1}.
    If missing, fallback to majority vote.
    """
    vals = batch.get("global_label", None)
    if vals is None:
        return compute_rho_global(rho, mask)

    rho_list = []
    mask_list = []
    for v in vals:
        if v is None:
            rho_list.append(0.0)
            mask_list.append(False)
        else:
            rho_list.append(1.0 if int(v) == 1 else -1.0)
            mask_list.append(True)

    rho_global = torch.tensor(rho_list, dtype=torch.float32, device=device)
    global_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)

    # fallback to majority for samples without explicit label
    if (~global_mask).any():
        rho_m, mask_m = compute_rho_global(rho, mask)
        rho_global = torch.where(global_mask, rho_global, rho_m)
        global_mask = global_mask | mask_m

    return rho_global, global_mask


def global_preference_loss(
    score:       torch.Tensor,   # [batch]  w^T · Delta
    rho_global:  torch.Tensor,   # [batch]  {-1, +1}  overall label
    global_mask: torch.Tensor,   # [batch]  bool — False if no clear majority
) -> torch.Tensor:
    """
    L_global = (1/N) * sum_i  softplus(-rho_global_i * score_i)

    Only computed over samples where a clear majority vote exists.
    """
    if global_mask.sum() == 0:
        return torch.tensor(0.0, device=score.device)

    margin    = rho_global * score                              # [batch]
    loss_elem = F.softplus(-margin) * global_mask.float()      # [batch]
    return loss_elem.sum() / global_mask.float().sum().clamp(min=1)


def compute_rho_global(
    rho:  torch.Tensor,   # [batch, K]  {-1, +1}
    mask: torch.Tensor,   # [batch, K]  bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Derive overall preference label by majority vote over valid objectives.

    rho_global_i = sign(mean(rho_i^k  for valid k))

    If mean > 0:  more objectives prefer A  →  rho_global = +1
    If mean < 0:  more objectives prefer B  →  rho_global = -1
    If mean = 0:  tie  →  masked out (global_mask = False)

    Returns:
        rho_global  : [batch]  {-1.0, +1.0}
        global_mask : [batch]  bool  (False on ties)
    """
    # Mean of rho over valid objectives
    rho_masked = rho * mask.float()                            # [batch, K]  0 where invalid
    valid_count = mask.float().sum(dim=1).clamp(min=1)         # [batch]
    rho_mean   = rho_masked.sum(dim=1) / valid_count           # [batch]

    rho_global  = torch.sign(rho_mean)                         # [batch]  {-1, 0, +1}
    global_mask = (rho_global != 0)                            # [batch]  bool — exclude ties

    return rho_global, global_mask


# ── Train epoch ───────────────────────────────────────────────────────────────

def train_epoch(
    model:         PythiaConflictRewardModel,
    loader:        DataLoader,
    optimizer:     torch.optim.Optimizer,
    device:        torch.device,
    grad_clip:     float,
    lambda_global: float  = 1.0,
    lambda_fail:   float  = 0.0,
    lambda_dro:    float  = 0.0,
    accum_steps:   int    = 1,
    scaler:        GradScaler | None = None,
) -> dict:
    model.train()
    meters = {
        "loss":        AverageMeter(),
        "loss_per":    AverageMeter(),
        "loss_global": AverageMeter(),
        "loss_fail":   AverageMeter(),
        "loss_dro":    AverageMeter(),
        **{f"loss_{o}": AverageMeter() for o in OBJECTIVES},
        **{f"fail_rate_{o}": AverageMeter() for o in OBJECTIVES},
    }

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        ids_a = batch["input_ids_a"].to(device)
        msk_a = batch["attention_mask_a"].to(device)
        ids_b = batch["input_ids_b"].to(device)
        msk_b = batch["attention_mask_b"].to(device)
        rho   = batch["rho"].to(device)     # [B, K]  {-1,+1}
        mask  = batch["mask"].to(device)    # [B, K]  bool

        with autocast(enabled=(scaler is not None)):
            # Per-objective reward gaps
            delta = model.get_reward_gap(ids_a, msk_a, ids_b, msk_b)  # [B, K]

            # Per-objective loss
            l_per, per_obj = per_objective_loss(delta, rho, mask)

            # Global score: w^T · Delta  (uses learned w_phi)
            w     = model.w                              # [K]  softmax(w_phi)
            score = (w * delta).sum(dim=-1)              # [B]

            # Majority vote overall label
            rho_global, global_mask = get_explicit_or_majority_global(batch, rho, mask, device)

            # Global loss
            l_global = global_preference_loss(score, rho_global, global_mask)

            # Failure-aware + DRO losses
            l_fail, l_dro, fail_rates, _ = failure_aware_losses(delta, rho, mask)

            loss = (
                l_per
                + lambda_global * l_global
                + lambda_fail * l_fail
                + lambda_dro * l_dro
            ) / accum_steps

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
        meters["loss_per"].update(l_per.item(), bs)
        meters["loss_global"].update(l_global.item(), bs)
        meters["loss_fail"].update(l_fail.item(), bs)
        meters["loss_dro"].update(l_dro.item(), bs)
        for i, o in enumerate(OBJECTIVES):
            meters[f"loss_{o}"].update(per_obj[i].item(), bs)
            meters[f"fail_rate_{o}"].update(fail_rates[i].item(), bs)

    return {k: v.avg for k, v in meters.items()}


# ── Val epoch ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(
    model:         PythiaConflictRewardModel,
    loader:        DataLoader,
    device:        torch.device,
    lambda_global: float = 1.0,
    lambda_fail:   float = 0.0,
    lambda_dro:    float = 0.0,
) -> dict:
    model.eval()
    meters = {
        "loss":        AverageMeter(),
        "loss_per":    AverageMeter(),
        "loss_global": AverageMeter(),
        "loss_fail":   AverageMeter(),
        "loss_dro":    AverageMeter(),
        **{f"loss_{o}": AverageMeter() for o in OBJECTIVES},
        **{f"fail_rate_{o}": AverageMeter() for o in OBJECTIVES},
        **{f"acc_{o}":  AverageMeter() for o in OBJECTIVES},
        "acc_global":  AverageMeter(),
    }

    for batch in loader:
        ids_a = batch["input_ids_a"].to(device)
        msk_a = batch["attention_mask_a"].to(device)
        ids_b = batch["input_ids_b"].to(device)
        msk_b = batch["attention_mask_b"].to(device)
        rho   = batch["rho"].to(device)
        mask  = batch["mask"].to(device)

        delta = model.get_reward_gap(ids_a, msk_a, ids_b, msk_b)

        l_per, per_obj = per_objective_loss(delta, rho, mask)

        w     = model.w
        score = (w * delta).sum(dim=-1)
        rho_global, global_mask = get_explicit_or_majority_global(batch, rho, mask, device)
        l_global = global_preference_loss(score, rho_global, global_mask)
        l_fail, l_dro, fail_rates, _ = failure_aware_losses(delta, rho, mask)

        loss = (
            l_per
            + lambda_global * l_global
            + lambda_fail * l_fail
            + lambda_dro * l_dro
        )

        # Per-objective accuracy
        correct = ((rho * delta) > 0).float() * mask.float()
        valid   = mask.float()

        # Global accuracy: sign(score) == rho_global
        global_correct = ((score * rho_global) > 0).float() * global_mask.float()

        bs = ids_a.size(0)
        meters["loss"].update(loss.item(), bs)
        meters["loss_per"].update(l_per.item(), bs)
        meters["loss_global"].update(l_global.item(), bs)
        meters["loss_fail"].update(l_fail.item(), bs)
        meters["loss_dro"].update(l_dro.item(), bs)

        for i, o in enumerate(OBJECTIVES):
            meters[f"loss_{o}"].update(per_obj[i].item(), bs)
            n_valid = valid[:, i].sum().item()
            if n_valid > 0:
                meters[f"acc_{o}"].update(correct[:, i].sum().item() / n_valid, int(n_valid))
                meters[f"fail_rate_{o}"].update(fail_rates[i].item(), int(n_valid))

        n_global = global_mask.float().sum().item()
        if n_global > 0:
            meters["acc_global"].update(global_correct.sum().item() / n_global, int(n_global))

    return {k: v.avg for k, v in meters.items()}


# ── Full training loop ────────────────────────────────────────────────────────

def train_conflict(
    model:    PythiaConflictRewardModel,
    train_dl: DataLoader,
    val_dl:   DataLoader,
    config:   dict,
    device:   torch.device,
) -> list[dict]:
    cfg      = config["training"]
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    lambda_global = config.get("lambda_global", 1.0)
    lambda_fail   = config.get("lambda_fail", 0.0)
    lambda_dro    = config.get("lambda_dro", 0.0)

    # Separate LRs: reward heads get standard LR, w_phi gets smaller LR
    # so w doesn't move too fast before the heads are trained
    head_params   = [p for n, p in model.named_parameters()
                     if p.requires_grad and "w_phi" not in n]
    w_phi_params  = [model.w_phi]

    optimizer = AdamW([
        {"params": head_params,  "lr": cfg["learning_rate"],        "weight_decay": cfg["weight_decay"]},
        {"params": w_phi_params, "lr": cfg["learning_rate"] * 0.1,  "weight_decay": 0.0},
    ])

    n_steps = len(train_dl) * cfg["num_epochs"]
    warmup  = cfg["warmup_steps"]
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=max(1, n_steps - warmup), eta_min=1e-6),
        ],
        milestones=[warmup],
    )

    use_amp = device.type == "cuda"
    scaler  = GradScaler() if use_amp else None
    accum   = cfg.get("grad_accum_steps", 1)

    best_val_loss = float("inf")
    history       = []

    print(f"\n{'='*65}")
    print(f"  Conflict Model Training  |  {cfg['num_epochs']} epochs")
    print(f"  lambda_global={lambda_global}  lambda_fail={lambda_fail}  lambda_dro={lambda_dro}")
    print(f"  accum={accum}  |  amp={use_amp}")
    print(f"  Initial w = {model.get_weights()}")
    print(f"{'='*65}\n")

    for epoch in range(1, cfg["num_epochs"] + 1):
        train_m = train_epoch(
            model, train_dl, optimizer, device,
            cfg["grad_clip"], lambda_global, lambda_fail, lambda_dro,
            accum, scaler
        )
        val_m = val_epoch(
            model, val_dl, device,
            lambda_global, lambda_fail, lambda_dro
        )
        scheduler.step()

        current_w = model.get_weights()

        print(
            f"Epoch {epoch:3d}/{cfg['num_epochs']}  "
            f"loss={val_m['loss']:.4f}  "
            f"(per={val_m['loss_per']:.3f} glob={val_m['loss_global']:.3f} "
            f"fail={val_m['loss_fail']:.3f} dro={val_m['loss_dro']:.3f})  "
            + "  ".join(f"acc_{o[:3]}={val_m[f'acc_{o}']:.3f}" for o in OBJECTIVES)
            + f"  acc_global={val_m['acc_global']:.3f}"
        )
        print(f"         w = {current_w}")

        record = {
            "epoch": epoch,
            "w":     current_w,
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
            print(f"  ✓ Best val_loss={best_val_loss:.4f}  w={current_w}")

    print(f"\nTraining complete.")
    print(f"Final learned weights: {model.get_weights()}")
    return history

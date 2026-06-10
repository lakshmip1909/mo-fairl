"""
main_conflict.py

MO-FAIRL Conflict Pipeline:
    1. Build real conflict pairs from GSM8K + Detoxify
    2. Train Pythia-410M with learnable w_phi + global preference loss
    3. Evaluate: per-objective, global, conflict-specific, weight trajectory
    4. Failure analysis

Usage:
    python main_conflict.py --config configs/conflict.yaml --mode all
    python main_conflict.py --config configs/conflict.yaml --mode generate
    python main_conflict.py --config configs/conflict.yaml --mode train
    python main_conflict.py --config configs/conflict.yaml --mode evaluate
    python main_conflict.py --config configs/conflict.yaml --mode analyze
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from src.utils                     import load_config, get_device, save_history, plot_training_history
from src.reward_model_pythia_conflict import build_conflict_model
from src.dataset_conflict          import build_conflict_dataloaders
from src.train_conflict            import train_conflict
from src.evaluate_conflict         import evaluate_conflict
from src.failure_analysis          import run_failure_analysis, categorise_sample


def parse_args():
    p = argparse.ArgumentParser(description="MO-FAIRL Conflict Pipeline")
    p.add_argument("--config",     default="configs/conflict.yaml")
    p.add_argument("--mode",       default="all",
                   choices=["generate", "train", "evaluate", "analyze", "all"])
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def step_generate():
    print("\n" + "="*60)
    print("  STEP 1: Building real conflict pairs")
    print("="*60)
    result = subprocess.run(
        [sys.executable, "scripts/real/build_conflict_pairs.py"]
    )
    if result.returncode != 0:
        print("ERROR: build_conflict_pairs.py failed.")
        sys.exit(1)
    print("  Done.")


def step_train(config: dict, device: torch.device) -> list[dict]:
    print("\n" + "="*60)
    print("  STEP 2: Training conflict model")
    print("="*60)

    train_dl, val_dl, test_dl = build_conflict_dataloaders(
        jsonl_path     = config["data"]["combined_path"],
        tokenizer_name = config["model"]["encoder"],
        batch_size     = config["training"]["batch_size"],
        train_split    = config["data"]["train_split"],
        val_split      = config["data"]["val_split"],
        seed           = config["data"]["seed"],
    )

    model = build_conflict_model(config).to(device)

    history = train_conflict(model, train_dl, val_dl, config, device)

    save_history(history, "outputs/metrics_conflict/training_history.json")
    plot_training_history(history, config["evaluation"]["plots_dir"])

    return history


def step_evaluate(config: dict, device: torch.device,
                  checkpoint: str | None = None, history: list[dict] | None = None):
    print("\n" + "="*60)
    print("  STEP 3: Evaluating conflict model")
    print("="*60)

    _, _, test_dl = build_conflict_dataloaders(
        jsonl_path     = config["data"]["combined_path"],
        tokenizer_name = config["model"]["encoder"],
        batch_size     = config["training"]["batch_size"],
        train_split    = config["data"]["train_split"],
        val_split      = config["data"]["val_split"],
        seed           = config["data"]["seed"],
    )

    model    = build_conflict_model(config).to(device)
    ckpt_path = checkpoint or Path(config["training"]["checkpoint_dir"]) / "best_model.pt"
    ckpt     = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded: {ckpt_path}")
    print(f"  Current w: {model.get_weights()}")

    # Load history for weight trajectory plot
    if history is None:
        hist_path = Path("outputs/metrics_conflict/training_history.json")
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)

    evaluate_conflict(model, test_dl, config, device, history)


def step_analyze(config: dict, device: torch.device, checkpoint: str | None = None):
    print("\n" + "="*60)
    print("  STEP 4: Failure analysis")
    print("="*60)

    raw_samples = []
    with open(config["data"]["combined_path"]) as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    _, _, test_dl = build_conflict_dataloaders(
        jsonl_path     = config["data"]["combined_path"],
        tokenizer_name = config["model"]["encoder"],
        batch_size     = config["training"]["batch_size"],
        train_split    = config["data"]["train_split"],
        val_split      = config["data"]["val_split"],
        seed           = config["data"]["seed"],
    )

    model    = build_conflict_model(config).to(device)
    ckpt_path = checkpoint or Path(config["training"]["checkpoint_dir"]) / "best_model.pt"
    ckpt     = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    n       = len(raw_samples)
    n_train = int(n * config["data"]["train_split"])
    n_val   = int(n * config["data"]["val_split"])
    test_raw = raw_samples[n_train + n_val:]

    out_dir = Path(config["failure_analysis"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    epsilon = config["failure_analysis"]["epsilon"]

    from src.reward_model_pythia_conflict import OBJECTIVES
    from collections import Counter
    import numpy as np

    all_categories = []
    conflicts      = []
    model.eval()
    sample_idx = 0

    with torch.no_grad():
        for batch in test_dl:
            ids_a = batch["input_ids_a"].to(device)
            msk_a = batch["attention_mask_a"].to(device)
            ids_b = batch["input_ids_b"].to(device)
            msk_b = batch["attention_mask_b"].to(device)
            rho   = batch["rho"].numpy()
            masks = batch["mask"].numpy()
            tasks = batch["task"]
            is_conflict_batch = batch["is_conflict"]

            deltas = model.get_reward_gap(ids_a, msk_a, ids_b, msk_b).cpu().numpy()
            w      = model.w.cpu().numpy()
            scores = (w * deltas).sum(axis=1)

            for b in range(deltas.shape[0]):
                cat  = categorise_sample(deltas[b], rho[b], masks[b], epsilon)
                all_categories.append(cat["category"])
                orig = test_raw[sample_idx] if sample_idx < len(test_raw) else {}
                info = {
                    "sample_idx":    sample_idx,
                    "task":          tasks[b],
                    "is_conflict":   bool(is_conflict_batch[b]),
                    "global_score":  round(float(scores[b]), 4),
                    "prompt":        orig.get("prompt","")[:200],
                    "conflict_type": orig.get("conflict_type",""),
                    "verified":      orig.get("verified", {}),
                    "analysis":      cat,
                }
                if cat["is_conflict"]:
                    conflicts.append(info)
                sample_idx += 1

    counter = Counter(all_categories)
    total   = len(all_categories)
    print(f"\n  Total test samples: {total}")
    for cat, count in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"  {cat:35s}: {count:5d}  ({100*count/total:.1f}%)")
    print(f"\n  Model-detected conflicts: {len(conflicts)}")
    print(f"  Final learned w: {model.get_weights()}")

    summary = {
        "total":      total,
        "categories": dict(counter),
        "conflicts":  len(conflicts),
        "learned_w":  model.get_weights(),
    }
    with open(out_dir / "failure_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "conflict_examples.json", "w") as f:
        json.dump(conflicts[:50], f, indent=2)
    print(f"  Saved to: {out_dir}")


def main():
    args   = parse_args()
    config = load_config(args.config)
    device = get_device(config["training"]["device"])
    mode   = args.mode

    history = None
    if mode in ("generate", "all"):
        step_generate()
    if mode in ("train", "all"):
        history = step_train(config, device)
    if mode in ("evaluate", "all"):
        step_evaluate(config, device, args.checkpoint, history)
    if mode in ("analyze", "all"):
        step_analyze(config, device, args.checkpoint)

    print("\n✓ Conflict pipeline complete.")


if __name__ == "__main__":
    main()

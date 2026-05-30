"""
main.py

MO-FAIRL: Multi-Objective Failure-Aware Inverse Reward Learning
Full pipeline entry point.

Modes:
    generate   — generate all preference pair datasets
    train      — train multi-objective reward model
    evaluate   — evaluate on test set
    analyze    — run failure analysis
    all        — run generate → train → evaluate → analyze

Usage:
    python main.py --config configs/default.yaml --mode all
    python main.py --config configs/default.yaml --mode train
    python main.py --config configs/default.yaml --mode evaluate --checkpoint outputs/checkpoints/best_model.pt
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

import torch

from src.utils        import load_config, get_device, save_history, plot_training_history
from src.reward_model import build_model
from src.dataset      import build_dataloaders
from src.train        import train
from src.evaluate     import evaluate
from src.failure_analysis import run_failure_analysis


def parse_args():
    p = argparse.ArgumentParser(description="MO-FAIRL pipeline")
    p.add_argument("--config",     type=str, default="configs/default.yaml")
    p.add_argument("--mode",       type=str, default="all",
                   choices=["generate", "train", "evaluate", "analyze", "all"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint (for evaluate/analyze mode)")
    return p.parse_args()


# ── Step 1: Generate data ──────────────────────────────────────────────────────

def step_generate():
    print("\n" + "="*60)
    print("  STEP 1: Generating preference pair datasets")
    print("="*60)

    scripts = [
        "scripts/generate_toxicity.py",
        "scripts/generate_math.py",
        "scripts/generate_code.py",
        "scripts/combine_data.py",
    ]
    for script in scripts:
        print(f"\n  Running: python {script}")
        result = subprocess.run([sys.executable, script], capture_output=False)
        if result.returncode != 0:
            print(f"  ERROR: {script} failed.")
            sys.exit(1)

    print("\n  Data generation complete.")


# ── Step 2: Train ──────────────────────────────────────────────────────────────

def step_train(config: dict, device: torch.device) -> list[dict]:
    print("\n" + "="*60)
    print("  STEP 2: Training multi-objective reward model")
    print("="*60)

    train_dl, val_dl, test_dl = build_dataloaders(
        jsonl_path   = config["data"]["combined_path"],
        encoder_name = config["model"]["encoder"],
        batch_size   = config["training"]["batch_size"],
        train_split  = config["data"]["train_split"],
        val_split    = config["data"]["val_split"],
        seed         = config["data"]["seed"],
    )

    model = build_model(config).to(device)
    print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train(model, train_dl, val_dl, config, device)

    save_history(history, "outputs/metrics/training_history.json")
    plot_training_history(history, config["evaluation"]["plots_dir"])

    return history


# ── Step 3: Evaluate ───────────────────────────────────────────────────────────

def step_evaluate(config: dict, device: torch.device, checkpoint: str | None = None):
    print("\n" + "="*60)
    print("  STEP 3: Evaluating on test set")
    print("="*60)

    _, _, test_dl = build_dataloaders(
        jsonl_path   = config["data"]["combined_path"],
        encoder_name = config["model"]["encoder"],
        batch_size   = config["training"]["batch_size"],
        train_split  = config["data"]["train_split"],
        val_split    = config["data"]["val_split"],
        seed         = config["data"]["seed"],
    )

    model = build_model(config).to(device)

    ckpt_path = checkpoint or Path(config["training"]["checkpoint_dir"]) / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded checkpoint: {ckpt_path}")

    metrics = evaluate(model, test_dl, config, device)
    return metrics


# ── Step 4: Failure analysis ───────────────────────────────────────────────────

def step_analyze(config: dict, device: torch.device, checkpoint: str | None = None):
    print("\n" + "="*60)
    print("  STEP 4: Failure analysis")
    print("="*60)

    import json
    raw_samples = []
    combined_path = config["data"]["combined_path"]
    if Path(combined_path).exists():
        with open(combined_path) as f:
            raw_samples = [json.loads(line) for line in f if line.strip()]

    _, _, test_dl = build_dataloaders(
        jsonl_path   = combined_path,
        encoder_name = config["model"]["encoder"],
        batch_size   = config["training"]["batch_size"],
        train_split  = config["data"]["train_split"],
        val_split    = config["data"]["val_split"],
        seed         = config["data"]["seed"],
    )

    model = build_model(config).to(device)

    ckpt_path = checkpoint or Path(config["training"]["checkpoint_dir"]) / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded checkpoint: {ckpt_path}")

    # Use the test split indices (approximate — use last n_test samples of raw)
    n = len(raw_samples)
    n_train = int(n * config["data"]["train_split"])
    n_val   = int(n * config["data"]["val_split"])
    test_raw = raw_samples[n_train + n_val:]

    summary = run_failure_analysis(model, test_dl, config, device, test_raw)
    return summary


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)
    device = get_device(config["training"]["device"])

    mode = args.mode

    if mode in ("generate", "all"):
        step_generate()

    if mode in ("train", "all"):
        step_train(config, device)

    if mode in ("evaluate", "all"):
        step_evaluate(config, device, args.checkpoint)

    if mode in ("analyze", "all"):
        step_analyze(config, device, args.checkpoint)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()

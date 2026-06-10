"""
main_pythia.py

MO-FAIRL Version 3: Pythia-410M + real datasets pipeline.

Usage:
    python main_pythia.py --config configs/v3_pythia.yaml --mode all
    python main_pythia.py --config configs/v3_pythia.yaml --mode generate
    python main_pythia.py --config configs/v3_pythia.yaml --mode train
    python main_pythia.py --config configs/v3_pythia.yaml --mode evaluate
    python main_pythia.py --config configs/v3_pythia.yaml --mode analyze
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

import torch

from src.utils              import load_config, get_device, save_history, plot_training_history
from src.reward_model_pythia import build_pythia_model
from src.dataset_pythia      import build_pythia_dataloaders
from src.train_pythia        import train_pythia


def parse_args():
    p = argparse.ArgumentParser(description="MO-FAIRL V3: Pythia pipeline")
    p.add_argument("--config",     default="configs/v3_pythia.yaml")
    p.add_argument("--mode",       default="all",
                   choices=["generate", "train", "evaluate", "analyze", "all"])
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def step_generate():
    print("\n" + "="*60)
    print("  STEP 1: Building real preference datasets")
    print("="*60)

    scripts = [
        "scripts/real/build_toxicity_pairs.py",
        "scripts/real/build_math_pairs.py",
        "scripts/real/build_code_pairs.py",
        "scripts/real/combine_real_data.py",
    ]
    for script in scripts:
        print(f"\n  Running: python {script}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"  ERROR: {script} failed.")
            sys.exit(1)
    print("\n  Data generation complete.")


def step_train(config: dict, device: torch.device):
    print("\n" + "="*60)
    print("  STEP 2: Training Pythia-410M reward model")
    print("="*60)

    train_dl, val_dl, test_dl = build_pythia_dataloaders(
        jsonl_path     = config["data"]["combined_path"],
        tokenizer_name = config["model"]["encoder"],
        batch_size     = config["training"]["batch_size"],
        train_split    = config["data"]["train_split"],
        val_split      = config["data"]["val_split"],
        seed           = config["data"]["seed"],
    )

    model = build_pythia_model(config).to(device)

    history = train_pythia(model, train_dl, val_dl, config, device)

    save_history(history, "outputs/metrics_v3/training_history.json")
    plot_training_history(history, config["evaluation"]["plots_dir"])
    return history


def step_evaluate(config: dict, device: torch.device, checkpoint: str | None = None):
    print("\n" + "="*60)
    print("  STEP 3: Evaluating on test set")
    print("="*60)

    from src.evaluate import evaluate as eval_fn

    _, _, test_dl = build_pythia_dataloaders(
        jsonl_path     = config["data"]["combined_path"],
        tokenizer_name = config["model"]["encoder"],
        batch_size     = config["training"]["batch_size"],
        train_split    = config["data"]["train_split"],
        val_split      = config["data"]["val_split"],
        seed           = config["data"]["seed"],
    )

    # Note: evaluate.py uses enc_a/enc_b keys — need a Pythia-compatible wrapper
    # Use the Pythia model's encode pass to get embeddings first
    model = build_pythia_model(config).to(device)
    ckpt_path = checkpoint or Path(config["training"]["checkpoint_dir"]) / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded checkpoint: {ckpt_path}")

    # Evaluate using Pythia-aware loop
    _evaluate_pythia(model, test_dl, config, device)


def _evaluate_pythia(model, test_dl, config, device):
    """Pythia-specific evaluation loop."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from src.reward_model_pythia import OBJECTIVES
    import json
    from pathlib import Path

    model.eval()
    all_deltas = []
    all_rho    = []
    all_masks  = []

    with torch.no_grad():
        for batch in test_dl:
            ids_a = batch["input_ids_a"].to(device)
            msk_a = batch["attention_mask_a"].to(device)
            ids_b = batch["input_ids_b"].to(device)
            msk_b = batch["attention_mask_b"].to(device)

            delta = model.get_reward_gap(ids_a, msk_a, ids_b, msk_b).cpu()
            all_deltas.append(delta)
            all_rho.append(batch["rho"])
            all_masks.append(batch["mask"])

    deltas = torch.cat(all_deltas).numpy()
    rho    = torch.cat(all_rho).numpy()
    masks  = torch.cat(all_masks).numpy()

    print("\n  Per-objective metrics:")
    results = {}
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

        results[obj] = {"accuracy": round(acc,4), "f1": round(f1,4), "auc": round(auc,4)}
        print(f"  [{obj:10s}]  n={valid.sum():4d}  acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")

    metrics_dir = Path(config["evaluation"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved metrics to {metrics_dir / 'test_metrics.json'}")


def step_analyze(config: dict, device: torch.device, checkpoint: str | None = None):
    print("\n" + "="*60)
    print("  STEP 4: Failure analysis")
    print("="*60)

    from src.failure_analysis import run_failure_analysis

    raw_samples = []
    with open(config["data"]["combined_path"]) as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    _, _, test_dl = build_pythia_dataloaders(
        jsonl_path     = config["data"]["combined_path"],
        tokenizer_name = config["model"]["encoder"],
        batch_size     = config["training"]["batch_size"],
        train_split    = config["data"]["train_split"],
        val_split      = config["data"]["val_split"],
        seed           = config["data"]["seed"],
    )

    model = build_pythia_model(config).to(device)
    ckpt_path = checkpoint or Path(config["training"]["checkpoint_dir"]) / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    n = len(raw_samples)
    n_train = int(n * config["data"]["train_split"])
    n_val   = int(n * config["data"]["val_split"])
    test_raw = raw_samples[n_train + n_val:]

    # failure_analysis.py expects enc_a/enc_b batches — wrap test_dl
    # with a Pythia encoder pass to produce pre-encoded batches
    from src.reward_model_pythia import OBJECTIVES
    import numpy as np
    from src.failure_analysis import categorise_sample
    from collections import Counter
    import json as json_mod
    from pathlib import Path as P

    out_dir = P(config["failure_analysis"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    epsilon = config["failure_analysis"]["epsilon"]

    all_categories = []
    conflicts = []
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

            deltas = model.get_reward_gap(ids_a, msk_a, ids_b, msk_b).cpu().numpy()

            for b in range(deltas.shape[0]):
                cat  = categorise_sample(deltas[b], rho[b], masks[b], epsilon)
                all_categories.append(cat["category"])
                orig = test_raw[sample_idx] if sample_idx < len(test_raw) else {}
                info = {
                    "sample_idx": sample_idx,
                    "task":       tasks[b],
                    "prompt":     orig.get("prompt","")[:200],
                    "analysis":   cat,
                }
                if cat["is_conflict"]:
                    conflicts.append(info)
                sample_idx += 1

    counter = Counter(all_categories)
    total   = len(all_categories)
    print(f"\n  Total test samples: {total}")
    for cat, count in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"  {cat:30s}: {count:5d}  ({100*count/total:.1f}%)")
    print(f"\n  Conflicts: {len(conflicts)}")

    summary = {"total": total, "categories": dict(counter), "conflicts": len(conflicts)}
    with open(out_dir / "failure_summary.json", "w") as f:
        json_mod.dump(summary, f, indent=2)
    with open(out_dir / "conflict_examples.json", "w") as f:
        json_mod.dump(conflicts[:50], f, indent=2)
    print(f"  Saved to {out_dir}")


def main():
    args   = parse_args()
    config = load_config(args.config)
    device = get_device(config["training"]["device"])
    mode   = args.mode

    if mode in ("generate", "all"):
        step_generate()
    if mode in ("train", "all"):
        step_train(config, device)
    if mode in ("evaluate", "all"):
        step_evaluate(config, device, args.checkpoint)
    if mode in ("analyze", "all"):
        step_analyze(config, device, args.checkpoint)

    print("\n✓ V3 Pipeline complete.")


if __name__ == "__main__":
    main()

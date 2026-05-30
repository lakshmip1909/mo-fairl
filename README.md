# MO-FAIRL: Multi-Objective Failure-Aware Inverse Reward Learning

A standalone research pipeline for multi-objective IRL applied to LLM alignment.

Instead of learning a single scalar reward from preferences, we decompose the reward into K objective-specific components:

```
R(x,y) = w₁·r_tox(x,y) + w₂·r_math(x,y) + w₃·r_code(x,y)
```

Each reward head is trained independently on labelled preference pairs, then combined via a fixed weight vector (baseline). This exposes per-objective failures and cross-objective conflicts hidden in standard RLHF.

---

## Objectives

| ID | Name | Description |
|----|------|-------------|
| 0  | toxicity | Safety / harmlessness |
| 1  | math | Mathematical / reasoning correctness |
| 2  | code | Code correctness (verified via unit tests) |

---

## Project Structure

```
mo-fairl/
├── data/                        # Generated JSONL preference pairs
├── scripts/
│   ├── generate_toxicity.py     # Generate toxicity preference pairs
│   ├── generate_math.py         # Generate math preference pairs
│   ├── generate_code.py         # Generate code preference pairs
│   └── combine_data.py          # Merge into combined_pairs.jsonl
├── src/
│   ├── dataset.py               # PreferenceDataset with null masking
│   ├── reward_model.py          # Shared encoder + K reward heads
│   ├── train.py                 # Training loop (masked BCE)
│   ├── evaluate.py              # Per-objective metrics
│   ├── failure_analysis.py      # Failure/conflict detection
│   └── utils.py                 # Helpers
├── configs/
│   └── default.yaml             # All hyperparameters
├── jobs/
│   ├── generate_pairs.pbs
│   ├── train_mo_fairl.pbs
│   ├── evaluate_mo_fairl.pbs
│   └── failure_analysis.pbs
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   ├── failure_examples/
│   └── plots/
├── main.py                      # Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate all preference pairs
python scripts/generate_toxicity.py
python scripts/generate_math.py
python scripts/generate_code.py
python scripts/combine_data.py

# 3. Train
python main.py --config configs/default.yaml --mode train

# 4. Evaluate
python main.py --config configs/default.yaml --mode evaluate

# 5. Failure analysis
python main.py --config configs/default.yaml --mode analyze
```

---

## Data Format

Each example is a JSONL line:

```json
{
  "prompt": "Solve 23 x 17",
  "response_a": "23 x 17 = 391",
  "response_b": "23 x 17 = 400",
  "labels": {
    "toxicity": null,
    "math": 1,
    "code": null
  },
  "task": "math"
}
```

Labels are binary `{0, 1}` or `null` (irrelevant for this sample). The loss ignores `null` labels.

---

## Loss Function

Per-objective BCE loss with null masking:

```
Lₖ = -(1/Nₖ) Σᵢ [ pᵢᵏ log σ(Δᵢᵏ) + (1-pᵢᵏ) log(1 - σ(Δᵢᵏ)) ]

where Δᵢᵏ = rₖ(xᵢ, yᵢᴬ) - rₖ(xᵢ, yᵢᴮ)
```

Total loss: `L = Σₖ Lₖ`

---

## PBS Workflow (HPC)

```bash
qsub jobs/generate_pairs.pbs
qsub jobs/train_mo_fairl.pbs
qsub jobs/evaluate_mo_fairl.pbs
qsub jobs/failure_analysis.pbs
```

---

## Metrics

- Per-objective accuracy, F1, AUC
- Reward gap distribution
- Failure rate per objective
- Conflict rate (samples where objectives disagree)
- Combined reward accuracy

---

## Citation / Reference

Based on the MO-IRL / RLHF decomposition framework. Extends Bradley-Terry preference modelling to multi-objective settings with structured per-objective labels.

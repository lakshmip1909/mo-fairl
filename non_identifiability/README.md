# Reward Non-Identifiability

This folder contains reproducible pipelines for studying reward non-identifiability across domains.

## Datasets

- `gsm8k/`: math reasoning reward non-identifiability.
- `toxicity/`: toxicity/safety reward non-identifiability.
- `code/`: code correctness reward non-identifiability.

## Common Pipeline

Each dataset follows the same structure:

1. Build preference pairs.
2. Extract frozen language-model features.
3. Train multiple linear reward models across random seeds and checkpoints.
4. Evaluate pairwise accuracy and reward gaps.
5. Analyse reward-vector cosine similarity for good/medium/poor model groups.

## Setup

From this folder:

    pip install -r requirements.txt

Then enter a dataset folder and run its pipeline:

    cd gsm8k
    ./run_pipeline.sh

or:

    cd toxicity
    ./run_pipeline.sh

or:

    cd code
    ./run_pipeline.sh

Large model checkpoints, extracted features, logs, and raw datasets are not intended to be committed.

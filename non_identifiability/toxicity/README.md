# Toxicity Reward Non-Identifiability

This folder contains a reproducible toxicity/safety reward non-identifiability pipeline.

## Pipeline

1. Build clean-vs-toxic preference pairs.
2. Extract response-only Pythia-410M features.
3. Train linear reward models with max-margin loss.
4. Evaluate train/test pairwise accuracy and reward gaps.
5. Analyse good/medium/poor reward-model groups by weight cosine similarity.

## How to run

From this folder:

    pip install -r ../requirements.txt
    ./run_pipeline.sh

The pipeline downloads toxicity data from HuggingFace, builds preference pairs, extracts frozen Pythia response features, trains linear reward models, and writes outputs to `results/`, `plots/`, `models/`, and `data/`.

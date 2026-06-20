# GSM8K Reward Non-Identifiability

This folder contains a reproducible GSM8K reward non-identifiability pipeline.

## Pipeline

1. Generate GSM8K base pairs.
2. Build filtered hard-negative GSM8K pairs.
3. Extract concat-only prompt-response features:
   phi(x,o) = [phi(x); phi(o)]
4. Train linear reward models with max-margin loss.
5. Evaluate train/OOD pairwise accuracy and reward gaps.
6. Analyse good/medium/poor reward-model groups by weight cosine similarity.

## How to run

From this folder:

    pip install -r ../requirements.txt
    ./run_pipeline.sh

The pipeline downloads GSM8K from HuggingFace, builds preference pairs, extracts frozen Pythia features, trains linear reward models, and writes outputs to `results/`, `plots/`, `models/`, and `data/`.

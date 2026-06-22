# Toxicity Reward Non-Identifiability

This folder contains the toxicity/safety reward non-identifiability pipeline.

## Pipeline

1. Build clean-vs-toxic preference pairs.
2. Extract response-only Pythia-410M features.
3. Train linear reward models with max-margin loss.
4. Evaluate train/test pairwise accuracy and reward gaps.
5. Analyse good/medium/poor reward-model groups by weight cosine similarity.

## Running

From this folder:

    pip install -r ../requirements.txt
    ./run_pipeline.sh

For PBS-based execution:

    qsub jobs/toxicity_linear_nonid.pbs

## Outputs

The pipeline writes generated data, extracted features, checkpoints, evaluation CSVs, and plots to:

- data/
- models/
- results/
- plots/

Example outputs are included in results/ and plots/.

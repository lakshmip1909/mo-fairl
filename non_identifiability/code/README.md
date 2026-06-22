# MBPP Code Reward Non-Identifiability

This folder contains the MBPP code reward non-identifiability pipeline.

## Pipeline

1. Build MBPP preference pairs.
2. Extract concat-only prompt-code features:
   phi(x,o) = [phi(x); phi(o)]
3. Train linear reward models with max-margin loss.
4. Evaluate train/test pairwise accuracy and reward gaps.
5. Analyse good/medium/poor reward-model groups by weight cosine similarity.

## Running

From this folder:

    pip install -r ../requirements.txt
    ./run_pipeline.sh

For PBS-based execution:

    qsub jobs/code_mbpp_full.pbs

## Outputs

The pipeline writes generated data, extracted features, checkpoints, evaluation CSVs, and plots to:

- data/
- models/
- results/
- plots/

Example outputs are included in results/ and plots/.

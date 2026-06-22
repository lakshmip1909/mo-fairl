# GSM8K Reward Non-Identifiability

This folder contains the GSM8K reward non-identifiability pipeline.

## Pipeline

1. Generate GSM8K base pairs.
2. Build filtered hard-negative GSM8K pairs.
3. Extract concat-only prompt-response features:
   phi(x,o) = [phi(x); phi(o)]
4. Train linear reward models with max-margin loss.
5. Evaluate train/OOD pairwise accuracy and reward gaps.
6. Analyse good/medium/poor reward-model groups by weight cosine similarity.

## Running

From this folder:

    pip install -r ../requirements.txt
    ./run_pipeline.sh

For PBS-based execution:

    qsub jobs/gsm8k_concat_pipeline.pbs

## Outputs

The pipeline writes generated data, extracted features, checkpoints, evaluation CSVs, and plots to:

- data/
- models/
- results/
- plots/

Example outputs are included in results/ and plots/.

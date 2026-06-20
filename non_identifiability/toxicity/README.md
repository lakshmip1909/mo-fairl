# Toxicity Reward Non-Identifiability

This folder contains the toxicity/safety reward non-identifiability pipeline.

## Pipeline

1. Build toxicity preference pairs from generated/scored outputs.
2. Extract response-only Pythia-410M features.
3. Train linear reward models with max-margin loss.
4. Evaluate train/test pairwise accuracy and reward gaps.
5. Analyse good/medium/poor reward-model groups by weight cosine similarity.

## Main Result

Overall:
- Train accuracy: 0.971 ± 0.033
- Test accuracy: 0.861 ± 0.009

Poor group:
- Train accuracy: 0.909 ± 0.015
- Test accuracy: 0.862 ± 0.010
- Mean cosine similarity: 0.123

Medium group:
- Train accuracy: 0.986 ± 0.004
- Test accuracy: 0.859 ± 0.009
- Mean cosine similarity: 0.181

Good group:
- Train accuracy: 1.000 ± 0.0004
- Test accuracy: 0.861 ± 0.009
- Mean cosine similarity: 0.238

This shows that reward models can achieve very similar external performance while corresponding to substantially different reward directions.

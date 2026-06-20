# GSM8K Reward Non-Identifiability

This folder contains the final GSM8K reward non-identifiability pipeline.

## Pipeline

1. Build filtered hard-negative GSM8K pairs.
2. Extract concat-only prompt-response features:
   phi(x,o) = [phi(x); phi(o)]
3. Train linear reward models with max-margin loss.
4. Evaluate train/OOD pairwise accuracy and reward gaps.
5. Analyse good/medium/poor reward-model groups by weight cosine similarity.

## Main Result

Concat-only GSM8K reward models produced a natural spread across checkpoints.

Overall:
- Train accuracy: 0.714 ± 0.041
- OOD accuracy: 0.560 ± 0.009

Poor group:
- Train accuracy: 0.640 ± 0.019
- Mean cosine similarity: 0.116

Medium group:
- Train accuracy: 0.725 ± 0.005
- Mean cosine similarity: 0.167

Good group:
- Train accuracy: 0.758 ± 0.003
- Mean cosine similarity: 0.236

This shows that reward models with similar performance can correspond to substantially different reward directions, supporting reward non-identifiability on GSM8K.

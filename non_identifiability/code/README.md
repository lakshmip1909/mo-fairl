# MBPP Code Reward Non-Identifiability

## Pipeline

1. Build MBPP preference pairs.
2. Extract prompt-code concat features.
3. Train 60 reward models (5 seeds × checkpoints).
4. Evaluate train/test accuracy.
5. Analyse reward-model similarity.

## Main Result

60 reward models trained.

Train accuracy:
0.803 ± 0.109

Test accuracy:
0.510 ± 0.021

Poor models:
train ≈ 0.608

Medium models:
train ≈ 0.847

Good models:
train ≈ 0.908

Mean cosine similarities remain low (~0.15–0.18),
indicating different reward directions despite similar
performance.

This supports reward non-identifiability in code
generation tasks.

# MO-FAIRL Core

Clean implementation of the core Multi-Objective Failure-Aware Inverse Reward Learning idea.

## Core idea

Instead of learning one scalar reward, the reward is decomposed into objective-specific rewards:

R(x,y) = sum_k w_k r_k(x,y)

where:

- r_k(x,y) is the reward for objective k
- w_k is the objective weight
- objectives used here are toxicity, math, and code

The project uses structured preference labels:

rho_i = (rho_i^tox, rho_i^math, rho_i^code)

For example:

safe but wrong answer vs toxic but correct answer

has labels:

toxicity = 1
math = 0
code = null

This means response A wins toxicity, but response B wins math.

## Implemented components

- Multi-objective reward decomposition
- Structured preference labels
- Conflict preference pairs
- Pythia-410M reward model backbone
- Objective-specific reward heads
- Learnable objective weights
- Global preference score using w^T Delta
- Weight-recovery experiments
- Conflict evaluation

## Main equations

Objective-specific reward gap:

Delta_i^k = r_k(x_i, y_i^+) - r_k(x_i, y_i^-)

Per-objective loss:

L_k = - log sigmoid(rho_i^k Delta_i^k)

Combined loss:

L = sum_k w_k L_k

Global preference score:

score_i = w^T Delta_i

where:

Delta_i = (Delta_i^tox, Delta_i^math, Delta_i^code)

## Repository structure

main_conflict.py
    Main entry point for training, evaluation and analysis.

src/dataset_conflict.py
    Dataset loader for structured multi-objective preference pairs.

src/reward_model_pythia_conflict.py
    Pythia-410M based multi-objective reward model.

src/train_conflict.py
    Training code for objective-specific losses, global preference loss and learnable weights.

src/evaluate_conflict.py
    Evaluation code for per-objective and global preference metrics.

scripts/real/build_conflict_pairs_v5.py
    Builds conflict pairs involving toxicity, math and code.

scripts/real/build_weight_recovery_pairs.py
    Builds controlled datasets for testing whether objective weights can be recovered.

configs/conflict_v5.yaml
    Configuration for full conflict experiment.

configs/weight_tox.yaml
    Configuration for toxicity-dominant weight recovery experiment.

jobs/run_conflict_v5.pbs
    PBS job for running the full conflict experiment.

jobs/run_weight_tox.pbs
    PBS job for toxicity-dominant weight recovery.

## Running on HPC

Activate the environment:

conda activate irlf

Run the full conflict experiment:

qsub jobs/run_conflict_v5.pbs

Monitor:

qstat -u ls1925
tail -f logs/pipeline_conflict_v5.log

Run toxicity-dominant weight recovery:

qsub jobs/run_weight_tox.pbs

Monitor:

qstat -u ls1925
tail -f logs/weight_tox.log

## Main results so far

### Real objective-specific reward learning

Using Pythia-410M with real datasets:

- Toxicity accuracy: 0.8802
- Math accuracy: 1.0000
- Code accuracy: 0.9700

### Full conflict experiment

Using conflict pairs involving toxicity, math and code:

- Toxicity accuracy: 0.83
- Math accuracy: 1.00
- Code accuracy: 1.00
- Global conflict accuracy: 0.4762

Learned weights:

toxicity = 0.3333
math = 0.3333
code = 0.3334

### Toxicity-dominant weight recovery

Learned weights:

toxicity = 0.3343
math = 0.3321
code = 0.3336

The movement is small, suggesting that the static weight model receives limited signal when objective-specific tasks are too easy.

## Current conclusion

The implementation validates the core formulation:

- rewards can be decomposed into objective-specific heads
- structured labels can represent conflicts
- conflict data can be generated and evaluated
- objective weights can be made learnable

However, the current conflict data is still too simple. The model learns the individual objectives almost perfectly, so the global weight vector remains close to uniform.

The next research step is to use harder and more realistic conflicts, or replace the static weight vector w with prompt-conditioned dynamic weights:

w(x) = softmax(g_phi(x))

This would allow the model to allocate importance differently depending on whether the prompt is mainly about safety, mathematics or code.


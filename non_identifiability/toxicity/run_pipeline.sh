#!/bin/bash
set -e

python scripts/00_build_jigsaw_pairs.py
python scripts/02_extract_response_features.py
python scripts/03_train_eval_toxicity_rewards.py
python scripts/04_analyse_toxicity_groups.py

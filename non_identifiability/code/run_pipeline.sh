#!/bin/bash
set -e

python scripts/00_build_mbpp_pairs.py
python scripts/01_extract_code_features.py
python scripts/02_train_eval_code_rewards.py
python scripts/03_analyse_code_groups.py

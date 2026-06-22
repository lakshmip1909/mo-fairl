#!/bin/bash
set -e

RUN=mbpp_random

python scripts/00_build_mbpp_pairs.py --run_name "$RUN" --strategy random --k 10
python scripts/01_extract_concat_features.py --run_name "$RUN"
python scripts/02_train_eval_code_rewards.py --run_name "$RUN"
python scripts/03_analyse_code_groups.py --run_name "$RUN"

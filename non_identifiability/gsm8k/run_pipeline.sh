#!/bin/bash
set -e

python scripts/00_generate_base_pairs.py
python scripts/01_build_filtered_hard_pairs.py
python scripts/02_extract_concat_features.py
python scripts/03_train_eval_concat.py
python scripts/04_analyse_good_poor_groups.py

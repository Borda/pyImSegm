#!/usr/bin/env bash

rm -r -f results && mkdir results
python experiments_ovary_centres/run_create_annotation.py
python experiments_ovary_centres/run_center_candidate_training.py
python experiments_ovary_centres/run_center_prediction.py
python experiments_ovary_centres/run_center_clustering.py
python experiments_ovary_centres/run_center_evaluation.py

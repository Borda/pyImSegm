#!/usr/bin/env bash

pip install --user git+https://github.com/Borda/morph-snakes.git
rm -r -f results && mkdir results
python experiments_ovary_detect/run_RG2Sp_estim_shape-models.py
python experiments_ovary_detect/run_ovary_egg-segmentation.py \
    -m ellipse_moments ellipse_ransac_mmt ellipse_ransac_crit GC_pixels-large GC_pixels-shape GC_slic-small GC_slic-shape rg2sp_greedy-single rg2sp_GC-mixture watershed_morph
python experiments_ovary_detect/run_ovary_segm_evaluation.py --visual
python experiments_ovary_detect/run_export_user-annot-segm.py
python experiments_ovary_detect/run_cut_segmented_objects.py
python experiments_ovary_detect/run_ellipse_annot_match.py
python experiments_ovary_detect/run_ellipse_cut_scale.py
python experiments_ovary_detect/run_egg_swap_orientation.py

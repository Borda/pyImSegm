#!/usr/bin/env bash

rm -r -f results && mkdir results
python experiments_segmentation/run_compute_stat_annot_segm.py \
    -a "data_images/drosophila_ovary_slice/annot_struct/*.png" \
    -s "data_images/drosophila_ovary_slice/segm/*.png" \
    --visual
python experiments_segmentation/run_segm_slic_model_graphcut.py \
    -i "data_images/drosophila_disc/image/img_[5,6].jpg" \
    -cfg ./experiments_segmentation/sample_config.yml \
    --visual
python experiments_segmentation/run_segm_slic_classif_graphcut.py \
    -l data_images/drosophila_ovary_slice/list_imgs-annot-struct_short.csv \
    -i "data_images/drosophila_ovary_slice/image/insitu41*.jpg" \
    -cfg ./experiments_segmentation/sample_config.yml \
    --visual
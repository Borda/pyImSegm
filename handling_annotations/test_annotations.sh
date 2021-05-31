#!/usr/bin/env bash

python handling_annotations/run_image_color_quantization.py \
    -imgs "./data-images/drosophila_ovary_slice/segm_rgb/*.png"
python handling_annotations/run_image_color_quantization.py \
    -imgs "./data-images/drosophila_ovary_slice/segm_rgb/*.png" \
    -m position
python handling_annotations/run_image_convert_label_color.py \
    -imgs "./data-images/drosophila_ovary_slice/segm/*.png" \
    -out ./data-images/drosophila_ovary_slice/segm_rgb
python handling_annotations/run_image_convert_label_color.py \
    -imgs "./data-images/drosophila_ovary_slice/segm_rgb/*.png" \
    -out ./data-images/drosophila_ovary_slice/segm
python handling_annotations/run_overlap_images_segms.py \
    -imgs "./data-images/drosophila_ovary_slice/image/*.jpg" \
    -segs ./data-images/drosophila_ovary_slice/segm \
    -out ./results/overlap_ovary_segment
python handling_annotations/run_segm_annot_inpaint.py \
    -imgs "./data-images/drosophila_ovary_slice/segm/*.png" \
    --label 0
python handling_annotations/run_segm_annot_relabel.py \
    -imgs "./data-images/drosophila_ovary_slice/center_levels/*.png" \
    -out ./results/relabel_center_levels

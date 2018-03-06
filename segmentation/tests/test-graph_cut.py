"""
Unit testing for particular segmentation module

Copyright (C) 2014-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import unittest
import logging

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('..', '..'))) # Add path to root
import segmentation.utils.data_samples as d_spl
import segmentation.utils.data_io as tl_io
import segmentation.superpixels as seg_spx
import segmentation.graph_cuts as seg_gc
import segmentation.labeling as seg_lb

# set the output put directory
PATH_OUTPUT = tl_io.update_path('output', absolute=True)


class TestGraphCut(unittest.TestCase):

    img = d_spl.load_sample_image(d_spl.IMAGE_DROSOPHILA_OVARY_2D)
    annot = d_spl.load_sample_image(d_spl.ANNOT_DROSOPHILA_OVARY_2D)

    def test_count_transitions_segment(self):
        img = self.img[:, :, 0]
        annot = self.annot.astype(int)

        slic = seg_spx.segment_slic_img2d(img, sp_size=15, rltv_compact=0.2)
        label_hist = seg_lb.histogram_regions_labels_norm(slic, annot)
        labels = np.argmax(label_hist, axis=1)
        trans = seg_gc.count_label_transitions_connected_segments({'a': slic},
                                                                  {'a': labels})
        path_csv = os.path.join(PATH_OUTPUT, 'labels_transitions.csv')
        pd.DataFrame(trans).to_csv(path_csv)
        gc_regul = seg_gc.compute_pairwise_cost_from_transitions(trans, 10.)

        np.random.seed(0)
        features = np.tile(labels, (5, 1)).T.astype(float)
        features += np.random.random(features.shape) - 0.5

        gmm = seg_gc.estim_class_model_gmm(features, 4)
        proba = gmm.predict_proba(features)

        seg_gc.segment_graph_cut_general(slic, proba, gc_regul)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

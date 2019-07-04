"""
Unit testing for particular segmentation module

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import unittest
import logging

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Add path to root
from imsegm.utilities.data_samples import (load_sample_image, IMAGE_DROSOPHILA_OVARY_2D,
                                           ANNOT_DROSOPHILA_OVARY_2D)
from imsegm.utilities.data_io import update_path
from imsegm.superpixels import segment_slic_img2d
from imsegm.graph_cuts import (count_label_transitions_connected_segments,
                               compute_pairwise_cost_from_transitions,
                               estim_class_model_gmm, segment_graph_cut_general)
from imsegm.labeling import histogram_regions_labels_norm

# set the output put directory
PATH_OUTPUT = update_path('output', absolute=True)


class TestGraphCut(unittest.TestCase):

    img = load_sample_image(IMAGE_DROSOPHILA_OVARY_2D)
    annot = load_sample_image(ANNOT_DROSOPHILA_OVARY_2D)

    def test_count_transitions_segment(self):
        img = self.img[:, :, 0]
        annot = self.annot.astype(int)

        slic = segment_slic_img2d(img, sp_size=15, relative_compact=0.2)
        label_hist = histogram_regions_labels_norm(slic, annot)
        labels = np.argmax(label_hist, axis=1)
        trans = count_label_transitions_connected_segments({'a': slic}, {'a': labels})
        path_csv = os.path.join(PATH_OUTPUT, 'labels_transitions.csv')
        pd.DataFrame(trans).to_csv(path_csv)
        gc_regul = compute_pairwise_cost_from_transitions(trans, 10.)

        np.random.seed(0)
        features = np.tile(labels, (5, 1)).T.astype(float)
        features += np.random.random(features.shape) - 0.5

        gmm = estim_class_model_gmm(features, 4)
        proba = gmm.predict_proba(features)

        segment_graph_cut_general(slic, proba, gc_regul)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

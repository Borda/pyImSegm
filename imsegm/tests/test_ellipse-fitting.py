"""
Unit testing for particular segmentation module

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Add path to root
from imsegm.utilities.data_io import update_path, load_image_2d
from imsegm.utilities.drawing import ellipse, figure_ellipse_fitting
from imsegm.ellipse_fitting import (get_slic_points_labels, ransac_segm, EllipseModelSegm,
                                    prepare_boundary_points_ray_edge)

# set some default paths
PATH_OUTPUT = update_path('output', absolute=True)
PATH_OVARY = os.path.join(update_path('data_images', absolute=True), 'drosophila_ovary_slice')
PATH_IMAGES = os.path.join(PATH_OVARY, 'image')
PATH_SEGM = os.path.join(PATH_OVARY, 'segm')
PATH_ANNOT = os.path.join(PATH_OVARY, 'annot_eggs')
PATH_CENTRE = os.path.join(PATH_OVARY, 'center_levels')
# color spaces for visualisations
COLORS = 'bgrmyck'
# set probability to be foreground / background
TABLE_FB_PROBA = [[0.01, 0.7, 0.95, 0.8],
                  [0.99, 0.3, 0.05, 0.2]]
MAX_FIGURE_SEIZE = 10


class TestEllipseFitting(unittest.TestCase):

    def test_ellipse_fitting(self, name='insitu7545',
                             table_prob=TABLE_FB_PROBA):
        """    """
        img, _ = load_image_2d(os.path.join(PATH_IMAGES, name + '.jpg'))
        seg, _ = load_image_2d(os.path.join(PATH_SEGM, name + '.png'))
        annot, _ = load_image_2d(os.path.join(PATH_ANNOT, name + '.png'))
        path_center = os.path.join(PATH_CENTRE, name + '.csv')
        centers = pd.read_csv(path_center, index_col=0).values[:, [1, 0]]

        slic, points_all, labels = get_slic_points_labels(seg, slic_size=20,
                                                          slic_regul=0.3)
        weights = np.bincount(slic.ravel())
        points_centers = prepare_boundary_points_ray_edge(seg, centers, close_points=5)

        segm = np.zeros(seg.shape)
        ellipses, crits = [], []
        for i, points in enumerate(points_centers):
            model, _ = ransac_segm(points, EllipseModelSegm, points_all,
                                   weights, labels, table_prob, min_samples=0.6,
                                   residual_threshold=15, max_trials=50)
            if not model:
                continue
            ellipses.append(model.params)
            crit = model.criterion(points_all, weights, labels, table_prob)
            crits.append(np.round(crit))
            logging.info('model params: %r', model.params)
            logging.info('-> crit: %f', crit)
            c1, c2, h, w, phi = model.params
            rr, cc = ellipse(int(c1), int(c2), int(h), int(w), phi, segm.shape)
            segm[rr, cc] = (i + 1)

        if img.ndim == 3:
            img = img[:, :, 0]
        fig = figure_ellipse_fitting(img, seg, ellipses, centers, crits)
        fig_name = 'ellipse-fitting_%s.pdf' % name
        fig.savefig(os.path.join(PATH_OUTPUT, fig_name),
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        score = adjusted_rand_score(annot.ravel(), segm.ravel())
        self.assertGreaterEqual(score, 0.5)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

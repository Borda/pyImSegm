"""
Unit testing for particular segmentation module

Copyright (C) 2014-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Add path to root
import imsegm.utils.data_io as tl_io
import imsegm.utils.drawing as tl_visu
import imsegm.ellipse_fitting as seg_fit

# set some default paths
PATH_OUTPUT = tl_io.update_path('output', absolute=True)
PATH_OVARY = os.path.join(tl_io.update_path('images', absolute=True),
                          'drosophila_ovary_slice')
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
        img, _ = tl_io.load_image_2d(os.path.join(PATH_IMAGES, name + '.jpg'))
        seg, _ = tl_io.load_image_2d(os.path.join(PATH_SEGM, name + '.png'))
        annot, _ = tl_io.load_image_2d(os.path.join(PATH_ANNOT, name + '.png'))
        path_center = os.path.join(PATH_CENTRE, name + '.csv')
        centers = pd.DataFrame.from_csv(path_center).values[:, [1, 0]]

        slic, points_all, labels = seg_fit.get_slic_points_labels(seg, slic_size=20,
                                                                  slic_regul=0.3)
        weights = np.bincount(slic.ravel())
        points_centers = seg_fit.prepare_boundary_points_ray_edge(seg, centers,
                                                                  close_points=5)

        segm = np.zeros(seg.shape)
        ellipses, crits = [], []
        for i, points in enumerate(points_centers):
            model, _ = seg_fit.ransac_segm(points, seg_fit.EllipseModelSegm,
                                           points_all, weights, labels,
                                           table_prob, min_samples=0.6,
                                           residual_threshold=15, max_trials=50)
            if model is None: continue
            ellipses.append(model.params)
            crit = model.criterion(points_all, weights, labels, table_prob)
            crits.append(np.round(crit))
            logging.info('model params: %s', repr(model.params))
            logging.info('-> crit: %f', crit)
            c1, c2, h, w, phi = model.params
            rr, cc = tl_visu.ellipse(int(c1), int(c2), int(h), int(w), phi,
                                     segm.shape)
            segm[rr, cc] = (i + 1)

        if img.ndim == 3:
            img = img[:, :, 0]
        fig = tl_visu.figure_ellipse_fitting(img, seg, ellipses, centers, crits)
        fig_name = 'ellipse-fitting_%s.pdf' % name
        fig.savefig(os.path.join(PATH_OUTPUT, fig_name),
                    bbox_inches='tight', pad_inches=0)

        score = adjusted_rand_score(annot.ravel(), segm.ravel())
        self.assertGreaterEqual(score, 0.5)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

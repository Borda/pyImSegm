"""
Unit testing for particular segmentation module

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import time
import logging
import unittest

import numpy as np
from skimage import draw, transform
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Add path to root
from imsegm.utilities.data_samples import (IMAGE_LENNA, load_sample_image,
                                           sample_color_image_rand_segment)
from imsegm.utilities.data_io import update_path
from imsegm.utilities.drawing import figure_ray_feature
from imsegm.descriptors import (cython_img2d_color_mean, create_filter_bank_lm_2d,
                                compute_ray_features_segm_2d, shift_ray_features,
                                reconstruct_ray_features_2d, FEATURES_SET_ALL,
                                compute_selected_features_color2d)
from imsegm.superpixels import segment_slic_img2d

# angular step for Ray features
ANGULAR_STEP = 15
# size of subfigure for visualise the Filter bank
SUBPLOT_SIZE_FILTER_BANK = 3
PATH_OUTPUT = update_path('output', absolute=True)
PATH_FIGURES_RAY = os.path.join(PATH_OUTPUT, 'temp_ray-features')
# create the folder for visualisations
if not os.path.exists(PATH_FIGURES_RAY):
    os.mkdir(PATH_FIGURES_RAY)


def export_ray_results(seg, center, points, ray_dist_raw, ray_dist, name):
    """ export result from Ray features extractions

    :param ndarray seg: segmentation
    :param tuple(int,int) center: center of the Ray features
    :param [[int, int]] points: list of reconstructed points
    :param list(list(int)) ray_dist_raw: list of raw Ray distances in regular step
    :param list(list(int)) ray_dist: list of normalised Ray distances in regular step
    :param str name: name of particular figure
    """
    fig = figure_ray_feature(seg, center, ray_dist_raw=ray_dist_raw,
                             ray_dist=ray_dist, points_reconst=points,
                             title=os.path.splitext(name)[0])
    fig_path = os.path.join(PATH_FIGURES_RAY, name)
    fig.savefig(fig_path)
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        plt.show()
    plt.close(fig)
    return fig_path


class TestFeatures(unittest.TestCase):

    def test_features_rgb(self):
        im, seg = sample_color_image_rand_segment()
        # Cython
        logging.info('running Cython code...')
        start = time.time()
        f = cython_img2d_color_mean(im, seg)
        logging.info('time elapsed: %f', time.time() - start)
        logging.debug('%r', f)
        # Python / Numba
        # logger.info('running Python code...')
        # start = time.time()
        # f = computeColourMeanRGB(im, seg)
        # logger.info('time elapsed: {}'.format(time.time() - start))
        # logger.debug(repr(f))

    def test_filter_banks(self, ax_size=SUBPLOT_SIZE_FILTER_BANK):
        filters, names = create_filter_bank_lm_2d()
        l_max, w_max = len(filters), max([f.shape[0] for f in filters])
        fig_size = (w_max * ax_size, l_max * ax_size)
        fig, axarr = plt.subplots(l_max, w_max, figsize=fig_size)
        for i in range(l_max):
            f = filters[i]
            for j in range(f.shape[0]):
                axarr[i, j].set_title(names[i][j])
                axarr[i, j].imshow(f[j, :, :], cmap=plt.cm.gray)
        fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        p_fig = os.path.join(PATH_OUTPUT, 'temp_filter-banks.png')
        fig.savefig(p_fig)
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            plt.show()
        plt.close(fig)
        self.assertTrue(os.path.exists(p_fig))

    def test_ray_features_circle(self):
        seg = np.ones((400, 600), dtype=bool)
        x, y = draw.circle(200, 250, 100, shape=seg.shape)
        seg[x, y] = False

        points = [(200, 250), (150, 200), (250, 200), (250, 300)]
        for i, point in enumerate(points):
            ray_dist_raw = compute_ray_features_segm_2d(seg, point,
                                                        angle_step=ANGULAR_STEP)
            ray_dist, shift = shift_ray_features(ray_dist_raw)
            points = reconstruct_ray_features_2d(point, ray_dist, shift)
            p_fig = export_ray_results(seg, point, points, ray_dist_raw, ray_dist,
                                       'circle-%i.png' % i)
            self.assertTrue(os.path.exists(p_fig))

    def test_ray_features_ellipse(self):
        seg = np.ones((400, 600), dtype=bool)
        x, y = draw.ellipse(200, 250, 120, 200, rotation=np.deg2rad(30),
                            shape=seg.shape)
        seg[x, y] = False

        points = [(200, 250), (150, 200), (250, 300)]
        for i, point in enumerate(points):
            ray_dist_raw = compute_ray_features_segm_2d(
                seg, point, angle_step=ANGULAR_STEP)
            # ray_dist, shift = seg_fts.shift_ray_features(ray_dist_raw)
            points = reconstruct_ray_features_2d(point, ray_dist_raw)
            p_fig = export_ray_results(seg, point, points, ray_dist_raw, [],
                                       'ellipse-%i.png' % i)
            self.assertTrue(os.path.exists(p_fig))

    def test_ray_features_circle_down_edge(self):
        seg = np.zeros((400, 600), dtype=bool)
        x, y = draw.circle(200, 250, 150, shape=seg.shape)
        seg[x, y] = True
        points = [(200, 250), (150, 200), (250, 200), (250, 300)]

        for i, point in enumerate(points):
            ray_dist_raw = compute_ray_features_segm_2d(seg, point, edge='down',
                                                        angle_step=ANGULAR_STEP)
            ray_dist, shift = shift_ray_features(ray_dist_raw)
            points_rt = reconstruct_ray_features_2d(point, ray_dist, shift)
            p_fig = export_ray_results(seg, point, points_rt, ray_dist_raw, ray_dist,
                                       'circle-full_edge-down-%i.png' % i)
            self.assertTrue(os.path.exists(p_fig))

        # insert white interior
        x, y = draw.circle(200, 250, 120, shape=seg.shape)
        seg[x, y] = False

        for i, point in enumerate(points):
            ray_dist_raw = compute_ray_features_segm_2d(seg, point, edge='down',
                                                        angle_step=ANGULAR_STEP)
            ray_dist, shift = shift_ray_features(ray_dist_raw)
            points_rt = reconstruct_ray_features_2d(point, ray_dist, shift)
            p_fig = export_ray_results(seg, point, points_rt, ray_dist_raw, ray_dist,
                                       'circle-inter_edge-down-%i.png' % i)
            self.assertTrue(os.path.exists(p_fig))

    def test_ray_features_polygon(self):
        seg = np.ones((400, 600), dtype=bool)
        x, y = draw.polygon(np.array([50, 170, 300, 250, 150, 150, 50]),
                            np.array([100, 270, 240, 150, 150, 80, 50]),
                            shape=seg.shape)
        seg[x, y] = False

        centres = [(150, 200), (200, 250), (250, 200), (120, 100)]
        for i, point in enumerate(centres):
            ray_dist_raw = compute_ray_features_segm_2d(seg, point,
                                                        angle_step=ANGULAR_STEP)
            ray_dist, shift = shift_ray_features(ray_dist_raw)
            points = reconstruct_ray_features_2d(point, ray_dist, shift)
            p_fig = export_ray_results(seg, point, points, ray_dist_raw, ray_dist,
                                       'polygon-%i.png' % i)
            self.assertTrue(os.path.exists(p_fig))

    def test_show_image_features_clr2d(self):
        img = load_sample_image(IMAGE_LENNA)
        img = transform.resize(img, (128, 128))
        slic = segment_slic_img2d(img, sp_size=10, relative_compact=0.2)

        features, names = compute_selected_features_color2d(img, slic, FEATURES_SET_ALL)

        path_dir = os.path.join(PATH_OUTPUT, 'temp_image-rgb2d-features')
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)

        for i in range(features.shape[1]):
            fts = features[:, i]
            im_fts = fts[slic]
            p_fig = os.path.join(path_dir, names[i] + '.png')
            plt.imsave(p_fig, im_fts)
            self.assertTrue(os.path.exists(p_fig))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

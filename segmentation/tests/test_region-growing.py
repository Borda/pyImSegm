"""

Copyright (C) 2014-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging
import os
import sys
import glob
import unittest
# import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join('..', '..'))) # Add path to root
import segmentation.utils.data_io as tl_io
import segmentation.utils.drawing as tl_visu
import segmentation.superpixels as tl_spx
import segmentation.region_growing as tl_rg

PATH_BASE = tl_io.update_path(os.path.join('images', 'drosophila_ovary_slice'),
                              absolute=True)
PATH_IMAGE = os.path.join(PATH_BASE, 'image')
PATH_SEGM = os.path.join(PATH_BASE, 'segm')
PATH_ANNOT = os.path.join(PATH_BASE, 'annot_eggs')
PATH_CENTRE = os.path.join(PATH_BASE, 'center_levels')
PATH_OUTPUT = tl_io.update_path('output', absolute=True)
NAME_RG2SP_MODEL = 'RG2SP_multi-model_mixture.npz'
PATH_PKL_MODEL = os.path.join(PATH_OUTPUT, NAME_RG2SP_MODEL)
LABELS_FG_PROB = [0.05, 0.7, 0.9, 0.9]
DEFAULT_RG2SP_THRESHOLDS = tl_rg.DEFAULT_RG2SP_THRESHOLDS


def compute_prior_map(cdist, size=(500, 800), step=5):
    prior_map = np.zeros(size)
    centre = np.array(size) / 2
    for i in np.arange(prior_map.shape[0], step=step):
        for j in np.arange(prior_map.shape[1], step=step):
            prior_map[i:i+step, j:j+step] = \
                tl_rg.compute_shape_prior_table_cdf([i, j], cdist, centre,
                                                    angle_shift=0)
    return prior_map


def load_inputs(name):
    img, _ = tl_io.load_image_2d(os.path.join(PATH_IMAGE, name + '.jpg'))
    seg, _ = tl_io.load_image_2d(os.path.join(PATH_SEGM, name + '.png'))
    annot, _ = tl_io.load_image_2d(os.path.join(PATH_ANNOT, name + '.png'))
    centers = pd.DataFrame.from_csv(
        os.path.join(PATH_CENTRE, name + '.csv')).values
    centers[:, [0, 1]] = centers[:, [1, 0]]

    slic = tl_spx.segment_slic_img2d(img, sp_size=25, rltv_compact=0.3)
    return img, seg, slic, centers, annot


def expert_segm(name, img, seg, segm_obj, annot, type='xxx'):
    FIG_SIZE = (12. * np.array(img.shape[:2]) / np.max(img.shape))
    fig, ax = plt.subplots(ncols=2, figsize=(FIG_SIZE[1] * 2, FIG_SIZE[0]))
    ax[0].set_title('Region growing - segmentation')
    ax[0].imshow(segm_obj, cmap=plt.cm.jet)
    ax[0].contour(seg, levels=np.unique(seg), colors='#bfbfbf')
    ax[1].set_title('Annotation')
    ax[1].imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
    ax[1].contour(annot, cmap=plt.cm.jet)
    fig.savefig(os.path.join(PATH_OUTPUT, '%s_%s.pdf' % (type, name)))


class TestRegionGrowing(unittest.TestCase):

    def test_shape_modeling(self, dir_annot=PATH_ANNOT):
        """    """
        list_paths = sorted(glob.glob(os.path.join(dir_annot, '*.png')))
        logging.info('nb images: %i SAMPLES: %s', len(list_paths),
                     repr([os.path.basename(p) for p in list_paths[:5]]))
        list_segms = []
        for path_seg in list_paths:
            seg, _ = tl_io.load_image_2d(path_seg)
            list_segms.append(seg)

        list_rays, list_shifts = tl_rg.compute_object_shapes(list_segms,
                             ray_step=25, interp_order='spline', smooth_coef=1)
        logging.info('nb eggs: %i nb rays: %i',
                     len(list_rays), len(list_rays[0]))

        model, list_mean_cdf = tl_rg.transform_rays_model_sets_mean_cdf_mixture(
            list_rays, 2)

        np.savez(PATH_PKL_MODEL, data={'name': 'set_cdfs',
                                       'cdfs': list_mean_cdf,
                                       'mix_model': model})
        # with open(PATH_PKL_MODEL, 'w') as fp:
        #     pickle.dump({'name': 'set_cdfs',
        #                  'cdfs': list_mean_cdf,
        #                  'mix_model': model}, fp)
        self.assertTrue(os.path.exists(PATH_PKL_MODEL))

        max_len = max([np.asarray(l_cdf).shape[1]
                       for _, l_cdf in list_mean_cdf])

        fig, axarr = plt.subplots(nrows=len(list_mean_cdf), ncols=2,
                                  figsize=(12, 3.5 * len(list_mean_cdf)))
        for i, (mean, list_cdf) in enumerate(list_mean_cdf):
            cdist = np.zeros((len(list_cdf), max_len))
            cdist[:, :len(list_cdf[0])] = np.array(list_cdf)
            axarr[i, 0].set_title('Inverse cumulative distribution')
            axarr[i, 0].imshow(cdist, aspect='auto')
            axarr[i, 0].set_xlim([0, max_len])
            axarr[i, 0].set_ylabel('Ray steps')
            axarr[i, 0].set_xlabel('Distance [px]')
            axarr[i, 1].set_title('Reconstructions')
            axarr[i, 1].imshow(compute_prior_map(cdist, step=10))

        fig.savefig(os.path.join(PATH_OUTPUT, 'RG2Sp_shape-modeling.pdf'))

    def test_region_growing_greedy(self, name='insitu7545'):
        """    """
        if not os.path.exists(PATH_PKL_MODEL):
            self.test_shape_modeling()

        # file_model = pickle.load(open(PATH_PKL_MODEL, 'r'))
        npz_file = np.load(PATH_PKL_MODEL)
        file_model = dict(npz_file[npz_file.files[0]].tolist())
        logging.info('loaded model: %s', repr(file_model.keys()))
        list_mean_cdf = file_model['cdfs']
        model = file_model['mix_model']

        img, seg, slic, centers, annot = load_inputs(name)

        dict_debug = {}
        labels_greedy = tl_rg.region_growing_shape_slic_greedy(
            seg, slic, centers, (model, list_mean_cdf), 'set_cdfs',
            LABELS_FG_PROB, coef_shape=5., coef_pairwise=15.,
            prob_label_trans=[0.1, 0.03], greedy_tol=3e-1, allow_obj_swap=True,
            dict_thresholds=DEFAULT_RG2SP_THRESHOLDS, nb_iter=250,
            dict_debug_history=dict_debug)

        segm_obj = labels_greedy[slic]
        logging.info('debug: %s', repr(dict_debug.keys()))

        FIG_SIZE = (12. * np.array(img.shape[:2]) / np.max(img.shape))
        for i in np.linspace(0, len(dict_debug['energy']) - 1, 5):
            fig, ax = plt.subplots(figsize=FIG_SIZE[::-1])
            tl_visu.draw_rg2sp_results(ax, seg, slic, dict_debug, int(i))
            fig_name = 'RG2Sp_greedy_%s_debug-%03d.pdf' % (name, i)
            fig.savefig(os.path.join(PATH_OUTPUT, fig_name))
            plt.close(fig)

        score = adjusted_rand_score(annot.ravel(), segm_obj.ravel())
        self.assertGreaterEqual(score, 0.5)

        expert_segm(name, img, seg, segm_obj, annot, type='RG2Sp_greedy')

    def test_region_growing_graphcut(self, name='insitu7545'):
        """    """
        if not os.path.exists(PATH_PKL_MODEL):
            self.test_shape_modeling()

        # file_model = pickle.load(open(PATH_PKL_MODEL, 'r'))
        npz_file = np.load(PATH_PKL_MODEL)
        file_model = dict(npz_file[npz_file.files[0]].tolist())
        logging.info('loaded model: %s', repr(file_model.keys()))
        list_mean_cdf = file_model['cdfs']
        model = file_model['mix_model']

        img, _ = tl_io.load_image_2d(os.path.join(PATH_IMAGE, name + '.jpg'))
        seg, _ = tl_io.load_image_2d(os.path.join(PATH_SEGM, name + '.png'))
        annot, _ = tl_io.load_image_2d(os.path.join(PATH_ANNOT, name + '.png'))
        centers = pd.DataFrame.from_csv(
            os.path.join(PATH_CENTRE, name + '.csv')).values
        centers[:, [0, 1]] = centers[:, [1, 0]]

        slic = tl_spx.segment_slic_img2d(img, sp_size=25, rltv_compact=0.3)

        dict_debug = {}
        labels_gc = tl_rg.region_growing_shape_slic_graphcut(
            seg, slic, centers, (model, list_mean_cdf), 'set_cdfs',
            LABELS_FG_PROB, coef_shape=5., coef_pairwise=15.,
            prob_label_trans=[0.1, 0.03], optim_global=False, nb_iter=65,
            allow_obj_swap=False, dict_thresholds=DEFAULT_RG2SP_THRESHOLDS,
            dict_debug_history=dict_debug)

        segm_obj = labels_gc[slic]
        logging.info('debug: %s', repr(dict_debug.keys()))

        for i in np.linspace(0, len(dict_debug['energy']) - 1, 5):
            fig = tl_visu.figure_rg2sp_debug_complete(seg, slic, dict_debug,
                                                      int(i), max_size=5)
            fig_name = 'RG2Sp_graph-cut_%s_debug-%03d.pdf' % (name, i)
            fig.savefig(os.path.join(PATH_OUTPUT, fig_name))
            plt.close(fig)

        score = adjusted_rand_score(annot.ravel(), segm_obj.ravel())
        self.assertGreaterEqual(score, 0.5)

        expert_segm(name, img, seg, segm_obj, annot, type='RG2Sp_graph-cut')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

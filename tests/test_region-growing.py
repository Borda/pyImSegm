"""
Unit testing for particular segmentation module

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
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

sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Add path to root
from imsegm.utilities.data_io import update_path, load_image_2d
from imsegm.utilities.drawing import draw_rg2sp_results, figure_rg2sp_debug_complete
from imsegm.superpixels import segment_slic_img2d
from imsegm.region_growing import (
    RG2SP_THRESHOLDS, compute_shape_prior_table_cdf, compute_object_shapes,
    transform_rays_model_sets_mean_cdf_mixture, compute_segm_prob_fg,
    region_growing_shape_slic_greedy, region_growing_shape_slic_graphcut)

PATH_OVARY = os.path.join(update_path('data_images', absolute=True),
                          'drosophila_ovary_slice')
PATH_IMAGE = os.path.join(PATH_OVARY, 'image')
PATH_SEGM = os.path.join(PATH_OVARY, 'segm')
PATH_ANNOT = os.path.join(PATH_OVARY, 'annot_eggs')
PATH_CENTRE = os.path.join(PATH_OVARY, 'center_levels')
PATH_OUTPUT = update_path('output', absolute=True)
NAME_RG2SP_MODEL = 'RG2SP_multi-model_mixture.npz'
PATH_PKL_MODEL = os.path.join(PATH_OUTPUT, NAME_RG2SP_MODEL)
LABELS_FG_PROB = (0.05, 0.7, 0.9, 0.9)
DEFAULT_RG2SP_THRESHOLDS = RG2SP_THRESHOLDS
FIG_SIZE = 12.


def compute_prior_map(cdist, size=(500, 800), step=5):
    prior_map = np.zeros(size)
    centre = np.array(size) / 2
    for i in np.arange(prior_map.shape[0], step=step):
        for j in np.arange(prior_map.shape[1], step=step):
            prior_map[i:i + step, j:j + step] = \
                compute_shape_prior_table_cdf([i, j], cdist, centre, angle_shift=0)
    return prior_map


def load_inputs(name):
    img, _ = load_image_2d(os.path.join(PATH_IMAGE, name + '.jpg'))
    seg, _ = load_image_2d(os.path.join(PATH_SEGM, name + '.png'))
    annot, _ = load_image_2d(os.path.join(PATH_ANNOT, name + '.png'))
    centers = pd.read_csv(os.path.join(PATH_CENTRE, name + '.csv'),
                          index_col=0).values
    centers[:, [0, 1]] = centers[:, [1, 0]]

    slic = segment_slic_img2d(img, sp_size=25, relative_compact=0.3)
    return img, seg, slic, centers, annot


def expert_segm(name, img, seg, segm_obj, annot, str_type='xxx'):
    fig_size = (FIG_SIZE * np.array(img.shape[:2]) / np.max(img.shape))
    fig, ax = plt.subplots(ncols=2, figsize=(fig_size[1] * 2, fig_size[0]))
    ax[0].set_title('Region growing - segmentation')
    ax[0].imshow(segm_obj, cmap=plt.cm.jet)
    ax[0].contour(seg, levels=np.unique(seg), colors='#bfbfbf')
    ax[1].set_title('Annotation')
    ax[1].imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
    ax[1].contour(annot, cmap=plt.cm.jet)
    fig.savefig(os.path.join(PATH_OUTPUT, '%s_%s.pdf' % (str_type, name)),
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


class TestRegionGrowing(unittest.TestCase):

    def test_shape_modeling(self, dir_annot=PATH_ANNOT):
        """    """
        list_paths = sorted(glob.glob(os.path.join(dir_annot, '*.png')))
        logging.info('nb images: %i SAMPLES: %r', len(list_paths),
                     [os.path.basename(p) for p in list_paths[:5]])
        list_segms = []
        for path_seg in list_paths:
            seg, _ = load_image_2d(path_seg)
            list_segms.append(seg)

        list_rays, _ = compute_object_shapes(list_segms, ray_step=25, smooth_coef=1,
                                             interp_order='spline')
        logging.info('nb eggs: %i nb rays: %i',
                     len(list_rays), len(list_rays[0]))

        model, list_mean_cdf = transform_rays_model_sets_mean_cdf_mixture(list_rays, 2)

        np.savez(PATH_PKL_MODEL, data={'name': 'set_cdfs',
                                       'cdfs': list_mean_cdf,
                                       'mix_model': model})
        # with open(PATH_PKL_MODEL, 'w') as fp:
        #     pickle.dump({'name': 'set_cdfs',
        #                  'cdfs': list_mean_cdf,
        #                  'mix_model': model}, fp)
        self.assertTrue(os.path.exists(PATH_PKL_MODEL))

        max_len = max([np.asarray(mc[1]).shape[1] for mc in list_mean_cdf])

        fig, axarr = plt.subplots(nrows=len(list_mean_cdf), ncols=2,
                                  figsize=(12, 3.5 * len(list_mean_cdf)))
        for i, (_, list_cdf) in enumerate(list_mean_cdf):
            cdist = np.zeros((len(list_cdf), max_len))
            cdist[:, :len(list_cdf[0])] = np.array(list_cdf)
            axarr[i, 0].imshow(cdist, aspect='auto')
            axarr[i, 0].set(title='Inverse cumulative distribution',
                            ylabel='Ray steps', xlabel='Distance [px]',
                            xlim=[0, max_len])
            axarr[i, 1].set_title('Reconstructions')
            axarr[i, 1].imshow(compute_prior_map(cdist, step=10))

        fig.savefig(os.path.join(PATH_OUTPUT, 'RG2Sp_shape-modeling.pdf'),
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def test_region_growing_greedy(self, name='insitu7545'):
        """    """
        if not os.path.exists(PATH_PKL_MODEL):
            self.test_shape_modeling()

        # file_model = pickle.load(open(PATH_PKL_MODEL, 'r'))
        npz_file = np.load(PATH_PKL_MODEL)
        file_model = dict(npz_file[npz_file.files[0]].tolist())
        logging.info('loaded model: %r', file_model.keys())
        list_mean_cdf = file_model['cdfs']
        model = file_model['mix_model']

        img, seg, slic, centers, annot = load_inputs(name)

        dict_debug = {}
        slic_prob_fg = compute_segm_prob_fg(slic, seg, LABELS_FG_PROB)
        labels_greedy = region_growing_shape_slic_greedy(
            slic, slic_prob_fg, centers, (model, list_mean_cdf), 'set_cdfs',
            coef_shape=5., coef_pairwise=15., prob_label_trans=[0.1, 0.03],
            greedy_tol=3e-1, allow_obj_swap=True,
            dict_thresholds=DEFAULT_RG2SP_THRESHOLDS, nb_iter=250,
            debug_history=dict_debug)

        segm_obj = labels_greedy[slic]
        logging.info('show debug: %r', dict_debug.keys())

        fig_size = (FIG_SIZE * np.array(img.shape[:2]) / np.max(img.shape))
        for i in np.linspace(0, len(dict_debug['criteria']) - 1, 5):
            fig, ax = plt.subplots(figsize=fig_size[::-1])
            draw_rg2sp_results(ax, seg, slic, dict_debug, int(i))
            fig_name = 'RG2Sp_greedy_%s_debug-%03d.pdf' % (name, i)
            fig.savefig(os.path.join(PATH_OUTPUT, fig_name),
                        bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        score = adjusted_rand_score(annot.ravel(), segm_obj.ravel())
        self.assertGreaterEqual(score, 0.5)

        expert_segm(name, img, seg, segm_obj, annot, str_type='RG2Sp_greedy')

    def test_region_growing_graphcut(self, name='insitu7545'):
        """    """
        if not os.path.exists(PATH_PKL_MODEL):
            self.test_shape_modeling()

        # file_model = pickle.load(open(PATH_PKL_MODEL, 'r'))
        npz_file = np.load(PATH_PKL_MODEL)
        file_model = dict(npz_file[npz_file.files[0]].tolist())
        logging.info('loaded model: %r', file_model.keys())
        list_mean_cdf = file_model['cdfs']
        model = file_model['mix_model']

        img, _ = load_image_2d(os.path.join(PATH_IMAGE, name + '.jpg'))
        seg, _ = load_image_2d(os.path.join(PATH_SEGM, name + '.png'))
        annot, _ = load_image_2d(os.path.join(PATH_ANNOT, name + '.png'))
        centers = pd.read_csv(os.path.join(PATH_CENTRE, name + '.csv'),
                              index_col=0).values
        centers[:, [0, 1]] = centers[:, [1, 0]]

        slic = segment_slic_img2d(img, sp_size=25, relative_compact=0.3)
        slic_prob_fg = compute_segm_prob_fg(slic, seg, LABELS_FG_PROB)

        dict_debug = {}
        labels_gc = region_growing_shape_slic_graphcut(
            slic, slic_prob_fg, centers, (model, list_mean_cdf), 'set_cdfs',
            coef_shape=5., coef_pairwise=15., prob_label_trans=[0.1, 0.03],
            optim_global=False, nb_iter=65, allow_obj_swap=False,
            dict_thresholds=DEFAULT_RG2SP_THRESHOLDS, debug_history=dict_debug)

        segm_obj = labels_gc[slic]
        logging.info('show debug: %r', dict_debug.keys())

        for i in np.linspace(0, len(dict_debug['criteria']) - 1, 5):
            fig = figure_rg2sp_debug_complete(seg, slic, dict_debug, int(i), max_size=5)
            fig_name = 'RG2Sp_graph-cut_%s_debug-%03d.pdf' % (name, i)
            fig.savefig(os.path.join(PATH_OUTPUT, fig_name),
                        bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        score = adjusted_rand_score(annot.ravel(), segm_obj.ravel())
        self.assertGreaterEqual(score, 0.5)

        expert_segm(name, img, seg, segm_obj, annot, str_type='RG2Sp_graph-cut')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

"""
Unit testing for particular segmentation module

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging
import os
import sys
import copy
import unittest

import matplotlib.pyplot as plt
from skimage.transform import resize

sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Add path to root
import imsegm.descriptors
from imsegm.utilities.data_samples import (load_sample_image, IMAGE_OBJECTS, IMAGE_STAR,
                                           IMAGE_DROSOPHILA_OVARY_2D, ANNOT_DROSOPHILA_OVARY_2D)
from imsegm.utilities.data_io import update_path
from imsegm.utilities.drawing import figure_image_segm_results, figure_segm_graphcut_debug
from imsegm.pipelines import (
    estim_model_classes_group, pipe_color2d_slic_features_model_graphcut,
    train_classif_color2d_slic_features, segment_color2d_slic_features_model_graphcut)
from imsegm.descriptors import FEATURES_SET_TEXTURE_SHORT

PATH_OUTPUT = update_path('output', absolute=True)
# set default feature extracted from image
FEATURES_TEXTURE = FEATURES_SET_TEXTURE_SHORT
DEFAULT_SEGM_PARAMS = {'nb_classes': 2,
                       'dict_features': {'color': ['mean']}}
imsegm.descriptors.USE_CYTHON = False


def show_segm_results_2d(img, seg, path_dir, fig_name='temp-segm_.png'):
    """ show and expert segmentation results

    :param ndarray img: input image
    :param ndarray seg: resulting segmentation
    :param str path_dir: path to the visualisations
    :param str fig_name: figure name
    """
    fig = figure_image_segm_results(img, seg)
    path_fig = os.path.join(path_dir, fig_name)
    fig.savefig(path_fig, bbox_inches='tight', pad_inches=0.1)

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        plt.show()
    plt.close(fig)


def show_segm_debugs_2d(images, path_dir, fig_name='temp-debug_.png'):
    """ show and expert partial segmentation results

    :param dict images:
    :param str path_dir: path to the visualisations
    :param str fig_name: figure name
    """
    if not images:
        return
    fig = figure_segm_graphcut_debug(images)

    path_fig = os.path.join(path_dir, fig_name)
    fig.savefig(path_fig, bbox_inches='tight', pad_inches=0.1)

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        plt.show()
    plt.close(fig)


def run_segm2d_gmm_gc(img2d, dir_name, params, types_edge=('model', 'const'),
                      list_regul=(0, 0.5, 1, 3, 5, 10)):
    """ perform several experiment with different edge type and GC regul.

    :param ndarray img2d: input image
    :param str dir_name: create the folder in output path
    :param list(str) types_edge: list of performed edge types
    :param dict params: segmentation parameters
    :param list(float) list_regul: list of performed edge types
    """
    path_dir = os.path.join(PATH_OUTPUT, dir_name)
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)

    model, _ = estim_model_classes_group([img2d], model_type='GMM', **params)
    params.pop('nb_classes', None)
    params.pop('pca_coef', None)

    for edge in types_edge:
        dict_imgs = dict()
        for regul in list_regul:
            seg, _ = segment_color2d_slic_features_model_graphcut(
                img2d, model, gc_regul=regul, gc_edge_type=edge,
                debug_visual=dict_imgs, **params)

            show_segm_debugs_2d(dict_imgs, path_dir,
                                'fig_regul-%.2f_edge-%s_debug.png' % (regul, edge))
            show_segm_results_2d(img2d, seg, path_dir,
                                 'fig_regul-%.2f_edge-%s.png' % (regul, edge))
            dict_imgs = None


class TestPipelinesGMM(unittest.TestCase):

    img_obj = load_sample_image(IMAGE_OBJECTS)
    img_star = load_sample_image(IMAGE_STAR)
    # img_islet = load_sample_image(IMAGE_LANGER_ISLET)
    # img_histo = load_sample_image(IMAGE_HISTOL_FLAGSHIP)
    # img_disc = load_sample_image(IMAGE_DROSOPHILA_DISC)
    # img_ovary = load_sample_image(IMAGE_DROSOPHILA_OVARY_2D)
    # img3d = get_image_path(IMAGE_DROSOPHILA_OVARY_3D)

    def test_segm_gmm_gc_objects(self):
        img = resize(self.img_obj, output_shape=(256, 256))
        logging.debug('dimension: {}'.format(img.shape))

        dict_imgs = dict()
        path_dir = os.path.join(PATH_OUTPUT, 'temp_segm-gmm-gc-objects')
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)
        seg, _ = pipe_color2d_slic_features_model_graphcut(
            img, nb_classes=4, dict_features={'color': ['mean']},
            sp_size=20, sp_regul=0.2, gc_regul=1., gc_edge_type='model',
            debug_visual=dict_imgs)

        show_segm_debugs_2d(dict_imgs, path_dir,
                            'fig_regul-%.2f_edge-%s_debug.png' % (1., 'model'))
        show_segm_results_2d(img, seg, path_dir,
                             'fig_regul-%.2f_edge-%s.png' % (1., 'model'))

    def test_segm_gmm_gc_stars(self):
        img = self.img_star
        logging.debug('dimension: {}'.format(img.shape))
        params = copy.deepcopy(DEFAULT_SEGM_PARAMS)
        params.update(dict(nb_classes=3, sp_regul=0.2, sp_size=25,
                           dict_features={'color': ['mean', 'std']}))
        run_segm2d_gmm_gc(img, 'temp_segm-gmm-gc-stars', params=params)

    # def sample_segm_gmm_gc_langer(self):
    #     img = resize(self.img_islet, (512, 512))
    #     params = copy.deepcopy(DEFAULT_SEGM_PARAMS)
    #     params.update(dict(sp_regul=0.15, sp_size=5))
    #     run_segm2d_gmm_gc(img, 'temp_segm-gmm-gc-langer', types_edge=['model_lT'],
    #                       list_regul=[0, 1], params=params)

    # def sample_segm_gmm_gc_histo(self):
    #     img = resize(self.img_histo, (512, 512))
    #     params = copy.deepcopy(DEFAULT_SEGM_PARAMS)
    #     params.update(dict(sp_regul=0.15, sp_size=15, pca_coef=0.98))
    #     run_segm2d_gmm_gc(img, 'temp_segm-gmm-gc-histology', types_edge=['model'],
    #                       list_regul=[0, 1, 5], params=params)

    # def sample_segm_gmm_gc_disc(self):
    #     img = resize(self.img_disc, (512, 512))
    #     params = copy.deepcopy(DEFAULT_SEGM_PARAMS)
    #     params.update(dict(sp_regul=0.2, sp_size=15, pca_coef=0.98))
    #     run_segm2d_gmm_gc(img, 'temp_segm-gmm-gc-disc', types_edge=['model_l2'],
    #                       list_regul=[0, 1, 5], params=params)

    # def sample_segm_gmm_gc_ovary_2d(self):
    #     img = resize(self.img_ovary[:, :, 0], (512, 512))
    #     # img = np.rollaxis(np.tile(img[:, :, 0], (3, 1, 1)), 0, 3)
    #     params = copy.deepcopy(DEFAULT_SEGM_PARAMS)
    #     params.update(dict(nb_classes=4, pca_coef=0.95, sp_regul=0.3, sp_size=10,
    #                        dict_features=FEATURES_SET_TEXTURE_SHORT))
    #     run_segm2d_gmm_gc(img, 'temp_segm-gmm-gc-ovary-2d',
    #                       list_regul=[0, 2, 10], params=params)

    # def sample_segm_gmm_gc_ovary_3d(self):
    #     # _ = self.img3d
    #     # TODO, add extension to 3D
    #     # seg = pipelines.pipe_gray3d_slic_features_model_graphcut(img)
    #     pass


class TestPipelinesClassif(unittest.TestCase):

    img = load_sample_image(IMAGE_DROSOPHILA_OVARY_2D)[:, :, 0]
    annot = load_sample_image(ANNOT_DROSOPHILA_OVARY_2D)
    img2 = load_sample_image(IMAGE_DROSOPHILA_OVARY_2D)[:, :, 0]

    def test_segm_supervised(self):
        img = resize(self.img, output_shape=(256, 256))
        annot = resize(self.annot, output_shape=(256, 256), order=0,
                       preserve_range=True).astype(int)
        img2 = resize(self.img2, output_shape=(256, 256))

        path_dir = os.path.join(PATH_OUTPUT, 'temp_segm-supervised_gc')
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)

        sp_size = 10
        tp_edge = ['model', 'const']
        list_regul = [0, 1, 10]
        dict_imgs = dict()
        name = 'fig-img%i_regul-%.2f_edge-%s.png'

        classif, _, _, _ = train_classif_color2d_slic_features(
            [img], [annot], sp_size=sp_size, dict_features=FEATURES_TEXTURE)

        segment_color2d_slic_features_model_graphcut(
            img, classif, sp_size=sp_size, gc_regul=0.,
            dict_features=FEATURES_TEXTURE, debug_visual=dict_imgs)
        show_segm_debugs_2d(dict_imgs, path_dir, name % (1, 0, '_debug'))

        for edge in tp_edge:
            dict_imgs = dict()
            for regul in list_regul:
                seg, _ = segment_color2d_slic_features_model_graphcut(
                    img, classif, dict_features=FEATURES_TEXTURE,
                    sp_size=sp_size, gc_regul=regul, gc_edge_type=edge)
                show_segm_results_2d(img, seg, path_dir, name % (1, regul, edge))

                seg, _ = segment_color2d_slic_features_model_graphcut(
                    img2, classif, dict_features=FEATURES_TEXTURE,
                    sp_size=sp_size, gc_regul=regul, gc_edge_type=edge,
                    debug_visual=dict_imgs)
                show_segm_results_2d(img2, seg, path_dir, name % (2, regul, edge))
                show_segm_debugs_2d(dict_imgs, path_dir, name % (2, regul, edge))
                dict_imgs = None


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

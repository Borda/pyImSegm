"""
Unit testing for particular segmentation module

Copyright (C) 2014-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging
import os
import sys
import unittest

import matplotlib.pyplot as plt
from scipy.misc import imresize

sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Add path to root
import segmentation.utils.data_samples as d_spl
import segmentation.utils.data_io as tl_io
import segmentation.utils.drawing as tl_visu
import segmentation.pipelines as pipelines
import segmentation.descriptors as seg_fts

PATH_OUTPUT = tl_io.update_path(os.path.join('output'))
# set default feature extracted from image
FEATURES_TEXTURE = seg_fts.FEATURES_SET_TEXTURE_SHORT
seg_fts.USE_CYTHON = False


def show_segm_results_2d(img, seg, path_dir, fig_name='test_segm_.png'):
    """ show and expert segmentation results

    :param ndarray img: input image
    :param ndarray seg: resulting segmentation
    :param str path_dir: path to the visualisations
    :param str fig_name: figure name
    """
    fig = tl_visu.figure_image_segm_results(img, seg)
    path_fig = os.path.join(path_dir, fig_name)
    fig.savefig(path_fig,  bbox_inches='tight', pad_inches=0.1)

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        plt.show()
    plt.close(fig)


def show_segm_debugs_2d(dict_imgs, path_dir, fig_name='test_debug_.png'):
    """ show and expert partial segmettaion results

    :param {str: ...} dict_imgs:
    :param str path_dir: path to the visualisations
    :param str fig_name: figure name
    """
    if dict_imgs is None:
        return
    fig = tl_visu.figure_segm_graphcut_debug(dict_imgs)

    path_fig = os.path.join(path_dir, fig_name)
    fig.savefig(path_fig, bbox_inches='tight', pad_inches=0.1)

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        plt.show()
    plt.close(fig)


def run_segm2d_gmm_gc(img2d, dir_name, types_edge=('model', 'const'),
                      list_regul=(0, 0.5, 1, 3, 5, 10),
                      dict_params=None):
    """ perform several experiment with different edge type and GC regul.

    :param ndarray img2d: input image
    :param str dir_name: create the folder in output path
    :param [str] types_edge: list of performed edge types
    :param [float] list_regul: list of performed edge types
    :param {str: ...} dict_params: segmentation parameters
    :return:
    """
    path_dir = os.path.join(PATH_OUTPUT, dir_name)
    if dict_params is None:
        dict_params = dict()
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    for edge in types_edge:
        dict_imgs = dict()
        for regul in list_regul:
            seg = pipelines.pipe_color2d_slic_features_gmm_graphcut(
                img2d, gc_regul=regul, gc_edge_type=edge,
                dict_debug_imgs=dict_imgs, **dict_params)
            show_segm_debugs_2d(dict_imgs, path_dir,
                        'fig_regul-%.2f_edge-%s_debug.png' % (regul, edge))
            show_segm_results_2d(img2d, seg, path_dir,
                         'fig_regul-%.2f_edge-%s.png' % (regul, edge))
            dict_imgs = None


class TestPipelinesGMM(unittest.TestCase):

    def test_segm_gmm_gc_objects(self):
        img = d_spl.load_sample_image(d_spl.IMAGE_OBJECTS)
        img = imresize(img, (256, 256))
        logging.debug('dimension: {}'.format(img.shape))
        params = {'nb_classes': 4,
                  'sp_size': 20, 'sp_regul': 0.2,
                  'dict_features': {'color': ['mean']}}
        run_segm2d_gmm_gc(img, 'test_segm_gmm_gc_objects',
                          list_regul=[0, 1, 5, 10],
                          dict_params=params)

    def test_segm_gmm_gc_stars(self):
        img = d_spl.load_sample_image(d_spl.IMAGE_STAR_2)
        logging.debug('dimension: {}'.format(img.shape))
        params = {'nb_classes': 3, 'pca_coef': 0.98,
                  'sp_regul': 0.2, 'sp_size': 25,
                  'dict_features': {'color': ['mean', 'std']}}
        run_segm2d_gmm_gc(img, 'test_segm_gmm_gc_stars', dict_params=params)

    # def test_segm_gmm_gc_langer(self):
    #     img = d_spl.load_sample_image(d_spl.IMAGE_LANGER_ISLET)
    #     img = imresize(img, (512, 512))
    #
    #     run_segm2d_gmm_gc(img, 'test_segm_gmm_gc_langer', list_regul=[0, 1, 5],
    #                       dict_params={'sp_regul': 0.15, 'sp_size': 5})

    # def test_segm_gmm_gc_histo(self):
    #     img = d_spl.load_sample_image(d_spl.IMAGE_HISTOL_FLAGSHIP)
    #     img = imresize(img, (512, 512))
    #     params = {'sp_regul': 0.15, 'sp_size': 15,
    #               'pca_coef': 0.98}
    #     run_segm2d_gmm_gc(img, 'test_segm_gmm_gc_histology',
    #                       types_edge=['model'], list_regul=[0, 1, 5],
    #                       dict_params=params)

    def test_segm_gmm_gc_disc(self):
        img = d_spl.load_sample_image(d_spl.IMAGE_DROSOPHILA_DISC)
        img = imresize(img, (512, 512))
        params = {'sp_regul': 0.2, 'sp_size': 15,
                  'pca_coef': 0.98}
        run_segm2d_gmm_gc(img, 'test_segm_gmm_gc_disc',
                          types_edge=['model_l2'], list_regul=[0, 1, 5],
                          dict_params=params)

    def test_segm_gmm_gc_ovary_2d(self):
        img = d_spl.load_sample_image(d_spl.IMAGE_DROSOPHILA_OVARY_2D)[:, :, 0]
        img = imresize(img, (512, 512))
        # img = np.rollaxis(np.tile(img[:, :, 0], (3, 1, 1)), 0, 3)
        params = {'nb_classes': 4, 'pca_coef': 0.95,
                  'sp_regul': 0.3, 'sp_size': 10,
                  'dict_features': seg_fts.FEATURES_SET_TEXTURE}
        run_segm2d_gmm_gc(img, 'test_segm_gmm_gc_ovary_2d',
                          list_regul=[0, 2, 10],
                          dict_params=params)

    def test_segm_gmm_gc_drosophila_3d(self):
        path_img = d_spl.get_image_path(d_spl.IMAGE_DROSOPHILA_OVARY_3D)
        # TODO, add extension to 3D
        # seg = pipelines.pipe_gray3d_slic_features_gmm_graphcut(img)


class TestPipelinesClassif(unittest.TestCase):

    def test_segm_supervised(self):
        img = d_spl.load_sample_image(d_spl.IMAGE_DROSOPHILA_OVARY_2D)[:, :, 0]
        img = imresize(img, (256, 256))
        annot = d_spl.load_sample_image(d_spl.ANNOT_DROSOPHILA_OVARY_2D)
        annot = imresize(annot, (256, 256), interp='nearest')
        img2 = d_spl.load_sample_image(d_spl.IMAGE_DROSOPHILA_OVARY_2D)[:, :, 0]
        img2 = imresize(img2, (256, 256))

        path_dir = os.path.join(PATH_OUTPUT, 'test_segm_supervised_gc')
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)

        sp_size = 10
        tp_edge = ['model', 'const']
        list_regul = [0, 1, 10]
        dict_imgs = dict()
        name = 'fig-img%i_regul-%.2f_edge-%s.png'

        classif, _, _, _ = pipelines.train_classif_color2d_slic_features(
            [img], [annot], sp_size, dict_features=FEATURES_TEXTURE)

        _ = pipelines.segment_color2d_slic_features_classif_graphcut(
                    img, classif, sp_size=sp_size, gc_regul=0.,
                    dict_features=FEATURES_TEXTURE, dict_debug_imgs=dict_imgs)
        show_segm_debugs_2d(dict_imgs, path_dir, name % (1, 0, '_debug'))

        for edge in tp_edge:
            dict_imgs = dict()
            for regul in list_regul:
                seg = pipelines.segment_color2d_slic_features_classif_graphcut(
                    img, classif, sp_size=sp_size, gc_regul=regul, gc_edge_type=edge,
                    dict_features=FEATURES_TEXTURE)
                show_segm_results_2d(img, seg, path_dir, name % (1, regul, edge))

                seg = pipelines.segment_color2d_slic_features_classif_graphcut(
                    img2, classif, sp_size=sp_size, gc_regul=regul, gc_edge_type=edge,
                    dict_features=FEATURES_TEXTURE, dict_debug_imgs=dict_imgs)
                show_segm_results_2d(img2, seg, path_dir, name % (2, regul, edge))
                show_segm_debugs_2d(dict_imgs, path_dir, name % (2, regul, edge))
                dict_imgs = None


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

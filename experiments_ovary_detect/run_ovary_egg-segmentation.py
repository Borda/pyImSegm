"""
Run experiments with several segmentation techniques for instance segmentation

Require installation of Morph. Snakes - https://github.com/Borda/morph-snakes ::

    pip install --user git+https://github.com/Borda/morph-snakes.git

Sample usage::

    python run_ovary_egg-segmentation.py \
        -list data_images/drosophila_ovary_slice/list_imgs-segm-center-points.csv \
        -out results -n ovary_slices --nb_workers 1 \
        -m ellipse_moments \
           ellipse_ransac_mmt \
           ellipse_ransac_crit \
           GC_pixels-large \
           GC_pixels-shape \
           GC_slic-shape \
           rg2sp_greedy-mixture \
           rg2sp_GC-mixture \
           watershed_morph

Copyright (C) 2016-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import time
import argparse
import logging
import pickle
import multiprocessing as mproc
from functools import partial

import matplotlib
if os.environ.get('DISPLAY', '') == '' and matplotlib.rcParams['backend'] != 'agg':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import segmentation, morphology
from skimage import measure, draw
# from sklearn.externals import joblib
# from sklearn import metrics, cross_validation
from skimage.measure.fit import EllipseModel

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.experiments as tl_expt
import imsegm.utilities.drawing as tl_visu
import imsegm.superpixels as seg_spx
import imsegm.region_growing as seg_rg
import imsegm.ellipse_fitting as ell_fit
from morphsnakes import morphsnakes, multi_snakes
# from libs import chanvese

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))
NAME_EXPERIMENT = 'experiment_egg-segment'
TYPE_LOAD_IMAGE = '2d_struct'
DIR_VISUAL_POSIX = '___visu'
DIR_CENTRE_POSIX = '___centres'
DIR_DEBUG_POSIX = '___debug'

# setting default file names
NAME_FIG_LABEL_HISTO = 'fig_histo_annot_segments.png'
NAME_CSV_SEGM_STAT_SLIC_ANNOT = 'statistic_segm_slic_annot.csv'
NAME_CSV_SEGM_STAT_RESULT = 'statistic_segm_results.csv'
NAME_CSV_SEGM_STAT_RESULT_GC = 'statistic_segm_results_gc.csv'

EACH_UNIQUE_EXPERIMENT = False
INIT_MASK_BORDER = 50.
# minimal diameter for estimating ellipse
MIN_ELLIPSE_DAIM = 25.
# subfigure size for experting images
MAX_FIGURE_SIZE = 14
# threshold if two segmentation overlap more, keep just one of them
SEGM_OVERLAP = 0.5
# paramters for SLIC segmentation
SLIC_SIZE = 40
SLIC_REGUL = 0.3
# Region Growing configuration
DEBUG_EXPORT = False

RG2SP_THRESHOLDS = {  # thresholds for updating between iterations
    'centre': 20,
    'shift': 10,
    'volume': 0.05,
    'centre_init': 50
}
COLUMNS_ELLIPSE = ('xc', 'yc', 'a', 'b', 'theta')

PATH_DATA = tl_data.update_path('data_images', absolute=True)
PATH_IMAGES = os.path.join(PATH_DATA, 'drosophila_ovary_slice')
# sample segmentation methods
LIST_SAMPLE_METHODS = (
    'ellipse_moments', 'ellipse_ransac_mmt', 'ellipse_ransac_crit',
    'GC_pixels-large', 'GC_pixels-shape', 'GC_slic-large', 'GC_slic-shape',
    'rg2sp_greedy-mixture', 'rg2sp_GC-mixture',
    'watershed_morph'
)
# default segmentation configuration
SEGM_PARAMS = {
    # ovary labels: background, funicular cells, nurse cells, cytoplasm
    'tab-proba_ellipse': [0.01, 0.95, 0.95, 0.85],
    'tab-proba_graphcut': [0.01, 0.6, 0.99, 0.75],
    'tab-proba_RG2SP': [0.01, 0.6, 0.95, 0.75],
    'path_single-model': os.path.join(PATH_DATA, 'RG2SP_eggs_single-model.pkl'),
    'path_multi-models': os.path.join(PATH_DATA, 'RG2SP_eggs_mixture-model.pkl'),
    'gc-pixel_regul': 3.,
    'gc-slic_regul': 2.,
    'RG2SP-shape': 5.,
    'RG2SP-pairwise': 3.,
    'RG2SP-swap': True,
    'label_trans': [0.1, 0.03],
    'overlap_theshold': SEGM_OVERLAP,
    'RG2SP_theshold': RG2SP_THRESHOLDS,
    'slic_size': SLIC_SIZE,
    'slic_regul': SLIC_REGUL,
    'path_list': os.path.join(PATH_IMAGES,
                              'list_imgs-segm-center-points_short.csv'),
    'path_out': tl_data.update_path('results', absolute=True)
}


def arg_parse_params(params):
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return {str: str}:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-list', '--path_list', type=str, required=False,
                        help='path to the list of image',
                        default=params['path_list'])
    parser.add_argument('-out', '--path_out', type=str, required=False,
                        help='path to the output directory',
                        default=params['path_out'])
    parser.add_argument('-n', '--name', type=str, required=False,
                        help='name of the experiment', default='ovary')
    parser.add_argument('-cfg', '--path_config', type=str, required=False,
                        help='path to the configuration', default=None)
    parser.add_argument('--nb_workers', type=int, required=False, default=NB_THREADS,
                        help='number of processes in parallel')
    parser.add_argument('-m', '--methods', type=str, required=False, nargs='+',
                        help='list of segment. methods', default=None)
    arg_params = vars(parser.parse_args())
    params.update(arg_params)
    if not isinstance(arg_params['path_config'], str) \
            or arg_params['path_config'].lower() == 'none':
        params['path_config'] = ''
    else:
        params['path_config'] = tl_data.update_path(params['path_config'])
        assert os.path.isfile(params['path_config']), \
            'missing file: %s' % params['path_config']
        ext = os.path.splitext(params['path_config'])[-1]
        assert (ext == '.yaml' or ext == '.yml'), \
            '"%s" should be YAML file' % os.path.basename(params['path_config'])
        data = tl_expt.load_config_yaml(params['path_config'])
        params.update(data)
        params.update(arg_params)
    for k in (k for k in arg_params if 'path' in k):
        if not arg_params[k]:
            continue
        params[k] = tl_data.update_path(arg_params[k], absolute=True)
        assert os.path.exists(params[k]), 'missing: %s' % params[k]
    # load saved configuration
    logging.info('ARG PARAMETERS: \n %r', params)
    return params


def load_image(path_img, img_type=TYPE_LOAD_IMAGE):
    """ load image from given path according specification

    :param str path_img:
    :param str img_type:
    :return ndarray:
    """
    path_img = os.path.abspath(os.path.expanduser(path_img))
    assert os.path.isfile(path_img), 'missing: "%s"' % path_img
    if img_type == 'segm':
        img = tl_data.io_imread(path_img)
    elif img_type == '2d_struct':
        img, _ = tl_data.load_img_double_band_split(path_img)
        assert img.ndim == 2, 'image can be only single color'
    else:
        logging.error('not supported loading img_type: %s', img_type)
        img = tl_data.io_imread(path_img)
    logging.debug('image shape: %r, value range %f - %f', img.shape,
                  img.min(), img.max())
    return img


def path_out_img(params, dir_name, name):
    return os.path.join(params['path_exp'], dir_name, name + '.png')


def export_draw_image_segm(path_fig, img, segm=None, segm_obj=None, centers=None):
    """ draw and export visualisation of image and segmentation

    :param str path_fig: path to the exported figure
    :param ndarray img:
    :param ndarray segm:
    :param ndarray segm_obj:
    :param ndarray centers:
    """
    size = np.array(img.shape[:2][::-1], dtype=float)
    fig, ax = plt.subplots(figsize=(size / size.max() * MAX_FIGURE_SIZE))
    ax.imshow(img, alpha=1., cmap=plt.cm.Greys)
    if segm is not None:
        ax.contour(segm)
    if segm_obj is not None:
        ax.imshow(segm_obj, alpha=0.1)
        assert len(np.unique(segm_obj)) < 1e2, \
            'too many labeled objects - %i' % len(np.unique(segm_obj))
        ax.contour(segm_obj, levels=np.unique(segm_obj).tolist(),
                   cmap=plt.cm.jet_r, linewidths=(10, ))
    if centers is not None:
        ax.plot(np.array(centers)[:, 1], np.array(centers)[:, 0], 'o', color='r')

    fig = tl_visu.figure_image_adjustment(fig, img.shape)

    fig.savefig(path_fig)
    plt.close(fig)


def segment_watershed(seg, centers, post_morph=False):
    """ perform watershed segmentation on input imsegm
    and optionally run some postprocessing using morphological operations

    :param ndarray seg: input image / segmentation
    :param [[int, int]] centers: position of centres / seeds
    :param bool post_morph: apply morphological postprocessing
    :return ndarray, [[int, int]]: resulting segmentation, updated centres
    """
    logging.debug('segment: watershed...')
    seg_binary = (seg > 0)
    seg_binary = ndimage.morphology.binary_fill_holes(seg_binary)
    # thr_area = int(0.05 * np.sum(seg_binary))
    # seg_binary = morphology.remove_small_holes(seg_binary, min_size=thr_area)
    distance = ndimage.distance_transform_edt(seg_binary)
    markers = np.zeros_like(seg)
    for i, pos in enumerate(centers):
        markers[int(pos[0]), int(pos[1])] = i + 1
    segm = morphology.watershed(-distance, markers, mask=seg_binary)

    # if morphological postprocessing was not selected, ends here
    if not post_morph:
        return segm, centers, None

    segm_clean = np.zeros_like(segm)
    for lb in range(1, np.max(segm) + 1):
        seg_lb = (segm == lb)
        # some morphology operartion for cleaning
        seg_lb = morphology.binary_closing(seg_lb, selem=morphology.disk(5))
        seg_lb = ndimage.morphology.binary_fill_holes(seg_lb)
        # thr_area = int(0.15 * np.sum(seg_lb))
        # seg_lb = morphology.remove_small_holes(seg_lb, min_size=thr_area)
        seg_lb = morphology.binary_opening(seg_lb, selem=morphology.disk(15))
        segm_clean[seg_lb] = lb
    return segm_clean, centers, None


def create_circle_center(img_shape, centers, radius=10):
    """ create initialisation from centres as small circles

    :param img_shape:
    :param [[int, int]] centers:
    :param int radius:
    :return:
    """
    mask_circle = np.zeros(img_shape, dtype=int)
    mask_perimeter = np.zeros(img_shape, dtype=int)
    center_circles = list()
    for i, pos in enumerate(centers):
        rr, cc = draw.circle(int(pos[0]), int(pos[1]), radius,
                             shape=img_shape[:2])
        mask_circle[rr, cc] = i + 1
        rr, cc = draw.circle_perimeter(int(pos[0]), int(pos[1]), radius,
                                       shape=img_shape[:2])
        mask_perimeter[rr, cc] = i + 1
        center_circles.append(np.array([rr, cc]).transpose())
    return center_circles, mask_circle, mask_perimeter


def segment_active_contour(img, centers):
    """ segmentation using acive contours

    :param ndarray img: input image / segmentation
    :param [[int, int]] centers: position of centres / seeds
    :return (ndarray, [[int, int]]): resulting segmentation, updated centres
    """
    logging.debug('segment: active_contour...')
    # http://scikit-image.org/docs/dev/auto_examples/edges/plot_active_contours.html
    segm = np.zeros(img.shape[:2])
    img_smooth = ndimage.filters.gaussian_filter(img, 5)
    center_circles, _, _ = create_circle_center(img.shape[:2], centers)
    for i, snake in enumerate(center_circles):
        snake = segmentation.active_contour(img_smooth, snake.astype(float),
                                            alpha=0.015, beta=10, gamma=0.001,
                                            w_line=0.0, w_edge=1.0,
                                            max_px_move=1.0,
                                            max_iterations=2500,
                                            convergence=0.2)
        seg = np.zeros(segm.shape, dtype=bool)
        x, y = np.array(snake).transpose().tolist()
        # rr, cc = draw.polygon(x, y)
        seg[map(int, x), map(int, y)] = True
        seg = morphology.binary_dilation(seg, selem=morphology.disk(3))
        bb_area = int((max(x) - min(x)) * (max(y) - min(y)))
        logging.debug('bounding box area: %d', bb_area)
        seg = morphology.remove_small_holes(seg, min_size=bb_area)
        segm[seg] = i + 1
    return segm, centers, None


def segment_morphsnakes(img, centers, init_center=True, smoothing=5,
                        lambdas=(3, 3), bb_dist=INIT_MASK_BORDER):
    """ segmentation using morphological snakes with some parameters

    :param ndarray img: input image / segmentation
    :param [[int, int]] centers: position of centres / seeds
    :param bool init_center:
    :param int smoothing:
    :param [int, int] lambdas:
    :param float bb_dist:
    :return (ndarray, [[int, int]]): resulting segmentation, updated centres
    """
    logging.debug('segment: morph-snakes...')
    if img.ndim == 3:
        img = img[:, :, 0]
    if init_center:
        _, mask, _ = create_circle_center(img.shape[:2], centers, radius=15)
    else:
        mask = np.zeros_like(img, dtype=int)
        mask[bb_dist:-bb_dist, bb_dist:-bb_dist] = 1
    # Morphological ACWE. Initialization of the level-set.
    params = dict(smoothing=smoothing, lambda1=lambdas[0], lambda2=lambdas[1])
    ms = multi_snakes.MultiMorphSnakes(img, mask, morphsnakes.MorphACWE, params)

    diag = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    ms.run(int(diag / 2.))
    segm = ms.levelset
    return segm, centers, None


# def segment_chanvese(img, centers, init_center=False, bb_dist=INIT_MASK_BORDER):
#     logging.debug('segment: chanvese...')
#     if img.ndim == 3:
#         img = img[:, :, 0]
#     if init_center:
#         _, mask, _ = create_circle_center(img.shape[:2], centers, radius=20)
#         init_mask = (mask > 0).astype(int)
#     else:
#         init_mask = np.zeros_like(img, dtype=int)
#         init_mask[bb_dist:-bb_dist, bb_dist:-bb_dist] = 1
#     nb_iter = int(sum(img.shape))
#     segm, phi, its = chanvese.chanvese(img, init_mask, alpha=0.2,
#                                        max_its=nb_iter, thresh=0)
#     segm = measure.label(segm)
#     return segm, centers, None


def segment_fit_ellipse(seg, centers, fn_preproc_points,
                        thr_overlap=SEGM_OVERLAP):
    """ segment eggs using ellipse fitting

    :param ndarray seg: input image / segmentation
    :param [[int, int]] centers: position of centres / seeds
    :param fn_preproc_points: function for detection boundary points
    :param float thr_overlap: threshold for removing overlapping segmentation
    :return (ndarray, [[int, int]]): resulting segmentation, updated centres
    """
    points_centers = fn_preproc_points(seg, centers)

    centres_new, ell_params = [], []
    segm = np.zeros_like(seg)
    for i, points in enumerate(points_centers):
        lb = i + 1
        ellipse = EllipseModel()
        ellipse.estimate(points)
        if not ellipse:
            continue
        logging.debug('ellipse params: %r', ellipse.params)
        segm = ell_fit.add_overlap_ellipse(segm, ellipse.params, lb, thr_overlap)

        if np.any(segm == lb):
            centres_new.append(centers[i])
            ell_params.append(ellipse.params)

    dict_export = {'ellipses.csv': pd.DataFrame(ell_params, columns=COLUMNS_ELLIPSE)}
    return segm, np.array(centres_new), dict_export


def segment_fit_ellipse_ransac(seg, centers, fn_preproc_points, nb_inliers=0.6,
                               thr_overlap=SEGM_OVERLAP):
    """ segment eggs using ellipse fitting and RANDSAC strategy

    :param ndarray seg: input image / segmentation
    :param [[int, int]] centers: position of centres / seeds
    :param fn_preproc_points: function for detection boundary points
    :param float nb_inliers: ratio of inliers for RANSAC
    :param float thr_overlap: threshold for removing overlapping segmentations
    :return (ndarray, [[int, int]]): resulting segmentation, updated centres
    """
    points_centers = fn_preproc_points(seg, centers)

    centres_new, ell_params = [], []
    segm = np.zeros_like(seg)
    for i, points in enumerate(points_centers):
        lb = i + 1
        nb_min = int(len(points) * nb_inliers)
        ransac_model, _ = measure.ransac(points, EllipseModel,
                                         min_samples=nb_min,
                                         residual_threshold=15,
                                         max_trials=250)
        if not ransac_model:
            continue
        logging.debug('ellipse params: %r', ransac_model.params)
        segm = ell_fit.add_overlap_ellipse(segm, ransac_model.params, lb,
                                           thr_overlap)

        if np.any(segm == lb):
            centres_new.append(centers[i])
            ell_params.append(ransac_model.params)

    dict_export = {'ellipses.csv': pd.DataFrame(ell_params, columns=COLUMNS_ELLIPSE)}
    return segm, np.array(centres_new), dict_export


def segment_fit_ellipse_ransac_segm(seg, centers, fn_preproc_points,
                                    table_p, nb_inliers=0.35,
                                    thr_overlap=SEGM_OVERLAP):
    """ segment eggs using ellipse fitting and RANDSAC strategy on segmentation

    :param ndarray seg: input image / segmentation
    :param [[int, int]] centers: position of centres / seeds
    :param fn_preproc_points: function for detection boundary points
    :param [[float]] table_p: table of probabilities being foreground / background
    :param float nb_inliers: ratio of inliers for RANSAC
    :param float thr_overlap: threshold for removing overlapping segmentations
    :return (ndarray, [[int, int]]): resulting segmentation, updated centres
    """
    slic, points_all, labels = ell_fit.get_slic_points_labels(seg, slic_size=15,
                                                              slic_regul=0.1)
    points_centers = fn_preproc_points(seg, centers)
    weights = np.bincount(slic.ravel())

    centres_new, ell_params = [], []
    segm = np.zeros_like(seg)
    for i, points in enumerate(points_centers):
        lb = i + 1
        ransac_model, _ = ell_fit.ransac_segm(points,
                                              ell_fit.EllipseModelSegm,
                                              points_all, weights,
                                              labels, table_p,
                                              min_samples=nb_inliers,
                                              residual_threshold=25,
                                              max_trials=250)
        if not ransac_model:
            continue
        logging.debug('ellipse params: %r', ransac_model.params)
        segm = ell_fit.add_overlap_ellipse(segm, ransac_model.params, lb,
                                           thr_overlap)

        if np.any(segm == lb):
            centres_new.append(centers[i])
            ell_params.append(ransac_model.params)

    dict_export = {'ellipses.csv': pd.DataFrame(ell_params, columns=COLUMNS_ELLIPSE)}
    return segm, np.array(centres_new), dict_export


def segment_graphcut_pixels(seg, centers, labels_fg_prob, gc_regul=1.,
                            seed_size=10, coef_shape=0.,
                            shape_mean_std=(50., 10.)):
    """ wrapper for segment global GraphCut optimisations

    :param ndarray seg: input image / segmentation
    :param [[int, int]] centers: position of centres / seeds
    :param labels_fg_prob:
    :param float gc_regul:
    :param int seed_size:
    :param float coef_shape:
    :param (float, float) shape_mean_std:
    :return (ndarray, [[int, int]]): resulting segmentation, updated centres
    """
    segm_obj = seg_rg.object_segmentation_graphcut_pixels(
        seg, centers, labels_fg_prob, gc_regul, seed_size, coef_shape,
        shape_mean_std=shape_mean_std)
    return segm_obj, centers, None


def segment_graphcut_slic(slic, seg, centers, labels_fg_prob, gc_regul=1.,
                          multi_seed=True, coef_shape=0., edge_weight=1.,
                          shape_mean_std=(50., 10.)):
    """ wrapper for segment global GraphCut optimisations on superpixels

    :param ndarray slic:
    :param ndarray seg: input image / segmentation
    :param [[int, int]] centers: position of centres / seeds
    :param labels_fg_prob:
    :param float gc_regul:
    :param bool multi_seed:
    :param float coef_shape:
    :param float edge_weight:
    :param shape_mean_std:
    :return (ndarray, [[int, int]]): resulting segmentation, updated centres
    """
    gc_labels = seg_rg.object_segmentation_graphcut_slic(
        slic, seg, centers, labels_fg_prob, gc_regul, edge_weight,
        add_neighbours=multi_seed, coef_shape=coef_shape,
        shape_mean_std=shape_mean_std)
    segm_obj = np.array(gc_labels)[slic]
    return segm_obj, centers, None


def segment_rg2sp_greedy(slic, seg, centers, labels_fg_prob, path_model,
                         coef_shape, coef_pairwise=5, allow_obj_swap=True,
                         prob_label_trans=(0.1, 0.03),
                         dict_thresholds=RG2SP_THRESHOLDS, debug_export=''):
    """ wrapper for region growing method with some debug exporting """
    if os.path.splitext(path_model)[-1] == '.npz':
        shape_model = np.load(path_model)
    else:
        shape_model = pickle.load(open(path_model, 'rb'))
    dict_debug = dict() if os.path.isdir(debug_export) else None

    slic_prob_fg = seg_rg.compute_segm_prob_fg(slic, seg, labels_fg_prob)
    labels_greedy = seg_rg.region_growing_shape_slic_greedy(
        slic, slic_prob_fg, centers, (shape_model['mix_model'], shape_model['cdfs']),
        shape_model['name'], coef_shape=coef_shape, coef_pairwise=coef_pairwise,
        prob_label_trans=prob_label_trans, greedy_tol=1e-1, allow_obj_swap=allow_obj_swap,
        dict_thresholds=dict_thresholds, nb_iter=1000, debug_history=dict_debug)

    if dict_debug is not None:
        nb_iter = len(dict_debug['energy'])
        for i in range(nb_iter):
            fig = tl_visu.figure_rg2sp_debug_complete(seg, slic, dict_debug, i)
            fig.savefig(os.path.join(debug_export, 'iter_%03d' % i))
            plt.close(fig)

    segm_obj = labels_greedy[slic]
    return segm_obj, centers, None


def segment_rg2sp_graphcut(slic, seg, centers, labels_fg_prob, path_model,
                           coef_shape, coef_pairwise=5, allow_obj_swap=True,
                           prob_label_trans=(0.1, 0.03),
                           dict_thresholds=RG2SP_THRESHOLDS, debug_export=''):
    """ wrapper for region growing method with some debug exporting """
    if os.path.splitext(path_model)[-1] == '.npz':
        shape_model = np.load(path_model)
    else:
        shape_model = pickle.load(open(path_model, 'rb'))
    dict_debug = dict() if os.path.isdir(debug_export) else None

    slic_prob_fg = seg_rg.compute_segm_prob_fg(slic, seg, labels_fg_prob)
    labels_gc = seg_rg.region_growing_shape_slic_graphcut(
        slic, slic_prob_fg, centers, (shape_model['mix_model'], shape_model['cdfs']),
        shape_model['name'], coef_shape=coef_shape, coef_pairwise=coef_pairwise,
        prob_label_trans=prob_label_trans, optim_global=True, allow_obj_swap=allow_obj_swap,
        dict_thresholds=dict_thresholds, nb_iter=250, debug_history=dict_debug)

    if dict_debug is not None:
        nb_iter = len(dict_debug['energy'])
        for i in range(nb_iter):
            fig = tl_visu.figure_rg2sp_debug_complete(seg, slic, dict_debug, i)
            fig.savefig(os.path.join(debug_export, 'iter_%03d' % i))
            plt.close(fig)

    segm_obj = labels_gc[slic]
    return segm_obj, centers, None


def simplify_segm_3cls(seg, lut=(0., 0.8, 1.), smooth=True):
    """ simple segmentation into 3 classes

    :param ndarray seg: input image / segmentation
    :param [float] lut:
    :param bool smooth:
    :return ndarray:
    """
    segm = seg.copy()
    segm[seg > 1] = 2
    if np.sum(seg > 0) > 0:
        seg_filled = ndimage.morphology.binary_fill_holes(seg > 0)
        segm[np.logical_and(seg == 0, seg_filled)] = 2
    segm = np.array(lut)[segm]
    if smooth:
        segm = ndimage.filters.gaussian_filter(segm, 5)
    return segm


def create_dict_segmentation(params, slic, segm, img, centers):
    """ create dictionary of segmentation function hash, function and parameters

    :param dict params:
    :param ndarray slic:
    :param ndarray segm:
    :param [[float]] centers:
    :return {str: (function, (...))}:
    """
    # parameters for Region Growing
    params_rg_single = (slic, segm, centers, params['tab-proba_RG2SP'],
                        params['path_single-model'], params['RG2SP-shape'],
                        params['RG2SP-pairwise'], params['RG2SP-swap'],
                        params['label_trans'], params['RG2SP_theshold'])
    params_rg_multi = (slic, segm, centers, params['tab-proba_RG2SP'],
                       params['path_multi-models'], params['RG2SP-shape'],
                       params['RG2SP-pairwise'], params['RG2SP-swap'],
                       params['label_trans'], params['RG2SP_theshold'])
    tab_proba_gc = params['tab-proba_graphcut']
    gc_regul_px = params['gc-pixel_regul']
    gc_regul_slic = params['gc-slic_regul']
    seg_simple = simplify_segm_3cls(segm) if segm is not None else None

    dict_segment = {
        'ellipse_moments': (segment_fit_ellipse,
                            (segm, centers,
                             ell_fit.prepare_boundary_points_ray_dist)),
        'ellipse_ransac_mmt': (segment_fit_ellipse_ransac,
                               (segm, centers,
                                ell_fit.prepare_boundary_points_ray_dist)),
        'ellipse_ransac_crit': (segment_fit_ellipse_ransac_segm,
                                (segm, centers,
                                 ell_fit.prepare_boundary_points_ray_edge,
                                 params['tab-proba_ellipse'])),

        'ellipse_ransac_crit2': (segment_fit_ellipse_ransac_segm,
                                 (segm, centers,
                                  ell_fit.prepare_boundary_points_ray_join,
                                  params['tab-proba_ellipse'])),
        'ellipse_ransac_crit3': (segment_fit_ellipse_ransac_segm,
                                 (segm, centers,
                                  ell_fit.prepare_boundary_points_ray_mean,
                                  params['tab-proba_ellipse'])),

        'GC_pixels-small': (segment_graphcut_pixels,
                            (segm, centers, tab_proba_gc, gc_regul_px, 10)),
        'GC_pixels-large': (segment_graphcut_pixels,
                            (segm, centers, tab_proba_gc, gc_regul_px, 30)),
        'GC_pixels-shape': (segment_graphcut_pixels, (segm, centers,
                            tab_proba_gc, gc_regul_px, 10, 0.1)),
        'GC_slic-small': (segment_graphcut_slic, (slic, segm, centers,
                          tab_proba_gc, gc_regul_slic, False)),
        'GC_slic-large': (segment_graphcut_slic, (slic, segm, centers,
                          tab_proba_gc, gc_regul_slic, True)),
        'GC_slic-shape': (segment_graphcut_slic,
                          (slic, segm, centers, tab_proba_gc, 1., False, 0.1)),

        'RG2SP_greedy-single': (segment_rg2sp_greedy, params_rg_single),
        'RG2SP_greedy-mixture': (segment_rg2sp_greedy, params_rg_multi),
        'RG2SP_GC-single': (segment_rg2sp_graphcut, params_rg_single),
        'RG2SP_GC-mixture': (segment_rg2sp_graphcut, params_rg_multi),

        'watershed': (segment_watershed, (segm, centers)),
        'watershed_morph': (segment_watershed, (segm, centers, True)),

        # NOTE, this method takes to long for run in CI
        'morph-snakes_seg': (segment_morphsnakes,
                             (seg_simple, centers, True, 3, [2, 1])),
        'morph-snakes_img': (segment_morphsnakes, (img, centers)),
    }
    if params['methods'] is not None:
        params['methods'] = [n.lower() for n in params['methods']]
        dict_segment_filter = {n: dict_segment[n] for n in dict_segment
                               if n.lower() in params['methods']}
    else:
        dict_segment_filter = dict_segment
    return dict_segment_filter


def image_segmentation(idx_row, params, debug_export=DEBUG_EXPORT):
    """ image segmentation which prepare inputs (imsegm, centres)
    and perform segmentation of various imsegm methods

    :param (int, str) idx_row: input image and centres
    :param dict params: segmentation parameters
    :return str: image name
    """
    _, row_path = idx_row
    for k in dict(row_path):
        if isinstance(k, str) and k.startswith('path_'):
            row_path[k] = tl_data.update_path(row_path[k], absolute=True)
    logging.debug('segmenting image: "%s"', row_path['path_image'])
    name = os.path.splitext(os.path.basename(row_path['path_image']))[0]

    img = load_image(row_path['path_image'])
    # make the image like RGB
    img_rgb = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)
    seg = load_image(row_path['path_segm'], 'segm')
    assert img_rgb.shape[:2] == seg.shape, \
        'image %r and segm %r do not match' % (img_rgb.shape[:2], seg.shape)
    if not os.path.isfile(row_path['path_centers']):
        logging.warning('no center was detected for "%s"', name)
        return name
    centers = tl_data.load_landmarks_csv(row_path['path_centers'])
    centers = tl_data.swap_coord_x_y(centers)
    if not list(centers):
        logging.warning('no center was detected for "%s"', name)
        return name
    # img = seg / float(seg.max())
    slic = seg_spx.segment_slic_img2d(img_rgb, sp_size=params['slic_size'],
                                      relative_compact=params['slic_regul'])

    path_segm = os.path.join(params['path_exp'], 'input', name + '.png')
    export_draw_image_segm(path_segm, img_rgb, segm_obj=seg, centers=centers)

    seg_simple = simplify_segm_3cls(seg)
    path_segm = os.path.join(params['path_exp'], 'simple', name + '.png')
    export_draw_image_segm(path_segm, seg_simple - 1.)

    dict_segment = create_dict_segmentation(params, slic, seg, img, centers)

    image_name = name + '.png'
    centre_name = name + '.csv'

    # iterate over segmentation methods and perform segmentation on this image
    for method in dict_segment:
        (fn, args) = dict_segment[method]
        logging.debug(' -> %s on "%s"', method, name)
        path_dir = os.path.join(params['path_exp'], method)  # n.split('_')[0]
        path_segm = os.path.join(path_dir, image_name)
        path_centre = os.path.join(path_dir + DIR_CENTRE_POSIX, centre_name)
        path_fig = os.path.join(path_dir + DIR_VISUAL_POSIX, image_name)
        path_debug = os.path.join(path_dir + DIR_DEBUG_POSIX, name)
        # assuming that segmentation may fail
        try:
            t = time.time()
            if debug_export and 'rg2sp' in method:
                os.mkdir(path_debug)
                segm_obj, centers, dict_export = fn(*args,
                                                    debug_export=path_debug)
            else:
                segm_obj, centers, dict_export = fn(*args)

            # also export ellipse params here or inside the segm fn
            if dict_export is not None:
                for k in dict_export:
                    export_partial(k, dict_export[k], path_dir, name)

            logging.info('running time of %r on image "%s" is %d s',
                         fn.__name__, image_name, time.time() - t)
            tl_data.io_imsave(path_segm, segm_obj.astype(np.uint8))
            export_draw_image_segm(path_fig, img_rgb, seg, segm_obj, centers)
            # export also centers
            centers = tl_data.swap_coord_x_y(centers)
            tl_data.save_landmarks_csv(path_centre, centers)
        except Exception:
            logging.exception('segment fail for "%s" via %s', name, method)

    return name


def export_partial(str_key, obj_content, path_dir, name):
    key, ext = os.path.splitext(str_key)
    path_out = os.path.join(path_dir + '___%s' % key)
    if not os.path.isdir(path_out):
        os.mkdir(path_out)
    path_file = os.path.join(path_out, name + ext)
    if ext.endswith('.csv'):
        obj_content.to_csv(path_file)
    return path_file


def main(params, debug_export=DEBUG_EXPORT):
    """ the main entry point

    :param dict params: segmentation parameters
    :param bool debug_export: whether export visualisations
    """
    logging.getLogger().setLevel(logging.DEBUG)

    params = tl_expt.create_experiment_folder(params, dir_name=NAME_EXPERIMENT,
                                              stamp_unique=EACH_UNIQUE_EXPERIMENT)
    tl_expt.set_experiment_logger(params['path_exp'])
    logging.info(tl_expt.string_dict(params, desc='PARAMETERS'))
    # tl_expt.create_subfolders(params['path_exp'], [FOLDER_IMAGE])

    df_paths = pd.read_csv(params['path_list'], index_col=0)
    logging.info('loaded %i items with columns: %r', len(df_paths),
                 df_paths.columns.tolist())
    df_paths.dropna(how='any', inplace=True)

    # create sub-folders if required
    tl_expt.create_subfolders(params['path_exp'], ['input', 'simple'])
    dict_segment = create_dict_segmentation(params, None, None, None, None)
    dirs_center = [n + DIR_CENTRE_POSIX for n in dict_segment]
    dirs_visu = [n + DIR_VISUAL_POSIX for n in dict_segment]
    tl_expt.create_subfolders(params['path_exp'],
                              [n for n in dict_segment] + dirs_center + dirs_visu)
    if debug_export:
        list_dirs = [n + DIR_DEBUG_POSIX for n in dict_segment if 'rg2sp' in n]
        tl_expt.create_subfolders(params['path_exp'], list_dirs)

    _wrapper_segment = partial(image_segmentation, params=params)
    iterate = tl_expt.WrapExecuteSequence(_wrapper_segment, df_paths.iterrows(),
                                          nb_workers=params['nb_workers'])
    list(iterate)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = arg_parse_params(SEGM_PARAMS)
    main(params)

    logging.info('DONE')

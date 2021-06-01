"""
Run supervised segmentation experiment with superpixels and training examples

1) train classifier on annotated images with some statistic - LPO
2) segment new images in specified folder

Segmentation pipeline:

 1. segment SLIC super-pixels
 2. compute features (color and texture)
 3. train classifier on training examples
 4. segment new images

The input is csv file with training images and related segmentation.
The output is set of segmented images.

.. note:: there are a few constants to that have an impact on the experiment,
 see them bellow with explanation for each of them.

Sample usage::

    python run_segm_slic_classif_graphcut.py \
        -l data-images/drosophila_ovary_slice/list_imgs-annot-struct.csv \
        -i "data-images/drosophila_ovary_slice/image/*.jpg" \
       --path_config experiments_segmentation/sample_config.yml \
        -o results -n Ovary --img_type 2d_split --visual

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import gc
import glob
import logging
import os
import sys
import time
from functools import partial

import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color as sk_color
from PIL import Image
# from llvmpy._api.llvm.CmpInst import FCMP_OLE
from skimage import segmentation
from sklearn import metrics

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from run_segm_slic_model_graphcut import (
    arg_parse_params,
    get_idx_name,
    load_image,
    parse_imgs_idx_path,
    write_skip_file,
)

import imsegm.classification as seg_clf
import imsegm.descriptors as seg_fts
import imsegm.graph_cuts as seg_gc
import imsegm.labeling as seg_label
import imsegm.pipelines as seg_pipe
import imsegm.superpixels as seg_spx
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.drawing as tl_visu
import imsegm.utilities.experiments as tl_expt

NAME_EXPERIMENT = 'experiment_segm-Supervised'
NB_WORKERS = tl_expt.nb_workers(0.9)

TYPES_LOAD_IMAGE = ['2d_rgb', '2d_split']
NAME_FIG_LABEL_HISTO = 'fig_histogram_annot_segments.png'
NAME_CSV_SEGM_STAT_SLIC_ANNOT = 'statistic_segm_slic_annot.csv'
NAME_CSV_SEGM_STAT_RESULT_LPO = 'statistic_segm_L-%i-O.csv'
NAME_CSV_SEGM_STAT_RESULT_LPO_GC = 'statistic_segm_L-%i-O_gc.csv'
NAME_CSV_SEGM_STAT_RESULTS = 'statistic_segm_results.csv'
NAME_DUMP_TRAIN_DATA = 'dump_training_data.npz'

# setting experiment sub-folders
FOLDER_IMAGE = 'images'
FOLDER_ANNOT = 'annotations'
FOLDER_SLIC = 'slic'
FOLDER_SLIC_ANNOT = 'annot_slic'
FOLDER_TRAIN = 'segmentation_trained'
SUFFIX_VISUAL = '___visual'
FOLDER_TRAIN_VISU = FOLDER_TRAIN + SUFFIX_VISUAL
FOLDER_LPO = 'segmentation_leave-P-out'
FOLDER_LPO_VISU = FOLDER_LPO + SUFFIX_VISUAL
LIST_FOLDERS_BASE = (FOLDER_IMAGE, FOLDER_ANNOT, FOLDER_SLIC, FOLDER_SLIC_ANNOT, FOLDER_TRAIN, FOLDER_LPO)
LIST_FOLDERS_DEBUG = (FOLDER_TRAIN_VISU, FOLDER_LPO_VISU)

# unique experiment means adding timestemp on the end of folder name
EACH_UNIQUE_EXPERIMENT = False
# showing some intermediate debug images from segmentation
SHOW_DEBUG_IMAGES = True
# relabel annotation such that labels are in sequence no gaps in between them
ANNOT_RELABEL_SEQUENCE = False
# whether skip loading config from previous fun
FORCE_RELOAD = False
# even you have dumped data from previous time, all wil be recomputed
FORCE_RECOMP_DATA = False
# even you have saved classif. data from previous time, all wil be retrained
FORCE_RETRAIN_CLASSIF = False
# ration of fold size for LPO for hyper-parameter search
CROSS_VAL_LEAVE_OUT_SEARCH = 0.2
# ration of fold size for LPO for evaluation
CROSS_VAL_LEAVE_OUT_EVAL = 0.1
# run prediction on training data, should be overfiting
RUN_TRAIN_PREDICT = False

FEATURES_SET_COLOR = {'color': ('mean', 'std', 'energy')}
FEATURES_SET_TEXTURE = {'tLM': ('mean', 'std', 'energy')}
FEATURES_SET_ALL = {
    'color': ('mean', 'std', 'median'),
    'tLM': ('mean', 'std', 'energy', 'meanGrad'),
}
FEATURES_SET_MIN = {
    'color': ('mean', 'std', 'energy'),
    'tLM_short': ('mean', ),
}
FEATURES_SET_MIX = {
    'color': ('mean', 'std', 'energy', 'median'),
    'tLM': ('mean', 'std'),
}
# Default parameter configuration
SEGM_PARAMS = {
    'name': 'ovary',
    'nb_classes': None,
    'img_type': '2d_split',
    'slic_size': 35,
    'slic_regul': 0.3,
    # 'spacing': (12, 1, 1),
    'features': FEATURES_SET_MIN,
    'label_purity': 0.95,  # training only superpixels with 0.9 label purity
    'balance': 'unique',
    'pca_coef': None,
    'classif': 'RandForest',  # 'GradBoost'
    'nb_classif_search': 50,
    'gc_regul': 5.0,
    'gc_edge_type': 'model',
    'gc_use_trans': False,
}
PATH_IMAGES = os.path.join(tl_data.update_path('data-images'), 'drosophila_ovary_slice')
PATH_RESULTS = tl_data.update_path('results', absolute=True)
SEGM_PARAMS.update({
    'path_train_list': os.path.join(PATH_IMAGES, 'list_imgs-annot-struct.csv'),
    'path_predict_imgs': os.path.join(PATH_IMAGES, 'image', 'insitu43*.tif'),
    # 'path_predict_imgs': '',
    'path_out': PATH_RESULTS,
})


def visu_histogram_labels(params, dict_label_hist, fig_name=NAME_FIG_LABEL_HISTO):
    """ draw histogram of superpixel-pixel annotation purity for each class

    :param {...} params:
    :param {...} dict_label_hist:
    :param str fig_name:
    """
    fig = tl_visu.figure_annot_slic_histogram_labels(dict_label_hist, params['slic_size'], params['slic_regul'])
    path_fig = os.path.join(params['path_exp'], fig_name)
    fig.savefig(path_fig)
    plt.close(fig)


def use_rgb_image(img, clr='rgb'):
    # clr = params.get('clr_space', 'rgb').lower()
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img_rgb = tl_data.convert_img_color_to_rgb(img, clr)
    elif img.ndim == 2:
        img_rgb = sk_color.gray2rgb(img)
    else:
        img_rgb = img.copy()
    return img_rgb


def load_image_annot_compute_features_labels(idx_row, params, show_debug_imgs=SHOW_DEBUG_IMAGES):
    """ load image and annotation, and compute superpixel features and labels

    :param (int, {...}) idx_row: row from table with paths
    :param dict params: segmentation parameters
    :param bool show_debug_imgs: whether show debug images
    :return (...):
    """

    def _path_out_img(params, dir_name, name):
        return os.path.join(params['path_exp'], dir_name, name + '.png')

    idx, row = idx_row
    idx_name = get_idx_name(idx, row['path_image'])
    img = load_image(row['path_image'], params['img_type'])
    annot = load_image(row['path_annot'], '2d_segm')
    logging.debug('.. processing: %s', idx_name)
    assert img.shape[:2] == annot.shape[:2], \
        'individual size of image %r and seg_pipe %r for "%s" - "%s"' % \
        (img.shape, annot.shape, row['path_image'], row['path_annot'])
    if show_debug_imgs:
        plt.imsave(_path_out_img(params, FOLDER_IMAGE, idx_name), img, cmap=plt.cm.gray)
        plt.imsave(_path_out_img(params, FOLDER_ANNOT, idx_name), annot)

    # duplicate gray band to be as rgb
    # if img.ndim == 2:
    #     img = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)
    slic = seg_spx.segment_slic_img2d(img, sp_size=params['slic_size'], relative_compact=params['slic_regul'])
    img = tl_data.convert_img_color_from_rgb(img, params.get('clr_space', 'rgb'))
    logging.debug('computed SLIC with %i labels', slic.max())
    if show_debug_imgs:
        img_rgb = use_rgb_image(img)
        img_slic = segmentation.mark_boundaries(img_rgb, slic, color=(1, 0, 0), mode='subpixel')
        plt.imsave(_path_out_img(params, FOLDER_SLIC, idx_name), np.clip(img_slic, 0, 1))
    slic_label_hist = seg_label.histogram_regions_labels_norm(slic, annot)
    labels = np.argmax(slic_label_hist, axis=1)
    slic_annot = labels[slic]
    if show_debug_imgs:
        plt.imsave(_path_out_img(params, FOLDER_SLIC_ANNOT, idx_name), np.clip(slic_annot, 0, slic_annot.max()))

    features, feature_names = seg_fts.compute_selected_features_img2d(img, slic, params['features'])
    return idx_name, img, annot, slic, features, labels, slic_label_hist, feature_names


def dataset_load_images_annot_compute_features(params, show_debug_imgs=SHOW_DEBUG_IMAGES):
    """ for all datasets perform the following steps:
    1) load image and annotation
    2) compute superpixel features and labels

    :param dict params: segmentation parameters
    :param bool show_debug_imgs: whether show debug images
    :return ({str: ndarray} * 6, list(str)):
    """
    dict_images, dict_annots = dict(), dict()
    dict_slics, dict_features, dict_labels, dict_label_hist = dict(), dict(), dict(), dict()
    feature_names = list()

    # compute features
    df_paths = pd.read_csv(params['path_train_list'], index_col=0)
    df_paths.reset_index(inplace=True)
    assert all(
        n in df_paths.columns for n in ['path_image', 'path_annot']
    ), 'missing required columns in loaded csv file'
    _wrapper_load_compute = partial(
        load_image_annot_compute_features_labels,
        params=params,
        show_debug_imgs=show_debug_imgs,
    )
    iterate = tl_expt.WrapExecuteSequence(
        _wrapper_load_compute,
        df_paths.iterrows(),
        nb_workers=params['nb_workers'],
        desc='extract training data',
    )
    for name, img, annot, slic, features, labels, label_hist, feature_names in iterate:
        dict_images[name] = img
        dict_annots[name] = annot
        dict_slics[name] = slic
        dict_features[name] = features
        dict_labels[name] = labels
        dict_label_hist[name] = label_hist

    # gc.collect(), time.sleep(1)
    return (dict_images, dict_annots, dict_slics, dict_features, dict_labels, dict_label_hist, feature_names)


def load_dump_data(path_dump_data):
    """ load dumped data from previous run of experiment

    :param str path_dump_data:
    :return ({str: ndarray} * 6, list(str)):
    """
    logging.info('loading dumped data "%s"', path_dump_data)
    # with open(os.path.join(path_out, NAME_DUMP_TRAIN_DATA), 'r') as f:
    #     dict_data = pickle.load(f)
    npz_file = np.load(path_dump_data, allow_pickle=True)
    dict_imgs = dict(npz_file['dict_images'].tolist())
    dict_annot = dict(npz_file['dict_annot'].tolist())
    dict_slics = dict(npz_file['dict_slics'].tolist())
    dict_label_hist = dict(npz_file['dict_label_hist'].tolist())
    dict_features = dict(npz_file['dict_features'].tolist())
    dict_labels = dict(npz_file['dict_labels'].tolist())
    feature_names = npz_file['feature_names'].tolist()
    return (dict_imgs, dict_annot, dict_slics, dict_features, dict_labels, dict_label_hist, feature_names)


def save_dump_data(path_dump_data, imgs, annot, slics, features, labels, label_hist, feature_names):
    """

    :param str path_dump_data:
    :param {str: ndarray} imgs: dictionary {name: data} of images
    :param {str: ndarray} annot: dictionary {name: data} of annotation
    :param {str: ndarray} slics: dictionary {name: data} of superpixels
    :param {str: ndarray} features: dictionary {name: data} of features
    :param {str: ndarray} labels: dictionary {name: data} of lables
    :param {str: ndarray} label_hist: dictionary {name: data} of
    :param list(str) feature_names: list of feature names
    """
    logging.info('save (dump) data to "%s"', path_dump_data)
    np.savez_compressed(
        path_dump_data,
        dict_images=imgs,
        dict_annot=annot,
        dict_slics=slics,
        dict_label_hist=label_hist,
        dict_features=features,
        dict_labels=labels,
        feature_names=feature_names,
    )


def export_draw_image_segm_contour(img, segm, path_out, name, suffix=''):
    logging.debug('export draw image segmentation countours: %s', name)
    fig = tl_visu.figure_image_segm_results(img, segm)
    fig.savefig(os.path.join(path_out, name + suffix + '.png'))
    plt.close(fig)


def segment_image(imgs_idx_path, params, classif, path_out, path_visu=None, show_debug_imgs=SHOW_DEBUG_IMAGES):
    """ perform image segmentation on input image with given paramters
    and trained classifier, and save results

    :param (int, str) imgs_idx_path:
    :param dict params: segmentation parameters
    :param obj classif: trained classifier
    :param str path_out: path for output
    :param str path_visu: the existing patch means export also visualisation
    :param bool show_debug_imgs: whether show debug images
    :return (str, ndarray, ndarray):
    """
    idx, path_img = parse_imgs_idx_path(imgs_idx_path)
    logging.debug('segmenting image: "%s"', path_img)
    idx_name = get_idx_name(idx, path_img)
    img = load_image(path_img, params['img_type'])

    debug_visual = dict() if show_debug_imgs else None

    gc_regul = params['gc_regul']
    if params['gc_use_trans']:
        label_penalty = seg_gc.compute_pairwise_cost_from_transitions(params['label_transitions'])
        gc_regul = (gc_regul * label_penalty)

    segm_gc, segm_soft = seg_pipe.segment_color2d_slic_features_model_graphcut(
        img,
        classif,
        sp_size=params['slic_size'],
        sp_regul=params['slic_regul'],
        dict_features=params['features'],
        gc_regul=gc_regul,
        gc_edge_type=params['gc_edge_type'],
        debug_visual=debug_visual,
    )
    segm_map = np.argmax(segm_soft, axis=-1)

    for segm, suffix in [(segm_gc, ''), (segm_map, '_MAP')]:
        path_img = os.path.join(path_out, idx_name + suffix + '.png')
        logging.debug('export segmentation: %s', path_img)
        if np.max(segm) <= 1:
            img_seg = Image.fromarray((segm * 255).astype(np.uint8))
        else:
            img_seg = Image.fromarray(segm.astype(np.uint8))
        img_seg.convert('L').save(path_img)
        # io.imsave(path_img, segm_gc)

    path_npz = os.path.join(path_out, idx_name + '.npz')
    np.savez_compressed(path_npz, segm_soft)

    # plt.imsave(os.path.join(path_out, idx_name + '_rgb.png'), seg_pipe)
    if params.get('visual', False) and path_visu is not None and os.path.isdir(path_visu):
        export_draw_image_segm_contour(img, segm_gc, path_visu, idx_name, '_GC')
        export_draw_image_segm_contour(img, segm_map, path_visu, idx_name, '_MAP')
        if show_debug_imgs and debug_visual is not None:
            path_fig = os.path.join(path_visu, str(idx_name) + '_debug.png')
            logging.debug('exporting (debug) visualization: %s', path_fig)
            fig = tl_visu.figure_segm_graphcut_debug(debug_visual)
            fig.savefig(path_fig, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
    gc.collect()
    time.sleep(1)
    return idx_name, segm_map, segm_gc


def eval_segment_with_annot(
    params,
    dict_annot,
    dict_segm,
    dict_label_hist=None,
    name_csv='statistic___.csv',
    drop_labels=None,
    nb_workers=1,
):
    """ evaluate the segmentation results according given annotation

    :param dict params:
    :param {str: ndarray} dict_annot:
    :param {str: ndarray} dict_segm:
    :param {str: ndarray} dict_label_hist:
    :param str name_csv:
    :param int nb_workers:
    :return:
    """
    if dict_label_hist is not None:
        visu_histogram_labels(params, dict_label_hist)
    assert sorted(dict_annot.keys()) == sorted(dict_segm.keys()), \
        'mismatch in dictionary keys: \n%s \n%s' % (sorted(dict_annot.keys()), sorted(dict_segm.keys()))
    list_annot = [dict_annot[n] for n in dict_annot]
    list_segm = [dict_segm[n] for n in dict_annot]
    df_stat = seg_clf.compute_stat_per_image(
        list_segm, list_annot, [n for n in dict_annot], nb_workers, drop_labels=drop_labels
    )

    path_csv = os.path.join(params['path_exp'], name_csv)
    logging.info('STATISTIC on segm and annot (%s):', name_csv)
    df_stat.to_csv(path_csv)

    logging.info(
        metrics.classification_report(
            seg_label.convert_segms_2_list(list_segm), seg_label.convert_segms_2_list(list_annot), digits=4
        )
    )
    logging.debug('%r', df_stat)
    return df_stat


def retrain_lpo_segment_image(
    list_imgs_idx_path, path_classif, path_dump, path_out, path_visu, show_debug_imgs=SHOW_DEBUG_IMAGES
):
    """ load the classifier, and dumped data, subtract the image,
    retrain the classif without it and do the segmentation

    :param list(str) list_imgs_idx_path: path to input image
    :param str path_classif: path to saved classifier
    :param str path_dump: path to dumped data
    :param, str path_out: path to segmentation outputs
    :param bool show_debug_imgs: whether show debug images
    :return (str, ndarray, ndarray):
    """
    dict_imgs, _, _, dict_features, dict_labels, _, _ = load_dump_data(path_dump)
    dict_classif = seg_clf.load_classifier(path_classif)
    classif = dict_classif['clf_pipeline']
    params = dict_classif['params']

    for idx, path_img in list_imgs_idx_path:
        idx_name = get_idx_name(idx, path_img)
        _ = dict_features.pop(idx_name, None)
        _ = dict_labels.pop(idx_name, None)
    assert (len(dict_imgs) - len(dict_features)) == len(list_imgs_idx_path), \
        'subset of %i images was not dropped, training set %i from total %i' \
        % (len(list_imgs_idx_path), len(dict_features), len(dict_imgs))

    features, labels, _ = seg_clf.convert_set_features_labels_2_dataset(
        dict_features,
        dict_labels,
        balance_type=params['balance'],
        drop_labels=[-1, np.nan] + params.get('drop_labels', [])
    )
    classif.fit(features, labels)

    dict_segm, dict_segm_gc = {}, {}
    for imgs_idx_path in list_imgs_idx_path:
        idx_name, segm, segm_gc = segment_image(
            imgs_idx_path, params, classif, path_out, path_visu, show_debug_imgs=show_debug_imgs
        )
        dict_segm[idx_name] = segm
        dict_segm_gc[idx_name] = segm_gc
    gc.collect()
    time.sleep(1)
    return dict_segm, dict_segm_gc


def get_summary(df, name, list_stat=('mean', 'std', 'median')):
    """ from particular segmentation results extract one global summary

    :param df:
    :param str name:
    :param [] list_stat:
    :return {str: float}:
    """
    df_summary = df.describe()
    cols = df.columns.tolist()
    dict_state = {'name': name, 'count': len(df)}
    for n in list_stat:
        col_names = ['%s (%s)' % (c, n) for c in cols]
        if n == 'median':
            vals = df.median(axis=0).values.tolist()
        else:
            vals = df_summary.T[n].values.tolist()
        dict_state.update(list(zip(col_names, vals)))
    return dict_state


def perform_train_predictions(params, paths_img, classif, show_debug_imgs=SHOW_DEBUG_IMAGES):
    logging.info('run prediction on training images...')
    imgs_idx_path = list(zip(range(1, len(paths_img) + 1), paths_img))

    dict_segms, dict_segms_gc = dict(), dict()
    path_out = os.path.join(params['path_exp'], FOLDER_TRAIN)
    path_visu = os.path.join(params['path_exp'], FOLDER_TRAIN_VISU)
    _wrapper_segment = partial(
        segment_image,
        params=params,
        classif=classif,
        path_out=path_out,
        path_visu=path_visu,
        show_debug_imgs=show_debug_imgs,
    )
    iterate = tl_expt.WrapExecuteSequence(
        _wrapper_segment,
        imgs_idx_path,
        nb_workers=params['nb_workers'],
        desc='image segm: prediction',
    )
    for name, segm, segm_gc in iterate:
        dict_segms[name] = segm
        dict_segms_gc[name] = segm_gc
    return dict_segms, dict_segms_gc


def experiment_lpo(
    params, df_stat, dict_annot, idx_paths_img, path_classif, path_dump, nb_holdout, show_debug_imgs=SHOW_DEBUG_IMAGES
):
    """ experiment Leave-P-samples-Out

    :param dict params:
    :param DF df_stat:
    :param {str: ndarray} dict_annot:
    :param list(str) paths_img:
    :param str path_classif:
    :param str path_dump:
    :param int nb_holdout:
    :param bool show_debug_imgs: whether show debug images
    :return dict:
    """
    logging.info('run prediction on training images as Leave-%i-Out...', nb_holdout)
    dict_segms, dict_segms_gc = dict(), dict()
    cv = seg_clf.CrossValidate(len(idx_paths_img), nb_hold_out=nb_holdout)
    test_imgs_idx_path = [[idx_paths_img[i] for i in ids] for _, ids in cv]
    path_out = os.path.join(params['path_exp'], FOLDER_LPO)
    path_visu = os.path.join(params['path_exp'], FOLDER_LPO_VISU) \
        if params.get('visual', False) else None
    _wrapper_segment = partial(
        retrain_lpo_segment_image,
        path_classif=path_classif,
        path_dump=path_dump,
        path_out=path_out,
        path_visu=path_visu,
        show_debug_imgs=show_debug_imgs,
    )
    iterate = tl_expt.WrapExecuteSequence(
        _wrapper_segment,
        test_imgs_idx_path,
        nb_workers=params['nb_workers'],
        desc='experiment LPO',
    )
    for dict_seg, dict_seg_gc in iterate:
        dict_segms.update(dict_seg)
        dict_segms_gc.update(dict_seg_gc)
    gc.collect()
    time.sleep(1)

    df = eval_segment_with_annot(
        params, dict_annot, dict_segms, None, NAME_CSV_SEGM_STAT_RESULT_LPO % nb_holdout,
        params.get('drop_labels', None), params['nb_workers']
    )
    df_stat = df_stat.append(get_summary(df, 'segm (L-%i-O)' % nb_holdout), ignore_index=True)
    df = eval_segment_with_annot(
        params, dict_annot, dict_segms_gc, None, NAME_CSV_SEGM_STAT_RESULT_LPO_GC % nb_holdout,
        params.get('drop_labels', None), params['nb_workers']
    )
    df_stat = df_stat.append(get_summary(df, 'segm GC (L-%i-O)' % nb_holdout), ignore_index=True)
    path_csv_stat = os.path.join(params['path_exp'], NAME_CSV_SEGM_STAT_RESULTS)
    df_stat.set_index(['name']).to_csv(path_csv_stat)
    return df_stat


def load_train_classifier(params, features, labels, feature_names, sizes, nb_holdout):
    logging.info('train classifier...')
    seg_clf.feature_scoring_selection(features, labels, feature_names, path_out=params['path_exp'])
    cv = seg_clf.CrossValidateGroups(sizes, nb_hold_out=nb_holdout)
    # feature norm & train classification
    fname_classif = seg_clf.TEMPLATE_NAME_CLF.format(params['classif'])
    path_classif = os.path.join(params['path_exp'], fname_classif)
    if os.path.isfile(path_classif) and not FORCE_RETRAIN_CLASSIF:
        logging.info('loading classifier: %s', path_classif)
        params_local = params.copy()
        dict_classif = seg_clf.load_classifier(path_classif)
        classif = dict_classif['clf_pipeline']
        params = dict_classif['params']
        params.update({k: params_local[k] for k in params_local if k.startswith('path_') or k.startswith('gc_')})
        logging.debug('loaded PARAMETERS: %r', params)
    else:
        classif, path_classif = seg_clf.create_classif_search_train_export(
            params['classif'],
            features,
            labels,
            cross_val=cv,
            params=params,
            feature_names=feature_names,
            pca_coef=params['pca_coef'],
            eval_metric=params.get('classif_metric', 'f1'),
            nb_search_iter=params.get('nb_classif_search', 1),
            nb_workers=params['nb_workers'],
            path_out=params['path_exp']
        )
    params['path_classif'] = path_classif
    cv = seg_clf.CrossValidateGroups(sizes, nb_hold_out=nb_holdout)
    seg_clf.eval_classif_cross_val_scores(
        params['classif'], classif, features, labels, cross_val=cv, path_out=params['path_exp']
    )
    seg_clf.eval_classif_cross_val_roc(
        params['classif'], classif, features, labels, cross_val=cv, path_out=params['path_exp']
    )
    return params, classif, path_classif


def wrapper_filter_labels(name_img_labels_slic_label_hist, label_purity, drop_labels=None, path_visu=None):
    name, img, labels, slic, label_hist = name_img_labels_slic_label_hist
    weights = np.max(label_hist, axis=1)
    if path_visu is not None and os.path.isdir(path_visu):
        used = np.zeros(len(weights))
        used[np.asarray(weights) >= label_purity] = 1
        if drop_labels is not None:
            for lb in drop_labels:
                used[labels == lb] = 0
        fig = tl_visu.figure_used_samples(img, labels, slic, used)
        path_fig = os.path.join(path_visu, name + '___training.jpg')
        fig.savefig(path_fig)
        plt.close(fig)

    labels[weights < label_purity] = -1

    return name, labels


def filter_train_with_purity(
    dict_imgs,
    dict_labels,
    dict_label_hist,
    label_purity,
    dict_slics,
    drop_labels=None,
    path_visu=None,
    nb_workers=NB_WORKERS,
):
    _w_filter = partial(wrapper_filter_labels, label_purity=label_purity, drop_labels=drop_labels, path_visu=path_visu)
    iter_vals = ((n, dict_imgs[n], dict_labels[n], dict_slics[n], dict_label_hist[n]) for n in dict_labels)
    iterate = tl_expt.WrapExecuteSequence(_w_filter, iter_vals, nb_workers=nb_workers, desc='filter labels (purity)')
    for n, lbs in iterate:
        dict_labels[n] = lbs

    return dict_labels


def main_train(params):
    """ the main composed from following steps:
    1) load already computed data (features and labels) or compute them now
    2) visualise labeled superpixels aka annotation
    3) load or train classifier with hyper-parameters search
    4) perform Leave-One-Out and Leave-P-Out experiments on images

    :param dict params:
    :return dict:
    """
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info('running TRAINING...')
    show_visual = params.get('visual', False)

    reload_dir_config = os.path.isfile(params.get('path_config', '')) or FORCE_RELOAD
    stamp_unique = params.get('unique', EACH_UNIQUE_EXPERIMENT)
    params = tl_expt.create_experiment_folder(
        params, dir_name=NAME_EXPERIMENT, stamp_unique=stamp_unique, skip_load=reload_dir_config
    )
    tl_expt.set_experiment_logger(params['path_exp'])
    logging.info(tl_expt.string_dict(params, desc='PARAMETERS'))
    tl_expt.create_subfolders(params['path_exp'], LIST_FOLDERS_BASE)
    if show_visual:
        tl_expt.create_subfolders(params['path_exp'], LIST_FOLDERS_DEBUG)
    df_stat = pd.DataFrame()

    path_dump = os.path.join(params['path_exp'], NAME_DUMP_TRAIN_DATA)
    if os.path.isfile(path_dump) and not FORCE_RECOMP_DATA:
        (dict_imgs, dict_annot, dict_slics, dict_features, dict_labels, dict_label_hist,
         feature_names) = load_dump_data(path_dump)
    else:
        (dict_imgs, dict_annot, dict_slics, dict_features, dict_labels,
         dict_label_hist, feature_names) = \
            dataset_load_images_annot_compute_features(params, show_visual)
        save_dump_data(
            path_dump, dict_imgs, dict_annot, dict_slics, dict_features, dict_labels, dict_label_hist, feature_names
        )
    assert len(dict_imgs) > 1, 'training require at least 2 images'

    dict_annot_slic = {n: np.asarray(dict_labels[n])[dict_slics[n]] for n in dict_annot}
    df = eval_segment_with_annot(
        params, dict_annot, dict_annot_slic, dict_label_hist, NAME_CSV_SEGM_STAT_SLIC_ANNOT,
        params.get('drop_labels', None), params['nb_workers']
    )
    df_stat = df_stat.append(get_summary(df, 'SLIC-annot'), ignore_index=True)
    path_csv_stat = os.path.join(params['path_exp'], NAME_CSV_SEGM_STAT_RESULTS)
    df_stat.set_index(['name']).to_csv(path_csv_stat)

    if params['gc_use_trans']:
        params['label_transitions'] = \
            seg_gc.count_label_transitions_connected_segments(dict_slics,
                                                              dict_labels)
        logging.info('summary on edge-label transitions: \n %r', params['label_transitions'])

    path_purity_visu = None
    if show_visual:
        path_purity_visu = os.path.join(params['path_exp'], FOLDER_SLIC_ANNOT)
    dict_labels = filter_train_with_purity(
        dict_imgs,
        dict_labels,
        dict_label_hist,
        params['label_purity'],
        dict_slics,
        drop_labels=params.get('drop_labels', None),
        path_visu=path_purity_visu,
        nb_workers=params['nb_workers']
    )

    logging.info('prepare features...')
    # concentrate features, labels
    features, labels, sizes = seg_clf.convert_set_features_labels_2_dataset(
        dict_features,
        dict_labels,
        balance_type=params['balance'],
        drop_labels=[-1, np.nan] + params.get('drop_labels', [])
    )
    # drop "do not care" label which are -1
    features = np.nan_to_num(features)

    nb_holdout = params.get('cross_val', CROSS_VAL_LEAVE_OUT_SEARCH)
    nb_holdout = max(1, int(round(len(sizes) * nb_holdout)))  # minimum is 1
    nb_holdout = min(nb_holdout, int(len(sizes) / 2))  # max is half of the set
    params, classif, path_classif = load_train_classifier(params, features, labels, feature_names, sizes, nb_holdout)

    def _path_expt(n):
        return os.path.join(params['path_exp'], n)

    # test classif. on  training images
    df_paths = pd.read_csv(params['path_train_list'], index_col=0)
    df_paths.reset_index(inplace=True)
    paths_img = df_paths['path_image'].tolist()
    if RUN_TRAIN_PREDICT:
        perform_train_predictions(params, paths_img, classif, show_debug_imgs=show_visual)
    else:
        write_skip_file(_path_expt(FOLDER_TRAIN))

    gc.collect()
    time.sleep(1)

    # LEAVE P OUT
    if params.get('run_LPO', True):
        idx_paths_img = list(zip(df_paths.index.tolist(), df_paths['path_image'].tolist()))
        df_stat = experiment_lpo(
            params,
            df_stat,
            dict_annot,
            idx_paths_img,
            path_classif,
            path_dump,
            nb_holdout,
            show_debug_imgs=show_visual,
        )
    else:
        write_skip_file(_path_expt(FOLDER_LPO))
        # write_skip_file(_path_expt(FOLDER_LPO_VISU))

    logging.info('Statistic: \n %r', df_stat.describe())
    logging.info('training DONE')
    return params


def prepare_output_dir(path_pattern_imgs, path_out, name, visual=True):
    """ prepare output directory for segmenting new images

    :param str path_pattern_imgs:
    :param str path_out:
    :param str name:
    :return (str, str):
    """
    # add last 2 dir names
    name += '_'.join(path_pattern_imgs.split(os.sep)[-3:-1])
    # params = tl_expt.create_experiment_folder(params, dir_name=name)
    path_out = os.path.join(path_out, name)
    if not os.path.isdir(path_out):
        os.mkdir(path_out)
    if visual:
        path_visu = path_out + SUFFIX_VISUAL
        if not os.path.isdir(path_visu):
            os.mkdir(path_visu)
    else:
        path_visu = None
    return path_out, path_visu


def try_segment_image(img_idx_path, params, classif, path_out, path_visu, show_debug_imgs=SHOW_DEBUG_IMAGES):
    try:
        return segment_image(img_idx_path, params, classif, path_out, path_visu, show_debug_imgs=show_debug_imgs)
    except Exception:
        logging.exception('segment_image')
        return '', None, None


def main_predict(path_classif, path_pattern_imgs, path_out, name='SEGMENT___', params_local=None):
    """ given trained classifier segment new images

    :param str path_classif:
    :param str path_pattern_imgs:
    :param str path_out:
    :param str name:
    """
    logging.getLogger().setLevel(logging.INFO)
    logging.info('running PREDICTION...')
    assert path_pattern_imgs is not None

    dict_classif = seg_clf.load_classifier(path_classif)
    classif = dict_classif['clf_pipeline']
    params = dict_classif['params']
    if params_local is not None:
        params.update({k: params_local[k] for k in params_local if k.startswith('path_') or k.startswith('gc_')})

    path_out, path_visu = prepare_output_dir(path_pattern_imgs, path_out, name, visual=params.get('visual', False))
    tl_expt.set_experiment_logger(path_out)
    logging.info(tl_expt.string_dict(params, desc='PARAMETERS'))

    paths_img = sorted(glob.glob(path_pattern_imgs))
    logging.info('found %i images on path "%s"', len(paths_img), path_pattern_imgs)

    logging.debug('run prediction...')
    show_debug_imgs = params.get('visual', False)
    _wrapper_segment = partial(
        try_segment_image,
        params=params,
        classif=classif,
        path_out=path_out,
        path_visu=path_visu,
        show_debug_imgs=show_debug_imgs,
    )
    list_img_path = list(zip([None] * len(paths_img), paths_img))
    iterate = tl_expt.WrapExecuteSequence(
        _wrapper_segment,
        list_img_path,
        nb_workers=params['nb_workers'],
        desc='segmenting images',
    )
    for _ in iterate:
        gc.collect()
        time.sleep(1)

    logging.info('prediction DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = arg_parse_params(SEGM_PARAMS)

    params = main_train(params)

    main_predict(params['path_classif'], params['path_predict_imgs'], params['path_exp'], params_local=params)

    logging.info('all DONE')

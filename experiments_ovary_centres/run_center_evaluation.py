"""
Attempt to detect egg centers in the segmented images from annotated data.
The output is list of potential center candidates

Sample usage::

    python run_center_evaluation.py -list none \
        -segs "data_images/drosophila_ovary_slice/segm/*.png" \
        -imgs "data_images/drosophila_ovary_slice/image/*.jpg" \
        -centers "results/detect-centers-predict_ovary/centers/*.csv" \
        -out results/detect-centers-predict_ovary

Copyright (C) 2016-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import time
import logging
import gc
from functools import partial

import pandas as pd
import numpy as np
from scipy import ndimage

import matplotlib
if os.environ.get('DISPLAY', '') == '' and matplotlib.rcParams['backend'] != 'agg':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.experiments as tl_expt
import imsegm.utilities.drawing as tl_visu
import imsegm.annotation as seg_annot
import run_center_candidate_training as run_train
import run_center_prediction as run_detect
import run_center_clustering as run_clust

# whether skip loading triplest CSV from previous run
FORCE_RELOAD = False
VISUAL_SEGM_CENTRES = True
EXPORT_ANNOT_EGGS = True
VISUAL_ANNOT_EGGS = True
# set distance in Z axis whetehr near sliuce may still bellong to the same egg
ANNOT_SLICE_DIST_TOL = seg_annot.ANNOT_SLICE_DIST_TOL
ANNOT_STAGES = [1, 2, 3, 4, 5]

FOLDER_ANNOT = 'annot_user_stage-%s'
FOLDER_ANNOT_VISUAL = 'annot_user_stage-%s___visual'
DEFAULT_PARAMS = run_train.CENTER_PARAMS
DEFAULT_PARAMS.update({
    'stages': [(1, 2, 3, 4, 5),
               (2, 3, 4, 5),
               (1, ), (2, ), (3, ), (4, ), (5, )],
    'path_list': '',
    'path_centers': os.path.join(os.path.dirname(DEFAULT_PARAMS['path_centers']),
                                 '*.csv'),
    'path_infofile': os.path.join(run_train.PATH_IMAGES,
                                  'info_ovary_images.txt'),
})

NAME_CSV_TRIPLES = run_train.NAME_CSV_TRIPLES
NAME_CSV_TRIPLES_TEMP = os.path.splitext(NAME_CSV_TRIPLES)[0] + '__TEMP.csv'
NAME_CSV_TRIPLES_STAT = os.path.splitext(NAME_CSV_TRIPLES)[0] + '__statistic.csv'
NAME_CSV_ANNOT_STAGE = 'annotation_user_stages_%s.csv'
NAME_CSV_STATISTIC = 'statistic_missed_annot_eggs.csv'
SLICE_NAME_GROUPING = 'stack_path'


def estimate_eggs_from_info(row_slice, mask_shape):
    """ finds all eggs for particular slice and mask them by ellipse annotated
    by ant, post and lat in the all info table

    :param row_slice:
    :param mask_shape:
    :return ndarray: ndarray
    """
    pos_ant, pos_lat, pos_post = tl_visu.parse_annot_rectangles(row_slice)
    list_masks = tl_visu.draw_eggs_rectangle(mask_shape, pos_ant, pos_lat,
                                             pos_post)
    mask_eggs = tl_visu.merge_object_masks(list_masks, overlap_thr=0.5)

    return mask_eggs


def compute_statistic_eggs_centres(dict_case, points, labels, mask_eggs,
                                   img=None, segm=None, path_out=None,
                                   col_prefix=''):
    """ compute statistic on missed detected eggs and multiple detection
    inside single egg

    :param dict_case:
    :param [[float]] points:
    :param [int] labels:
    :param ndarray mask_eggs:
    :param ndarray img: optional for visualisation purposes
    :param ndarray segm: optional for visualisation purposes
    :param str path_out: path to the output directory
    :param str col_prefix: column prefix
    :return {str: int}:
    """
    unique_eggs = [int(lb) for lb in np.unique(mask_eggs) if lb != 0]
    dict_case[col_prefix + 'eggs annot.'] = len(unique_eggs)
    centers = np.array(points)[labels == 1].astype(int)
    labels_eggs = [1] * len(centers)
    dict_case[col_prefix + 'eggs missed'] = 0
    dict_case[col_prefix + 'eggs multiple'] = 0
    for lb in unique_eggs:
        mask = (mask_eggs == lb)
        inside = mask[centers[:, 0], centers[:, 1]]
        if sum(inside) == 0:
            pos = ndimage.measurements.center_of_mass(mask)
            dict_case[col_prefix + 'eggs missed'] += 1
            centers = np.vstack((centers, list(map(int, pos))))
            labels_eggs.append(-1)
        elif sum(inside) > 1:
            dict_case[col_prefix + 'eggs multiple'] += 1
    labels_eggs = np.array(labels_eggs)

    # visualise missing eggs from annotation
    if os.path.isdir(path_out) and img is not None and segm is not None:
        run_train.export_show_image_points_labels(
            path_out, dict_case['image'], img, segm, centers, labels_eggs, None,
            mask_eggs, '_stat_eggs', dict_label_marker=tl_visu.DICT_LABEL_MARKER_FN_FP)
    return dict_case


def load_center_evaluate(idx_row, df_annot, path_annot, path_visu=None,
                         col_prefix=''):
    """ complete pipeline fon input image and seg_pipe, such that load them,
    generate points, compute features and using given classifier predict labels

    :param (int, DF:row) idx_row:
    :param df_annot:
    :param str path_annot:
    :param str path_visu:
    :param str col_prefix:
    :return {str: float}:
    """
    idx, row = idx_row
    dict_row = dict(row)
    dict_row['image'] = os.path.splitext(os.path.basename(dict_row['path_image']))[0]

    if idx not in df_annot.index:
        logging.debug('particular image/slice "%s" does not contain eggs '
                      'of selected stage %s', idx, col_prefix)
        return dict_row

    name, img, segm, centres = run_train.load_image_segm_center((None, row))
    if centres is None:
        logging.debug('center missing "%s"', idx)
        return dict_row

    assert all(c in df_annot.columns for c in tl_visu.COLUMNS_POSITION_EGG_ANNOT), \
        'some required columns %r are missing for %s' % \
        (tl_visu.COLUMNS_POSITION_EGG_ANNOT, df_annot.columns)
    mask_eggs = estimate_eggs_from_info(df_annot.loc[idx], img.shape[:2])

    try:
        if EXPORT_ANNOT_EGGS:
            path_img = os.path.join(path_annot, idx + '.png')
            tl_data.io_imsave(path_img, mask_eggs.astype(np.uint8))

        if VISUAL_ANNOT_EGGS:
            fig = tl_visu.figure_image_segm_results(img, mask_eggs)
            fig.savefig(os.path.join(path_visu, idx + '_eggs.png'))
            plt.close(fig)

        if VISUAL_SEGM_CENTRES:
            run_clust.export_draw_image_centers_clusters(path_visu, name, img,
                                                         centres, segm=segm)
        labels = np.array([1] * len(centres))
        dict_stat = compute_statistic_eggs_centres(dict_row, centres, labels,
                                                   mask_eggs, img, segm,
                                                   path_visu, col_prefix)
    except Exception:
        logging.exception('load_center_evaluate')
        dict_stat = dict_row
    return dict_stat


def evaluate_detection_stage(df_paths, stage, path_info, path_out, nb_workers=1):
    """ evaluate center detection for particular list of stages

    :param df_paths:
    :param [int] stage:
    :param str path_info:
    :param str path_out:
    :param int nb_workers:
    :return DF:
    """
    logging.info('evaluate stages: %r', stage)
    str_stage = '-'.join(map(str, stage))

    path_csv = os.path.join(path_out, NAME_CSV_ANNOT_STAGE % str_stage)
    if not os.path.exists(path_csv) or FORCE_RELOAD:
        df_slices_info = seg_annot.load_info_group_by_slices(path_info, stage)
        logging.debug('export slices_info to "%s"', path_csv)
        df_slices_info.to_csv(path_csv)
    else:
        logging.debug('loading slices_info from "%s"', path_csv)
        df_slices_info = pd.read_csv(path_csv, index_col=0)

    if df_slices_info.empty:
        return df_paths

    # df_paths = pd.merge(df_paths, df_slices_info, how='inner',
    #                     left_index=True, right_index=True)

    df_eval = pd.DataFrame()
    path_annot = os.path.join(path_out, FOLDER_ANNOT % str_stage)
    path_visu = os.path.join(path_out, FOLDER_ANNOT_VISUAL % str_stage)
    list_dirs = [os.path.basename(p) for p in [path_annot, path_visu]]
    logging.debug('create sub-dirs: %r', list_dirs)
    tl_expt.create_subfolders(path_out, list_dirs)

    # perfom on new images
    stage_prefix = '[stage-%s] ' % str_stage
    logging.info('start section %s - load_center_evaluate ...', stage_prefix)
    _wrapper_detection = partial(load_center_evaluate, df_annot=df_slices_info,
                                 path_annot=path_annot, path_visu=path_visu,
                                 col_prefix=stage_prefix)
    iterate = tl_expt.WrapExecuteSequence(_wrapper_detection,
                                          df_paths.iterrows(),
                                          nb_workers=nb_workers)
    for dict_eval in iterate:
        df_eval = df_eval.append(dict_eval, ignore_index=True)
        df_eval.to_csv(os.path.join(path_out, NAME_CSV_TRIPLES_TEMP))
        # gc.collect(), time.sleep(1)
    return df_eval


def main(params):
    """ PIPELINE for new detections

    :param dict params:
    """
    params['path_expt'] = os.path.join(params['path_output'],
                                       run_detect.FOLDER_EXPERIMENT % params['name'])
    tl_expt.set_experiment_logger(params['path_expt'])
    # tl_expt.create_subfolders(params['path_expt'], LIST_SUBDIRS)
    logging.info(tl_expt.string_dict(params, desc='PARAMETERS'))

    path_csv = os.path.join(params['path_expt'], NAME_CSV_TRIPLES)
    df_paths = run_detect.get_csv_triplets(params['path_list'], path_csv,
                                           params['path_images'],
                                           params['path_segms'],
                                           params['path_centers'], FORCE_RELOAD)

    df_eval = df_paths.copy(deep=True)
    for stage in params['stages']:
        df_eval = evaluate_detection_stage(df_eval, stage,
                                           params['path_infofile'],
                                           params['path_expt'],
                                           params['nb_workers'])
        if not df_eval.empty and 'image' in df_eval.columns:
            df_eval.set_index('image', inplace=True)
        df_eval.to_csv(os.path.join(params['path_expt'], NAME_CSV_TRIPLES_STAT))
        gc.collect()
        time.sleep(1)

    if not df_eval.empty:
        df_stat = df_eval.describe().transpose()
        logging.info('STATISTIC: \n %r', df_stat)
        df_stat.to_csv(os.path.join(params['path_expt'], NAME_CSV_STATISTIC))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('running...')

    params = run_train.arg_parse_params(DEFAULT_PARAMS)
    main(params)

    logging.info('DONE')

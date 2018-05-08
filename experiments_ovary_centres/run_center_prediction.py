"""
Attempt to detect egg centers in the segmented images from annotated data.
The output is list of potential center candidates.

SAMPLE run:
>> python run_center_prediction.py -list none \
    -segs "data_images/drosophila_ovary_slice/segm/*.png" \
    -imgs "data_images/drosophila_ovary_slice/image/*.jpg" \
    -centers results/detect-centers-train_ovary/classifier_RandForest.pkl \
    -out results -n ovary

Copyright (C) 2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import time, gc
import logging
import traceback
# import multiprocessing as mproc
from functools import partial

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')] # Add path to root
import imsegm.utils.experiments as tl_expt
import imsegm.utils.data_io as tl_data
import imsegm.classification as seg_clf
import run_center_candidate_training as run_train
import run_center_clustering as run_clust

FORCE_RERUN = False
NAME_CSV_TRIPLES = run_train.NAME_CSV_TRIPLES
NAME_CSV_TRIPLES_TEMP = os.path.splitext(NAME_CSV_TRIPLES)[0] + '__TEMP.csv'

FOLDER_INPUTS = 'inputs'
FOLDER_POINTS = run_train.FOLDER_POINTS
FOLDER_POINTS_VISU = run_train.FOLDER_POINTS_VISU
FOLDER_CENTRE = run_clust.FOLDER_CENTER
FOLDER_CLUSTER_VISUAL = run_clust.FOLDER_CLUSTER_VISUAL
LIST_SUBFOLDER = [FOLDER_INPUTS, FOLDER_POINTS, FOLDER_POINTS_VISU,
                  FOLDER_CENTRE, FOLDER_CLUSTER_VISUAL]
FOLDER_EXPERIMENT = 'detect-centers-predict_%s'

# This sampling only influnece the number of point to be evaluated in the image
DEFAULT_PARAMS = run_train.CENTER_PARAMS
DEFAULT_PARAMS.update(run_clust.CLUSTER_PARAMS)
DEFAULT_PARAMS['path_centers'] = os.path.join(DEFAULT_PARAMS['path_output'],
                                              run_train.FOLDER_EXPERIMENT % DEFAULT_PARAMS['name'],
                                      'classifier_RandForest.pkl')


def load_compute_detect_centers(idx_row, params, classif=None, path_classif='',
                                path_output=''):
    """ complete pipeline fon input image and seg_pipe, such that load them,
    generate points, compute features and using given classifier predict labels

    :param (int, DF:row) idx_row:
    :param {} params:
    :param obj classif:
    :param str path_classif:
    :param str path_output:
    :return {str: float}:
    """
    _, row = idx_row
    dict_center = dict(row)

    if classif is None:
        dict_classif = seg_clf.load_classifier(path_classif)
        classif = dict_classif['clf_pipeline']

    try:
        path_show_in = os.path.join(path_output, FOLDER_INPUTS)
        name, img, segm, _ = run_train.load_image_segm_center((None, row),
                                                              path_show_in,
                                                              params['dict_relabel'])
        t_start = time.time()
        _, slic, points, features, feature_names = \
                run_train.estim_points_compute_features(name, img, segm, params)
        dict_detect = run_train.detect_center_candidates(name, img, segm, None,
                                                         slic, points, features,
                                                         feature_names, params,
                                                         path_output, classif)
        dict_detect['time elapsed'] = time.time() - t_start
        dict_center.update(dict_detect)

        dict_center = run_clust.cluster_points_draw_export(dict_center, params,
                                                           path_output)
    except Exception:
        logging.error(traceback.format_exc())
    gc.collect()
    time.sleep(1)
    return dict_center


def get_csv_triplets(path_csv, path_csv_out, path_imgs, path_segs,
                     path_centers=None, force_reload=False):
    """ load triplets from CSV if it exists, otherwise crete such triplets
    from paths on particular directories

    :param str path_csv: path to existing triplets
    :param str path_csv_out:
    :param str path_imgs:
    :param str path_segs:
    :param str path_centers:
    :param bool force_reload:
    :return DF:
    """
    if os.path.isfile(path_csv):
        logging.info('loading path pairs "%s"', path_csv)
        df_paths = pd.read_csv(path_csv, index_col=0)
        df_paths['image'] = df_paths['path_image'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0])
        df_paths.set_index('image', inplace=True)
    elif os.path.isfile(path_csv_out) and not force_reload:
        logging.info('loading path pairs "%s"', path_csv_out)
        df_paths = pd.read_csv(path_csv_out, index_col=0)
    else:
        logging.info('estimating own triples')
        df_paths = run_train.find_match_images_segms_centers(
                                        path_imgs, path_segs, path_centers)
        df_paths['image'] = df_paths['path_image'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0])
        df_paths.set_index('image', inplace=True)
    for col in (c  for c in df_paths.columns if c.startswith('path_')):
        df_paths[col] = df_paths[col].apply(tl_data.update_path)
    df_paths.to_csv(path_csv_out)
    return df_paths


def main(params):
    """ PIPELINE for new detections

    :param {str: str} paths:
    """
    logging.info('running...')
    params = run_train.prepare_experiment_folder(params, FOLDER_EXPERIMENT)

    # run_train.check_pathes_patterns(paths)
    tl_expt.set_experiment_logger(params['path_expt'])
    logging.info('COMPUTER: \n%s', repr(os.uname()))
    logging.info(tl_expt.string_dict(params, desc='PARAMETERS'))

    tl_expt.create_subfolders(params['path_expt'], LIST_SUBFOLDER)

    path_csv = os.path.join(params['path_expt'], NAME_CSV_TRIPLES)
    df_paths = get_csv_triplets(params['path_list'], path_csv,
                                params['path_images'], params['path_segms'],
                                force_reload=FORCE_RERUN)

    dict_classif = seg_clf.load_classifier(params['path_classif'])
    params_clf = dict_classif['params']
    params_clf.update(params)
    logging.info(tl_expt.string_dict(params, desc='UPDATED PARAMETERS'))

    # perform on new images
    df_stat = pd.DataFrame()
    _wrapper_detection = partial(load_compute_detect_centers, params=params_clf,
                                 path_classif=params['path_classif'],
                                 path_output=params['path_expt'])
    iterate = tl_expt.WrapExecuteSequence(_wrapper_detection, df_paths.iterrows(),
                                          nb_jobs=params['nb_jobs'])
    for dict_center in iterate:
        df_stat = df_stat.append(dict_center, ignore_index=True)
        df_stat.to_csv(os.path.join(params['path_expt'], NAME_CSV_TRIPLES_TEMP))

    df_stat.set_index(['image'], inplace=True)
    df_stat.to_csv(os.path.join(params['path_expt'], NAME_CSV_TRIPLES))
    logging.info('STATISTIC: \n %s', repr(df_stat.describe()))

    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    params = run_train.arg_parse_params(DEFAULT_PARAMS)

    params['path_classif'] = params['path_centers']
    assert os.path.isfile(params['path_classif']), \
        'missing classifier: %s' % params['path_classif']

    main(params)

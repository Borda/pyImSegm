"""
Attempt to detect egg centers in the segmented images from annotated data
The inputs are:

1. 4-class segmentation of ovary images
    (background, nurse, follicular cells and cytoplasm)
2. annotation of egg centers as
  2a) csv list of centers
  2b) 3-class annotation:
        (i) for close center,
        (iii) too far and,
        (ii) something in between

The output is list of potential center candidates

Sample usage::

    python run_center_candidate_training.py -list none \
        -imgs "data_images/drosophila_ovary_slice/image/*.jpg" \
        -segs "data_images/drosophila_ovary_slice/segm/*.png" \
        -centers "data_images/drosophila_ovary_slice/center_levels/*.png" \
        -out results -n ovary

Copyright (C) 2016-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import argparse
import multiprocessing as mproc
from functools import partial

import tqdm
import pandas as pd
import numpy as np
from scipy import spatial

import matplotlib
if os.environ.get('DISPLAY', '') == '' and matplotlib.rcParams['backend'] != 'agg':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.experiments as tl_expt
import imsegm.utilities.drawing as tl_visu
import imsegm.superpixels as seg_spx
import imsegm.descriptors as seg_fts
import imsegm.classification as seg_clf
import imsegm.labeling as seg_lbs

# whether skip loading triplest CSV from previous run
FORCE_RELOAD = False
# even you have dumped data from previous time, all wil be recomputed
FORCE_RECOMP_DATA = False
EXPORT_TRAINING_DATA = True
# perform the Leave-One-Out experiment
RUN_LEAVE_ONE_OUT = True
# Set experiment folders
FOLDER_EXPERIMENT = 'detect-centers-train_%s'
FOLDER_INPUT = 'inputs_annot'
FOLDER_POINTS = 'candidates'
FOLDER_POINTS_VISU = 'candidates_visul'
FOLDER_POINTS_TRAIN = 'points_train'
LIST_SUBDIRS = [FOLDER_INPUT, FOLDER_POINTS,
                FOLDER_POINTS_VISU, FOLDER_POINTS_TRAIN]

NAME_CSV_TRIPLES = 'list_images_segms_centers.csv'
NAME_CSV_STAT_TRAIN = 'statistic_train_centers.csv'
NAME_YAML_PARAMS = 'configuration.yaml'
NAME_DUMP_TRAIN_DATA = 'dump_training_data.npz'

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))
# position is label in loaded segm and nb are out labels
LUT_ANNOT_CENTER_RELABEL = [0, 0, -1, 1]
CROSS_VAL_LEAVE_OUT_SEARCH = 0.2
CROSS_VAL_LEAVE_OUT_EVAL = 0.1

CENTER_PARAMS = {
    'computer': os.uname(),
    'slic_size': 25,
    'slic_regul': 0.3,
    # 'fts_hist_diams': None,
    # 'fts_hist_diams': [10, 25, 50, 75, 100, 150, 200, 250, 300],
    'fts_hist_diams': [10, 50, 100, 200, 300],
    # 'fts_ray_step': None,
    'fts_ray_step': 15,
    'fts_ray_types': [('up', [0])],
    # 'fts_ray_types': [('up', [0]), ('down', [1])],
    'fts_ray_closer': True,
    'fts_ray_smooth': 0,
    'pca_coef': None,
    # 'pca_coef': 0.99,
    'balance': 'unique',
    'classif': 'RandForest',
    # 'classif': 'SVM',
    'nb_classif_search': 50,
    'dict_relabel': None,
    # 'dict_relabel': {0: [0], 1: [1], 2: [2, 3]},
    'center_dist_thr': 50,  # distance to from annotated center as a point
}

PATH_IMAGES = os.path.join(tl_data.update_path('data_images'),
                           'drosophila_ovary_slice')
PATH_RESULTS = tl_data.update_path('results', absolute=True)
CENTER_PARAMS.update({
    'path_list': os.path.join(PATH_IMAGES,
                              'list_imgs-segm-center-levels_short.csv'),
    'path_images': '',
    'path_segms': '',
    'path_centers': '',
    # 'path_images': os.path.join(PATH_IMAGES, 'image', '*.jpg'),
    # 'path_segms': os.path.join(PATH_IMAGES, 'segm', '*.png'),
    # 'path_centers': os.path.join(PATH_IMAGES, 'center_levels', '*.png'),
    'path_infofile': '',
    'path_output': PATH_RESULTS,
    'name': 'ovary',
})


def arg_parse_params(params):
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return dict:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-list', '--path_list', type=str, required=False,
                        help='path to the list of input files',
                        default=params['path_list'])
    parser.add_argument('-imgs', '--path_images', type=str, required=False,
                        help='path to directory & name pattern for images',
                        default=params['path_images'])
    parser.add_argument('-segs', '--path_segms', type=str, required=False,
                        help='path to directory & name pattern for segmentation',
                        default=params['path_segms'])
    parser.add_argument('-centers', '--path_centers', type=str, required=False,
                        help='path to directory & name pattern for centres',
                        default=params['path_centers'])
    parser.add_argument('-info', '--path_infofile', type=str, required=False,
                        help='path to the global information file',
                        default=params['path_infofile'])
    parser.add_argument('-out', '--path_output', type=str, required=False,
                        help='path to the output directory',
                        default=params['path_output'])
    parser.add_argument('-n', '--name', type=str, required=False,
                        help='name of the experiment', default='ovary')
    parser.add_argument('-cfg', '--path_config', type=str, required=False,
                        help='path to the configuration', default=None)
    parser.add_argument('--nb_workers', type=int, required=False, default=NB_THREADS,
                        help='number of processes in parallel')
    params.update(vars(parser.parse_args()))
    paths = {}
    for k in (k for k in params if 'path' in k):
        if not isinstance(params[k], str) or params[k].lower() == 'none':
            paths[k] = ''
            continue
        if k in ['path_images', 'path_segms', 'path_centers', 'path_expt']:
            p_dir = tl_data.update_path(os.path.dirname(params[k]))
            paths[k] = os.path.join(p_dir, os.path.basename(params[k]))
        else:
            paths[k] = tl_data.update_path(params[k], absolute=True)
            p_dir = paths[k]
        assert os.path.exists(p_dir), 'missing (%s) %s' % (k, p_dir)
    # load saved configuration
    if params['path_config'] is not None:
        ext = os.path.splitext(params['path_config'])[-1]
        assert (ext == '.yaml' or ext == '.yml'), \
            'wrong extension for %s' % params['path_config']
        data = tl_expt.load_config_yaml(params['path_config'])
        params.update(data)
    params.update(paths)
    logging.info('ARG PARAMETERS: \n %r', params)
    return params


def is_drawing(path_out):
    """ check if the out folder exist and also if the process is in debug mode

    :param str path_out:
    :return bool:
    # """
    bool_res = path_out is not None and os.path.exists(path_out) \
        and logging.getLogger().isEnabledFor(logging.DEBUG)
    return bool_res


def find_match_images_segms_centers(path_pattern_imgs, path_pattern_segms,
                                    path_pattern_center=None):
    """ walk over dir with images and segmentation and pair those with the same
    name and if the folder with centers exists also add to each par a center

    .. note:: returns just paths

    :param str path_pattern_imgs:
    :param str path_pattern_segms:
    :param str path_pattern_center:
    :return DF: DF<path_img, path_segm, path_center>
    """
    logging.info('find match images-segms-centres...')
    list_paths = [path_pattern_imgs, path_pattern_segms, path_pattern_center]
    df_paths = tl_data.find_files_match_names_across_dirs(list_paths)

    if not path_pattern_center:
        df_paths.columns = ['path_image', 'path_segm']
        df_paths['path_centers'] = ''
    else:
        df_paths.columns = ['path_image', 'path_segm', 'path_centers']
    df_paths.index = range(1, len(df_paths) + 1)
    return df_paths


def get_idx_name(idx, path_img):
    """ create string identifier for particular image

    :param int idx: image index
    :param str path_img: image path
    :return str: identifier
    """
    im_name = os.path.splitext(os.path.basename(path_img))[0]
    if idx is not None:
        return '%03d_%s' % (idx, im_name)
    else:
        return im_name


def load_image_segm_center(idx_row, path_out=None, dict_relabel=None):
    """ by paths load images and segmentation and weather centers exist,
    load them if the path out is given redraw visualisation of inputs

    :param (int, DF:row) idx_row: tuple of index and row
    :param str path_out: path to output directory
    :param dict dict_relabel: look-up table for relabeling
    :return(str, ndarray, ndarray, [[int, int]]): idx_name, img_rgb, segm, centers
    """
    idx, row_path = idx_row
    for k in ['path_image', 'path_segm', 'path_centers']:
        row_path[k] = tl_data.update_path(row_path[k])
        assert os.path.exists(row_path[k]), 'missing %s' % row_path[k]

    idx_name = get_idx_name(idx, row_path['path_image'])
    img_struc, img_gene = tl_data.load_img_double_band_split(row_path['path_image'],
                                                             im_range=None)
    # img_rgb = np.array(Image.open(row_path['path_img']))
    img_rgb = tl_data.merge_image_channels(img_struc, img_gene)
    if np.max(img_rgb) > 1:
        img_rgb = img_rgb / float(np.max(img_rgb))

    seg_ext = os.path.splitext(os.path.basename(row_path['path_segm']))[-1]
    if seg_ext == '.npz':
        with np.load(row_path['path_segm']) as npzfile:
            segm = npzfile[npzfile.files[0]]
        if dict_relabel is not None:
            segm = seg_lbs.merge_probab_labeling_2d(segm, dict_relabel)
    else:
        segm = tl_data.io_imread(row_path['path_segm'])
        if dict_relabel is not None:
            segm = seg_lbs.relabel_by_dict(segm, dict_relabel)

    if row_path['path_centers'] is not None \
            and os.path.isfile(row_path['path_centers']):
        ext = os.path.splitext(os.path.basename(row_path['path_centers']))[-1]
        if ext == '.csv':
            centers = tl_data.load_landmarks_csv(row_path['path_centers'])
            centers = tl_data.swap_coord_x_y(centers)
        elif ext == '.png':
            centers = tl_data.io_imread(row_path['path_centers'])
            # relabel loaded segm into relevant one
            centers = np.array(LUT_ANNOT_CENTER_RELABEL)[centers]
        else:
            logging.warning('not supported file format %s', ext)
            centers = None
    else:
        centers = None

    if is_drawing(path_out):
        export_visual_input_image_segm(path_out, idx_name, img_rgb, segm, centers)

    return idx_name, img_rgb, segm, centers


def export_visual_input_image_segm(path_out, img_name, img, segm, centers=None):
    """ visualise the input image and segmentation in common frame

    :param str path_out: path to output directory
    :param str img_name: image name
    :param ndarray img: np.array
    :param ndarray segm: np.array
    :param centers: [(int, int)] or np.array
    """
    fig = tl_visu.figure_image_segm_centres(img, segm, centers)
    fig.savefig(os.path.join(path_out, img_name + '.png'),
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def compute_min_dist_2_centers(centers, points):
    """ compute distance toclosestt center and mark which center it is

    :param [int, int] centers:
    :param [int, int] points:
    :return (float, int):
    """
    dists = spatial.distance.cdist(np.array(points), np.array(centers),
                                   metric='euclidean')
    dist = np.min(dists, axis=1)
    idx = np.argmin(dists, axis=1)
    return dist, idx


def export_show_image_points_labels(path_out, img_name, img, seg, points,
                                    labels=None, slic=None, seg_centers=None,
                                    fig_suffix='', dict_label_marker=tl_visu.DICT_LABEL_MARKER):
    """ export complete visualisation of labeld point over rgb image and segm

    :param str path_out:
    :param str img_name:
    :param img: np.array
    :param seg: np.array
    :param [(int, int)] points:
    :param [int] labels:
    :param slic: np.array
    :param seg_centers:
    :param str fig_suffix:
    :param dict_label_marker:
    """
    points = np.array(points)

    fig, axarr = plt.subplots(ncols=2, figsize=(9 * 2, 6))
    img = img / float(np.max(img)) if np.max(img) > 1 else img
    tl_visu.draw_image_segm_points(axarr[0], img, points, labels,
                                   seg_contour=seg_centers,
                                   lut_label_marker=dict_label_marker)
    tl_visu.draw_image_segm_points(axarr[1], seg, points, labels, slic,
                                   seg_contour=seg_centers,
                                   lut_label_marker=dict_label_marker)
    fig.tight_layout()
    fig.savefig(os.path.join(path_out, img_name + fig_suffix + '.png'),
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def estim_points_compute_features(name, img, segm, params):
    """ determine points (center candidates) using slic
    and for each compute feature vector with their names

    :param str name:
    :param ndarray img:
    :param ndarray segm:
    :param {str: any} params:
    :return (str, ndarray, [(int, int)], [[float]], list(str)):
    """
    # superpixels on image
    assert img.shape[:2] == segm.shape[:2], \
        'not matching shapes: %r : %r' % (img.shape, segm.shape)
    slic = seg_spx.segment_slic_img2d(img, params['slic_size'], params['slic_regul'])
    slic_centers = seg_spx.superpixel_centers(slic)
    # slic_edges = seg_spx.make_graph_segm_connect_grid2d_conn4(slic)

    features, feature_names = compute_points_features(segm, slic_centers,
                                                      params)

    return name, slic, slic_centers, features, feature_names


def compute_points_features(segm, points, params):
    """ for each point in segmentation compute relevant features according params

    :param ndarray segm: segmentations
    :param [(int, int)] points: positions in image
    :param {str: any} params: parameters
    :return ([[float]], list(str)): [[float] * nb_features] * nb_points, list(str) * nb_features
    """
    features, feature_names = np.empty((len(points), 0)), list()

    # segmentation histogram
    if 'fts_hist_diams' in params and params['fts_hist_diams'] is not None:
        features_hist, names_hist = seg_fts.compute_label_histograms_positions(
            segm, points, diameters=params['fts_hist_diams'])
        features = np.hstack((features, features_hist))
        feature_names += names_hist

    names_ray = list()  # default empty, require some at leas one compute
    # Ray features
    if 'fts_ray_step' in params and params['fts_ray_step'] is not None:
        list_features_ray = []
        perform_closer = all((params.get('fts_ray_closer', False),
                              len(params['fts_ray_types']) > 1))
        shifting = not perform_closer
        for ray_edge, ray_border in params['fts_ray_types']:
            features_ray, _, names_ray = seg_fts.compute_ray_features_positions(
                segm, points, angle_step=params['fts_ray_step'], edge=ray_edge,
                border_labels=ray_border, smooth_ray=params['fts_ray_smooth'],
                shifting=shifting)

            # if closer, save all in temporray array else add to feature space
            if perform_closer:
                list_features_ray.append(features_ray)
            else:
                features = np.hstack((features, features_ray))
                feature_names += names_ray

        # take the closest ray and then perform the shifting
        if perform_closer:
            features_ray = [seg_fts.shift_ray_features(ray)[0] for ray
                            in np.min(np.array(list_features_ray), axis=0)]
            features = np.hstack((features, np.array(features_ray)))
            feature_names += names_ray

    return features, feature_names


def wrapper_estim_points_compute_features(name_img_segm, params):
    name, img, segm = name_img_segm
    return estim_points_compute_features(name, img, segm, params)


def label_close_points(centers, points, params):
    """ label points whether they are close to center by distance to real center
    or from annotation of close center regions

    :param ndarray|[(int, int)] centers:
    :param [(int, int)] points: positions in image
    :param {str: any} params: parameters
    :return [int]:
    """
    if isinstance(centers, list):
        min_dist, _ = compute_min_dist_2_centers(centers, points)
        labels = (min_dist <= params['center_dist_thr'])
    elif isinstance(centers, np.ndarray):
        mx_points = np.array(points, dtype=int)
        labels = centers[mx_points[:, 0], mx_points[:, 1]]
    else:
        logging.warning('not relevant centers info of type "%s"', type(centers))
        labels = [-1] * len(points)
    assert len(points) == len(labels), \
        'not equal lenghts of points (%i) and labels (%i)' \
        % (len(points), len(labels))
    return labels


def wrapper_draw_export_slic_centers(args):
    return export_show_image_points_labels(*args)


def dataset_load_images_segms_compute_features(params, df_paths, nb_workers=NB_THREADS):
    """ create whole dataset composed from loading input data, computing features
    and label points by label whether its positive or negative center candidate

    :param {str: any} params: parameters
    :param DF df_paths: DataFrame
    :param int nb_workers: parallel
    :return dict:
    """
    dict_imgs, dict_segms, dict_center = dict(), dict(), dict()
    logging.info('loading input data (images, segmentation and centers)')
    path_show_in = os.path.join(params['path_expt'], FOLDER_INPUT)
    _wrapper_load = partial(load_image_segm_center, path_out=path_show_in,
                            dict_relabel=params['dict_relabel'])
    iterate = tl_expt.WrapExecuteSequence(_wrapper_load, df_paths.iterrows(),
                                          nb_workers=nb_workers,
                                          desc='loading input data')
    for name, img, seg, center in iterate:
        dict_imgs[name] = img
        dict_segms[name] = seg
        dict_center[name] = center

    dict_slics, dict_points, dict_features = dict(), dict(), dict()
    logging.info('estimate candidate points and compute features')
    gene_name_img_seg = ((name, dict_imgs[name], dict_segms[name])
                         for name in dict_imgs)
    _wrapper_pnt_features = partial(wrapper_estim_points_compute_features,
                                    params=params)
    feature_names = None
    iterate = tl_expt.WrapExecuteSequence(_wrapper_pnt_features,
                                          gene_name_img_seg, nb_workers=nb_workers,
                                          desc='estimate candidates & features')
    for name, slic, points, features, feature_names in iterate:
        dict_slics[name] = slic
        dict_points[name] = points
        dict_features[name] = features
    logging.debug('computed features:\n %r', feature_names)

    dict_labels = dict()
    logging.info('assign labels according close distance to center')
    path_points_train = os.path.join(params['path_expt'], FOLDER_POINTS_TRAIN)
    tqdm_bar = tqdm.tqdm(total=len(dict_center), desc='labels assignment')
    for name in dict_center:
        dict_labels[name] = label_close_points(dict_center[name],
                                               dict_points[name], params)
        points = np.asarray(dict_points[name])[np.asarray(dict_labels[name]) == 1]
        path_csv = os.path.join(path_points_train, name + '.csv')
        tl_data.save_landmarks_csv(path_csv, points)

        tqdm_bar.update()
    tqdm_bar.close()

    return (dict_imgs, dict_segms, dict_slics, dict_points, dict_center,
            dict_features, dict_labels, feature_names)


def export_dataset_visual(path_output, dict_imgs, dict_segms, dict_slics,
                          dict_points, dict_labels, nb_workers=NB_THREADS):
    """ visualise complete training dataset by marking labeld points
    over image and input segmentation

    :param {str: ndarray} dict_imgs:
    :param {str: ndarray} dict_segms:
    :param {str: ndarray} dict_slics:
    :param {str: ndarray} dict_points:
    :param {str: ndarray} dict_labels:
    :param int nb_workers: number processing in parallel
    """
    logging.info('export training visualisations')

    path_out = os.path.join(path_output, FOLDER_POINTS_TRAIN)
    gener_args = ((path_out, name, dict_imgs[name], dict_segms[name],
                   dict_points[name], dict_labels[name], dict_slics[name],
                   None, '_train') for name in dict_imgs)
    iterate = tl_expt.WrapExecuteSequence(wrapper_draw_export_slic_centers,
                                          gener_args, nb_workers=nb_workers,
                                          desc='exporting visualisations')
    list(iterate)


def compute_statistic_centers(dict_stat, img, segm, center, slic, points, labels,
                              params, path_out=''):
    """ compute statistic on centers

    :param {str: float} dict_stat:
    :param ndarray img:
    :param ndarray segm:
    :param center:
    :param ndarray slic:
    :param points:
    :param labels:
    :param dict params:
    :param str path_out:
    :return dict:
    """
    labels_gt = label_close_points(center, points, params)

    mask_valid = (labels_gt != -1)
    points = np.asarray(points)[mask_valid, :].tolist()
    labels = labels[mask_valid]
    # proba = proba[mask_valid, :]
    labels_gt = labels_gt[mask_valid].astype(int)

    dict_stat.update(seg_clf.compute_classif_metrics(labels_gt, labels,
                                                     metric_averages=['binary']))
    dict_stat['points all'] = len(labels)
    dict_stat['points FP'] = np.sum(np.logical_and(labels == 1, labels_gt == 0))
    dict_stat['points FN'] = np.sum(np.logical_and(labels == 0, labels_gt == 1))
    # compute FP and FN to annotation
    labels_fn_fp = labels.copy()
    labels_fn_fp[np.logical_and(labels == 1, labels_gt == 0)] = -2
    labels_fn_fp[np.logical_and(labels == 0, labels_gt == 1)] = -1
    # visualise FP and FN to annotation
    if os.path.isdir(path_out):
        export_show_image_points_labels(path_out, dict_stat['image'], img, segm,
                                        points, labels_fn_fp, slic, center,
                                        '_FN-FP', tl_visu.DICT_LABEL_MARKER_FN_FP)
    return dict_stat


def detect_center_candidates(name, image, segm, centers_gt, slic, points,
                             features, feature_names, params, path_out, classif):
    """ for loaded or computer all necessary data, classify centers_gt candidates
    and if we have annotation validate this results

    :param str name:
    :param ndarray image:
    :param ndarray segm:
    :param centers_gt:
    :param slic: np.array
    :param [(int, int)] points:
    :param features:
    :param list(str) feature_names:
    :param dict params:
    :param str path_out:
    :param classif: obj
    :return dict:
    """
    labels = classif.predict(features)
    # proba = classif.predict_proba(features)

    candidates = np.asarray(points)[np.asarray(labels) == 1]

    path_points = os.path.join(path_out, FOLDER_POINTS)
    path_visu = os.path.join(path_out, FOLDER_POINTS_VISU)

    path_csv = os.path.join(path_points, name + '.csv')
    tl_data.save_landmarks_csv(path_csv, tl_data.swap_coord_x_y(candidates))
    export_show_image_points_labels(path_visu, name, image, segm, points,
                                    labels, slic, centers_gt)

    dict_centers = {'image': name,
                    'path_points': path_csv}
    if centers_gt is not None:
        dict_centers = compute_statistic_centers(dict_centers, image, segm,
                                                 centers_gt, slic, points, labels,
                                                 params, path_visu)
    return dict_centers


def wrapper_detect_center_candidates(data, params, path_output, classif):
    name, img, segm, center, slic, points, features, feature_names = data
    return detect_center_candidates(name, img, segm, center, slic, points,
                                    features, feature_names, params,
                                    path_output, classif)


def load_dump_data(path_dump_data):
    """ loading saved data prom previous stages

    :param path_dump_data:
    :return dict:
    """
    logging.info('loading dumped data "%s"', path_dump_data)
    # with open(os.path.join(path_out, NAME_DUMP_TRAIN_DATA), 'r') as f:
    #     dict_data = pickle.load(f)
    npz_file = np.load(path_dump_data, encoding='bytes')
    dict_imgs = dict(npz_file['dict_images'].tolist())
    dict_segms = dict(npz_file['dict_segms'].tolist())
    dict_slics = dict(npz_file['dict_slics'].tolist())
    dict_points = dict(npz_file['dict_points'].tolist())
    dict_features = dict(npz_file['dict_features'].tolist())
    dict_labels = dict(npz_file['dict_labels'].tolist())
    dict_centers = dict(npz_file['dict_centers'].tolist())
    feature_names = npz_file['feature_names'].tolist()
    return (dict_imgs, dict_segms, dict_slics, dict_points, dict_centers,
            dict_features, dict_labels, feature_names)


def save_dump_data(path_dump_data, imgs, segms, slics, points, centers,
                   features, labels, feature_names):
    """ loading saved data prom previous stages  """
    logging.info('save (dump) data to "%s"', path_dump_data)
    np.savez_compressed(path_dump_data, dict_images=imgs, dict_segms=segms,
                        dict_slics=slics, dict_points=points, dict_centers=centers,
                        dict_features=features, dict_labels=labels,
                        feature_names=feature_names, encoding='bytes')


def experiment_loo(classif, dict_imgs, dict_segms, dict_centers, dict_slics,
                   dict_points, dict_features, feature_names, params):
    logging.info('run LOO prediction on training data...')
    # test classif on images
    gener_data = ((n, dict_imgs[n], dict_segms[n], dict_centers[n],
                   dict_slics[n], dict_points[n], dict_features[n],
                   feature_names) for n in dict_imgs)
    _wrapper_detection = partial(wrapper_detect_center_candidates,
                                 params=params, classif=classif,
                                 path_output=params['path_expt'])
    df_stat = pd.DataFrame()
    iterate = tl_expt.WrapExecuteSequence(_wrapper_detection,
                                          gener_data, nb_workers=params['nb_workers'],
                                          desc='detect center candidates')
    for dict_stat in iterate:
        df_stat = df_stat.append(dict_stat, ignore_index=True)
        df_stat.to_csv(os.path.join(params['path_expt'], NAME_CSV_STAT_TRAIN))

    df_stat.set_index(['image'], inplace=True)
    df_stat.to_csv(os.path.join(params['path_expt'], NAME_CSV_STAT_TRAIN))
    logging.info('STATISTIC: \n %r', df_stat.describe().transpose())


def prepare_experiment_folder(params, dir_template):
    params['path_expt'] = os.path.join(params['path_output'],
                                       dir_template % params['name'])
    if not os.path.exists(params['path_expt']):
        assert os.path.isdir(os.path.dirname(params['path_expt'])), \
            'missing: %s' % os.path.dirname(params['path_expt'])
        logging.debug('creating missing folder: %s', params['path_expt'])
        os.mkdir(params['path_expt'])
    return params


def load_df_paths(params):
    path_csv = os.path.join(params['path_expt'], NAME_CSV_TRIPLES)
    if os.path.isfile(path_csv) and not FORCE_RELOAD:
        logging.info('loading path pairs "%s"', path_csv)
        df_paths = pd.read_csv(path_csv, encoding='utf-8', index_col=0)
    else:
        if os.path.isfile(params['path_list']):
            df_paths = pd.read_csv(params['path_list'], index_col=0,
                                   encoding='utf-8')
        else:
            df_paths = find_match_images_segms_centers(params['path_images'],
                                                       params['path_segms'],
                                                       params['path_centers'])
        df_paths.to_csv(path_csv, encoding='utf-8')
    df_paths.index = list(range(len(df_paths)))
    return df_paths, path_csv


def main_train(params):
    """ PIPELINE for training
    0) load triplets or create triplets from path to images, annotations
    1) load precomputed data or compute them now
    2) train classifier with hyper-parameters
    3) perform Leave-One-Out experiment

    :param {str: any} params:
    """
    params = prepare_experiment_folder(params, FOLDER_EXPERIMENT)

    tl_expt.set_experiment_logger(params['path_expt'])
    logging.info(tl_expt.string_dict(params, desc='PARAMETERS'))
    tl_expt.save_config_yaml(os.path.join(params['path_expt'], NAME_YAML_PARAMS), params)
    tl_expt.create_subfolders(params['path_expt'], LIST_SUBDIRS)

    df_paths, _ = load_df_paths(params)

    path_dump_data = os.path.join(params['path_expt'], NAME_DUMP_TRAIN_DATA)
    if not os.path.isfile(path_dump_data) or FORCE_RECOMP_DATA:
        (dict_imgs, dict_segms, dict_slics, dict_points, dict_centers,
         dict_features, dict_labels, feature_names) = \
            dataset_load_images_segms_compute_features(params, df_paths, params['nb_workers'])
        assert len(dict_imgs) > 0, 'missing images'
        save_dump_data(path_dump_data, dict_imgs, dict_segms, dict_slics, dict_points,
                       dict_centers, dict_features, dict_labels, feature_names)
    else:
        (dict_imgs, dict_segms, dict_slics, dict_points, dict_centers, dict_features,
         dict_labels, feature_names) = load_dump_data(path_dump_data)

    if is_drawing(params['path_expt']) and EXPORT_TRAINING_DATA:
        export_dataset_visual(params['path_expt'], dict_imgs, dict_segms, dict_slics,
                              dict_points, dict_labels, params['nb_workers'])

    # concentrate features, labels
    features, labels, sizes = seg_clf.convert_set_features_labels_2_dataset(
        dict_features, dict_labels, drop_labels=[-1], balance_type=params['balance'])
    # remove all bad values from features space
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = -1
    assert np.sum(sizes) == len(labels), \
        'not equal sizes (%d) and labels (%i)' \
        % (int(np.sum(sizes)), len(labels))

    # feature norm & train classification
    nb_holdout = int(np.ceil(len(sizes) * CROSS_VAL_LEAVE_OUT_SEARCH))
    cv = seg_clf.CrossValidateGroups(sizes, nb_holdout)
    classif, params['path_classif'] = seg_clf.create_classif_search_train_export(
        params['classif'], features, labels, cross_val=cv, params=params,
        feature_names=feature_names, nb_search_iter=params['nb_classif_search'],
        pca_coef=params.get('pca_coef', None), nb_workers=params['nb_workers'],
        path_out=params['path_expt'])
    nb_holdout = int(np.ceil(len(sizes) * CROSS_VAL_LEAVE_OUT_EVAL))
    cv = seg_clf.CrossValidateGroups(sizes, nb_holdout)
    seg_clf.eval_classif_cross_val_scores(params['classif'], classif, features, labels,
                                          cross_val=cv, path_out=params['path_expt'])
    seg_clf.eval_classif_cross_val_roc(params['classif'], classif, features, labels,
                                       cross_val=cv, path_out=params['path_expt'])

    if RUN_LEAVE_ONE_OUT:
        experiment_loo(classif, dict_imgs, dict_segms, dict_centers, dict_slics,
                       dict_points, dict_features, feature_names, params)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('run TRAINING...')

    params = arg_parse_params(CENTER_PARAMS)
    main_train(params)

    logging.info('DONE')

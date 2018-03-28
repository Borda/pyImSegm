"""
Evaluate superpixels quality regarding given annotation

SAMPLE run:
>> python run_eval_superpixels.py \
    -imgs "images/drosophila_ovary_slice/image/*.jpg" \
    -segm "images/drosophila_ovary_slice/annot_eggs/*.png" \
    --img_type 2d_gray \
    --slic_size 20 --slic_regul 0.25 --slico 0

Copyright (C) 2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import argparse
import logging
import multiprocessing as mproc
from functools import partial

import numpy as np
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utils.data_io as tl_data
import imsegm.utils.experiments as tl_expt
import imsegm.utils.drawing as tl_visu
import imsegm.superpixels as seg_spx
import imsegm.labeling as seg_lbs
from run_segm_slic_model_graphcut import load_image
from run_segm_slic_model_graphcut import TYPES_LOAD_IMAGE


NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))
PATH_IMAGES = tl_data.update_path(os.path.join('images', 'drosophila_ovary_slice'))
PATH_RESULTS = tl_data.update_path('results', absolute=True)
NAME_CSV_DISTANCES = 'measured_boundary_distances.csv'
PARAMS = {
    'path_images': os.path.join(PATH_IMAGES, 'image', '*.jpg'),
    'path_segms': os.path.join(PATH_IMAGES, 'annot_eggs', '*.png'),
    'path_out': os.path.join(PATH_RESULTS, 'compute_boundary_distances'),
    'img_type': '2d_gray',
}


def arg_parse_params(params):
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return {str: ...}:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', '--path_images', type=str, required=False,
                        help='path to directory & name pattern for image',
                        default=params['path_images'])
    parser.add_argument('-segm', '--path_segms', type=str, required=False,
                        help='path to directory & name pattern for annotation',
                        default=params['path_segms'])
    parser.add_argument('-out', '--path_out', type=str, required=False,
                        help='path to the output directory',
                        default=params['path_out'])
    parser.add_argument('--img_type', type=str, required=False,
                        default=params['img_type'], choices=TYPES_LOAD_IMAGE,
                        help='type of image to be loaded')
    parser.add_argument('--slic_size', type=int, required=False,
                        default=20, help='superpixels size')
    parser.add_argument('--slic_regul', type=float, required=False,
                        default=0.25, help='superpixel regularization')
    parser.add_argument('--slico', action='store_true', required=False,
                        default=False, help='using SLICO (ASLIC)')
    parser.add_argument('--nb_jobs', type=int, required=False, default=NB_THREADS,
                        help='number of processes in parallel')
    params = vars(parser.parse_args())
    logging.info('ARG PARAMETERS: \n %s', repr(params))
    for k in (k for k in params if 'path' in k):
        params[k] = tl_data.update_path(params[k])
        if k == 'path_out' and not os.path.isdir(params[k]):
            params[k] = ''
            continue
        p = os.path.dirname(params[k]) if '*' in params[k] else params[k]
        assert os.path.exists(p), 'missing: (%s) "%s"' % (k, p)
    # if the config path is set load the it otherwise use default
    return params


def compute_boundary_distance(idx_row, params, path_out=''):
    """ compute nearest distance between two segmentation contours

    :param (int, str) idx_row:
    :param {} params:
    :param str path_out:
    :return (str, float):
    """
    _, row = idx_row
    name = os.path.splitext(os.path.basename(row['path_image']))[0]
    img = load_image(row['path_image'], params['img_type'])
    segm = load_image(row['path_segm'], 'segm')

    logging.debug('segment SLIC...')
    slic = seg_spx.segment_slic_img2d(img,
                                      params['slic_size'],
                                      params['slic_regul'],
                                      params['slico'])
    _, dists = seg_lbs.compute_boundary_distances(segm, slic)

    if os.path.isdir(path_out):
        logging.debug('visualise results...')
        fig = tl_visu.figure_segm_boundary_dist(segm, slic)
        fig.savefig(os.path.join(path_out, name + '.jpg'))

    return name, np.mean(dists)


def main(params):
    """ compute the distance among segmented superpixels and given annotation

    :param {str: ...} params:
    """
    logging.info('running...')

    if os.path.isdir(params['path_out']):
        logging.info('Missing output dir -> no visual export & results table.')

    list_paths = [params['path_images'], params['path_segms']]
    df_paths = tl_data.find_files_match_names_across_dirs(list_paths)
    df_paths.columns = ['path_image', 'path_segm']

    df_dist = pd.DataFrame()

    wrapper_eval = partial(compute_boundary_distance, params=params,
                           path_out=params['path_out'])
    iterate = tl_expt.WrapExecuteSequence(wrapper_eval, df_paths.iterrows(),
                                          nb_jobs=params['nb_jobs'],
                                          desc='evaluate SLIC')
    for name, dist in iterate:
        df_dist = df_dist.append({'name': name, 'mean boundary distance': dist},
                                 ignore_index=True)
    df_dist.set_index('name', inplace=True)

    if os.path.isdir(params['path_out']):
        df_dist.to_csv(os.path.join(params['path_out'], NAME_CSV_DISTANCES))
    logging.info('STATISTIC:')
    logging.info(df_dist.describe())

    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    params = arg_parse_params(PARAMS)

    main(params)

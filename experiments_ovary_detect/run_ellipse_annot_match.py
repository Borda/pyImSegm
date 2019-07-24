"""
Attempt to find best matching between annotation and ellipses
- reconstruct the user annotation and compare it with estimated ellipses

Sample usage::

    python run_ellipse_annot_match.py \
        -info ~/Medical-drosophila/all_ovary_image_info_for_prague.txt \
        -ells ~/Medical-drosophila/RESULTS/3_ellipse_ransac_crit_params/*.csv \
        -out ~/Medical-drosophila/RESULTS

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import logging
import argparse
from functools import partial

import pandas as pd
import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.experiments as tl_expt
import imsegm.utilities.drawing as tl_visu
# import segmentation.annotation as seg_annot
import imsegm.ellipse_fitting as ell_fit

NAME_CSV_RESULTS = 'info_ovary_images_ellipses.csv'
OVERLAP_THRESHOLD = 0.

NB_WORKERS = tl_expt.nb_workers(0.8)
PATH_IMAGES = tl_data.update_path(os.path.join('data_images', 'drosophila_ovary_slice'))

DEFAULT_PARAMS = {
    'path_ellipses': os.path.join(PATH_IMAGES, 'ellipse_fitting', '*.csv'),
    'path_infofile': os.path.join(PATH_IMAGES, 'info_ovary_images.txt'),
    'path_output': tl_data.update_path('results', absolute=True),
}


def arg_parse_params(params):
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return dict:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', '--path_images', type=str, required=False,
                        help='path to directory & name pattern for images',
                        default=params.get('path_images', None))
    parser.add_argument('-ells', '--path_ellipses', type=str, required=False,
                        help='path to directory & name pattern for ellipses',
                        default=params.get('path_ellipses', None))
    parser.add_argument('-info', '--path_infofile', type=str, required=False,
                        help='path to the global information file',
                        default=params.get('path_infofile', None))
    parser.add_argument('-out', '--path_output', type=str, required=False,
                        help='path to the output directory',
                        default=params.get('path_output', None))
    parser.add_argument('--nb_workers', type=int, required=False, default=NB_WORKERS,
                        help='number of processes in parallel')
    arg_params = vars(parser.parse_args())
    params.update(arg_params)
    for k in (k for k in params if 'path' in k and params[k] is not None):
        params[k] = tl_data.update_path(params[k], absolute=True)
    logging.info('ARG PARAMETERS: \n %r', params)
    return params


def select_optimal_ellipse(idx_row, path_dir_csv, overlap_thr=OVERLAP_THRESHOLD):
    """ reconstruct the user annotation and compare it with estimated ellipses

    :param (int, row) idx_row: index and row with user annotation
    :param str path_dir_csv: path to list of ellipse parameters
    :param float overlap_thr: skip all annot. which Jaccard lower then threshold
    :return dict:
    """
    _, row = idx_row
    dict_row = dict(row)
    path_csv = os.path.join(path_dir_csv, row['image_name'] + '.csv')
    df_ellipses = pd.read_csv(path_csv, index_col=0)

    pos = row[list[tl_visu.COLUMNS_POSITION_EGG_ANNOT]]
    max_size = 2 * max(pos) + min(pos)

    pos_ant = [[row['ant_x'], row['ant_y']]]
    pos_lat = [[row['lat_x'], row['lat_y']]]
    pos_post = [[row['post_x'], row['post_y']]]
    mask_ref = tl_visu.draw_eggs_rectangle((max_size, max_size),
                                           pos_ant, pos_lat, pos_post)[0]

    list_jaccard = []
    for _, ell_row in df_ellipses.iterrows():
        mask_ell = ell_fit.add_overlap_ellipse(np.zeros(mask_ref.shape),
                                               ell_row.values.tolist(), 1)
        union = np.sum(np.logical_and(mask_ref, mask_ell))
        intersect = np.sum(np.logical_or(mask_ref, mask_ell))
        list_jaccard.append(union / float(intersect))

    # if no match with annotation
    if max(list_jaccard) < overlap_thr:
        dict_row['ellipse_Jaccard'] = max(list_jaccard)
        return dict_row

    idx_best = np.argmax(list_jaccard)
    ell_best = dict(df_ellipses.iloc[idx_best])

    # optional swap main axis and rotate by 90 deg
    if ell_best['b'] > ell_best['a']:
        ell_best['a'], ell_best['b'] = ell_best['b'], ell_best['a']
        ell_best['theta'] += np.deg2rad(90)

    ell_best['Jaccard'] = max(list_jaccard)
    # add to each name ellipse prefix
    dict_row.update({'ellipse_' + n: ell_best[n] for n in ell_best})

    return dict_row


def filter_table(df_info, path_pattern):
    """ filter the table according existing files in given folder

    :param df_info:
    :param str path_pattern: path and pattern to selected images
    :return DF: filterd dataframe
    """
    list_name = [os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(path_pattern) if os.path.isfile(p)]
    logging.info('loaded item in table %i and found %i in dir'
                 % (len(df_info), len(list_name)))

    df_info['image_name'] = [os.path.splitext(p)[0]
                             for p in df_info['image_path']]
    df_info = df_info[df_info['image_name'].isin(list_name)]

    return df_info


def main(params):
    """ PIPELINE for matching

    :param {str: str} params:
    """
    logging.info(tl_expt.string_dict(params, desc='PARAMETERS'))

    df_info = pd.read_csv(params['path_infofile'], sep='\t', index_col=0)
    df_info = filter_table(df_info, params['path_ellipses'])
    logging.info('filtered %i item in table' % len(df_info))
    path_csv = os.path.join(params['path_output'], NAME_CSV_RESULTS)

    list_evals = []
    # get the folder
    path_dir_csv = os.path.dirname(params['path_ellipses'])
    _wrapper_match = partial(select_optimal_ellipse,
                             path_dir_csv=path_dir_csv)
    iterate = tl_expt.WrapExecuteSequence(_wrapper_match, df_info.iterrows(),
                                          nb_workers=params['nb_workers'])
    for i, dict_row in enumerate(iterate):
        list_evals.append(dict_row)
        # every hundreds iteration do export
        if i % 100 == 0:
            pd.DataFrame(list_evals).to_csv(path_csv)

    df_ellipses = pd.DataFrame(list_evals)
    df_ellipses.to_csv(path_csv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('running...')

    params = arg_parse_params(DEFAULT_PARAMS)
    main(params)

    logging.info('DONE')

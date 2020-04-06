"""
Estimate the normal size per stage, cut these images and norm them

Sample usage::

    python run_ellipse_cut_scale.py \
        -info ~/Medical-drosophila/RESULTS/info_ovary_images_ellipses.csv \
        -imgs ~/Medical-drosophila/RESULTS/0_input_images_png/*.png \
        -out ~/Medical-drosophila/RESULTS/images_cut_ellipse_stages

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import glob
import logging
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from skimage import transform

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.experiments as tl_expt
import imsegm.ellipse_fitting as ell_fit
import run_ellipse_annot_match as r_match

COLUMNS_ELLIPSE = ['ellipse_xc', 'ellipse_yc',
                   'ellipse_a', 'ellipse_b',
                   'ellipse_theta']
OVERLAP_THRESHOLD = 0.45
NORM_FUNC = np.median  # other options - mean, max, ...

NB_WORKERS = tl_expt.nb_workers(0.8)
PATH_IMAGES = tl_data.update_path(os.path.join('data_images', 'drosophila_ovary_slice'))
PATH_RESULTS = tl_data.update_path('results', absolute=True)

DEFAULT_PARAMS = {
    'path_images': os.path.join(PATH_IMAGES, 'image', '*.jpg'),
    'path_infofile': os.path.join(PATH_IMAGES, 'info_ovary_images_ellipses.csv'),
    'path_output': os.path.join(PATH_RESULTS, 'cut_stages'),
}


def extract_ellipse_object(idx_row, path_images, path_out, norm_size):
    """ cut the image selection according ellipse parameters
    and scale it into given size to have all image in the end the same sizes

    :param (int, row) idx_row: index and row with ellipse parameters
    :param str path_images: path to the image folder
    :param str path_out: path to output folder
    :param (int, int) norm_size: output image size
    """
    _, row = idx_row
    # select image with this name and any extension
    list_imgs = glob.glob(os.path.join(path_images, row['image_name'] + '.*'))
    path_img = sorted(list_imgs)[0]
    img, _ = tl_data.load_image_2d(path_img)

    # create mask according to chosen ellipse
    ell_params = row[COLUMNS_ELLIPSE].tolist()
    mask = ell_fit.add_overlap_ellipse(np.zeros(img.shape[:2], dtype=int),
                                       ell_params, 1)

    # cut the particular image
    img_cut = tl_data.cut_object(img, mask, 0, use_mask=True, bg_color=None)

    # scaling according to the normal size
    img_norm = transform.resize(img_cut, norm_size)

    path_img = os.path.join(path_out, os.path.basename(path_img))
    tl_data.export_image(path_img, img_norm)


def perform_stage(df_group, stage, path_images, path_out):
    """ perform cutting images for a particular development stage
    and nom them into common image size

    :param df_group: input dataframe with ellipse parameters
    :param int stage: index of development stage
    :param str path_images: path to the image folder
    :param str path_out: path to the output folder
    """
    logging.info('stage %i listing %i items' % (stage, len(df_group)))
    stat_a = NORM_FUNC(df_group['ellipse_a'])
    stat_b = NORM_FUNC(df_group['ellipse_b'])
    norm_size = (int(stat_b), int(stat_a))
    logging.info('normal dimension is %r' % norm_size)

    path_out_stage = os.path.join(path_out, str(stage))
    if not os.path.isdir(path_out_stage):
        os.mkdir(path_out_stage)

    _wrapper_object = partial(extract_ellipse_object, path_images=path_images,
                              path_out=path_out_stage, norm_size=norm_size)
    desc = 'stage %i - size %s' % (stage, norm_size)
    iterate = tl_expt.WrapExecuteSequence(_wrapper_object, df_group.iterrows(),
                                          nb_workers=params['nb_workers'],
                                          desc=desc)
    list(iterate)


def main(params):
    """ PIPELINE for matching

    :param {str: str} params:
    """
    # tl_expt.set_experiment_logger(params['path_expt'])
    # tl_expt.create_subfolders(params['path_expt'], LIST_SUBDIRS)
    logging.info(tl_expt.string_dict(params, desc='PARAMETERS'))

    if not os.path.isdir(params['path_output']):
        os.mkdir(params['path_output'])

    df_info = pd.read_csv(params['path_infofile'], index_col=0)
    df_info = r_match.filter_table(df_info, params['path_images'])
    df_info.dropna(inplace=True)
    df_info = df_info[df_info['ellipse_Jaccard'] >= OVERLAP_THRESHOLD]
    logging.info('filtered %i item in table' % len(df_info))

    # execute over groups per stage
    path_dir_imgs = os.path.dirname(params['path_images'])
    for stage, df_stage in df_info.groupby('stage'):
        perform_stage(df_stage, stage, path_dir_imgs, params['path_output'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = r_match.arg_parse_params(DEFAULT_PARAMS)
    main(params)

    logging.info('DONE')

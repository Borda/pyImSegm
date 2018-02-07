"""
Rotate the extracted eggs according major mass in main diagonal

SAMPLE run:
>> python run_egg_swap_orientation.py \
    -imgs ~/Medical-drosophila/RESULTS/images_cut_ellipse_stages/2/*.png \
    -out ~/Medical-drosophila/RESULTS/images_cut_ellipse_stages/2

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import logging
import multiprocessing as mproc
from functools import partial

import tqdm
import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import segmentation.utils.data_io as tl_io
import segmentation.utils.experiments as tl_expt
import run_ellipse_annot_match as r_match

IMAGE_CHANNEL = 0  # image channel for mass extraction

NB_THREADS = max(1, int(mproc.cpu_count() * 0.8))
PATH_IMAGES = tl_io.update_path(os.path.join('images', 'drosophila_ovary_slice'))
PATH_RESULTS = tl_io.update_path('results', absolute=True)

PARAMS = {
    'path_images': os.path.join(PATH_IMAGES, 'image_cut-stage-2', '*.png'),
    'path_output': os.path.join(PATH_RESULTS, 'image_cut-stage-2'),
}


def perform_orientation_swap(path_img, path_out):
    """ compute the density in front adn back part of the egg rotate eventually
    we split the egg into thirds instead half because the middle part variate

    :param str path_img:
    :param str path_out:
    """
    img, _ = tl_io.load_image_2d(path_img)

    part = img.shape[1] / 3
    sel_mask = img[:, :, IMAGE_CHANNEL] > np.min(img[:, :, IMAGE_CHANNEL])
    norm_val = np.mean(img[sel_mask, IMAGE_CHANNEL])
    val_left = np.sum(img[:, :part, IMAGE_CHANNEL] > norm_val)
    val_fight = np.sum(img[:, -part:, IMAGE_CHANNEL] > norm_val)
    ration = val_left / float(val_fight)
    # ration = STAT_FUNC(img[:, :half, IMAGE_CHANNEL]) \
    #          / float(STAT_FUNC(img[:, half:, IMAGE_CHANNEL]))

    if ration > 1.:
        img = img[::-1, ::-1, :]

    path_img = os.path.join(path_out, os.path.basename(path_img))
    tl_io.export_image(path_img, img)


def main(params):
    """ PIPELINE for rotation

    :param {str: str} params:
    """
    logging.info('running...')

    logging.info(tl_expt.string_dict(params, desc='PARAMETERS'))

    list_imgs = sorted([p for p in glob.glob(params['path_images'])
                        if os.path.isfile(p)])
    logging.info('found images: %i' % len(list_imgs))

    if not os.path.isdir(params['path_output']):
        os.mkdir(params['path_output'])

    tqdm_bar = tqdm.tqdm(total=len(list_imgs),
                         desc=os.path.dirname(params['path_images']))
    if params['nb_jobs'] > 1:
        wrapper_object = partial(perform_orientation_swap,
                                 path_out=params['path_output'])
        mproc_pool = mproc.Pool(params['nb_jobs'])
        for _ in mproc_pool.imap_unordered(wrapper_object, list_imgs):
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()
    else:
        for p_img in list_imgs:
            perform_orientation_swap(p_img, path_out=params['path_output'])
            tqdm_bar.update()

    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    params = r_match.arg_parse_params(PARAMS)
    main(params)

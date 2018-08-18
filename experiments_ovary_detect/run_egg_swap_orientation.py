"""
Rotate the extracted eggs according major mass in main diagonal

SAMPLE run:
>> python run_egg_swap_orientation.py \
    -imgs "~/Medical-drosophila/RESULTS/images_cut_ellipse_stages/2/*.png" \
    -out ~/Medical-drosophila/RESULTS/images_cut_ellipse_stages/2

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import logging
import multiprocessing as mproc
from functools import partial

import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.experiments as tl_expt
import run_ellipse_annot_match as r_match

IMAGE_CHANNEL = 0  # image channel for mass extraction

NB_THREADS = max(1, int(mproc.cpu_count() * 0.8))
PATH_IMAGES = os.path.join(tl_data.update_path('data_images'),
                           'drosophila_ovary_slice')
PATH_RESULTS = tl_data.update_path('results', absolute=True)
SWAP_CONDITION = 'cc'
DEFAULT_PARAMS = {
    'path_images': os.path.join(PATH_IMAGES, 'image_cut-stage-2', '*.png'),
    'path_output': os.path.join(PATH_RESULTS, 'image_cut-stage-2'),
}


def perform_orientation_swap(path_img, path_out, img_template,
                             swap_type=SWAP_CONDITION):
    """ compute the density in front adn back part of the egg rotate eventually
    we split the egg into thirds instead half because the middle part variate

    :param str path_img: path to input image
    :param str path_out: path to output folder
    :param ndarray img_template: template / mean image
    :param str swap_type: used swap condition
    """
    img, _ = tl_data.load_image_2d(path_img)
    # cut the same image
    img_size = img_template.shape
    img = img[:img_size[0], :img_size[1]]

    if swap_type == 'cc':
        b_swap = condition_swap_correl(img, img_template)
    else:
        b_swap = condition_swap_density(img)

    if b_swap:
        img = img[::-1, ::-1, :]

    path_img = os.path.join(path_out, os.path.basename(path_img))
    tl_data.export_image(path_img, img)


def condition_swap_density(img):
    part = int(img.shape[1] / 3)
    sel_mask = img[:, :, IMAGE_CHANNEL] > np.min(img[:, :, IMAGE_CHANNEL])
    norm_val = np.mean(img[sel_mask, IMAGE_CHANNEL])
    val_left = np.sum(img[:, :part, IMAGE_CHANNEL] > norm_val)
    val_fight = np.sum(img[:, -part:, IMAGE_CHANNEL] > norm_val)
    ration = val_left / float(val_fight)
    # ration = STAT_FUNC(img[:, :half, IMAGE_CHANNEL]) \
    #          / float(STAT_FUNC(img[:, half:, IMAGE_CHANNEL]))
    return ration > 1.


def condition_swap_correl(img, img_template):
    cc = correlation_coefficient(img[:, :, IMAGE_CHANNEL], img_template)
    cc_swap = correlation_coefficient(img[::-1, ::-1, IMAGE_CHANNEL],
                                      img_template)
    return cc < cc_swap


def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


def compute_mean_image(list_img_paths):
    iterate = tl_expt.WrapExecuteSequence(tl_data.load_image_2d, list_img_paths,
                                          desc='compute mean image')
    imgs = [im[:, :, IMAGE_CHANNEL] for im, _ in iterate]
    min_size = np.min([img.shape for img in imgs], axis=0)
    imgs = [img[:min_size[0], :min_size[1]] for img in imgs]
    img_mean = np.median(imgs, axis=0)
    return img_mean


def main(params):
    """ PIPELINE for rotation

    :param {str: str} params:
    """
    logging.info(tl_expt.string_dict(params, desc='PARAMETERS'))

    list_img_paths = sorted([p for p in glob.glob(params['path_images'])
                             if os.path.isfile(p)])
    logging.info('found images: %i' % len(list_img_paths))

    if not os.path.isdir(params['path_output']):
        os.mkdir(params['path_output'])

    img_mean = compute_mean_image(list_img_paths)

    _wrapper_object = partial(perform_orientation_swap,
                              path_out=params['path_output'],
                              img_template=img_mean)
    dir_name = os.path.dirname(params['path_images'])
    iterate = tl_expt.WrapExecuteSequence(_wrapper_object, list_img_paths,
                                          nb_jobs=params['nb_jobs'],
                                          desc=dir_name)
    list(iterate)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = r_match.arg_parse_params(DEFAULT_PARAMS)
    main(params)

    logging.info('DONE')

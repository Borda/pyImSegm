"""
Quantize annotation, common application:
 * remove some noise in the image
 * gradients along edges

NOTE, for JPEG there is always some smoothing so only allowed format is PNG

SAMPLE run:
>> python run_image_color_quantization.py \
    -imgs "images/drosophila_ovary_slice/segm_rgb/*.png" \
    -m position

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import glob
import logging
import argparse
import multiprocessing as mproc
from functools import partial

import numpy as np
import tqdm

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utils.data_io as tl_data
import imsegm.utils.experiments as tl_expt
import imsegm.annotation as seg_annot

PATH_IMAGES = os.path.join('images', 'drosophila_ovary_slice', 'segm_rgb', '*.png')
NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))
THRESHOLD_INVALID_PIXELS = 5e-3


def parse_arg_params():
    """ create simple arg parser with default values (input, results, dataset)

    :return obj: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', '--path_images', type=str, required=True,
                        help='path to dir with images', default=PATH_IMAGES)
    parser.add_argument('-m', '--method', type=str, required=False,
                        help='method for quantisation color/position', default='color',
                        choices=['color', 'position'])
    parser.add_argument('-thr', '--px_threshold', type=float, required=False,
                        help='percentage of pixels of a color to be removed',
                        default=THRESHOLD_INVALID_PIXELS)
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of jobs in parallel', default=NB_THREADS)
    args = vars(parser.parse_args())
    p_dir = tl_data.update_path(os.path.dirname(args['path_images']))
    assert os.path.isdir(p_dir), 'missing folder: %s' % args['path_images']
    args['path_images'] = os.path.join(p_dir, os.path.basename(args['path_images']))
    logging.info(tl_expt.string_dict(args, desc='ARG PARAMETERS'))
    return args


def see_images_color_info(path_images, px_thr=THRESHOLD_INVALID_PIXELS):
    """ look to the folder on all images and estimate most frequent colours

    :param path_dir: str
    :param im_pattern: str
    :param px_th: float, percentage of nb clr pixels to be assumed as important
    :return {}:
    """
    if not os.path.isdir(os.path.dirname(path_images)):
        logging.error('input folder does not exist')
        return {}
    paths_img = sorted(glob.glob(path_images))
    logging.debug('found %i images', len(paths_img))
    dict_colors = seg_annot.dir_images_frequent_colors(paths_img, px_thr)
    return dict_colors


def perform_quantize_image(path_image, list_colors, method='color'):
    """ perform the quantization together with loading and exporting

    :param str path_image:
    :param [(int, int, int)] list_colors: list of possible colours
    """
    logging.debug('quantize img: "%s"', path_image)
    im = tl_data.io_imread(path_image)
    assert im.ndim == 3, 'not valid color image of dims %s' % repr(im.shape)
    im = im[:, :, :3]
    # im = io.imread(path_image)[:, :, :3]
    if method == 'color':
        im_q = seg_annot.quantize_image_nearest_color(im, list_colors)
    elif method == 'position':
        im_q = seg_annot.quantize_image_nearest_pixel(im, list_colors)
    else:
        logging.error('not implemented method "%s"', method)
    path_image = os.path.splitext(path_image)[0] + '.png'
    tl_data.io_imsave(path_image, im_q.astype(np.uint8))
    # io.imsave(path_image, im_q)
    # plt.subplot(121), plt.imshow(im)
    # plt.subplot(122), plt.imshow(im_q)
    # plt.show()


def quantize_folder_images(path_images, list_colors=None, method='color',
                           px_threshold=THRESHOLD_INVALID_PIXELS, nb_jobs=1):
    """ perform single or multi thread image quantisation

    :param str method:
    :param float px_threshold:
    :param str path_images:, input directory and image pattern for loading
    :param list_colors: [(int, int, int)], list of possible colours
    :param str method: interpolation method
    :param float px_threshold:
    :param nb_jobs: int
    """
    path_imgs = sorted(glob.glob(path_images))
    logging.info('found %i images', len(path_imgs))
    if list_colors is None:
        dict_colors = see_images_color_info(path_images, px_thr=px_threshold)
        list_colors = [c for c in dict_colors]

    wrapper_quantize_img = partial(perform_quantize_image,
                                   method=method, list_colors=list_colors)
    tqdm_bar = tqdm.tqdm(total=len(path_imgs), desc='quantize images')

    if nb_jobs > 1:
        logging.debug('perform_sequence in %i threads', nb_jobs)
        mproc_pool = mproc.Pool(nb_jobs)
        for _ in mproc_pool.imap_unordered(wrapper_quantize_img, path_imgs):
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()
    else:
        for _ in map(wrapper_quantize_img, path_imgs):
            tqdm_bar.update()


def main(params):
    """ the main_train entry point   """
    logging.info('running...')
    quantize_folder_images(params['path_images'], method=params['method'],
                           px_threshold=params['px_threshold'],
                           nb_jobs=params['nb_jobs'])
    logging.info('DONE')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = parse_arg_params()
    main(params)
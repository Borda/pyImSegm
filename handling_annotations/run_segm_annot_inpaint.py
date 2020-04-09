"""
Remove a label a inpant these pixels

SAMPLE run::

    python run_image_annot_inpaint.py \
        -imgs "data_images/drosophila_ovary_slice/segm/*.png" \
        --label 4 --nb_workers 2

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import argparse
import glob
import logging
import os
import sys
from functools import partial

import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.experiments as tl_expt
import imsegm.annotation as seg_annot

PATH_IMAGES = os.path.join('data_images', 'drosophila_ovary_slice', 'segm', '*.png')
NB_WORKERS = tl_expt.nb_workers(0.9)


def parse_arg_params():
    """ create simple arg parser with default values (input, results, dataset)

    :return obj: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', '--path_images', type=str, required=True,
                        help='path to dir with annot', default=PATH_IMAGES)
    parser.add_argument('--label', type=int, required=False, nargs='+',
                        help='labels to be replaced', default=[-1])
    parser.add_argument('--nb_workers', type=int, required=False,
                        help='number of jobs in parallel', default=NB_WORKERS)
    args = vars(parser.parse_args())
    p_dir = tl_data.update_path(os.path.dirname(args['path_images']))
    assert os.path.isdir(p_dir), 'missing folder: %s' % args['path_images']
    args['path_images'] = os.path.join(p_dir,
                                       os.path.basename(args['path_images']))
    logging.info(tl_expt.string_dict(args, desc='ARG PARAMETERS'))
    return args


def perform_img_inpaint(path_img, labels):
    """ perform the quantization together with loading and exporting

    :param str path_img: image path
    """
    logging.debug('repaint labels %r for image: "%s"', labels, path_img)
    img = np.array(tl_data.io.imread(path_img), dtype=np.float)

    for label in labels:
        img[img == label] = np.nan

    # interpolate nearest fo label
    valid_mask = ~np.isnan(img)
    im_paint = seg_annot.image_inpaint_pixels(img, valid_mask)

    tl_data.io_imsave(path_img, im_paint.astype(np.uint8))
    # plt.subplot(121), plt.imshow(img)
    # plt.subplot(122), plt.imshow(im_paint)
    # plt.show()


def quantize_folder_images(path_images, label, nb_workers=1):
    """ perform single or multi thread image quantisation

    :param list(str) path_images: list of image paths
    :param int nb_workers:
    """
    assert os.path.isdir(os.path.dirname(path_images)), \
        'input folder does not exist: %s' % os.path.dirname(path_images)
    path_imgs = sorted(glob.glob(path_images))
    logging.info('found %i images', len(path_imgs))

    _wrapper_img_inpaint = partial(perform_img_inpaint, labels=label)
    iterate = tl_expt.WrapExecuteSequence(_wrapper_img_inpaint, path_imgs,
                                          nb_workers=nb_workers,
                                          desc='quantise images')
    list(iterate)


def main(params):
    """ the main_train entry point   """
    logging.info('running...')
    quantize_folder_images(params['path_images'], params['label'],
                           nb_workers=params['nb_workers'])
    logging.info('DONE')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = parse_arg_params()
    main(params)

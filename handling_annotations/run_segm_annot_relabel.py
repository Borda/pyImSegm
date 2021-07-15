"""
Replace labels in annotation

SAMPLE run::

    python run_segm_annot_relabel.py \
        -imgs "data-images/drosophila_ovary_slice/center_levels/*.png" \
        -out results/relabel_center_levels \
        --label_old 2 3 --label_new 1 1 --nb_workers 2

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

PATH_IMAGES = os.path.join('data-images', 'drosophila_ovary_slice', 'center_levels', '*.png')
PATH_OUTPUT = os.path.join('results', 'relabel_center_levels')
NB_WORKERS = tl_expt.get_nb_workers(0.9)


def parse_arg_params():
    """ create simple arg parser with default values (input, results, dataset)

    :return obj: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', '--path_images', type=str, required=True, help='path to images', default=PATH_IMAGES)
    parser.add_argument(
        '-out', '--path_output', type=str, required=True, help='path to output dir', default=PATH_OUTPUT
    )
    parser.add_argument('--label_old', type=int, required=False, nargs='+', help='labels to be replaced', default=[0])
    parser.add_argument(
        '--label_new', type=int, required=False, nargs='+', help='new labels after replacing', default=[0]
    )
    parser.add_argument('--nb_workers', type=int, required=False, help='number of jobs in parallel', default=NB_WORKERS)
    args = vars(parser.parse_args())
    for k in ['path_images', 'path_output']:
        p_dir = tl_data.update_path(os.path.dirname(args[k]))
        if not os.path.isdir(p_dir):
            raise FileNotFoundError('missing folder: %s' % args[k])
        args[k] = os.path.join(p_dir, os.path.basename(args[k]))
    if len(args['label_old']) != len(args['label_new']):
        raise ValueError(
            'length of old (%i) and new (%i) labels should be same' % (len(args['label_old']), len(args['label_new']))
        )
    logging.info(tl_expt.string_dict(args, desc='ARG PARAMETERS'))
    return args


def perform_image_relabel(path_img, path_out, labels_old, labels_new):
    """ perform the quantization together with loading and exporting

    :param str path_img:
    :param str path_out:
    :param str labels_new:
    :param str labels_old:
    """
    logging.debug('repaint labels %r -> %r for image: "%s"', labels_old, labels_new, path_img)
    img = np.array(tl_data.io.imread(path_img), dtype=int)

    max_label = int(max(img.max(), max(labels_old)))
    lut = np.array(range(max_label + 1))
    for label_old, label_new in zip(labels_old, labels_new):
        lut[label_old] = label_new
    img = lut[img]

    # for label_old, label_new in zip(labels_old, labels_new):
    #     img[img == label_old] = label_new

    logging.debug('resulting image labels: %r', np.unique(img).tolist())
    path_img_out = os.path.join(path_out, os.path.basename(path_img))
    tl_data.io_imsave(path_img_out, img)
    # plt.subplot(121), plt.imshow(img)
    # plt.subplot(122), plt.imshow(im_paint)
    # plt.show()


def relabel_folder_images(path_images, path_out, labels_old, labels_new, nb_workers=1):
    """ perform single or multi thread image quantisation

    :param [int] labels_old:
    :param [int] labels_new:
    :param list(str) path_images: list of input images
    :param path_out: output directory
    :param [int] labels_old: list of labels to be replaced
    :param [int] labels_new: list of new labels
    :param int nb_workers:
    """
    if not os.path.isdir(os.path.dirname(path_images)):
        raise FileNotFoundError('missing folder: %s' % path_images)
    if not os.path.isdir(path_out):
        raise FileNotFoundError('missing output folder: %s' % path_out)

    path_imgs = sorted(glob.glob(path_images))
    logging.info('found %i images', len(path_imgs))

    _wrapper_img_relabel = partial(
        perform_image_relabel,
        path_out=path_out,
        labels_old=labels_old,
        labels_new=labels_new,
    )
    iterate = tl_expt.WrapExecuteSequence(
        _wrapper_img_relabel,
        path_imgs,
        nb_workers=nb_workers,
        desc='relabel images',
    )
    list(iterate)


def main(params):
    """ the main_train entry point   """
    logging.info('running...')

    if not os.path.exists(params['path_output']):
        dir_out = os.path.dirname(params['path_output'])
        if not os.path.isdir(dir_out):
            raise FileNotFoundError('missing folder: %s' % dir_out)
        os.mkdir(params['path_output'])

    relabel_folder_images(
        params['path_images'], params['path_output'], params['label_old'], params['label_new'], params['nb_workers']
    )

    logging.info('DONE')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cli_params = parse_arg_params()
    main(cli_params)

"""
Convert label image to colors and other way around

SAMPLE run::

    python run_image_convert_label_color.py \
        -imgs "data_images/drosophila_ovary_slice/segm/*.png" \
        -out data_images/drosophila_ovary_slice/segm_rgb \
        -clrs data_images/drosophila_ovary_slice/segm_rgb/dict_label-color.json

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import glob
import json
import logging
import argparse
import multiprocessing as mproc
from functools import partial

import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.experiments as tl_expt
import imsegm.annotation as seg_annot

PATH_INPUT = os.path.join('data_images', 'drosophila_ovary_slice', 'segm', '*.png')
PATH_OUTPUT = os.path.join('data_images', 'drosophila_ovary_slice', 'segm_rgb')
NAME_JSON_DICT = 'dictionary_label-color.json'
NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))


def parse_arg_params():
    """ create simple arg parser with default values (input, results, dataset)

    :return obj: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', '--path_images', type=str, required=True,
                        help='path to dir with images', default=PATH_INPUT)
    parser.add_argument('-out', '--path_out', type=str, required=True,
                        help='path to output dir', default=PATH_OUTPUT)
    parser.add_argument('-clrs', '--path_colors', type=str, required=False,
                        help='json with colour-label dict', default=None)
    parser.add_argument('--nb_workers', type=int, required=False,
                        help='number of jobs in parallel', default=NB_THREADS)
    args = vars(parser.parse_args())
    for n in ['path_images', 'path_out']:
        p_dir = tl_data.update_path(os.path.dirname(args[n]))
        assert os.path.isdir(p_dir), 'missing: %s' % args[n]
        args[n] = os.path.join(p_dir, os.path.basename(args[n]))
    if args['path_colors'] is not None:
        args['path_colors'] = tl_data.update_path(args['path_colors'])
    logging.info(tl_expt.string_dict(args, desc='ARG PARAMETERS'))
    return args


def load_dict_colours(path_json):
    if path_json is not None and os.path.isfile(path_json):
        with open(path_json, 'r') as fp:
            dict_colors = json.load(fp)
    else:
        dict_colors = {}
    # convert to correct type
    dict_colors = {int(lb): tuple(dict_colors[lb]) for lb in dict_colors}
    return dict_colors


def convert_labels_2_colors(img, dict_colors, path_out):
    img_labels = np.unique(img)
    if not all(lb in dict_colors.keys() for lb in img_labels):
        for lb in (l for l in img_labels if l not in dict_colors.keys()):
            dict_colors[lb] = tuple(np.random.randint(255, size=3))
        with open(os.path.join(path_out, NAME_JSON_DICT), 'w') as f:
            json.dump(dict_colors, f)
    img_labels = seg_annot.convert_img_labels_to_colors(img, dict_colors)
    return img_labels


def convert_colors_2_labels(img, dict_colors, path_out):
    img_colors = seg_annot.unique_image_colors(img)
    if not all(c in dict_colors.values() for c in img_colors):
        for clr in (c for c in img_colors if c not in dict_colors.values()):
            max_idx = max(dict_colors.keys()) if len(dict_colors) > 0 else -1
            dict_colors[max_idx + 1] = clr
        with open(os.path.join(path_out, NAME_JSON_DICT), 'w') as f:
            json.dump(dict_colors, f)
    img_rgb = seg_annot.convert_img_colors_to_labels(img, dict_colors)
    return img_rgb


def perform_img_convert(path_img, path_out, dict_colors):
    """ perform the quantization together with loading and exporting

    :param str path_img:
    :param str path_out:
    :param {} dict_colors:
    """
    img = tl_data.io.imread(path_img)

    if img.ndim == 2:
        if not dict_colors:
            dict_colors = seg_annot.DICT_COLOURS
        img_new = convert_labels_2_colors(img, dict_colors, path_out)
    elif img.ndim == 3:
        if img.shape[2] > 3:
            # for some 4 chanel images remove the last one (alpha)
            img = img[:, :, :3]
        img_new = convert_colors_2_labels(img, dict_colors, path_out)
    else:
        logging.warning('not supported image format %r', img.shape)
        img_new = None

    if img_new is not None:
        img_new = img_new.astype(np.uint8)
        path_img_out = os.path.join(path_out, os.path.basename(path_img))
        logging.debug('export "%s"', path_img_out)
        tl_data.io_imsave(path_img_out, img_new)
    # plt.subplot(121), plt.imshow(img)
    # plt.subplot(122), plt.imshow(im_paint)
    # plt.show()


def convert_folder_images(path_images, path_out, path_json=None, nb_workers=1):
    """ perform single or multi thread image quantisation

    :param [str] path_images: list of input images
    :param str path_out: output directory
    :param str path_json: path to json file
    :param int int nb_workers:
    """
    assert os.path.isdir(os.path.dirname(path_images)), \
        'input folder does not exist'
    path_imgs = sorted(glob.glob(path_images))
    logging.info('found %i images', len(path_imgs))
    if not os.path.exists(path_out):
        assert os.path.isdir(os.path.dirname(path_out)), \
            'missing folder: %s' % os.path.dirname(path_out)
        os.mkdir(path_out)

    dict_colors = load_dict_colours(path_json)
    logging.debug('loaded dictionary %r', dict_colors)
    _wrapper_img_convert = partial(perform_img_convert, path_out=path_out,
                                   dict_colors=dict_colors)
    iterate = tl_expt.WrapExecuteSequence(_wrapper_img_convert, path_imgs,
                                          nb_workers=nb_workers,
                                          desc='convert images')
    list(iterate)


def main(params):
    """ the main_train entry point   """
    logging.info('running...')

    if not os.path.exists(params['path_out']):
        assert os.path.isdir(os.path.dirname(params['path_out'])), \
            'missing folder: %s' % os.path.dirname(params['path_out'])
        os.mkdir(params['path_out'])

    convert_folder_images(params['path_images'], params['path_out'],
                          params['path_colors'], params['nb_workers'])

    logging.info('DONE')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = parse_arg_params()
    main(params)

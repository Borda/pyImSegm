"""
Cut out images according given object segmentation

Sample usage::

    python run_cut_segmented_objects.py \
        -annot "data-images/drosophila_ovary_slice/annot_eggs/*.png" \
        -img "data-images/drosophila_ovary_slice/segm/*.png" \
        -out results/cut_images --padding 20

"""

import argparse
import logging
import os
import sys
from functools import partial

import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.experiments as tl_expt

NB_WORKERS = tl_expt.nb_workers(0.9)
PATH_IMAGES = tl_data.update_path(os.path.join('data-images', 'drosophila_ovary_slice'))
PATH_RESULTS = tl_data.update_path('results', absolute=True)
PATHS = {
    'annot': os.path.join(PATH_IMAGES, 'annot_eggs', '*.png'),
    'image': os.path.join(PATH_IMAGES, 'image', '*.jpg'),
    'output': os.path.join(PATH_RESULTS, 'cut_images'),
}


def arg_parse_params(dict_paths):
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return ({str: str}, int):
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-annot', '--path_annot', type=str, required=False, help='annotations', default=dict_paths['annot']
    )
    parser.add_argument(
        '-imgs',
        '--path_image',
        type=str,
        required=False,
        help='path to directory & name pattern for images',
        default=dict_paths['image']
    )
    parser.add_argument(
        '-out',
        '--path_output',
        type=str,
        required=False,
        help='path to the output directory',
        default=dict_paths['output']
    )
    parser.add_argument('--padding', type=int, required=False, help='crop padding [px]', default=25)
    parser.add_argument('--mask', type=int, required=False, help='mask by the segmentation', default=1)
    parser.add_argument(
        '-bg', '--background', type=int, required=False, help='using background color', default=None, nargs='+'
    )
    parser.add_argument(
        '--nb_workers', type=int, required=False, default=NB_WORKERS, help='number of processes in parallel'
    )
    args = vars(parser.parse_args())
    logging.info('ARG PARAMETERS: \n %r', args)

    _fn_path = lambda k: os.path.join(tl_data.update_path(os.path.dirname(args[k])), os.path.basename(args[k]))
    dict_paths = {k.split('_')[-1]: _fn_path(k) for k in args if k.startswith('path_')}
    for k in dict_paths:
        assert os.path.exists(os.path.dirname(dict_paths[k])), 'missing (%s) "%s"' % (k, os.path.dirname(dict_paths[k]))
    return dict_paths, args


def export_cut_objects(df_row, path_out, padding, use_mask=True, bg_color=None):
    """ cut and expert objects in image according given segmentation

    :param df_row:
    :param str path_out: path for exporting image
    :param int padding: set padding around segmented object
    """
    annot, _ = tl_data.load_image_2d(df_row['path_1'])
    img, name = tl_data.load_image_2d(df_row['path_2'])
    assert annot.shape[:2] == img.shape[:2], 'image sizes not match %r vs %r' % (annot.shape, img.shape)

    uq_objects = np.unique(annot)
    if len(uq_objects) == 1:
        return

    for idx in uq_objects[1:]:
        img_new = tl_data.cut_object(img, annot == idx, padding, use_mask, bg_color)
        path_img = os.path.join(path_out, '%s_%i.png' % (name, idx))
        logging.debug('saving image "%s"', path_img)
        tl_data.io_imsave(path_img, img_new)


def main(dict_paths, padding=0, use_mask=False, bg_color=None, nb_workers=NB_WORKERS):
    """ the main executable

    :param dict_paths:
    :param int padding:
    :param int nb_workers:
    """
    if not os.path.isdir(dict_paths['output']):
        assert os.path.isdir(os.path.dirname(dict_paths['output'])), '"%s" should be folder' % dict_paths['output']
        logging.debug('creating dir: %s', dict_paths['output'])
        os.mkdir(dict_paths['output'])

    list_dirs = [dict_paths['annot'], dict_paths['image']]
    df_paths = tl_data.find_files_match_names_across_dirs(list_dirs)

    logging.info('start cutting images')
    _wrapper_cutting = partial(
        export_cut_objects,
        path_out=dict_paths['output'],
        padding=padding,
        use_mask=use_mask,
        bg_color=bg_color,
    )
    iterate = tl_expt.WrapExecuteSequence(
        _wrapper_cutting,
        (row for idx, row in df_paths.iterrows()),
        nb_workers=nb_workers,
    )
    list(iterate)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    dict_paths, args = arg_parse_params(PATHS)
    main(dict_paths, args['padding'], args['mask'], args['background'], args['nb_workers'])

    logging.info('DONE')

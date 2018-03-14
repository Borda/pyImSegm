"""
Cut out images according given object segmentation

SAMPLE run:
>> python run_cut_segmented_objects.py \
    -annot "images/drosophila_ovary_slice/annot_eggs/*.png" \
    -img "images/drosophila_ovary_slice/segm/*.png" \
    -out results/cut_images --padding 20

"""

import os
import sys
import logging
import argparse
import multiprocessing as mproc
from functools import partial

import tqdm
import numpy as np
from PIL import Image

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utils.data_io as tl_io

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))
PATH_IMAGES = tl_io.update_path(os.path.join('images', 'drosophila_ovary_slice'))
PATH_RESULTS = tl_io.update_path('results', absolute=True)
PATHS = {
    'annot': os.path.join(PATH_IMAGES, 'annot_eggs', '*.png'),
    'image': os.path.join(PATH_IMAGES, 'image', '*.jpg'),
    'output': os.path.join(PATH_RESULTS, 'cut_images'),
}


def arg_parse_params(dict_paths):
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return: {str: str}, int
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-annot', '--path_annot', type=str, required=False,
                        help='annotations',
                        default=dict_paths['annot'])
    parser.add_argument('-imgs', '--path_image', type=str, required=False,
                        help='path to directory & name pattern for images',
                        default=dict_paths['image'])
    parser.add_argument('-out', '--path_out', type=str, required=False,
                        help='path to the output directory',
                        default=dict_paths['output'])
    parser.add_argument('--padding', type=int, required=False,
                        help='crop padding [px]', default=25)
    parser.add_argument('--mask', type=int, required=False,
                        help='mask by the segmentation', default=1)
    parser.add_argument('-bg', '--background', type=int, required=False,
                        help='using background color', default=None, nargs='+')
    parser.add_argument('--nb_jobs', type=int, required=False, default=NB_THREADS,
                        help='number of processes in parallel')
    args = parser.parse_args()
    logging.info('ARG PARAMETERS: \n %s', repr(args))
    dict_paths = {
        'annot': tl_io.update_path(args.path_annot),
        'image': tl_io.update_path(args.path_image),
        'output': tl_io.update_path(args.path_out),
    }
    for k in dict_paths:
        if dict_paths[k] == '' or k == 'output':
            continue
        p = os.path.dirname(dict_paths[k]) \
            if '*' in dict_paths[k] else dict_paths[k]
        assert os.path.exists(p), 'missing (%s) "%s"' % (k, p)
    return dict_paths, args


def export_cut_objects(df_row, path_out, padding, use_mask=True, bg_color=None):
    """ cut and expert objects in image according given segmentation

    :param df_row:
    :param str path_out: path for exporting image
    :param int padding: set padding around segmented object
    """
    annot, _ = tl_io.load_image_2d(df_row['path_1'])
    img, name = tl_io.load_image_2d(df_row['path_2'])
    assert annot.shape[:2] == img.shape[:2], \
        'image sizes not match %s vs %s' % (repr(annot.shape), repr(img.shape))

    uq_objects = np.unique(annot)
    if len(uq_objects) == 1:
        return

    for idx in uq_objects[1:]:
        img_new = tl_io.cut_object(img, annot == idx, padding, use_mask, bg_color)
        path_img = os.path.join(path_out, '%s_%i.png' % (name, idx))
        logging.debug('saving image "%s"', path_img)
        Image.fromarray(img_new).save(path_img)


def main(dict_paths, padding=0, use_mask=False, bg_color=None,
         nb_jobs=NB_THREADS):
    """ the main executable

    :param dict_paths:
    :param int padding:
    :param int nb_jobs:
    """
    logging.info('running...')
    if not os.path.isdir(dict_paths['output']):
        assert os.path.isdir(os.path.dirname(dict_paths['output'])), \
            '"%s" should be folder' % dict_paths['output']
        logging.debug('creating dir: %s', dict_paths['output'])
        os.mkdir(dict_paths['output'])

    list_dirs = [dict_paths['annot'], dict_paths['image']]
    df_paths = tl_io.find_files_match_names_across_dirs(list_dirs)

    logging.info('start cutting images')
    tqdm_bar = tqdm.tqdm(total=len(df_paths))
    wrapper_cutting = partial(export_cut_objects, path_out=dict_paths['output'],
                              padding=padding, use_mask=use_mask, bg_color=bg_color)
    mproc_pool = mproc.Pool(nb_jobs)
    for _ in mproc_pool.imap_unordered(wrapper_cutting,
                                       (row for idx, row in df_paths.iterrows())):
        tqdm_bar.update()
    mproc_pool.close()
    mproc_pool.join()

    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dict_paths, args = arg_parse_params(PATHS)
    main(dict_paths, args.padding, args.mask, args.background, args.nb_jobs)

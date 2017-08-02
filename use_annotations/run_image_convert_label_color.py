"""
Quantize annotation and so remove some noise in the image and gradients along edges

RUN:
>> python run_image_convert_label_color.py \
    -in ~/Medical-drosophila/egg_segmentation/mask_2d_slice_egg_center_levels \
    -out ~/Medical-drosophila/egg_segmentation/mask_2d_slice_egg_center_levels_rgb \
    --im_pattern *.png --nb_jobs 2

>> python run_image_convert_label_color.py \
    -in ~/Medical-drosophila/TEMPORARY_OVARY/segment_ovary_selected_slices_rgb \
    -out ~/Medical-drosophila/TEMPORARY_OVARY/segment_ovary_selected_slices \
    -colors ~/Medical-drosophila/TEMPORARY_OVARY/segment_ovary_selected_slices/dict_label_color.json
    --im_pattern *.png

"""


import os
import sys
import glob
import json
import logging
import argparse
import traceback
import multiprocessing as mproc
from functools import partial

import numpy as np
import tqdm
from skimage import io

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
matplotlib.use('Agg')

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import segmentation.utils.data_samples as tl_spl
import segmentation.annotation as tl_annot

PATH_BASE = tl_spl.PATH_IMAGES
PATH_DATA = PATH_BASE + 'drosophila_egg_2D/annot'
PATH_OUTPUT = PATH_BASE + 'drosophila_egg_2D/annot_rgb'
IM_PATTERN = '*.png'
NAME_JSON_DICT = 'dict_label_color.json'
NB_THREADS = int(mproc.cpu_count() * .8)

DICT_COLOURS = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (255, 229, 0),  # yellow
    4: (142, 68, 173),  # purple
    5: (127, 140, 141),  # gray
    6: (0, 212, 255),  # blue
    7: (128, 0, 0),  # brown
}


def parse_arg_params():
    """ create simple arg parser with default values (input, results, dataset)

    :return: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--path_in', type=str, required=False,
                        help='path to dir with annot', default=PATH_DATA)
    parser.add_argument('-out', '--path_out', type=str, required=False,
                        help='path to dir with annot', default=PATH_OUTPUT)
    parser.add_argument('--im_pattern', type=str, required=False,
                        help='pattern of image names', default=IM_PATTERN)
    parser.add_argument('-colors', '--dict_color', type=str, required=False,
                        help='json wiith colour-label dict', default=None)
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number running in parallel', default=NB_THREADS)
    args = vars(parser.parse_args())
    for k in ['path_in', 'path_out']:
        args[k] = os.path.expanduser(args[k])
        assert os.path.exists(args[k]), '%s' % args[k]
    logging.info('PARAMS: %s', repr(args))
    return args


def load_dict_colours(path_json):
    if path_json is not None and os.path.exists(path_json):
        with open(path_json, 'r') as fp:
            dict_colors = json.load(fp)
    else:
        dict_colors = DICT_COLOURS
    # convert to correct type
    dict_colors = {int(lb): tuple(clr) for lb, clr in dict_colors.iteritems()}
    return dict_colors


def convert_labels_2_colours(img, dict_colors, path_out):
    img_labels = np.unique(img)
    if not all(lb in dict_colors.keys() for lb in img_labels):
        for lb in (l for l in img_labels if not l in dict_colors.keys()):
            dict_colors[lb] = tuple(np.random.randint(255, size=3))
        with open(os.path.join(path_out, NAME_JSON_DICT), 'w') as f:
            json.dump(dict_colors, f)
    img_labels = tl_annot.convert_img_labels_to_colors(img, dict_colors)
    return img_labels


def convert_colours_2_labels(img, dict_colors, path_out):
    img_colors = tl_annot.unique_image_colors(img)
    if not all(c in dict_colors.values() for c in img_colors):
        for clr in (c for c in img_colors if not c in dict_colors.values()):
            dict_colors[max(dict_colors.keys()) + 1] = clr
        with open(os.path.join(path_out, NAME_JSON_DICT), 'w') as f:
            json.dump(dict_colors, f)
    img_rgb = tl_annot.convert_img_colors_to_labels(img, dict_colors)
    return img_rgb


def perform_img_convert(path_img, path_out, dict_colors):
    """ perform the quantization together with loading and exporting

    :param path_img: str
    """
    img = np.array(io.imread(path_img))

    if img.ndim == 2:
        img_new = convert_labels_2_colours(img, dict_colors, path_out)
    elif img.ndim == 3:
        if img.shape[2] > 3:
            # for some 4 chanel images remove the last one (alpha)
            img = img[:, :, :3]
        img_new = convert_colours_2_labels(img, dict_colors, path_out)
    else:
        logging.warning('not supported image format 5s', repr(img.shape))
        img_new = None

    if img_new is not None:
        img_new = img_new.astype(np.uint8)
        path_img_out = os.path.join(path_out, os.path.basename(path_img))
        logging.debug('export "%s"', path_img_out)
        io.imsave(path_img_out, img_new)
    # plt.subplot(121), plt.imshow(img)
    # plt.subplot(122), plt.imshow(im_paint)
    # plt.show()


def quantize_folder_imgs(path_in, path_out, im_pattern, path_json=None,
                         nb_jobs=1):
    """ perform single or multi thread image quantisation

    :param path_dir: str, input directory
    :param im_pattern: str, image pattern for loading
    :param nb_jobs: int
    """
    assert os.path.exists(path_in), 'input folder does not exist'
    path_imgs = sorted(glob.glob(os.path.join(path_in, im_pattern)))
    logging.info('found %i images', len(path_imgs))

    dict_colors = load_dict_colours(path_json)
    logging.debug('loaded dictionary %s', repr(dict_colors))
    wrapper_img_convert = partial(perform_img_convert, path_out=path_out,
                                  dict_colors=dict_colors)
    tqdm_bar = tqdm.tqdm(total=len(path_imgs))

    if nb_jobs > 1:
        logging.debug('perform_sequence in %i threads', nb_jobs)
        mproc_pool = mproc.Pool(nb_jobs)
        for r in mproc_pool.imap_unordered(wrapper_img_convert, path_imgs):
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()
    else:
        for r in map(wrapper_img_convert, path_imgs):
            tqdm_bar.update()


def main():
    """ the main_train entry point   """
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = parse_arg_params()
    quantize_folder_imgs(params['path_in'], params['path_out'], params['im_pattern'],
                         params['dict_color'], params['nb_jobs'])

    logging.info('DONE')


if __name__ == "__main__":
    main()

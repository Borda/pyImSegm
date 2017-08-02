"""
Quantize annotation and so remove some noise in the image and gradients along edges

NOTE, for JPEG there is always some smoothing so only allowed format is PNG

RUN:
>> python run_image_colour_quantization.py \
    -p ~/Medical-drosophila/TEMPORARY_OVARY/mask_2d_slice_egg_center_levels_rgb \
    --im_pattern *.png --nb_jobs 2

>> python run_image_colour_quantization.py \
    -p ~/Medical-drosophila//egg_segmentation/mask_2d_4class/stage1_rgb \
    --im_pattern *.png -m position

"""


import os
import sys
import glob
import logging
import argparse
import multiprocessing as mproc
from functools import partial

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
matplotlib.use('Agg')

from PIL import Image
import numpy as np
import tqdm

sys.path += [os.path.abspath('.'), os.path.abspath('..')] # Add path to root
import segmentation.utils.data_samples as tl_spl
import segmentation.annotation as tl_annot


PATH_DATA = os.path.dirname(tl_spl.get_image_path(tl_spl.ANNOT_DROSOPHILA_DISC_RGB))
IM_PATTERN = '*.png'
NB_THREADS = int(mproc.cpu_count() * .8)
THRESHOLD_INVALID_PIXELS = 5e-2


def parse_arg_params():
    """ create simple arg parser with default values (input, results, dataset)

    :return: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_dir', type=str, required=False,
                        help='path to dir with annot', default=PATH_DATA)
    parser.add_argument('-imgs', '--im_pattern', type=str, required=False,
                        help='pattern of image names', default=IM_PATTERN)
    parser.add_argument('-m', '--method', type=str, required=False,
                        help='method for quantisation', default='color',
                        choices=['color', 'position'])
    parser.add_argument('--px_threshold', type=float, required=False,
                        help='number of pixels to remove a color',
                        default=THRESHOLD_INVALID_PIXELS)
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number running in parallel', default=NB_THREADS)
    args = vars(parser.parse_args())
    args['path_dir'] = os.path.expanduser(args['path_dir'])
    logging.info('PARAMS: %s', repr(args))
    assert os.path.exists(args['path_dir']), '%s' % args['path_dir']
    return args


def see_imgs_clr_info(path_dir=PATH_DATA, im_pattern=IM_PATTERN,
                      px_th=THRESHOLD_INVALID_PIXELS):
    """ look to the folder on all images and estimate most frequent colours

    :param path_dir: str
    :param im_pattern: str
    :param px_th: float, percentage of nb clr pixels to be assumed as important
    :return:
    """
    if not os.path.exists(path_dir):
        logging.error('input folder does not exist')
        return
    paths_img = sorted(glob.glob(os.path.join(path_dir, im_pattern)))
    logging.debug('found %i images', len(paths_img))
    dict_clrs = tl_annot.dir_images_frequent_colors(paths_img, px_th)
    return dict_clrs


def perform_quantize_img(path_img, list_colors, method='color'):
    """ perform the quantization together with loading and exporting

    :param path_img: str
    :param list_colors: [(int, int, int)], list of possible colours
    """
    logging.debug('quantize img: "%s"', path_img)
    im = np.array(Image.open(path_img))
    assert im.ndim == 3, 'not valid color image of dims %s' % repr(im.shape)
    im = im[:, :, :3]
    # im = io.imread(path_img)[:, :, :3]
    if method == 'color':
        im_q = tl_annot.quantize_image_nearest_color(im, list_colors)
    elif method == 'position':
        im_q = tl_annot.quantize_image_nearest_pixel(im, list_colors)
    else:
        logging.error('not implemented method "%s"', method)
    path_img = os.path.splitext(path_img)[0] + '.png'
    Image.fromarray(im_q.astype(np.uint8)).save(path_img)
    # io.imsave(path_img, im_q)
    # plt.subplot(121), plt.imshow(im)
    # plt.subplot(122), plt.imshow(im_q)
    # plt.show()


def quantize_folder_imgs(path_dir=PATH_DATA, im_pattern=IM_PATTERN,
                         list_colors=None, method='color',
                         px_threshold=THRESHOLD_INVALID_PIXELS, nb_jobs=1):
    """ perform single or multi thread image quantisation

    :param path_dir: str, input directory
    :param im_pattern: str, image pattern for loading
    :param list_colors: [(int, int, int)], list of possible colours
    :param nb_jobs: int
    """
    assert os.path.exists(path_dir), 'input folder does not exist'
    path_imgs = sorted(glob.glob(os.path.join(path_dir, im_pattern)))
    logging.info('found %i images', len(path_imgs))
    if list_colors is None:
        dict_clrs = see_imgs_clr_info(path_dir, im_pattern, px_th=px_threshold)
        list_colors = dict_clrs.keys()

    wrapper_quantize_img = partial(perform_quantize_img,
                                   method=method, list_colors=list_colors)
    tqdm_bar = tqdm.tqdm(total=len(path_imgs))

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


def main():
    """ the main_train entry point   """
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = parse_arg_params()
    quantize_folder_imgs(params['path_dir'], params['im_pattern'],
                         method=params['method'], nb_jobs=params['nb_jobs'])

    logging.info('DONE')


if __name__ == "__main__":
    main()
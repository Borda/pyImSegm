"""
Quantize annotation and so remove some noise in the image and gradients along edges

RUN:
>> python run_image_annot_inpaint.py \
    -p ~/py_ImageProcessing/images/drosophila_ovary_2D_annot \
    --im_pattern *.png --label 4 --nb_jobs 2

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

import numpy as np
import tqdm
from skimage import io

# import matplotlib.pyplot as plt

sys.path += [os.path.abspath('.'), os.path.abspath('..')] # Add path to root
import segmentation.utils.data_samples as tl_spl
import segmentation.annotation as tl_annot

PATH_BASE = tl_spl.PATH_IMAGES
PATH_DATA = PATH_BASE + 'drosophila_egg_2D/annot'
IM_PATTERN = '*.png'
NB_THREADS = int(mproc.cpu_count() * .8)


def parse_arg_params():
    """ create simple arg parser with default values (input, results, dataset)

    :return: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_dir', type=str, required=True,
                        help='path to dir with annot', default=PATH_DATA)
    parser.add_argument('--im_pattern', type=str, required=False,
                        help='pattern of image names', default=IM_PATTERN)
    parser.add_argument('--label', type=int, required=False, nargs='+',
                        help='labels to be replaced', default=[-1])
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number running in parallel', default=NB_THREADS)
    args = vars(parser.parse_args())
    args['path_dir'] = os.path.expanduser(args['path_dir'])
    logging.info('PARAMS: %s', repr(args))
    assert os.path.exists(args['path_dir']), '%s' % args['path_dir']
    return args


def perform_img_inpaint(path_img, labels):
    """ perform the quantization together with loading and exporting

    :param path_img: str
    """
    logging.debug('repaint labels %s for image: "%s"', repr(labels), path_img)
    img = np.array(io.imread(path_img), dtype=np.float)

    for label in labels:
        img[img == label] = np.nan

    # interpolate nearest fo label
    valid_mask = ~np.isnan(img)
    im_paint = tl_annot.image_inpaint_pixels(img, valid_mask)

    io.imsave(path_img, im_paint.astype(np.uint8))
    # plt.subplot(121), plt.imshow(img)
    # plt.subplot(122), plt.imshow(im_paint)
    # plt.show()


def quantize_folder_imgs(path_dir, im_pattern, label, nb_jobs=1):
    """ perform single or multi thread image quantisation

    :param path_dir: str, input directory
    :param im_pattern: str, image pattern for loading
    :param nb_jobs: int
    """
    assert os.path.exists(path_dir), 'input folder does not exist'
    path_imgs = sorted(glob.glob(os.path.join(path_dir, im_pattern)))
    logging.info('found %i images', len(path_imgs))

    wrapper_img_inpaint = partial(perform_img_inpaint, labels=label)
    tqdm_bar = tqdm.tqdm(total=len(path_imgs))

    if nb_jobs > 1:
        logging.debug('perform_sequence in %i threads', nb_jobs)
        mproc_pool = mproc.Pool(nb_jobs)
        for r in mproc_pool.imap_unordered(wrapper_img_inpaint, path_imgs):
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()
    else:
        for r in map(wrapper_img_inpaint, path_imgs):
            tqdm_bar.update()


def main():
    """ the main_train entry point   """
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = parse_arg_params()
    quantize_folder_imgs(params['path_dir'], params['im_pattern'], params['label'],
                         nb_jobs=params['nb_jobs'])

    logging.info('DONE')


if __name__ == "__main__":
    main()

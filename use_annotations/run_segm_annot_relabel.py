"""
Quantize annotation and so remove some noise in the image and gradients along edges

RUN:
>> python run_image_annot_relabel.py \
    -in ~/Dropbox/Workspace/py_ImageProcessing/images/drosophila_ovary_2D_annot \
    -out ~/Dropbox/Workspace/py_ImageProcessing/images/drosophila_ovary_2D_binary \
    --im_pattern *.png --label_old 2 3 --label_new 1 1 --nb_jobs 2

>> python run_segm_annot_relabel.py \
    -in ~/Dropbox/Langer-Islet/segment_3CL_RF-GC_HSV \
    -out ~/Dropbox/Langer-Islet/segment_3CL_RF-GC_HSV_binary \
    --im_pattern *.png --label_old 127 --label_new 0

"""


import os
# import sys
import glob
import logging
import argparse
import multiprocessing as mproc
from functools import partial

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
matplotlib.use('Agg')

import tqdm
import numpy as np
from skimage import io

IM_PATTERN = '*.png'
NB_THREADS = int(mproc.cpu_count() * .8)


def parse_arg_params():
    """ create simple arg parser with default values (input, results, dataset)

    :return: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--path_in', type=str, required=True,
                        help='path to dir with annot')
    parser.add_argument('-out', '--path_out', type=str, required=True,
                        help='path to dir with annot')
    parser.add_argument('--im_pattern', type=str, required=False,
                        help='pattern of image names', default=IM_PATTERN)
    parser.add_argument('--label_old', type=int, required=True, nargs='+',
                        help='labels to be replaced', default=[0])
    parser.add_argument('--label_new', type=int, required=True, nargs='+',
                        help='labels to be replaced', default=[0])
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number running in parallel', default=NB_THREADS)
    args = vars(parser.parse_args())
    for k in ['path_in', 'path_out']:
        args[k] = os.path.expanduser(args[k])
        assert os.path.exists(args[k]), '%s' % args[k]
    logging.info('PARAMS: %s', repr(args))
    assert len(args['label_old']) == len(args['label_new'])
    return args


def perform_img_relabel(path_img, path_out, labels_old, labels_new):
    """ perform the quantization together with loading and exporting

    :param str path_img:
    :param str path_out:
    :param str labels_new:
    :param str labels_old:
    """
    logging.debug('repaint labels %s -> %s for image: "%s"',
                  repr(labels_old), repr(labels_new), path_img)
    img = np.array(io.imread(path_img), dtype=int)

    max_label = int(max(img.max(), max(labels_old)))
    lut = np.array(range(max_label + 1))
    for label_old, label_new in zip(labels_old, labels_new):
        lut[label_old] = label_new
    img = lut[img]

    # for label_old, label_new in zip(labels_old, labels_new):
    #     img[img == label_old] = label_new

    logging.debug('resulting image labels: i%s', repr(np.unique(img).tolist()))
    path_img_out = os.path.join(path_out, os.path.basename(path_img))
    io.imsave(path_img_out, img)
    # plt.subplot(121), plt.imshow(img)
    # plt.subplot(122), plt.imshow(im_paint)
    # plt.show()


def quantize_folder_imgs(path_in, path_out, im_pattern, labels_old, labels_new,
                         nb_jobs=1):
    """ perform single or multi thread image quantisation

    :param str path_dir: input directory
    :param str im_pattern: image pattern for loading
    :param int nb_jobs:
    """
    assert os.path.exists(path_in), 'input folder does not exist'
    assert os.path.exists(path_out), 'output folder does not exist'

    path_imgs = sorted(glob.glob(os.path.join(path_in, im_pattern)))
    logging.info('found %i images', len(path_imgs))

    wrapper_img_relabel = partial(perform_img_relabel, path_out=path_out,
                                  labels_old=labels_old, labels_new=labels_new)
    tqdm_bar = tqdm.tqdm(total=len(path_imgs))

    if nb_jobs > 1:
        logging.debug('perform_sequence in %i threads', nb_jobs)
        mproc_pool = mproc.Pool(nb_jobs)
        for r in mproc_pool.imap_unordered(wrapper_img_relabel, path_imgs):
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()
    else:
        for r in map(wrapper_img_relabel, path_imgs):
            tqdm_bar.update()


def main(params):
    """ the main_train entry point   """
    logging.info('running...')

    quantize_folder_imgs(params['path_in'], params['path_out'], params['im_pattern'],
                         params['label_old'], params['label_new'], params['nb_jobs'])

    logging.info('DONE')


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    params = parse_arg_params()
    main(params)

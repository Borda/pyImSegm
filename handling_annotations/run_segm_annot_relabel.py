"""
Replace labels in annotation

SAMPLE run:
>> python run_segm_annot_relabel.py \
    -imgs "images/drosophila_ovary_slice/center_levels/*.png" \
    -out results/relabel_center_levels \
    --label_old 2 3 --label_new 1 1 --nb_jobs 2

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import glob
import logging
import argparse
import multiprocessing as mproc
from functools import partial

import tqdm
import numpy as np
from skimage import io

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utils.data_io as tl_io
import imsegm.utils.experiments as tl_expt

PATH_IMAGES = os.path.join('images', 'drosophila_ovary_slice', 'center_levels', '*.png')
PATH_OUTPUT = os.path.join('results', 'relabel_center_levels')
NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))


def parse_arg_params():
    """ create simple arg parser with default values (input, results, dataset)

    :return: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', '--path_images', type=str, required=True,
                        help='path to images', default=PATH_IMAGES)
    parser.add_argument('-out', '--path_output', type=str, required=True,
                        help='path to output dir', default=PATH_OUTPUT)
    parser.add_argument('--label_old', type=int, required=False, nargs='+',
                        help='labels to be replaced', default=[0])
    parser.add_argument('--label_new', type=int, required=False, nargs='+',
                        help='new labels after replacing', default=[0])
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of jobs in parallel', default=NB_THREADS)
    args = vars(parser.parse_args())
    for k in ['path_images', 'path_output']:
        p_dir = tl_io.update_path(os.path.dirname(args[k]))
        assert os.path.isdir(p_dir), '%s' % args[k]
        args[k] = os.path.join(p_dir, os.path.basename(args[k]))
    assert len(args['label_old']) == len(args['label_new'])
    logging.info(tl_expt.string_dict(args, desc='ARG PARAMETERS'))
    return args


def perform_image_relabel(path_img, path_out, labels_old, labels_new):
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


def relabel_folder_images(path_images, path_out, labels_old, labels_new,
                          nb_jobs=1):
    """ perform single or multi thread image quantisation

    :param [int] labels_old:
    :param [int] labels_new:
    :param [str] path_images: list of input images
    :param path_out: output directory
    :param [int] labels_old: list of labels to be replaced
    :param [int] labels_new: list of new labels
    :param int nb_jobs:
    """
    assert os.path.isdir(os.path.dirname(path_images)), '%s' % path_images
    assert os.path.isdir(path_out), 'missing ouput folder %s' % path_out

    path_imgs = sorted(glob.glob(path_images))
    logging.info('found %i images', len(path_imgs))

    wrapper_img_relabel = partial(perform_image_relabel, path_out=path_out,
                                  labels_old=labels_old, labels_new=labels_new)
    tqdm_bar = tqdm.tqdm(total=len(path_imgs), desc='relabel images')

    if nb_jobs > 1:
        logging.debug('perform_sequence in %i threads', nb_jobs)
        mproc_pool = mproc.Pool(nb_jobs)
        for _ in mproc_pool.imap_unordered(wrapper_img_relabel, path_imgs):
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()
    else:
        for _ in map(wrapper_img_relabel, path_imgs):
            tqdm_bar.update()


def main(params):
    """ the main_train entry point   """
    logging.info('running...')

    if not os.path.exists(params['path_output']):
        assert os.path.isdir(os.path.dirname(params['path_output']))
        os.mkdir(params['path_output'])

    relabel_folder_images(params['path_images'], params['path_output'],
                          params['label_old'], params['label_new'],
                          params['nb_jobs'])

    logging.info('DONE')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    params = parse_arg_params()
    main(params)

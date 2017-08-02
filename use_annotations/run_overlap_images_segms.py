"""
taking folder with images and also related segmentation and generate to another
folder a figure for each image as image with overlapped contour of the segmentation


>> python run_overlap_images_segms.py \
    -imgs "~/Medical-drosophila/egg_segmentation/ovary_2d/stage5_png/*.png" \
    -segs ~/Medical-drosophila/egg_segmentation/mask_2d_4class/stage5 \
    -out ~/Medical-drosophila/egg_segmentation/mask_2d_4class/stage5_visu

Copyright (C) 2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import glob
import logging
import argparse
import traceback
import multiprocessing as mproc
from functools import partial

import matplotlib
if os.environ.get('DISPLAY','') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
matplotlib.use('Agg')

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure, segmentation

sys.path += [os.path.abspath('.'), os.path.abspath('..')] # Add path to root
import segmentation.utils.visualisation as tl_visu

NB_THREADS = int(mproc.cpu_count() * .9)
BOOL_IMAGE_RESCALE_INTENSITY = False
BOOL_SAVE_IMAGE_CONTOUR = True
BOOL_SHOW_SEGM_BINARY = False
BOOL_ANNOT_RELABEL = True
SIZE_SUB_FIGURE = 9
COLOR_CONTOUR = (0, 0, 1)


def parse_arg_params():
    """ create simple arg parser with default values (input, output, dataset)

    :param dict_params: {str: ...}
    :return: object argparse<in, out, ant, name>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', '--path_imgs', type=str, required=True,
                        help='path to the input images')
    parser.add_argument('-segs', '--path_segs', type=str, required=True,
                        help='path to the input segms')
    parser.add_argument('-out', '--path_out', type=str, required=True,
                        help='path to the output')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of jobs in parallel', default=NB_THREADS)
    args = parser.parse_args()
    paths = dict(zip(['images', 'segms', 'output'],
                     [args.path_imgs, args.path_segs, args.path_out]))
    for k in paths:
        paths[k] = os.path.abspath(os.path.expanduser(paths[k]))
        p = os.path.dirname(paths[k]) if '*' in paths[k] else paths[k]
        assert os.path.exists(p), '%s' % p
    return paths, args.nb_jobs


def visualise_overlap(path_img, path_seg, path_out,
                      b_img_scale=BOOL_IMAGE_RESCALE_INTENSITY,
                      b_img_contour=BOOL_SAVE_IMAGE_CONTOUR,
                      b_relabel=BOOL_ANNOT_RELABEL):
    # TODO: loading TIFF
    img = np.array(Image.open(path_img))
    seg = np.array(Image.open(path_seg))

    if b_relabel:
        seg, _, _ = segmentation.relabel_sequential(seg)

    if img.ndim == 2:  # for gray images of ovary
        img = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)

    if b_img_scale:
        p_low, p_high = np.percentile(img, (3, 98))
        # plt.imshow(255 - img, cmap='Greys')
        img = exposure.rescale_intensity(img, in_range=(p_low, p_high),
                                         out_range='uint8')

    if b_img_contour:
        path_im_visu = os.path.splitext(path_out)[0] + '_contour.png'
        img_contour = segmentation.mark_boundaries(img[:, :, :3], seg,
                                       color=COLOR_CONTOUR, mode='subpixel')
        plt.imsave(path_im_visu, img_contour)
    # else:  # for colour images of disc
    #     mask = (np.sum(img, axis=2) == 0)
    #     img[mask] = [255, 255, 255]

    fig = tl_visu.draw_image_segm_results(img, seg, SIZE_SUB_FIGURE)
    fig.savefig(path_out)
    plt.close(fig)


def perform_visu_overlap(path_img, paths):
    # create the rest of paths
    img_name = os.path.splitext(os.path.basename(path_img))[0]
    path_seg = os.path.join(paths['segms'], img_name + '.png')
    logging.debug('input image: "%s" (%s) & seg_pipe: "%s" (%s)',
                  path_img, os.path.exists(path_img),
                  path_seg, os.path.exists(path_seg))
    path_out = os.path.join(paths['output'], img_name + '.png')

    if not all(os.path.exists(p) for p in (path_img, path_seg)):
        logging.debug('missing seg_pipe (image)')
        return False

    try:
        visualise_overlap(path_img, path_seg, path_out)
    except:
        logging.error(traceback.format_exc())
        return False
    return True


def main(paths, nb_jobs=NB_THREADS):
    logging.info('running...')

    logging.info('PATHS: \n %s', repr(paths))

    paths_imgs = glob.glob(paths['images'])
    logging.info('found %i images in dir "%s"', len(paths_imgs), paths['images'])

    warped_overlap = partial(perform_visu_overlap, paths=paths)

    created = []
    tqdm_bar = tqdm.tqdm(total=len(paths_imgs))
    if nb_jobs > 1:
        mproc_pool = mproc.Pool(nb_jobs)
        for r in mproc_pool.imap_unordered(warped_overlap, paths_imgs):
            created.append(r)
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()
    else:
        for r in map(warped_overlap, paths_imgs):
            created.append(r)
            tqdm_bar.update()

    logging.info('matched and created %i overlaps', np.sum(created))
    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    paths, nb_jobs = parse_arg_params()
    main(paths, nb_jobs)

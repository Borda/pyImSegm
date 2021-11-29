"""
Taking folder with images and also related segmentation and generate to another
folder a figure for each image as image with overlapped contour of the segmentation

SAMPLE run::

    python run_overlap_images_segms.py \
        -imgs "data-images/drosophila_ovary_slice/image/*.jpg" \
        -segs data-images/drosophila_ovary_slice/segm \
        -out results/overlap_ovary_segment

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import argparse
import glob
import logging
import os
import sys
from functools import partial

import matplotlib

if os.environ.get('DISPLAY', '') == '' and matplotlib.rcParams['backend'] != 'agg':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, segmentation

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.drawing as tl_visu
import imsegm.utilities.experiments as tl_expt

NB_WORKERS = tl_expt.get_nb_workers(0.9)
BOOL_IMAGE_RESCALE_INTENSITY = False
BOOL_SAVE_IMAGE_CONTOUR = False
BOOL_SHOW_SEGM_BINARY = False
BOOL_ANNOT_RELABEL = True
SIZE_SUB_FIGURE = 9
COLOR_CONTOUR = (0., 0., 1.)
MIDDLE_ALPHA_OVERLAP = 0.
MIDDLE_IMAGE_GRAY = False


def parse_arg_params():
    """ create simple arg parser with default values (input, output, dataset)

    :return obj: object argparse<in, out, ant, name>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-imgs', '--path_images', type=str, required=True, help='path to the input images + name pattern'
    )
    parser.add_argument('-segs', '--path_segms', type=str, required=True, help='path to the input segms')
    parser.add_argument('-out', '--path_output', type=str, required=True, help='path to the output')
    parser.add_argument('--overlap', type=float, required=False, help='alpha for segmentation', default=0.)
    parser.add_argument('--nb_workers', type=int, required=False, help='number of jobs in parallel', default=NB_WORKERS)
    args = parser.parse_args()
    paths = dict(zip(['images', 'segms', 'output'], [args.path_images, args.path_segms, args.path_output]))
    for k in paths:
        p_dir = tl_data.update_path(os.path.dirname(paths[k]))
        paths[k] = os.path.join(p_dir, os.path.basename(paths[k]))
        if not os.path.isdir(p_dir):
            raise FileNotFoundError('missing: %s' % paths[k])
    return paths, args


def visualise_overlap(
    path_img,
    path_seg,
    path_out,
    b_img_scale=BOOL_IMAGE_RESCALE_INTENSITY,
    b_img_contour=BOOL_SAVE_IMAGE_CONTOUR,
    b_relabel=BOOL_ANNOT_RELABEL,
    segm_alpha=MIDDLE_ALPHA_OVERLAP,
):
    img, _ = tl_data.load_image_2d(path_img)
    seg, _ = tl_data.load_image_2d(path_seg)

    # normalise alpha in range (0, 1)
    segm_alpha = tl_visu.norm_aplha(segm_alpha)

    if b_relabel:
        seg, _, _ = segmentation.relabel_sequential(seg.copy())

    if img.ndim == 2:  # for gray images of ovary
        img = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)

    if b_img_scale:
        p_low, p_high = np.percentile(img, q=(3, 98))
        # plt.imshow(255 - img, cmap='Greys')
        img = exposure.rescale_intensity(img, in_range=(p_low, p_high), out_range='uint8')

    if b_img_contour:
        path_im_visu = os.path.splitext(path_out)[0] + '_contour.png'
        img_contour = segmentation.mark_boundaries(img[:, :, :3], seg, color=COLOR_CONTOUR, mode='subpixel')
        plt.imsave(path_im_visu, img_contour)
    # else:  # for colour images of disc
    #     mask = (np.sum(img, axis=2) == 0)
    #     img[mask] = [255, 255, 255]

    fig = tl_visu.figure_image_segm_results(
        img, seg, SIZE_SUB_FIGURE, mid_labels_alpha=segm_alpha, mid_image_gray=MIDDLE_IMAGE_GRAY
    )
    fig.savefig(path_out)
    plt.close(fig)


def perform_visu_overlap(path_img, paths, segm_alpha=MIDDLE_ALPHA_OVERLAP):
    # create the rest of paths
    img_name = os.path.splitext(os.path.basename(path_img))[0]
    path_seg = os.path.join(paths['segms'], img_name + '.png')
    logging.debug(
        'input image: "%s" (%s) & seg_pipe: "%s" (%s)', path_img, os.path.isfile(path_img), path_seg,
        os.path.isfile(path_seg)
    )
    path_out = os.path.join(paths['output'], img_name + '.png')

    if not all(os.path.isfile(p) for p in (path_img, path_seg)):
        logging.debug('missing seg_pipe (image)')
        return False

    try:
        visualise_overlap(path_img, path_seg, path_out, segm_alpha=segm_alpha)
    except Exception:
        logging.exception('visualise_overlap')
        return False
    return True


def main(paths, nb_workers=NB_WORKERS, segm_alpha=MIDDLE_ALPHA_OVERLAP):
    logging.info('running...')
    if paths['segms'] == paths['output']:
        raise RuntimeError('overwriting segmentation dir')
    if os.path.basename(paths['images']) == paths['output']:
        raise RuntimeError('overwriting image dir')

    logging.info(tl_expt.string_dict(paths, desc='PATHS'))
    if not os.path.exists(paths['output']):
        dir_out = os.path.dirname(paths['output'])
        if not os.path.isdir(dir_out):
            raise FileNotFoundError('missing folder: %s' % dir_out)
        os.mkdir(paths['output'])

    paths_imgs = glob.glob(paths['images'])
    logging.info('found %i images in dir "%s"', len(paths_imgs), paths['images'])

    _warped_overlap = partial(perform_visu_overlap, paths=paths, segm_alpha=segm_alpha)

    created = []
    iterate = tl_expt.WrapExecuteSequence(_warped_overlap, paths_imgs, nb_workers=nb_workers, desc='overlapping')
    for r in iterate:
        created.append(r)

    logging.info('matched and created %i overlaps', np.sum(created))
    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli_paths, cli_args = parse_arg_params()
    main(cli_paths, cli_args.nb_workers, cli_args.overlap)

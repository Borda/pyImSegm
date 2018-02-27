"""
This script takes folder with some segmentation and identify all independent
regions (eggs), and fo each region it compute cennter of mass and all centers
export into single csv file.

Note, it keep same names fo the in and out dir has to be different

SAMPLE run:
>> python run_create_annot_centers.py \
    -segs "~/Medical-drosophila/RESULTS/segment_ovary_slices_selected/*.png" \
    -out ~/Medical-drosophila/egg_segmentation/mask_center_levels

Visualize:
>> python handling_annotations/run_overlap_images_segms.py \
    -imgs "~/Medical-drosophila/ovary_all_slices/png/*.png" \
    -segs ~/Medical-drosophila/egg_segmentation/mask_center_levels \
    -out ~/Medical-drosophila/RESULTS/visu_mask_center_levels

Copyright (C) 2015-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import logging
import multiprocessing as mproc
from functools import partial

import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from scipy import ndimage
from skimage import morphology, measure, draw


sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import run_center_candidate_training as run_train

NAME_DIR = 'annot_centres'
PARAMS = run_train.CENTER_PARAMS
PARAMS.update({
    'path_segms': os.path.join(run_train.PATH_IMAGES, 'annot_eggs', '*.png'),
})
# setting relative distances from object boundary for 3 levels of annotation
DISTANCE_LEVELS = [0., 0.4, 0.8]


def compute_eggs_center(seg_label):
    """ given object (egg) segmentation copute centers as center of mass
    assuming that each label is just one object

    :param ndarray seg_label:
    :return: DF
    """
    df_center = pd.DataFrame()
    for lb in range(1, seg_label.max() + 1):
        im_bin = (seg_label == lb)
        if np.sum(im_bin) > 0:
            pos = ndimage.measurements.center_of_mass(seg_label == lb)
            df_center = df_center.append({'X': pos[1], 'Y': pos[0]},
                                         ignore_index=True)
    return df_center


def load_correct_segm(path_img):
    """ load segmentation and correct it with simple morphological operations

    :param str path_img:
    :return:
    """
    assert os.path.isfile(path_img), 'missing: %s' % path_img
    logging.debug('loading image: %s', path_img)
    img = np.array(Image.open(path_img))
    seg = (img > 0)
    seg = morphology.binary_opening(seg, selem=morphology.disk(25))
    seg = morphology.remove_small_objects(seg)
    seg_lb = measure.label(seg)
    seg_lb[seg == 0] = 0
    return seg, seg_lb


def draw_circle(pos_center, radius, img_shape):
    """ create empty image and draw a circle with specific radius

    :param [int, int] pos_center:
    :param int radius:
    :param [int, int] img_shape:
    :return ndarray:
    """
    im = np.zeros(img_shape)
    x, y = draw.circle(pos_center[0], pos_center[1], radius, shape=im.shape[:2])
    im[x, y] = True
    return im


def segm_set_center_levels(name, seg_labels, path_out, levels=DISTANCE_LEVELS):
    """ set segmentation levels according distance inside object segmentation

    :param str name: image name
    :param ndarray seg_labels:
    :param str path_out: path for output
    :param [float] levels: distance levels fro segmentation levels
    """
    seg = np.zeros_like(seg_labels)

    # set bourders to 0
    # seg_labels = set_boundary_values(seg_labels)

    for obj_id in range(1, seg_labels.max() + 1):
        im_bin = (seg_labels == obj_id)
        if np.sum(im_bin) == 0:
            continue
        distance = ndimage.distance_transform_edt(im_bin)
        probab = distance / np.max(distance)
        pos_center = ndimage.measurements.center_of_mass(im_bin)
        # logging.debug('object %i with levels: %s', obj_id, repr(levels))
        for i, level in enumerate(levels):
            mask = probab > level
            if i > 0:
                radius = int(np.sqrt(np.sum(mask) / np.pi))
                im_level = draw_circle(pos_center, radius, mask.shape)
                mask = np.logical_and(mask, im_level)
                sel = morphology.disk(int(radius * 0.15))
                mask = morphology.binary_opening(mask, sel)
            seg[mask] = i + 1

    path_seg = os.path.join(path_out, name)
    Image.fromarray(seg.astype(np.uint8)).save(path_seg)


def create_annot_centers(path_img, path_out_seg, path_out_csv):
    """ create and export annotation (levels, centres)
    from object segmentation (annotation)

    :param str path_img: path to image - object annotation
    :param str path_out_seg: path to output level annotation
    :param str path_out_csv: path to csv with centers
    """
    name = os.path.basename(path_img)
    _, seg_labeled = load_correct_segm(path_img)

    # # just convert the strange tif to common png
    # name = name.replace('slice_', 'insitu').replace('-label.tif', '.png')
    # path_seg = os.path.join(os.path.dirname(path_img), name)
    # Image.fromarray((255 * seg).astype(np.uint8)).save(path_seg)

    segm_set_center_levels(name, seg_labeled, path_out_seg)

    df_center = compute_eggs_center(seg_labeled)
    df_center.to_csv(os.path.join(path_out_csv, name.replace('.png', '.csv')))


def main(path_segs, path_out, nb_jobs):
    """ the main for creating annotations

    :param str path_segs: path with image pattern of images - obj segmentation
    :param str path_out:
    :param int nb_jobs: number of processes in parallel
    :return:
    """
    logging.info('running...')

    assert os.path.dirname(path_segs) != path_out, \
        'the output dir has to be different then the input object segmentation'
    list_imgs = glob.glob(path_segs)
    logging.info('found %i images', len(list_imgs))

    if not os.path.exists(path_out):
        assert os.path.isdir(os.path.dirname(path_out)), \
            'missing: %s' % path_out
        os.mkdir(path_out)

    tqdm_bar = tqdm.tqdm(total=len(list_imgs), desc='annotating images')
    wrapper_create_annot_centers = partial(create_annot_centers,
                                           path_out_seg=path_out,
                                           path_out_csv=path_out)
    if nb_jobs > 1:
        pool = mproc.Pool(nb_jobs)
        for _ in pool.imap_unordered(wrapper_create_annot_centers, list_imgs):
            tqdm_bar.update()
        pool.close()
        pool.join()
    else:
        for _ in map(wrapper_create_annot_centers, list_imgs):
            tqdm_bar.update()

    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    params = run_train.arg_parse_params(PARAMS)
    path_out = os.path.join(params['path_output'], NAME_DIR)
    main(params['path_segms'], path_out, params['nb_jobs'])

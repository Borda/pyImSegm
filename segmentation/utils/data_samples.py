# -*- coding: utf-8 -*-
"""
Sandbox with some sample images

Copyright (C) 2015-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import logging

from PIL import Image
import numpy as np

import segmentation.utils.data_io as tl_io

SAMPLE_SEG_SIZE_2D_SMALL = (20, 10)
SAMPLE_SEG_SIZE_2D_NORM = (150, 100)
SAMPLE_SEG_NB_CLASSES = 3
SAMPLE_SEG_SIZE_3D_SMALL = (10, 5, 6)

PATH_IMAGES = tl_io.update_path('images')
IMAGE_LENNA = os.path.join('others', 'lena.png')
IMAGE_OBJECTS = os.path.join('synthetic', 'reference.jpg')
IMAGE_3CLS = os.path.join('textures', 'sample_rgb_3cls.jpg')
IMAGE_STAR_1 = os.path.join('see_starfish', 'star_nb1-b.jpg')
IMAGE_STAR_2 = os.path.join('see_starfish', 'stars_nb2.jpg')
IMAGE_HISTOL_CIMA = \
    os.path.join('histology_CIMA', '29-041-Izd2-w35-CD31-3-les1.jpg')
IMAGE_HISTOL_FLAGSHIP = \
    os.path.join('histology_Flagship', 'Case001_Cytokeratin.jpg')
IMAGE_DROSOPHILA_DISC = \
    os.path.join('drosophila_disc', 'image', 'img_6.jpg')
ANNOT_DROSOPHILA_DISC = \
    os.path.join('drosophila_disc', 'annot', 'img_6.png')
IMAGE_DROSOPHILA_OVARY_2D = \
    os.path.join('drosophila_ovary_slice', 'image', 'insitu7545.jpg')
ANNOT_DROSOPHILA_OVARY_2D = \
    os.path.join('drosophila_ovary_slice', 'annot_struct', 'insitu7545.png')
IMAGE_DROSOPHILA_OVARY_3D = \
    os.path.join('drosophila_ovary_3D', 'AU10-13_f0011.tif')
IMAGE_LANGER_ISLET = \
    os.path.join('langerhans_islets', 'image', 'gtExoIsl_21.jpg')

LIST_ALL_IMAGES = [
    IMAGE_LENNA, IMAGE_3CLS, IMAGE_OBJECTS,
    IMAGE_STAR_1, IMAGE_STAR_2,
    IMAGE_HISTOL_CIMA, IMAGE_HISTOL_FLAGSHIP, IMAGE_LANGER_ISLET,
    IMAGE_DROSOPHILA_DISC, ANNOT_DROSOPHILA_DISC,
    IMAGE_DROSOPHILA_OVARY_2D, ANNOT_DROSOPHILA_OVARY_2D,
    IMAGE_DROSOPHILA_OVARY_3D,
]

for p in LIST_ALL_IMAGES:
    p = os.path.join(PATH_IMAGES, p)
    assert os.path.exists(p), 'missing: %s' % p


def sample_segment_vertical_2d(seg_size=SAMPLE_SEG_SIZE_2D_SMALL,
                               nb_labels=SAMPLE_SEG_NB_CLASSES):
    """ create sample segmentation with vertical stripes

    :param (int, int) seg_size:
    :param int nb_labels:
    :return ndarray:

    >>> sample_segment_vertical_2d((7, 5), 2)
    array([[0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1]])
    """
    cls_vals = []
    cls_size = (seg_size[1], int(seg_size[0] / nb_labels))
    for l in range(nb_labels):
        cls_vals.append(l * np.ones(cls_size))
    seg = np.hstack(tuple(cls_vals))
    seg = np.array(seg, dtype=np.int)
    return seg


def sample_segment_vertical_3d(seg_size=SAMPLE_SEG_SIZE_3D_SMALL,
                               nb_labels=SAMPLE_SEG_NB_CLASSES, levels=2):
    """ create sample regular 3D segmentation

    :param (int, int) seg_size:
    :param int nb_labels:
    :param int levels:
    :return ndarray:

    >>> im =  sample_segment_vertical_3d((10, 5, 6), 3)
    >>> im[:, :, 3]
    array([[1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [4, 4, 4, 4, 4],
           [4, 4, 4, 4, 4],
           [4, 4, 4, 4, 4]])
    """
    seg = []
    for l in range(int(levels)):
        seg_2d = sample_segment_vertical_2d(seg_size[:2], nb_labels)
        for _ in range(int(seg_size[2] / levels)):
            seg.append(seg_2d.copy() + l * nb_labels)
    seg = np.array(seg, dtype=np.int)
    return seg


def sample_color_image_rand_segment(im_size=SAMPLE_SEG_SIZE_2D_NORM,
                                    nb_classes=SAMPLE_SEG_NB_CLASSES,
                                    rand_seed=None):
    """ create samoe image and segmentation

    :param (int, int) im_size:
    :param int nb_classes:
    :param rand_seed:
    :return:

    >>> im, seg = sample_color_image_rand_segment((5, 6), 2, rand_seed=0)
    >>> im.shape
    (5, 6, 3)
    >>> seg
    array([[1, 1, 0, 0, 1, 0],
           [0, 1, 1, 0, 1, 0],
           [0, 1, 0, 0, 0, 1],
           [1, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 1, 0]])
    """
    assert len(im_size) == 2
    np.random.seed(rand_seed)
    im_size_rgb = (im_size[0], im_size[1], 3)
    img = np.random.random_integers(0, 255, im_size_rgb)
    seg = np.random.random_integers(0, nb_classes - 1, im_size)
    for lb in range(int(nb_classes)):
        val_step = 255 / nb_classes
        im = np.random.random_integers(val_step * lb, val_step * (lb + 1),
                                       im_size_rgb)
        img[seg == lb] = im[seg == lb]
    # img = Image.fromarray(np.array(im, dtype=np.uint8), 'RGB')
    return img, seg


def get_image_path(name_img, path_base=PATH_IMAGES):
    """ merge default image path and sample image

    :param str name_img:
    :return str:

    >>> p = get_image_path(IMAGE_LENNA)
    >>> os.path.isfile(p)
    True
    >>> os.path.basename(p)
    'lena.png'
    """
    path_img = os.path.join(path_base, name_img)
    path_img = tl_io.update_path(path_img)
    return path_img


def load_sample_image(name_img=IMAGE_LENNA):
    """ load sample image

    :param str name_img:
    :return ndarray:

    >>> img = load_sample_image(IMAGE_LENNA)
    >>> img.shape
    (512, 512, 3)
    """
    path_img = get_image_path(name_img)
    assert os.path.exists(path_img), 'missing "%s"' % path_img
    logging.debug('image (%s): %s', os.path.exists(path_img), path_img)
    img = np.array(Image.open(path_img, 'r'))
    return img

# -*- coding: utf-8 -*-
"""
Sandbox with some sample images

Some images are taken from following sources:

    * [ANHIR challenge](https://anhir.grand-challenge.org)
    * [PhD Thesis](https://www.researchgate.net/publication/323120618)

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import logging

# from PIL import Image
import numpy as np

from imsegm.utilities.data_io import update_path, io_imread

#: image size for small 2D sample image
SAMPLE_SEG_SIZE_2D_SMALL = (20, 10)
#: image size for normal 2D sample image
SAMPLE_SEG_SIZE_2D_NORM = (150, 100)
#: number of classes for segmentation
SAMPLE_SEG_NB_CLASSES = 3
#: image size for small 3D sample image
SAMPLE_SEG_SIZE_3D_SMALL = (10, 5, 6)

#: path to the folder with all sample image/data
PATH_IMAGES = update_path('data_images')
#: sample Lenna image
IMAGE_LENNA = os.path.join('others', 'lena.png')
#: sample image with three color objects of different shape
IMAGE_OBJECTS = os.path.join('synthetic', 'reference.jpg')
#: sample image with three different textures
IMAGE_3CLS = os.path.join('synthetic', 'texture_rgb_3cls.jpg')
#: sample image of sea starfish
IMAGE_STAR = os.path.join('others', 'sea_starfish-2.jpg')
#: sample image of histology tissue from CIMA dataset - ANHIR challenge
IMAGE_HISTOL_CIMA = os.path.join('histology_CIMA', '29-041-Izd2-w35-CD31-3-les1.jpg')
#: sample image of histology tissue
IMAGE_HISTOL_FLAGSHIP = os.path.join('histology_Flagship', 'Case001_Cytokeratin.jpg')
#: sample microscopy image of Drosophila imaginal disc
IMAGE_DROSOPHILA_DISC = os.path.join('drosophila_disc', 'image', 'img_6.jpg')
#: three class tissue annotation of microscopy image of Drosophila imaginal disc
ANNOT_DROSOPHILA_DISC = os.path.join('drosophila_disc', 'annot', 'img_6.png')
#: sample microscopy image of Drosophila ovary
IMAGE_DROSOPHILA_OVARY_2D = os.path.join(
    'drosophila_ovary_slice', 'image', 'insitu7545.jpg')
#: four class tissue annotation of microscopy image of Drosophila ovary
ANNOT_DROSOPHILA_OVARY_2D = os.path.join(
    'drosophila_ovary_slice', 'annot_struct', 'insitu7545.png')
#: sample 3D microscopy image of Drosophila ovary
IMAGE_DROSOPHILA_OVARY_3D = os.path.join('drosophila_ovary_3D', 'AU10-13_f0011.tif')
#: sample microscopy image of Langernat islets
IMAGE_LANGER_ISLET = os.path.join('langerhans_islets', 'image', 'gtExoIsl_21.jpg')
#: list of all default images
LIST_ALL_IMAGES = [
    IMAGE_LENNA, IMAGE_3CLS, IMAGE_OBJECTS, IMAGE_STAR,
    IMAGE_HISTOL_CIMA, IMAGE_HISTOL_FLAGSHIP, IMAGE_LANGER_ISLET,
    IMAGE_DROSOPHILA_DISC, ANNOT_DROSOPHILA_DISC,
    IMAGE_DROSOPHILA_OVARY_2D, ANNOT_DROSOPHILA_OVARY_2D,
    IMAGE_DROSOPHILA_OVARY_3D,
]


def sample_segment_vertical_2d(seg_size=SAMPLE_SEG_SIZE_2D_SMALL,
                               nb_labels=SAMPLE_SEG_NB_CLASSES):
    """ create sample segmentation with vertical stripes

    :param tuple(int,int) seg_size:
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

    :param tuple(int,int)|(int, int, int) seg_size:
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

    :param tuple(int,int) im_size:
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
    assert len(im_size) == 2, \
        'required image dimension is 2 to instead %r' % im_size
    np.random.seed(rand_seed)
    im_size_rgb = (im_size[0], im_size[1], 3)
    img = np.random.random_integers(0, 255, im_size_rgb)
    seg = np.random.random_integers(0, nb_classes - 1, im_size)
    for lb in range(int(nb_classes)):
        val_step = 255 / nb_classes
        im = np.random.random_integers(int(val_step * lb),
                                       int(val_step * (lb + 1)),
                                       im_size_rgb)
        img[seg == lb] = im[seg == lb]
    # img = Image.fromarray(np.array(im, dtype=np.uint8), 'RGB')
    return img, seg


def get_image_path(name_img, path_base=PATH_IMAGES):
    """ merge default image path and sample image

    :param str name_img:
    :param str path_base:
    :return str:

    >>> p = get_image_path(IMAGE_LENNA)
    >>> os.path.isfile(p)
    True
    >>> os.path.basename(p)
    'lena.png'
    """
    path_img = os.path.join(path_base, name_img)
    path_img = update_path(path_img)
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
    assert os.path.exists(path_img), 'missing: "%s"' % path_img
    logging.debug('image (%s): %s', os.path.exists(path_img), path_img)
    img = io_imread(path_img)
    return img

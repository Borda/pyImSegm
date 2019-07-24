"""
Framework for feature extraction
 * color and gray 3D images
 * color and texture features
 * Ray features
 * label histogram

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import itertools
import logging

import numpy as np
from scipy import ndimage, interpolate, optimize, spatial
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d, gaussian_laplace
from sklearn import preprocessing
from skimage import morphology
# from numba.decorators import jit
# from numba import int32, int64, float32

from imsegm.utilities.data_io import convert_img_color_from_rgb
try:
    import imsegm.features_cython as fts_cython
    # logging.debug('try to load Cython implementation')  # CRASH logger
    USE_CYTHON = True
except Exception:
    # NOTE: in some cases following warning may crash all message logging
    print('descriptors: using pure python libraries')
    USE_CYTHON = False

#: define all available statistic computed on superpixels
NAMES_FEATURE_FLAGS = ('mean', 'std', 'energy', 'median', 'meanGrad')
#: define sigmas for Lewen-Malik filter bank
DEFAULT_FILTERS_SIGMAS = (np.sqrt(2), 2, 2 * np.sqrt(2), 4)
#: define small list/range of sigmas for Lewen-Malik filter bank
SHORT_FILTERS_SIGMAS = (np.sqrt(2), 2, 4)
#: define the richest version of computed superpixel features
FEATURES_SET_ALL = {'color': ('mean', 'std', 'energy', 'median', 'meanGrad'),
                    'tLM': ('mean', 'std', 'energy', 'median', 'meanGrad')}
#: define basic color features for supepixels
FEATURES_SET_COLOR = {'color': ('mean', 'std', 'energy')}
#: define basic texture features (complete LM filter bank) for supepixels
FEATURES_SET_TEXTURE = {'tLM': ('mean', 'std', 'energy')}
#: define basic color features for (small LM filter bank) supepixels
FEATURES_SET_TEXTURE_SHORT = {'tLM_short': ('mean', 'std', 'energy')}
#: define circular diamters for computing label histogram
HIST_CIRCLE_DIAGONALS = (10, 20, 30, 40, 50)
#: maximal response is bounded by fix number to prevent overflowing (for LM filer bank)
MAX_SIGNAL_RESPONSE = 1.e6

# Wavelets:
# * http://www.pybytes.com/pywavelets/
# * https://pypi.python.org/pypi/dtcwt/0.10.0


# NUMBA code
# @jit
# def computeColourMeanRGB(im, seg):
#     """
#     compute mean colour in RGB per segment
#    :param im: input RGB image
#    :param seg: segmentation og the image
#    :return:[][3] vector of mean colour per segmrnt
#     """
#     img = np.array(im)
#     nbSegments = np.max(seg[:]) +1
#     logging.info('Computing RGB means for {} segments'.format(nbSegments))
#     features = np.zeros([nbSegments, 3])
#     count = np.zeros([nbSegments, 1])
#     w, h = im.shape[:2]
#     for x in range(w):
#         for y in range(h):
#             count[seg[x,y], 0] += 1
#             features[seg[x,y],:] += img[x, y,:]
#     logging.debug('features > ' +repr(features))
#     logging.debug('counts > ' +repr(count))
#     features = features / np.asarray(count, dtype=np.float)
#     return features


# NUMBA code
# @jit
# def computeColourMean(im, seg):
#     """
#     compute mean colour in RGB per segment
#    :param im: input RGB image
#    :param seg: segmentation og the image
#    :return:[][3] vector of mean colour per segmrnt
#     """
#     img = np.array(im)
#     uniqueLbs = np.unique(seg[:])
#     logging.info('Computing RGB means for %d segments', uniqueLbs.max())
#     features = np.zeros([uniqueLbs.max()+1, 3])
#     for l in uniqueLbs:
#         features[l] = np.mean(img[ seg==l ])
#     logging.debug('features > ' +repr(features))
#     return features


# def cython_mean_img2d_rgb(im, seg):
#     """ wrapper for fast implementation of colour features
#
#     :param ndarray im: input RGB image
#     :param ndarray seg: segmentation og the image
#     :return: np.array<nb_lbs, 3> matrix features per segment
#
#     >>> image = np.zeros((2, 10, 3))
#     >>> image[:, 2:6, 0] = 1
#     >>> image[:, 3:7, 1] = 3
#     >>> image[:, 4:9, 2] = 2
#     >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#     ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
#     >>> cython_mean_img2d_rgb(image, segm)
#     array([[ 0.6,  1.2,  0.4],
#            [ 0.2,  1.2,  1.6]])
#     """
#     logging.debug('Cython: computing RGB means for %i segments', np.max(seg))
#     if im.shape[:2] != seg.shape:
#         raise ValueError('arrays - image and segm do not match %s vs %s'
#                          % (repr(im.shape), repr(seg.shape)))
#     means = fts_cython.getColourMeanImg2dRGB(np.array(im, dtype=np.int16),
#                                              np.array(seg, dtype=np.int16))
#     return np.array(means)


def _check_color_image_segm(image, segm):
    """ verify image - segmentation compatibility

    :param ndarray image: image
    :param ndarray segm: segmentation
    :return bool:

    >>> _check_color_image_segm(np.zeros((125, 150, 3)), np.zeros((150, 125)))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: ndarrays - image and segmentation do not match (125, 150, 3) vs (150, 125)
    """
    if image.shape[:2] != segm.shape:
        raise ValueError('ndarrays - image and segmentation do not match %r vs %r'
                         % (image.shape, segm.shape))
    return True


def _check_gray_image_segm(image, segm):
    """ verify image - segmentation compatibility

    :param ndarray image: image
    :param ndarray segm: segmentation
    :return bool:

    >>> _check_gray_image_segm(np.zeros((125, 150)), np.zeros((150, 125)))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: ndarrays - image and segmentation do not match (125, 150) vs (150, 125)
    """
    if image.shape != segm.shape:
        raise ValueError('ndarrays - image and segmentation do not match %r vs %r'
                         % (image.shape, segm.shape))
    return True


def _check_color_image(image):
    """ verify proper image

    :param ndarray image:
    :return bool:

    >>> _check_color_image(np.zeros((200, 250, 1)))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: image is not RGB with dims (200, 250, 1)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image is not RGB with dims %s' % repr(image.shape))
    return True


def _check_unrecognised_feature_group(feature_flags):
    """ search for not defined flags

    :param dict feature_flags: input
    :return list(str): unrecognised

    >>> _check_unrecognised_feature_group({'color': [], 'texture': []})
    ['texture']
    """
    unknown = [k for k in feature_flags
               if not (k.startswith('color') or k.startswith('tLM'))]
    if unknown:
        logging.warning('unrecognised following feature groups: %r', unknown)
    return unknown


def _check_unrecognised_feature_names(feature_flags):
    """ search for not defined flags

    :param list(str) feature_flags: input
    :return list(str): unrecognised

    >>> _check_unrecognised_feature_names(['mean', 'average'])
    ['average']
    """
    unknown = [k for k in feature_flags if k not in NAMES_FEATURE_FLAGS]
    if unknown:
        logging.warning('unrecognised following feature names: %r', unknown)
    return unknown


def cython_img2d_color_mean(img, seg):
    """ wrapper for fast implementation of colour features

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    .. seealso:: :func:`imsegm.descriptors.numpy_img2d_color_mean`

    >>> image = np.zeros((2, 10, 3))
    >>> image[:, 2:6, 0] = 1
    >>> image[:, 3:7, 1] = 3
    >>> image[:, 4:9, 2] = 2
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> cython_img2d_color_mean(image, segm)
    array([[ 0.6,  1.2,  0.4],
           [ 0.2,  1.2,  1.6]])
    """
    logging.debug('Cython: computing Colour means for image %r & segm %r with'
                  ' %i segments', img.shape, seg.shape, np.max(seg))
    _check_color_image_segm(img, seg)

    means = fts_cython.computeColorImage2dMean(np.array(img, dtype=np.float32),
                                               np.array(seg, dtype=np.int32))
    return np.array(means)


def cython_img2d_color_energy(img, seg):
    """  wrapper for fast implementation of colour features

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    .. seealso:: :func:`imsegm.descriptors.numpy_img2d_color_energy`

    >>> image = np.zeros((2, 10, 3))
    >>> image[:, 2:6, 0] = 1
    >>> image[:, 3:7, 1] = 3
    >>> image[:, 4:9, 2] = 2
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> cython_img2d_color_energy(image, segm)
    array([[ 0.6,  3.6,  0.8],
           [ 0.2,  3.6,  3.2]])
    """
    logging.debug('Cython: computing Colour energy for image %r & segm %r with'
                  ' %i segments', img.shape, seg.shape, np.max(seg))
    _check_color_image_segm(img, seg)

    energy = fts_cython.computeColorImage2dEnergy(np.array(img, dtype=np.float32),
                                                  np.array(seg, dtype=np.int32))
    return np.array(energy)


def cython_img2d_color_std(img, seg, means=None):
    """ wrapper for fast implementation of colour features

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :param ndarray means: precomputed feature means
    :return: np.array<nb_lbs, 3> matrix features per segment

    .. seealso:: :func:`imsegm.descriptors.numpy_img2d_color_std`

    >>> image = np.zeros((2, 10, 3))
    >>> image[:, 2:6, 0] = 1
    >>> image[:, 3:7, 1] = 3
    >>> image[:, 4:9, 2] = 2
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> cython_img2d_color_std(image, segm)
    array([[ 0.48989794,  1.46969383,  0.80000003],
           [ 0.40000001,  1.46969383,  0.80000001]])
    """
    logging.debug('Cython: computing Colour STD for image %r & segm %r with'
                  ' %i segments', img.shape, seg.shape, np.max(seg))
    _check_color_image_segm(img, seg)

    if means is None:
        means = cython_img2d_color_mean(img, seg)
    var = fts_cython.computeColorImage2dVariance(np.array(img, dtype=np.float32),
                                                 np.array(seg, dtype=np.int32),
                                                 np.array(means, dtype=np.float32))
    std = np.sqrt(var)
    return std


def numpy_img2d_color_mean(img, seg):
    """ compute color means by numpy

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    .. seealso:: :func:`imsegm.descriptors.cython_img2d_color_mean`

    >>> image = np.zeros((2, 10, 3))
    >>> image[:, 2:6, 0] = 1
    >>> image[:, 3:8, 1] = 3
    >>> image[:, 4:9, 2] = 2
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> numpy_img2d_color_mean(image, segm)
    array([[ 0.6,  1.2,  0.4],
           [ 0.2,  1.8,  1.6]])
    """
    logging.debug('computing Colour mean for image %r & segm %r with'
                  ' %i segments', img.shape, seg.shape, np.max(seg))
    _check_color_image_segm(img, seg)

    nb_labels = np.max(seg) + 1
    means = np.zeros((nb_labels, 3))
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            lb = seg[i, j]
            means[lb, :] += img[i, j, :]
            counts[lb] += 1
    # prevent dividing by 0
    counts[counts == 0] = -1
    means = (means / np.tile(counts, (3, 1)).T.astype(float))
    # preventing negative zeros
    # means[means == 0] = 0
    return means


def numpy_img2d_color_std(img, seg, means=None):
    """ compute color STD by numpy

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :param ndarray means: precomputed feature means
    :return: np.array<nb_lbs, 3> matrix features per segment

    .. seealso:: :func:`imsegm.descriptors.cython_img2d_color_std`

    >>> image = np.zeros((2, 10, 3))
    >>> image[:, 2:6, 0] = 1
    >>> image[:, 3:8, 1] = 3
    >>> image[:, 4:9, 2] = 2
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> numpy_img2d_color_std(image, segm)
    array([[ 0.48989795,  1.46969385,  0.8       ],
           [ 0.4       ,  1.46969385,  0.8       ]])
    """
    logging.debug('computing Colour STD for image %r & segm %r with'
                  ' %i segments', img.shape, seg.shape, np.max(seg))
    _check_color_image_segm(img, seg)

    if means is None:
        means = numpy_img2d_color_mean(img, seg)

    nb_labels = np.max(seg) + 1
    assert len(means) >= nb_labels, \
        'number of means (%i) should be equal to number of labels (%i)' \
        % (len(means), nb_labels)
    variations = np.zeros((nb_labels, 3))
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            lb = seg[i, j]
            variations[lb, :] += (img[i, j, :] - means[lb, :]) ** 2
            counts[lb] += 1
    # prevent dividing by 0
    counts[counts == 0] = -1
    variations = (variations / np.tile(counts, (3, 1)).T.astype(float))
    # preventing negative zeros
    variations[variations == 0] = 0
    stds = np.sqrt(variations)
    return stds


def numpy_img2d_color_energy(img, seg):
    """ compute color energy by numpy

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    .. seealso:: :func:`imsegm.descriptors.cython_img2d_color_energy`

    >>> image = np.zeros((2, 10, 3))
    >>> image[:, 2:6, 0] = 1
    >>> image[:, 3:8, 1] = 3
    >>> image[:, 4:9, 2] = 2
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> numpy_img2d_color_energy(image, segm)
    array([[ 0.6,  3.6,  0.8],
           [ 0.2,  5.4,  3.2]])
    """
    logging.debug('computing Colour energy for image %r & segm %r with'
                  ' %i segments', img.shape, seg.shape, np.max(seg))
    _check_color_image_segm(img, seg)

    nb_labels = np.max(seg) + 1
    energy = np.zeros((nb_labels, 3))
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            lb = seg[i, j]
            energy[lb, :] += img[i, j, :] ** 2
            counts[lb] += 1
    # prevent dividing by 0
    counts[counts == 0] = -1
    energy = (energy / np.tile(counts, (3, 1)).T.astype(float))
    # preventing negative zeros
    # energy[energy == 0] = 0
    return energy


def numpy_img2d_color_median(img, seg):
    """ compute color median by numpy

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    .. seealso:: :func:`imsegm.descriptors.cython_img2d_color_median`

    >>> image = np.zeros((2, 10, 3))
    >>> image[:, 2:6, 0] = 1
    >>> image[:, 3:8, 1] = 3
    >>> image[:, 4:9, 2] = 2
    >>> segm = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    >>> numpy_img2d_color_median(image, segm)
    array([[ 0.5,  0. ,  0. ],
           [ 0. ,  3. ,  2. ]])
    """
    logging.debug('computing Colour median for image %r & segm %r with'
                  ' %i segments', img.shape, seg.shape, np.max(seg))
    _check_color_image_segm(img, seg)

    nb_labels = np.max(seg) + 1
    list_values = [([], [], []) for _ in range(nb_labels)]

    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            lb = seg[i, j]
            for k in range(3):
                list_values[lb][k].append(img[i, j, k])

    medians = np.zeros((nb_labels, 3))
    for i in range(nb_labels):
        for k in range(3):
            medians[i, k] = np.median(list_values[i][k])
    return medians


def cython_img3d_gray_mean(img, seg):
    """ wrapper for fast implementation of colour features

    WARNING: the Z dimension is parallel and without sync,
    multiple equal labels across Z dim may lead to not mistakes in summing

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 1> vector of mean colour per segment

    .. seealso:: :func:`imsegm.descriptors.numpy_img3d_gray_mean`

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> segm.shape
    (2, 3, 8)
    >>> cython_img3d_gray_mean(image, segm)
    array([ 0.5 ,  0.5 ,  0.75,  2.25])
    """
    logging.debug('Cython: computing Gray means for image %r and segm %r with'
                  ' %i segments', img.shape, seg.shape, np.max(seg))
    _check_gray_image_segm(img, seg)

    means = fts_cython.computeGrayImage3dMean(np.array(img, dtype=np.float32),
                                              np.array(seg, dtype=np.int32))
    return np.array(means)


def cython_img3d_gray_energy(img, seg):
    """ wrapper for fast implementation of colour features

    WARNING: the Z dimension is parallel and without sync,
    multiple equal labels across Z dim may lead to not mistakes in summing

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :return:np.array<nb_lbs, 1> vector of mean colour per segment

    .. seealso:: :func:`imsegm.descriptors.numpy_img3d_gray_energy`

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> cython_img3d_gray_energy(image, segm)
    array([ 0.5 ,  0.5 ,  2.25,  6.75])
    """
    logging.debug('Cython: computing Gray energy for image %r and segm %r with'
                  ' %i segments', img.shape, seg.shape, np.max(seg))
    _check_gray_image_segm(img, seg)

    energy = fts_cython.computeGrayImage3dEnergy(np.array(img, dtype=np.float32),
                                                 np.array(seg, dtype=np.int32))
    return np.array(energy)


def cython_img3d_gray_std(img, seg, mean=None):
    """ wrapper for fast implementation of colour features

    WARNING: the Z dimension is parallel and without sync,
    multiple equal labels across Z dim may lead to not mistakes in summing

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :param ndarray mean: precomputed feature means
    :return:np.array<nb_lbs, 1> vector of mean colour per segment

    .. seealso:: :func:`imsegm.descriptors.numpy_img3d_gray_std`

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> cython_img3d_gray_std(image, segm)
    array([ 0.5       ,  0.5       ,  1.29903811,  1.29903811])
    """
    logging.debug('Cython: computing Gray STD for image %r and segm %r with'
                  ' %i segments', img.shape, seg.shape, np.max(seg))
    _check_gray_image_segm(img, seg)

    if mean is None:
        mean = cython_img3d_gray_mean(img, seg)
    var = fts_cython.computeGrayImage3dVariance(np.array(img, dtype=np.float32),
                                                np.array(seg, dtype=np.int32),
                                                np.array(mean, dtype=np.float32))
    std = np.sqrt(var)
    return std


def numpy_img3d_gray_mean(img, seg):
    """ compute gray (3D) means by numpy

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    .. seealso:: :func:`imsegm.descriptors.cython_img3d_gray_mean`

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> numpy_img3d_gray_mean(image, segm)
    array([ 0.5 ,  0.5 ,  0.75,  2.25])
    """
    logging.debug('computing Gray mean for %i segments', np.max(seg))
    _check_gray_image_segm(img, seg)

    nb_labels = np.max(seg) + 1
    means = np.zeros(nb_labels)
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                lb = seg[i, j, k]
                means[lb] += img[i, j, k]
                counts[lb] += 1
    # just for not dividing by 0
    counts[counts == 0] = -1
    means = (means / counts.astype(float))
    # preventing negative zeros
    # means[means == 0] = 0
    return means


def numpy_img3d_gray_std(img, seg, means=None):
    """ compute gray (3D) STD by numpy

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :param ndarray means: precomputed feature means
    :return: np.array<nb_lbs, 3> matrix features per segment

    .. seealso:: :func:`imsegm.descriptors.cython_img3d_gray_std`

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> numpy_img3d_gray_std(image, segm)
    array([ 0.5       ,  0.5       ,  1.29903811,  1.29903811])
    """
    logging.debug('computing Gray mean for %i segments', np.max(seg))
    _check_gray_image_segm(img, seg)

    if means is None:
        means = numpy_img3d_gray_mean(img, seg)

    nb_labels = np.max(seg) + 1
    assert len(means) >= nb_labels, \
        'number of means (%i) should be equal to number of labels (%i)' \
        % (len(means), nb_labels)
    variances = np.zeros(nb_labels)
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                lb = seg[i, j, k]
                variances[lb] += (img[i, j, k] - means[lb]) ** 2
                counts[lb] += 1
    # just for not dividing by 0
    counts[counts == 0] = -1
    variances = (variances / counts.astype(float))
    # preventing negative zeros
    variances[variances == 0] = 0
    stds = np.sqrt(variances)
    return stds


def numpy_img3d_gray_energy(img, seg):
    """ compute gray (3D) energy by numpy

    :param img: input RGB image
    :param seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    .. seealso:: :func:`imsegm.descriptors.cython_img3d_gray_energy`

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> numpy_img3d_gray_energy(image, segm)
    array([ 0.5 ,  0.5 ,  2.25,  6.75])
    """
    logging.debug('computing Gray energy for %i segments', np.max(seg))
    _check_gray_image_segm(img, seg)

    nb_labels = np.max(seg) + 1
    energy = np.zeros(nb_labels)
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                lb = seg[i, j, k]
                energy[lb] += img[i, j, k] ** 2
                counts[lb] += 1
    # just for not dividing by 0
    counts[counts == 0] = -1
    energy = (energy / counts.astype(float))
    # preventing negative zeros
    # energy[energy == 0] = 0
    return energy


def numpy_img3d_gray_median(img, seg):
    """ compute gray (3D) median by numpy

    :param ndarray img: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    .. seealso:: :func:`imsegm.descriptors.cython_img3d_gray_median`

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> numpy_img3d_gray_median(image, segm)
    array([ 0.5,  0.5,  0. ,  3. ])
    """
    logging.debug('computing Gray median for %i segments', np.max(seg))
    _check_gray_image_segm(img, seg)

    nb_labels = np.max(seg) + 1
    list_values = [[] for _ in range(nb_labels)]

    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                list_values[seg[i, j, k]].append(img[i, j, k])

    medians = np.zeros(nb_labels)
    for i in range(nb_labels):
        medians[i] = np.median(list_values[i])
    return medians


def compute_image3d_gray_statistic(image, segm,
                                   feature_flags=NAMES_FEATURE_FLAGS,
                                   ch_name='gray'):
    """ compute complete descriptors / statistic on gray (3D) images

    :param ndarray image:
    :param ndarray segm: segmentation
    :param list(str) feature_flags:
    :param str ch_name: channel name
    :return tuple(ndarray,list(str)): np.ndarray<nb_samples, nb_features>

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 5, 5, 5, 5]] * 3])
    >>> segm.shape
    (2, 3, 8)
    >>> features, names = compute_image3d_gray_statistic(image, segm)
    >>> features.shape
    (6, 5)
    >>> np.round(features, 3)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.25 ],
           [ 0.5  ,  0.5  ,  0.5  ,  0.5  , -0.25 ],
           [ 0.75 ,  1.299,  2.25 ,  0.   ,  0.75 ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 2.25 ,  1.299,  6.75 ,  3.   , -1.125]])
    >>> names  # doctest: +NORMALIZE_WHITESPACE
    ['gray_mean',
     'gray_std',
     'gray_energy',
     'gray_median',
     'gray_meanGrad']
    """
    _check_gray_image_segm(image, segm)

    assert list(feature_flags), 'some features has to be selected'
    image = np.nan_to_num(image)
    features = []
    # nb_fts = image.shape[0]
    # ch_names = ['%s-ch%i' % (ch_name, i + 1) for i in range(nb_fts)]

    _fn_mean = cython_img3d_gray_mean if USE_CYTHON else numpy_img3d_gray_mean
    _fn_std = cython_img3d_gray_std if USE_CYTHON else numpy_img3d_gray_std
    _fn_energy = cython_img3d_gray_energy if USE_CYTHON else numpy_img3d_gray_energy

    # MEAN
    mean = None
    if 'mean' in feature_flags:
        mean = _fn_mean(image, segm)
        features.append(mean)
    # Standard Deviation
    if 'std' in feature_flags:
        features.append(_fn_std(image, segm, mean))
    # ENERGY
    if 'energy' in feature_flags:
        features.append(_fn_energy(image, segm))
    # MEDIAN
    if 'median' in feature_flags:
        features.append(numpy_img3d_gray_median(image, segm))
    # mean Gradient
    if 'meanGrad' in feature_flags:
        grad_matrix = np.zeros_like(image)
        for i in range(image.shape[0]):
            grad_matrix[i, :, :] = np.sum(np.gradient(image[i]), axis=0)
        features.append(_fn_mean(grad_matrix, segm))

    names = ['%s_%s' % (ch_name, fts_name)
             for fts_name in ('mean', 'std', 'energy', 'median', 'meanGrad')
             if fts_name in feature_flags]
    _check_unrecognised_feature_names(feature_flags)

    features = np.concatenate(tuple([fts] for fts in features), axis=0)
    features = np.nan_to_num(features).T
    # normalise +/- zeros as set all as positive
    features[features == 0] = 0
    assert features.shape[1] == len(names), \
        'features: %r and names %r' % (features.shape, names)
    return features, names


def compute_image2d_color_statistic(image, segm,
                                    feature_flags=NAMES_FEATURE_FLAGS,
                                    color_name='color'):
    """ compute complete descriptors / statistic on color (2D) images

    :param ndarray image:
    :param ndarray segm: segmentation
    :param list(str) feature_flags:
    :param str color_name: channel name
    :return tuple(ndarray,list(str)): np.ndarray<nb_samples, nb_features>

    >>> image = np.zeros((2, 10, 3))
    >>> image[:, 2:6, 0] = 1
    >>> image[:, 3:7, 1] = 3
    >>> image[:, 4:9, 2] = 2
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> features, names = compute_image2d_color_statistic(image, segm)
    >>> names  # doctest: +NORMALIZE_WHITESPACE
    ['color-ch1_mean', 'color-ch2_mean', 'color-ch3_mean',
     'color-ch1_std', 'color-ch2_std', 'color-ch3_std',
     'color-ch1_energy', 'color-ch2_energy', 'color-ch3_energy',
     'color-ch1_median', 'color-ch2_median', 'color-ch3_median',
     'color-ch1_meanGrad', 'color-ch2_meanGrad', 'color-ch3_meanGrad']
    >>> features.shape
    (2, 15)
    >>> np.round(features, 1).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [[0.6, 1.2, 0.4, 0.5, 1.5, 0.8, 0.6, 3.6, 0.8, 1.0, 0.0, 0.0, 0.2, 0.6, 0.4],
     [0.2, 1.2, 1.6, 0.4, 1.5, 0.8, 0.2, 3.6, 3.2, 0.0, 0.0, 2.0, -0.2, -0.6, -0.6]]
    """
    _check_color_image(image)
    _check_color_image_segm(image, segm)

    image = np.nan_to_num(image)
    features = np.empty((np.max(segm) + 1, 0))
    ch_names = ['%s-ch%i' % (color_name, i + 1) for i in range(3)]

    _fn_mean = cython_img2d_color_mean if USE_CYTHON else numpy_img2d_color_mean
    _fn_std = cython_img2d_color_std if USE_CYTHON else numpy_img2d_color_std
    _fn_energy = cython_img2d_color_energy if USE_CYTHON else numpy_img2d_color_energy

    # MEAN
    mean = None
    if 'mean' in feature_flags:
        mean = _fn_mean(image, segm)
        features = np.hstack((features, mean))
    # Standard Deviation
    if 'std' in feature_flags:
        features = np.hstack((features, _fn_std(image, segm, mean)))
    # ENERGY
    if 'energy' in feature_flags:
        features = np.hstack((features, _fn_energy(image, segm)))
    # MEDIAN
    if 'median' in feature_flags:
        features = np.hstack((features, numpy_img2d_color_median(image, segm)))
    # mean Gradient
    if 'meanGrad' in feature_flags:
        grad_matrix = np.zeros_like(image)
        for i in range(image.shape[-1]):
            grad_matrix[:, :, i] = np.sum(np.gradient(image[:, :, i]), axis=0)
        features = np.hstack((features, _fn_mean(grad_matrix, segm)))

    feature_names = ('mean', 'std', 'energy', 'median', 'meanGrad')
    names = list(itertools.chain.from_iterable(
        ['%s_%s' % (n, fts_name) for n in ch_names]
        for fts_name in feature_names if fts_name in feature_flags))
    _check_unrecognised_feature_names(feature_flags)
    # mean Gradient
    # G = np.zeros_like(image)
    # for i in range(image.shape[0]):
    #     G[i,:,:] = np.sum(np.gradient(image[i]), axis=0)
    # grad = cython_img3d_gray_mean(G, segm)
    features = np.nan_to_num(features)
    # normalise +/- zeros as set all as positive
    features[features == 0] = 0
    assert features.shape[1] == len(names), \
        'features: %r and names %r' % (features.shape, names)
    return features, names


def norm_features(features, scaler=None):
    """ normalise features to be in range(0;1)

    :param ndarray features: vector of features
    :param obj scaler:
    :return list(list(float)):
    """
    if not scaler:
        scaler = preprocessing.StandardScaler()
        scaler.fit(features)
    features = scaler.transform(features)
    return features, scaler


def make_gaussian_filter1d(vals, sigma, order=0):
    if order > 2:
        raise ValueError("Only orders up to 2 are supported")
    # compute unnormalized Gaussian response
    response = np.exp(-vals ** 2 / (2. * sigma ** 2))
    if order == 1:
        response = - response * vals
    elif order == 2:
        response = (response * (vals ** 2 - sigma ** 2))
    # normalize
    response /= np.abs(response).sum()
    return response


def make_edge_filter2d(sig, phase, points, sup):
    gx = make_gaussian_filter1d(points[0, :], sigma=3 * sig)
    gy = make_gaussian_filter1d(points[1, :], sigma=sig, order=phase)
    ft = (gx * gy).reshape(sup, sup)
    # normalize
    ft /= np.abs(ft).sum()
    return ft


def create_filter_bank_lm_2d(radius=16, sigmas=DEFAULT_FILTERS_SIGMAS,
                             nb_orient=8):
    """ create filter bank with  rotation, Gaussian, Laplace-Gaussian, ...

    :param radius:
    :param sigmas:
    :param nb_orient:
    :return np.ndarray<nb_samples, nb_features>, list(str):

    >>> filters, names = create_filter_bank_lm_2d(6, SHORT_FILTERS_SIGMAS, 2)
    >>> [f.shape for f in filters]  # doctest: +NORMALIZE_WHITESPACE
    [(2, 13, 13), (2, 13, 13), (1, 13, 13), (1, 13, 13), (1, 13, 13),
     (2, 13, 13), (2, 13, 13), (1, 13, 13), (1, 13, 13), (1, 13, 13),
     (2, 13, 13), (2, 13, 13), (1, 13, 13), (1, 13, 13), (1, 13, 13)]
    >>> names  # doctest: +NORMALIZE_WHITESPACE
    ['sigma1.4-edge', 'sigma1.4-bar',
     'sigma1.4-Gauss', 'sigma1.4-GaussLap', 'sigma1.4-GaussLap2',
     'sigma2.0-edge', 'sigma2.0-bar',
     'sigma2.0-Gauss', 'sigma2.0-GaussLap', 'sigma2.0-GaussLap2',
     'sigma4.0-edge', 'sigma4.0-bar',
     'sigma4.0-Gauss', 'sigma4.0-GaussLap', 'sigma4.0-GaussLap2']
    """
    logging.debug('creating Leung-Malik filter bank')
    support = 2 * radius + 1
    x, y = np.mgrid[-radius:radius + 1, radius:-radius - 1:-1]
    org_pts = np.vstack([x.ravel(), y.ravel()])
    a = np.zeros((support, support))
    a[radius, radius] = 1

    filters, names = [], []
    for sigma in sigmas:
        orient_edge, orient_bar = [], []
        for orient in range(nb_orient):
            # Not 2pi as filters have symmetry
            angle = np.pi * orient / nb_orient
            c, s = np.cos(angle), np.sin(angle)
            rot_points = np.dot(np.array([[c, -s], [s, c]]), org_pts)
            orient_edge.append(make_edge_filter2d(sigma, 1, rot_points, support))
            orient_bar.append(make_edge_filter2d(sigma, 2, rot_points, support))
        filters.append(np.asarray(orient_edge))
        filters.append(np.asarray(orient_bar))

        filters.append(gaussian_filter(a, sigma)[np.newaxis, :, :])
        filters.append(gaussian_laplace(a, sigma)[np.newaxis, :, :])
        filters.append(gaussian_laplace(a, sigma ** 2)[np.newaxis, :, :])
        names += ['sigma%.1f-%s' % (sigma, n)
                  for n in ['edge', 'bar', 'Gauss', 'GaussLap', 'GaussLap2']]
    return filters, names


def compute_img_filter_response2d(img, filter_battery):
    """ compute image filter response in 2D

    :param [[float]] img: image
    :param [[[float]]] filter_battery: filters
    :return [[float]]:
    """
    if filter_battery.ndim != 3:
        raise ValueError('wrong battery dim %r' % filter_battery.shape)
    responses = np.array([ndimage.convolve(img, fl) for fl in filter_battery])
    if filter_battery.shape[0] > 1:
        # usually for rotational edge detectors and we tae the maximal response
        response = np.max(responses, axis=0)
    else:
        response = responses[0]
    return response


def compute_img_filter_response3d(img, filter_battery):
    """ compute image filter response in 3D

    :param ndarray img:
    :param ndarray filter_battery:
    :return:
    """
    logging.debug('compute image filter response in 3D')
    response = np.array([compute_img_filter_response2d(img[i, :, :],
                                                       filter_battery)
                         for i in range(img.shape[0])])
    return response


def image_subtract_gauss_smooth(img, sigma):
    """ smoothing by fist dimension assuming the in dim 0. image is independent

    :param ndarray img:
    :param sigma:
    :return:
    """
    if sigma <= 0:
        return img
    img_smooth = np.zeros(img.shape)
    for i in range(img.shape[0]):
        img_smooth[i, :, :] = gaussian_filter(img[i, :, :].astype(float), sigma)
    img = (img - img_smooth)
    return img


def compute_texture_desc_lm_img3d_val(img, seg, feature_flags, bank_type='normal'):
    """ compute texture descriptors as mean / std / ...
    on Lewen-Malik filter bank response

    :param [[[float]]] img: image
    :param [[[int]]] seg: segmentation
    :param list(str) feature_flags: list of feature flags
    :param str bank_type: define used LM filter bank ['short', 'normal']
    :return tuple(ndarray,list(str)): np.ndarray<nb_samples, nb_features>, names

    .. seealso:: :func:`imsegm.descriptors.compute_texture_desc_lm_img2d_clr`
    """
    _check_gray_image_segm(img, seg)

    logging.debug('compute texture descriptors using Leung-Malik')
    img = image_subtract_gauss_smooth(img, 150)
    if bank_type == 'short':
        filters, fl_names = create_filter_bank_lm_2d(sigmas=SHORT_FILTERS_SIGMAS,
                                                     nb_orient=4)
    else:
        filters, fl_names = create_filter_bank_lm_2d()
    features, names = [], []
    for battery, fl_name in zip(filters, fl_names):
        response = compute_img_filter_response3d(img, battery)
        # cut too large values
        response[response > MAX_SIGNAL_RESPONSE] = MAX_SIGNAL_RESPONSE
        # norm responses
        l_n = np.sqrt(np.sum(np.power(response, 2)))
        if l_n == 0 or abs(l_n) == np.Inf:
            response = np.zeros(response.shape)
        else:
            response = (response * (np.log(1 + l_n) / 0.03)) / l_n
        fts, n = compute_image3d_gray_statistic(response, seg,
                                                feature_flags, fl_name)
        features += [fts]
        names += n
    features = np.concatenate(tuple(features), axis=1)
    features = np.nan_to_num(features)
    # normalise +/- zeros as set all as positive
    features[features == 0] = 0
    names = ['tLM_%s' % name for name in names]
    assert features.shape[1] == len(names), \
        'features: %r and names %r' % (features.shape, names)
    return features, names


def compute_texture_desc_lm_img2d_clr(img, seg, feature_flags, bank_type='normal'):
    """ compute texture descriptors via Lewen-Malik filter response

    :param ndarray img: image
    :param ndarray seg: segmentation
    :param list(str) feature_flags:
    :param str bank_type: define used LM filter bank ['short', 'normal']
    :return tuple(np.ndarray<nb_samples, nb_features>, list(str)):

    .. seealso:: :func:`imsegm.descriptors.compute_texture_desc_lm_img3d_val`

    >>> h, w, step = 30, 20, 5
    >>> np.random.seed(0)
    >>> seg = np.zeros((h, w), dtype=int)
    >>> for i in range(int(np.ceil(h / float(step)))):
    ...     for j in range(int(np.ceil(w / float(step)))):
    ...         val = i * (w / step) + j
    ...         i_step, j_step = int(i * step), int(j * step)
    ...         seg[i_step:int(i_step + step), j_step:int(j_step + step)] = val
    >>> img = np.random.random((h, w, 3))
    >>> features, names = compute_texture_desc_lm_img2d_clr(img, seg,
    ...                          ['mean', 'std', 'median'], bank_type='short')
    >>> features.shape
    (24, 135)
    >>> names  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['tLM_sigma1.4-edge-ch1_mean', ..., 'tLM_sigma1.4-edge-ch3_mean',
     'tLM_sigma1.4-edge-ch1_std', ..., 'tLM_sigma1.4-edge-ch3_std',
     'tLM_sigma1.4-edge-ch1_median', ..., 'tLM_sigma1.4-edge-ch3_median',
     'tLM_sigma1.4-bar-ch1_mean', ..., 'tLM_sigma1.4-bar-ch3_median',
     'tLM_sigma1.4-Gauss-ch1_mean', ..., 'tLM_sigma1.4-Gauss-ch3_median',
     'tLM_sigma1.4-GaussLap-ch1_mean', ..., 'tLM_sigma1.4-GaussLap-ch3_median',
     'tLM_sigma1.4-GaussLap2-ch1_mean', ..., 'tLM_sigma1.4-GaussLap2-ch3_median',
     'tLM_sigma2.0-edge-ch1_mean', ..., 'tLM_sigma2.0-GaussLap2-ch3_median',
     'tLM_sigma4.0-edge-ch1_mean', ..., 'tLM_sigma4.0-GaussLap2-ch3_median']
    """
    _check_color_image(img)
    logging.debug('compute texture descriptors using Leung-Malik')
    img = (img - gaussian_filter(img.astype(float), 150))
    img_roll = np.rollaxis(img, -1, 0)
    if bank_type == 'short':
        filters, fl_names = create_filter_bank_lm_2d(sigmas=SHORT_FILTERS_SIGMAS,
                                                     nb_orient=4)
    else:
        filters, fl_names = create_filter_bank_lm_2d()
    features, names = [], []
    for fl_battery, fl_name in zip(filters, fl_names):
        response_roll = compute_img_filter_response3d(img_roll, fl_battery)
        # cut too large values
        response_roll[response_roll > MAX_SIGNAL_RESPONSE] = MAX_SIGNAL_RESPONSE
        # norm responses
        norm = np.sqrt(np.sum(response_roll ** 2))
        if norm == 0 or abs(norm) == np.inf:
            response_roll = np.zeros(response_roll.shape)
        else:
            response_roll = (response_roll * (np.log(1 + norm) / 0.03)) / norm
        response = np.rollaxis(response_roll, 0, 3)
        fts, ns = compute_image2d_color_statistic(response, seg,
                                                  feature_flags, fl_name)
        features += [fts]
        names += ns
    features = np.concatenate(tuple(features), axis=1)
    features = np.nan_to_num(features)
    # normalise +/- zeros as set all as positive
    features[features == 0] = 0
    names = ['tLM_%s' % name for name in names]
    assert features.shape[1] == len(names), \
        'features: %r and names %r' % (features.shape, names)
    return features, names


def compute_selected_features_gray3d(img, segments, feature_flags=FEATURES_SET_COLOR):
    """ compute selected features on gray 3D image

    :param ndarray img: image
    :param ndarray segments: segmentation
    :param dict(list(str)) feature_flags: dictionary of feature flags
    :return tuple(np.ndarray<nb_samples, nb_features>, list(str)):

    >>> np.random.seed(0)
    >>> img = np.random.random((2, 10, 15))
    >>> slic = np.zeros((2, 10, 15), dtype=int)
    >>> slic[:, :, :7] += 1
    >>> slic[1, :, :] += 2
    >>> fts, names = compute_selected_features_gray3d(
    ...     img, slic, {'color': ('mean', 'std', 'median')})
    >>> fts.shape
    (4, 3)
    >>> names  # doctest: +NORMALIZE_WHITESPACE
    ['gray_mean', 'gray_std', 'gray_median']
    >>> _ = compute_selected_features_gray3d(
    ...     img, slic, {'tLM': ('median', 'std', 'energy')})
    >>> fts, names = compute_selected_features_gray3d(
    ...     img, slic, {'tLM_short': ('mean', 'std', 'energy')})
    >>> fts.shape
    (4, 45)
    >>> names  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['tLM_sigma1.4-edge_mean', ..., 'tLM_sigma4.0-GaussLap2_energy']

    """
    _check_gray_image_segm(img, segments)
    assert feature_flags, 'some features has to be selected'

    features, names = [], []
    # COLOR FEATURES
    if any(k.startswith('color') for k in feature_flags):
        flags = np.unique([feature_flags[k]
                           for k in feature_flags if k.startswith('color')])
        fts, ns = compute_image3d_gray_statistic(img, segments, flags)
        features.append(fts)
        names += ns

    # TEXTURE - LEWEN-MALIK
    k_text = [k for k in feature_flags if k.startswith('tLM')]
    if k_text:
        for k in k_text:
            bank_type = k.split('_')[-1] if '_' in k else 'normal'
            fts, ns = compute_texture_desc_lm_img3d_val(img, segments, feature_flags[k],
                                                        bank_type)
            features.append(fts)
            names += ns
    _check_unrecognised_feature_group(feature_flags)

    if not features:
        logging.error('not supported features: %r', feature_flags)
    features = np.concatenate(tuple(features), axis=1)
    features = np.nan_to_num(features)
    # normalise +/- zeros as set all as positive
    features[features == 0] = 0
    assert features.shape[1] == len(names), \
        'features: %r and names %r' % (features.shape, names)
    return features, names


def compute_selected_features_gray2d(img, segments, features_flags=FEATURES_SET_ALL):
    """ compute selected features for gray image 2D

    :param ndarray img: image
    :param ndarray segments: segmentation
    :param dict(list(str)) feature_flags: dictionary of feature flags
    :return tuple(np.ndarray<nb_samples, nb_features>, list(str)):

    >>> image = np.zeros((2, 10))
    >>> image[0, 2:6] = 1
    >>> image[1, 3:7] = 3
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> features, names = compute_selected_features_gray2d(
    ...     image, segm, {'color': ('mean', 'std', 'median')})
    >>> np.round(features, 3)
    array([[ 0.9  ,  1.136,  0.5  ],
           [ 0.7  ,  1.187,  0.   ]])
    >>> _ = compute_selected_features_gray2d(
    ...     image, segm, {'tLM': ('mean', 'std', 'median')})
    >>> features, names = compute_selected_features_gray2d(
    ...     image, segm, {'tLM_short': ('mean', 'std', 'energy')})
    >>> features.shape
    (2, 45)
    >>> features, names = compute_selected_features_gray2d(image, segm)
    >>> features.shape
    (2, 105)
    """
    _check_gray_image_segm(img, segments)

    features, names = compute_selected_features_gray3d(img[np.newaxis, ...],
                                                       segments[np.newaxis, ...],
                                                       features_flags)
    assert features.shape[1] == len(names), \
        'features: %r and names %r' % (features.shape, names)
    return features, names


def compute_selected_features_color2d(img, segments, feature_flags=FEATURES_SET_ALL):
    """ compute selected features color image 2D

    :param ndarray img: image
    :param ndarray segments: segmentation
    :param dict(list(str)) feature_flags: dictionary of feature flags
    :return tuple(np.ndarray<nb_samples, nb_features>, list(str)):

    >>> image = np.zeros((2, 10, 3))
    >>> image[:, 2:6, 0] = 1
    >>> image[:, 3:7, 1] = 3
    >>> image[:, 4:9, 2] = 2
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> features, names = compute_selected_features_color2d(image, segm,
    ...                                   {'color': ('mean', 'std', 'median')})
    >>> np.round(features, 3)
    array([[ 0.6 ,  1.2 ,  0.4 ,  0.49,  1.47,  0.8 ,  1.  ,  0.  ,  0.  ],
           [ 0.2 ,  1.2 ,  1.6 ,  0.4 ,  1.47,  0.8 ,  0.  ,  0.  ,  2.  ]])
    >>> features, names = compute_selected_features_color2d(image, segm,
    ...                                   {'color_hsv': ('mean', 'std')})
    >>> np.round(features, 3)
    array([[ 0.139,  0.533,  1.4  ,  0.176,  0.452,  1.356],
           [ 0.439,  0.733,  2.   ,  0.244,  0.389,  1.095]])
    >>> _ = compute_selected_features_color2d(image, segm,
    ...                                   {'tLM': ('mean', 'std', 'energy')})
    >>> features, names = compute_selected_features_color2d(image, segm,
    ...                                   {'tLM_short': ('mean', 'energy')})
    >>> features.shape
    (2, 90)
    >>> features, names = compute_selected_features_color2d(image, segm)
    >>> features.shape
    (2, 315)
    """
    _check_color_image(img)
    features, names = [], []
    # COLOR SPACES
    k_color = [k for k in feature_flags if k.startswith('color')]
    if k_color:
        for k in k_color:
            clr = k.split('_')[-1] if '_' in k else 'rgb'
            img_color = convert_img_color_from_rgb(img, clr) if '_' in k else img
            fts, ns = compute_image2d_color_statistic(img_color, segments, feature_flags[k],
                                                      color_name=clr)
            features.append(fts)
            names += ns
    # TEXTURE - LEWEN-MALIK
    k_text = [k for k in feature_flags if k.startswith('tLM')]
    if k_text:
        for k in k_text:
            bank_type = k.split('_')[-1] if '_' in k else 'normal'
            fts, ns = compute_texture_desc_lm_img2d_clr(img, segments, feature_flags[k],
                                                        bank_type)
            features.append(fts)
            names += ns

    _check_unrecognised_feature_group(feature_flags)
    features = np.concatenate(tuple(features), axis=1)
    features = np.nan_to_num(features)
    # normalise +/- zeros as set all as positive
    features[features == 0] = 0
    if not features.size:
        logging.error('not supported features: %r', feature_flags)
    assert features.shape[1] == len(names), \
        'features: %r and names %r' % (features.shape, names)
    return features, names


def compute_selected_features_img2d(image, segm, features_flags=FEATURES_SET_COLOR):
    """ compute features

    :param ndarray img: image
    :param ndarray segments: segmentation
    :param dict(list(str)) feature_flags: dictionary of feature flags
    :return:
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return compute_selected_features_color2d(image, segm, features_flags)
    elif image.ndim == 2:
        return compute_selected_features_gray2d(image, segm, features_flags)
    else:
        logging.error('invalid image size - %r', image.shape)


def compute_label_histograms_positions(segm, positions,
                                       diameters=HIST_CIRCLE_DIAGONALS,
                                       nb_labels=None):
    """ compute the histogram features doe consecutive growing diameter
    of inter circle neighbouring around given points in the segmentation

    :param ndarray segm: np.array<height, width>
    :param [(int, int)] positions: list of positions
    :param list(int) diameters: circular diameters
    :param int nb_labels:
    :return tuple(ndarray,list(str)): ndarray<nb_samples, nb_features>, names

    >>> segm = np.zeros((10, 10), dtype=int)
    >>> segm[1:9, 2:8] = 1
    >>> segm[3:7, 4:6] = 2
    >>> points = [[3, 3], [4, 4], [2, 7], [6, 6]]
    >>> hists, names = compute_label_histograms_positions(segm, points, [1, 2, 4])
    >>> names  # doctest: +NORMALIZE_WHITESPACE
    ['hist-d_1-lb_0', 'hist-d_1-lb_1', 'hist-d_1-lb_2', \
     'hist-d_2-lb_0', 'hist-d_2-lb_1', 'hist-d_2-lb_2', \
     'hist-d_4-lb_0', 'hist-d_4-lb_1', 'hist-d_4-lb_2']
    >>> hists.shape
    (4, 9)
    >>> np.round(hists, 2)
    array([[ 0.  ,  0.8 ,  0.2 ,  0.12,  0.62,  0.25,  0.44,  0.41,  0.15],
           [ 0.  ,  0.2 ,  0.8 ,  0.  ,  0.62,  0.38,  0.22,  0.75,  0.03],
           [ 0.2 ,  0.8 ,  0.  ,  0.5 ,  0.5 ,  0.  ,  0.46,  0.33,  0.21],
           [ 0.  ,  0.8 ,  0.2 ,  0.12,  0.62,  0.25,  0.44,  0.41,  0.15]])
    >>> segm = np.zeros((10, 10, 2), dtype=int)
    >>> segm[3:7, 4:6, 1] = 1
    >>> segm[:, :, 0] = 1 - segm[:, :, 0]
    >>> points = [[3, 3], [4, 4], [2, 7], [6, 6]]
    >>> hists, names = compute_label_histograms_positions(segm, points, [1, 2, 4])
    >>> np.round(hists, 2)
    array([[ 1.  ,  0.2 ,  1.  ,  0.25,  1.  ,  0.15],
           [ 1.  ,  0.8 ,  1.  ,  0.38,  1.  ,  0.03],
           [ 1.  ,  0.  ,  1.  ,  0.  ,  1.  ,  0.21],
           [ 1.  ,  0.2 ,  1.  ,  0.25,  1.  ,  0.15]])
    """
    pos_dim = np.asarray(positions).shape[1]
    assert (segm.ndim - pos_dim) in (0, 1), \
        'dimension %r and %r difference should be 0 or 1' % (segm.ndim, pos_dim)

    if nb_labels is None:
        if segm.ndim == pos_dim:
            nb_labels = segm.max() + 1
        elif segm.ndim == (pos_dim + 1):
            nb_labels = segm.shape[-1]
        else:
            logging.error('estimate nb labels failed')

    logging.debug('prepare extended segm. and struc. elements')
    list_struct_elems = [morphology.disk(d) for d in diameters]

    pos_hists = list()
    logging.debug('compute circular histogram')
    # for each position compute features
    for pos in positions:
        hist_inter = list()
        hist_last = np.zeros(nb_labels)
        sel_size_last = np.zeros(1)
        for sel in list_struct_elems:
            # hist_new = segm_convol[diam, :, pos[1], pos[0]]
            if segm.ndim == len(pos):
                hist, sel_size = compute_label_hist_segm(segm, pos, sel, nb_labels)
            else:
                hist, sel_size = compute_label_hist_proba(segm, pos, sel)
            inter_size = sel_size - sel_size_last
            assert inter_size > 0, 'norm or element should be positive'
            assert np.all(hist >= hist_last), \
                'outer elem should have more labels %r then the inter %r' \
                % (hist.tolist(), hist_last.tolist())
            hist_inter += ((hist - hist_last) / float(inter_size)).tolist()
            hist_last = hist
            sel_size_last = sel_size
        pos_hists.append(hist_inter)

    feature_names = ['hist-d_%i-lb_%i' % (d, lb)
                     for d in diameters for lb in range(nb_labels)]
    pos_hists = np.array(pos_hists)
    assert pos_hists.shape[1] == len(feature_names), \
        'histogram: %r and names %r' % (pos_hists.shape, feature_names)
    return np.array(pos_hists), feature_names


def adjust_bounding_box_crop(image_size, bbox_size, position):
    """ adjust the bounding box according image sizes and position

    :param tuple(int,int)|[int, int] image_size: image size
    :param tuple(int,int)|[int, int] bbox_size: size of the bounding box
    :param tuple(int,int)|[int, int] position: position in yhe image
    :return (), (), (), (): im_begin, im_end, bb_begin, bb_end

    >>> adjust_bounding_box_crop((50, 50), (7, 7), (20, 20))
    ((17, 17), (24, 24), (0, 0), (7, 7))
    >>> adjust_bounding_box_crop((50, 50), (15, 15), (20, 45))
    ((13, 38), (28, 50), (0, 0), (15, 12))
    >>> adjust_bounding_box_crop((50, 50), (15, 15), (5, 5))
    ((0, 0), (13, 13), (2, 2), (15, 15))
    >>> adjust_bounding_box_crop((50, 50), (80, 80), (20, 20))
    ((0, 0), (50, 50), (20, 20), (70, 70))
    """
    assert len(image_size) == len(bbox_size), \
        'incompatible sizes %r != %r' % (image_size, bbox_size)
    im_size, pos = np.asarray(image_size), np.asarray(position)
    bb_size = np.asarray(bbox_size)

    im_begin = pos - np.floor(bb_size / 2.).astype(int)
    im_begin[im_begin < 0] = 0
    im_end = pos + np.ceil(bb_size / 2.).astype(int)
    im_end = [im_size[i] if end > im_size[i] else end
              for i, end in enumerate(im_end)]

    bb_begin, bb_end = np.zeros(len(im_size), dtype=int), bb_size
    for i, bb in enumerate(bb_size):
        if im_begin[i] == 0:
            bb_begin[i] = (np.floor(bb / 2.) - pos[i]).astype(int)
        if im_end[i] == im_size[i]:
            bb_end[i] = (np.floor(bb / 2.) + (im_size[i] - pos[i])).astype(int)

    assert np.array_equal((im_end - im_begin), (bb_end - bb_begin)), \
        'different sizes of image %r and bounding box %r mask' \
        % (im_end - im_begin, bb_end - bb_begin)
    return tuple(im_begin), tuple(im_end), tuple(bb_begin), tuple(bb_end)


def compute_label_hist_segm(segm, position, struc_elem, nb_labels):
    """ compute histogram of labels for set of centric annulus

    :param ndarray segm: np.array<height, width>
    :param tuple(float,float) position: position in the segmentation
    :param ndarray struc_elem: np.array<height, width>
    :param int nb_labels: total number of labels in the segmentation
    :return list(float):

    .. seealso:: :func:`imsegm.descriptors.cython_label_hist_seg2d`

    >>> segm = np.zeros((10, 10), dtype=int)
    >>> segm[1:9, 2:8] = 1
    >>> segm[3:7, 4:6] = 2
    >>> segm
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 2, 2, 1, 1, 0, 0],
           [0, 0, 1, 1, 2, 2, 1, 1, 0, 0],
           [0, 0, 1, 1, 2, 2, 1, 1, 0, 0],
           [0, 0, 1, 1, 2, 2, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> compute_label_hist_segm(segm, [6, 6], np.ones((3, 3)), 3)
    (array([ 0.,  7.,  2.]), 9.0)
    >>> compute_label_hist_segm(segm, [4, 4], np.ones((5, 5)), 3)
    (array([  0.,  17.,   8.]), 25.0)
    """
    assert segm.ndim == len(position), \
        'dim of position %r should match the segmentation %r dim' % (position, segm.shape)
    position = [int(p) for p in position]
    # take selection around point with size of struc. element
    im_begin, im_end, bb_begin, bb_end = \
        adjust_bounding_box_crop(segm.shape, struc_elem.shape, position)
    segm_select = segm[im_begin[0]:im_end[0], im_begin[1]:im_end[1]]
    struc_elem = struc_elem[bb_begin[0]:bb_end[0], bb_begin[1]:bb_end[1]]
    assert segm_select.shape == struc_elem.shape, \
        'segmentation %s and element %s should match' % (segm_select.shape, struc_elem.shape)
    if USE_CYTHON:
        hist = cython_label_hist_seg2d(segm_select, struc_elem, nb_labels)
    else:  # use standard python code
        hist = np.zeros(nb_labels)
        for lb in range(nb_labels):
            hist[lb] = np.sum(np.logical_and(segm_select == lb, struc_elem == 1))
    size = np.sum(struc_elem)
    return hist, size


def cython_label_hist_seg2d(segm_select, struc_elem, nb_labels):
    """ compute histogram of labels for set of centric annulus

    :param ndarray segm: np.array<height, width>
    :param tuple(float,float) position: position in the segmentation
    :param ndarray struc_elem: np.array<height, width>
    :param int nb_labels: total number of labels in the segmentation
    :return list(float):

    .. seealso:: :func:`imsegm.descriptors.compute_label_hist_segm`

    .. note:: output of this function should be equal to
    ```
    for lb in range(nb_labels):
        hist[lb] = np.sum(np.logical_and(segm_select == lb, struc_elem == 1))
    ```

    >>> segm = np.zeros((10, 10), dtype=int)
    >>> segm[1:9, 2:8] = 1
    >>> segm[3:7, 4:6] = 2
    >>> cython_label_hist_seg2d(segm[2:5, 4:7], np.ones((3, 3)), 3)
    array([ 0.,  5.,  4.])
    >>> cython_label_hist_seg2d(segm[1:6, 3:8], np.ones((5, 5)), 3)
    array([  0.,  19.,   6.])
    """
    assert np.array_equal(segm_select.shape, struc_elem.shape), \
        'segm. %r and mask %r sizes do not match' % (segm_select.shape, struc_elem.shape)
    # removing NaN which are converted as 0
    segm_select[np.isnan(segm_select)] = -1
    # assert nb_labels >= (np.nanmax(segm_select) + 1)
    hist = fts_cython.computeLabelHistogram2d(np.array(segm_select, dtype=np.int16),
                                              np.array(struc_elem, dtype=np.int16),
                                              int(nb_labels))
    return np.array(hist, dtype=float)


def compute_label_hist_proba(segm, position, struc_elem):
    """ compute histogram of labels for set of centric annulus
    expecting that each label has own layer

    :param ndarray segm: np.array<height, width>
    :param tuple(float,float) position:
    :param ndarray struc_elem: np.array<height, width>
    :return list(float):

    >>> seg = np.zeros((50, 50, 2), dtype=float)
    >>> seg[15:35, 20:40, 1] = 1
    >>> seg[:, :, 0] = 1 - seg[:, :, 1]
    >>> compute_label_hist_proba(seg, (15, 20), np.ones((12, 13), dtype=int))
    (array([ 114.,   42.]), 156)
    """
    assert segm.ndim == (len(position) + 1), \
        'segment. (%r) should have larger (+1) dim than position %i' \
        % (segm.shape, len(position))
    position = list(map(int, position))
    # take selection around point with size of struc. element
    im_begin, im_end, bb_begin, bb_end = adjust_bounding_box_crop(
        segm.shape[:struc_elem.ndim], struc_elem.shape, position)
    segm_select = segm[im_begin[0]:im_end[0], im_begin[1]:im_end[1], :]
    struc_elem = struc_elem[bb_begin[0]:bb_end[0], bb_begin[1]:bb_end[1]]
    assert segm_select.shape[:-1] == struc_elem.shape, \
        'initial dim of segmentation %r should match element %r' \
        % (segm_select.shape, struc_elem)
    tile_struc_elem = np.tile(struc_elem, (segm_select.shape[-1], 1, 1))
    segm_mask = np.rollaxis(segm_select, -1, 0) * tile_struc_elem
    hist = np.sum(segm_mask, axis=tuple(range(1, segm_mask.ndim)))
    size = np.sum(struc_elem)
    return hist, size


# def compute_conv_segm_hist(segm, diameters, nb_labels):
#     logging.debug('compute convolution images for %i labels and %s diams',
#                   nb_labels, repr(diameters))
#     segm_convol = np.empty((len(diameters), nb_labels)).tolist()
#     for lb in range(nb_labels):
#         seg_lb = (segm == lb)
#         for i, diam in enumerate(diameters):
#             sel = morphology.disk(diam)
#             segm_convol[i][lb] = signal.convolve2d(seg_lb, sel)
#     # Ttodo, roll axis,  subtract diameters,  normalise
#     logging.error('not finished yet')
#     return None


def compute_ray_features_segm_2d_vectors(seg_binary, position, angle_step=5.,
                                         smooth_coef=0, edge='up'):
    """ USES WHOLE IMAGE ROTATION SO IT IS VERY SLOW
    compute ray features vector , shift them to be startig from larges
    and smooth_coef them by gauss filter
    (from fiven point the close distance to boundary)

    :param str edge: pointing to the up of down edge o
    :param int smooth_coef:
    :param ndarray seg_binary: np.array<height, width>
    :param tuple(int,int) position:
    :param float angle_step:
    :return list(float):

    .. seealso:: :func:`imsegm.descriptors.compute_ray_features_segm_2d`

    .. note:: for more examples, see unittests

    >>> from skimage import draw
    >>> seg = np.ones((100, 100), dtype=bool)
    >>> x, y = draw.circle(45, 55, 30, shape=seg.shape)
    >>> seg[x, y] = False
    >>> compute_ray_features_segm_2d_vectors(seg, (50, 50), 45)
    array([35, 29, 25, 23, 24, 29, 34, 36])
    >>> compute_ray_features_segm_2d_vectors(seg, (60, 40), 30, smooth_coef=1)
    array([35, 27, 18, 12, 10,  9, 12, 18, 27, 37, 45, 49])
    >>> compute_ray_features_segm_2d_vectors(seg, (40, 60), 20).tolist()
    [25, 27, 29, 32, 34, 35, 37, 36, 36, 34, 32, 29, 27, 25, 24, 23, 24, 24]
    """
    seg_binary = seg_binary.astype(bool)
    if (90 % angle_step) == 0:
        angle_range = 90
    else:
        angle_range = 180
    nb_steps = int(angle_range / angle_step)
    ray_dist = np.array([-1] * int(nb_steps * 2 * (180 / angle_range)))

    # in case the position is inside the border lable
    label_position = seg_binary[int(position[0]), int(position[1])]
    if bool(label_position) and edge == 'up':
        return ray_dist * 0

    pos_center = np.array(seg_binary.shape) / 2
    shift = (pos_center - np.asarray(position))
    shift_abs = abs(shift).astype(int)

    size = np.array(seg_binary.shape)
    # add some extra spce that the segm is alwais complete for given shifting
    seg_ext = np.zeros((size + 2 * shift_abs).astype(int))
    seg_ext[shift_abs[0]:shift_abs[0] + size[0],
            shift_abs[1]:shift_abs[1] + size[1]] = seg_binary
    # sfift the segm to have the actial position in centre
    seg_shift = ndimage.shift(seg_ext, shift.tolist(), order=0, cval=True)

    for i, ang in enumerate(np.arange(0, angle_range, angle_step)):
        # rotate ndimage
        seg_rot = ndimage.rotate(seg_shift, ang + 90, order=0,
                                 reshape=True, cval=True)
        pos_new = (np.array(seg_rot.shape) / 2).astype(int)
        # extract vector projections
        vec_lines = [
            seg_rot[:pos_new[0], pos_new[1]].tolist()[::-1],
            seg_rot[pos_new[0], pos_new[1]:].tolist(),
            seg_rot[pos_new[0]:, pos_new[1]].tolist(),
            seg_rot[pos_new[0], :pos_new[1]].tolist()[::-1],
        ]
        # in case rotation by 180 skip the ortogonal axis
        if angle_range == 180:
            vec_lines = [vec_lines[0], vec_lines[2]]
        # compute distances
        for j, vec in enumerate(vec_lines):
            ray = i + (j * nb_steps)
            if True not in vec:
                continue
            if edge == 'up':
                ray_dist[ray] = vec.index(True)
            elif edge == 'down':
                dist_up = vec.index(True)
                if False in vec[dist_up:]:
                    dist_down = vec[dist_up:].index(False)
                    ray_dist[ray] = dist_up + dist_down

    if smooth_coef > 0:
        ray_dist = gaussian_filter1d(ray_dist, smooth_coef)

    return np.array(ray_dist)


def cython_ray_features_seg2d(seg_binary, position, angle_step=5., edge='up'):
    """ computing the Ray features from a segmentation and given position

    :param ndarray seg_binary: np.array<height, width>
    :param tuple(int,int) position: integer position in the segmentation
    :param float angle_step: angular step for ray features
    :param str edge: pointing to the up of down edge of an boundary
    :return list(float): ray distances

    .. seealso:: :func:`imsegm.descriptors.numpy_ray_features_seg2d`

    >>> seg_empty = np.zeros((100, 150), dtype=bool)
    >>> cython_ray_features_seg2d(seg_empty, (50, 75), 90)  # doctest: +ELLIPSIS
    array([-1., -1., -1., -1.]...)
    >>> from skimage import draw
    >>> seg = np.ones((100, 150), dtype=bool)
    >>> x, y = draw.circle(50, 75, 40, shape=seg.shape)
    >>> seg[x, y] = False
    >>> cython_ray_features_seg2d(seg, (50, 75), 45).astype(int)  # doctest: +ELLIPSIS
    array([40, 41, 40, 41, 40, 41, 40, 41]...)
    >>> cython_ray_features_seg2d(seg, (60, 40), 30).astype(int).tolist()
    [74, 55, 28, 10, 5, 4, 4, 5, 9, 30, 57, 75]
    >>> cython_ray_features_seg2d(seg, (40, 60), 20).astype(int).tolist()
    [54, 57, 58, 55, 50, 43, 38, 31, 26, 24, 22, 22, 23, 26, 29, 34, 41, 48]
    """
    edge_int = {'down': -1, 'up': 1}[edge]
    ray_dist = fts_cython.computeRayFeaturesBinary2d(np.array(seg_binary, dtype=np.int8),
                                                     np.array(position, dtype=np.int32),
                                                     float(angle_step), int(edge_int))
    return np.array(ray_dist)


def numpy_ray_features_seg2d(seg_binary, position, angle_step=5., edge='up'):
    """ computing the Ray features from a segmentation and given position

    :param ndarray seg_binary: np.array<height, width>
    :param tuple(int,int) position: integer position in the segmentation
    :param float angle_step: angular step for ray features
    :param str edge: pointing to the up of down edge of an boundary
    :return list(float): ray distances

    .. seealso:: :func:`imsegm.descriptors.cython_ray_features_seg2d`

    >>> seg_empty = np.zeros((100, 150), dtype=bool)
    >>> numpy_ray_features_seg2d(seg_empty, (50, 75), 90)  # doctest: +ELLIPSIS
    array([-1., -1., -1., -1.]...)
    >>> from skimage import draw
    >>> seg = np.ones((100, 150), dtype=bool)
    >>> x, y = draw.circle(50, 75, 40, shape=seg.shape)
    >>> seg[x, y] = False
    >>> numpy_ray_features_seg2d(seg, (50, 75), 45).astype(int)  # doctest: +ELLIPSIS
    array([40, 41, 40, 41, 40, 41, 40, 41]...)
    >>> numpy_ray_features_seg2d(seg, (60, 40), 30).astype(int).tolist()
    [74, 55, 28, 10, 5, 4, 4, 5, 9, 30, 57, 75]
    >>> numpy_ray_features_seg2d(seg, (40, 60), 20).astype(int).tolist()
    [54, 57, 58, 55, 50, 43, 38, 31, 26, 24, 22, 22, 23, 26, 29, 34, 41, 48]
    """
    angles = np.arange(0, 360, angle_step)
    ray_dist = np.array([-1.] * len(angles))

    # in case the position is inside the border label
    if bool(seg_binary[position[0], position[1]]) and edge == 'up':
        return ray_dist * 0
    width, height = seg_binary.shape[1], seg_binary.shape[0]
    segm_diag = int(np.sqrt(width ** 2 + height ** 2))

    for i, ang in enumerate(angles):
        pos = np.array(position, dtype=float)
        rad = np.deg2rad(ang)
        grad = np.array([np.sin(rad), np.cos(rad)])
        grad /= max(np.abs(grad))
        last = seg_binary[position[0], position[1]]
        for _ in range(segm_diag):
            pos += grad
            if pos[0] < 0 or round(pos[0]) >= height \
                    or pos[1] < 0 or round(pos[1]) >= width:
                break
            actual = seg_binary[int(round(pos[0])), int(round(pos[1]))]
            if (edge == 'up' and actual) or (edge == 'down' and last and not actual):
                diff = np.asarray(pos) - np.asarray(position)
                ray_dist[i] = np.sqrt(np.sum(diff ** 2))
                break
            last = actual
    return ray_dist


def compute_ray_features_segm_2d(seg_binary, position, angle_step=5.,
                                 smooth_coef=0, edge='up'):
    """ compute ray features vector , shift them to be starting from larges
    and smooth_coef them by gauss filter
    (from given point the close distance to boundary)

    :param ndarray seg_binary: np.array<height, width>
    :param tuple(int,int) position: integer position in the segmentation
    :param float angle_step: angular step for ray features
    :param str edge: pointing to the up of down edge of an boundary
    :param int smooth_coef: smoothing the final ray features
    :return list(float): ray distances

    .. seealso:: :func:`imsegm.descriptors.compute_ray_features_segm_2d_vectors`

    .. note:: for more examples, see unittests

    >>> seg_empty = np.zeros((100, 150), dtype=bool)
    >>> compute_ray_features_segm_2d(seg_empty, (50, 75), 90)  # doctest: +ELLIPSIS
    array([-1., -1., -1., -1.]...)
    >>> from skimage import draw
    >>> seg = np.ones((100, 150), dtype=bool)
    >>> x, y = draw.circle(50, 75, 40, shape=seg.shape)
    >>> seg[x, y] = False
    >>> np.round(compute_ray_features_segm_2d(seg, (50, 75), 45))  # doctest: +ELLIPSIS
    array([ 40.,  41.,  40.,  41.,  40.,  41.,  40.,  41.]...)
    >>> np.round(compute_ray_features_segm_2d(seg, (60, 40), 30, smooth_coef=1)).tolist()
    [66.0, 52.0, 32.0, 16.0, 8.0, 5.0, 5.0, 8.0, 16.0, 33.0, 53.0, 67.0]
    >>> ray_fts = compute_ray_features_segm_2d(seg, (40, 60), 20)
    >>> np.round(ray_fts).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [54.0, 57.0, 59.0, 55.0, 51.0, 44.0, 38.0, 31.0, 27.0, 24.0, 22.0, 22.0,
     23.0, 26.0, 29.0, 35.0, 42.0, 49.0]
    """
    assert seg_binary.ndim == len(position), \
        'Segmentation dim of %r and position (%i) does not match' \
        % (seg_binary.ndim, len(position))
    seg_binary = seg_binary.astype(bool)
    position = tuple(map(int, position))

    fn_compute = cython_ray_features_seg2d if USE_CYTHON else numpy_ray_features_seg2d
    ray_dist = fn_compute(seg_binary, position, angle_step, edge)

    if smooth_coef is not None and smooth_coef > 0:
        ray_dist = gaussian_filter1d(ray_dist, smooth_coef)

    return ray_dist


def shift_ray_features(ray_dist, method='phase'):
    """ shift Ray features ti the global maxim to be rotation invariant

    :param list(float) ray_dist: array of features
    :param str method: use method for estimate shift maxima (phase or max)
    :return list(float):

    >>> vec = np.array([43, 46, 44, 39, 28, 18, 12, 10,  9, 12, 22, 28])
    >>> ray, shift = shift_ray_features(vec)
    >>> shift   # doctest: +ELLIPSIS
    41.50...
    >>> ray
    array([46, 44, 39, 28, 18, 12, 10,  9, 12, 22, 28, 43])
    >>> ray2, shift = shift_ray_features(ray)
    >>> shift  # doctest: +ELLIPSIS
    11.50...
    >>> np.array_equal(ray, ray2)
    True
    >>> ray, shift = shift_ray_features(vec, method='max')
    >>> shift   # doctest: +ELLIPSIS
    30.0...
    """
    angle_step = 360 / len(ray_dist)
    # https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example
    # https://www.gaussianwaves.com/2015/11/interpreting-fft-results-obtaining-magnitude-and-phase-information
    if method == 'phase':
        # use major phase from FFT, see following
        ray_dist_ext = np.hstack([ray_dist] * 5)
        spectrum = np.fft.fft(ray_dist_ext - np.mean(ray_dist_ext)) / float(
            len(ray_dist_ext))
        # freq = np.fft.fftfreq(len(ray_dist_ext), angle_step)
        magnitude = np.abs(spectrum)[:len(ray_dist_ext) // 2]
        idx_max_mag = np.argmax(magnitude)
        phase = np.angle(spectrum)[:len(ray_dist_ext) // 2]
        shift = np.rad2deg(- phase[idx_max_mag])
        shift = (360 + shift) if shift < 0 else shift
    else:
        max_loc = np.argmax(ray_dist)
        shift = float(max_loc * angle_step)
    # round the shift to dicreate angular steps
    shift_discrete = int(round(shift / angle_step))
    ray_dist_shift = ray_dist[shift_discrete:].tolist() + ray_dist[:shift_discrete].tolist()
    return np.array(ray_dist_shift), shift


def compute_ray_features_positions(segm, list_positions, angle_step=5.,
                                   border_labels=None, segm_open=None,
                                   smooth_ray=None, shifting=True, edge='up'):
    """ compute ray features fo multiple points in the segmentation
    with given boundary labels and step angle

    :param ndarray segm: np.array<height, width>
    :param [(int, int)] list_positions:
    :param float angle_step:
    :param list(int) border_labels: all labels to be set as boundaries
    :param int segm_open:
    :param float smooth_ray:
    :param bool shifting:
    :param str edge: type of edge up/down
    :return:

    .. note:: for more examples, see unittests

    >>> from skimage import draw
    >>> np.random.seed(0)
    >>> seg = np.zeros((100, 100), dtype=int)
    >>> x, y = draw.circle(45, 55, 30, shape=seg.shape)
    >>> seg[x, y] = 1
    >>> x, y = draw.circle(55, 45, 10, shape=seg.shape)
    >>> seg[x, y] = 2
    >>> points = [(50, 50), (60, 40), (44, 55)]
    >>> ray_dist, shift, _ = compute_ray_features_positions(seg, points, 20)
    >>> shift  # doctest: +ELLIPSIS
    [314.3..., 314.7..., 90.0...]
    >>> ray_dist.astype(int).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [[37, 37, 35, 32, 30, 27, 25, 24, 23, 23, 24, 25, 26, 30, 31, 33, 35, 38],
     [50, 47, 41, 31, 23, 17, 13, 10, 9, 9, 9, 11, 14, 19, 27, 37, 45, 50],
     [31, 31, 31, 30, 30, 29, 30, 30, 29, 29, 30, 30, 29, 30, 30, 31, 31, 31]]
    >>> noise_pos = np.random.randint(10, 80, (2, 300))
    >>> seg[noise_pos[0], noise_pos[1]] = 0  # add random noise
    >>> ray_dist, shift, names = compute_ray_features_positions(seg, points, 45,
    ...                                                         segm_open=10)
    >>> names  # doctest: +NORMALIZE_WHITESPACE
    ['ray-lb_0-agl_0', 'ray-lb_0-agl_45', 'ray-lb_0-agl_90',
     'ray-lb_0-agl_135', 'ray-lb_0-agl_180', 'ray-lb_0-agl_225',
     'ray-lb_0-agl_270', 'ray-lb_0-agl_315']
    >>> shift  # doctest: +ELLIPSIS
    [315.0..., 315.0..., 90.0...]
    >>> ray_dist.astype(int)
    array([[38, 35, 29, 25, 24, 25, 29, 35],
           [52, 41, 21, 11,  9, 11, 21, 41],
           [31, 31, 30, 29, 29, 29, 30, 31]])
    """
    logging.debug('compute Ray features with border label=%r and angle step=%f',
                  border_labels, angle_step)
    pos_dim = np.asarray(list_positions).shape[1]
    assert (segm.ndim - pos_dim) in (0, 1), \
        'dimension %s and %s difference should be 0 or 1' % (segm.ndim, pos_dim)
    border_labels = border_labels if border_labels is not None else [0]
    if segm.ndim > pos_dim:
        # set label segment from probab
        segm = np.argmax(segm, axis=-1)

    seg_binary = np.zeros(segm.shape, dtype=bool)
    for lb in border_labels:
        seg_binary[segm == lb] = True

    # filter binary image
    if isinstance(segm_open, int):
        seg_binary = morphology.opening(seg_binary, morphology.disk(segm_open))

    pos_rays, pos_shift, ray_dist = [], [], []
    for pos in list_positions:
        # logging.debug('position %s', repr(pos))
        ray_dist = compute_ray_features_segm_2d(seg_binary, pos, angle_step,
                                                smooth_ray, edge)
        if shifting:
            ray_dist, shift = shift_ray_features(ray_dist)
        else:
            shift = 0
        pos_rays.append(ray_dist)
        pos_shift.append(float(shift))

    feature_names = ['ray-lb_%s-agl_%i' % (''.join(map(str, border_labels)), int(a))
                     for a in np.linspace(0, 360 - angle_step, len(ray_dist))]
    pos_rays = np.array(pos_rays)
    assert pos_rays.shape[1] == len(feature_names), \
        'Ray features: %r and names %r' % (pos_rays.shape, feature_names)
    return pos_rays, pos_shift, feature_names


def interpolate_ray_dist(ray_dists, order='spline'):
    """ interpolate ray distances

    :param list(float) ray_dists:
    :param str|int order: degree of interpolation
    :return list(float):

    >>> interpolate_ray_dist([-1] * 5)
    array([-1, -1, -1, -1, -1])
    >>> vals = np.sin(np.linspace(0, 2 * np.pi, 20)) * 10
    >>> np.round(vals).astype(int).tolist()
    [0, 3, 6, 8, 10, 10, 9, 7, 5, 2, -2, -5, -7, -9, -10, -10, -8, -6, -3, 0]
    >>> vals[3:7] = -1
    >>> vals[16:] = -1
    >>> vals_interp = interpolate_ray_dist(vals, order=3)
    >>> np.round(vals_interp).astype(int).tolist()
    [0, 3, 6, 9, 10, 10, 8, 7, 5, 2, -2, -5, -7, -9, -10, -10, -10, -8, -4, 1]
    >>> vals_interp = interpolate_ray_dist(vals, order='spline')
    >>> np.round(vals_interp).astype(int).tolist()
    [0, 3, 6, 8, 9, 10, 9, 7, 5, 2, -2, -5, -7, -9, -10, -10, -9, -7, -5, -3]
    >>> vals_interp = interpolate_ray_dist(vals, order='cos')
    >>> np.round(vals_interp).astype(int).tolist()
    [0, 3, 6, 8, 10, 10, 9, 7, 5, 2, -2, -5, -7, -9, -10, -10, -8, -6, -3, 0]
    """
    x_space = np.arange(len(ray_dists))
    ray_dists = np.array(ray_dists)
    missing = ray_dists == -1
    x_train = x_space[ray_dists != -1]
    y_train = ray_dists[ray_dists != -1]
    if not np.asarray(y_train).size:
        return ray_dists
    # set 3x range from -N to 2N
    x_train_ext = np.hstack((x_train - len(x_space),
                             x_train,
                             x_train + len(x_space)))
    y_train_ext = np.array(y_train.tolist() * 3)

    if isinstance(order, int):
        # model = pipeline.make_pipeline(
        #     preprocessing.PolynomialFeatures(order), linear_model.Ridge())
        # model.fit(x_space[ray_dists != -1], ray_dists[ray_dists != -1])
        # ray_dists[ray_dists == -1] = model.predict(x_space[ray_dists == -1])
        z = np.polyfit(x_train, y_train, order)
        fn_interp = np.poly1d(z)
        ray_dists[missing] = fn_interp(x_space[missing])
    elif order == 'spline':
        uinterp_us = interpolate.InterpolatedUnivariateSpline(x_train_ext,
                                                              y_train_ext)
        ray_dists[missing] = uinterp_us(x_space[missing])
    elif order == 'cos':
        def _fn_cos(x, t):
            return x[0] + x[1] * np.sin(x[2] + x[3] * t)

        def _fn_cos_residual(x, t, y):
            return _fn_cos(x, t) - y

        x0 = np.array([np.mean(y_train), (y_train.max() - y_train.min()) / 2.,
                       0, len(x_space) / np.pi])
        lsm_res = optimize.least_squares(_fn_cos_residual, x0, gtol=1e-1,
                                         # loss='soft_l1', f_scale=0.1,
                                         args=(x_train, y_train))
        ray_dists[missing] = _fn_cos(lsm_res.x, x_space[missing])

    return ray_dists


def reconstruct_ray_features_2d(position, ray_features, shift=0):
    """ reconstruct ray features for 2D image

    :param tuple(int,int)|tuple(float,float) position:
    :param list(float) ray_features:
    :param float shift:
    :return [[float, float]]:

    .. note:: for more examples, see unittests

    >>> reconstruct_ray_features_2d((10., 10), np.array([1] * 4))
    array([[ 10.,  11.],
           [ 11.,  10.],
           [ 10.,   9.],
           [  9.,  10.]])
    >>> reconstruct_ray_features_2d((10., 10), np.array([-1, 0, 1, np.inf]))
    array([[ 10.,  10.],
           [ 10.,   9.]])
    """
    assert len(position) == 2, 'positions has to have 2 coordinates'
    assert len(ray_features) > 2, 'required at least 2 features'

    angles = np.linspace(0, 2 * np.pi, len(ray_features), endpoint=False)
    angles = (np.pi / 2.) - angles - np.deg2rad(shift)

    mask = np.logical_and(np.array(ray_features) >= 0,
                          ~ np.isinf(ray_features))
    angles = angles[mask]
    ray_features = ray_features[mask]
    dx = np.cos(angles) * ray_features
    dy = np.sin(angles) * ray_features

    positions = np.tile(position, (len(ray_features), 1))
    points = positions + np.array([dx, dy]).T
    # points = points[mask, :]

    return points


def reduce_close_points(points, dist_thr):
    """ reduce remove points with smaller internal distance then treshold
    assumption, the points are in sequence geometrically ordered)

    :param [[float, float]] points:
    :param float dist_thr: distance threshold
    :return [[float, float]]:

    >>> points = np.array([range(10), range(10)]).T
    >>> reduce_close_points(points, 2)
    array([[0, 0],
           [2, 2],
           [4, 4],
           [6, 6],
           [8, 8]])
    >>> points = np.array([[0, 0], [1, 1], [0, 2]])
    >>> reduce_close_points(points, 2)
    array([[0, 0],
           [0, 2]])
    >>> reduce_close_points(np.ones((10, 2)), 2)
    array([[ 1.,  1.]])
    """
    assert len(points) > 2, 'too few point to be reduced'

    dist = spatial.distance.cdist(points, points, metric='euclidean')
    for i in range(len(points)):
        dist[i, i] = np.Inf

    while np.min(dist) < dist_thr and len(points) > 0:
        coord = np.unravel_index(dist.argmin(), dist.shape)
        max_coord = max(coord)
        points = np.delete(points, max_coord, axis=0)
        dist = np.delete(dist, max_coord, axis=0)
        dist = np.delete(dist, max_coord, axis=1)

    return points

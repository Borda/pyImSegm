"""
Framework for feature extraction
 * color and gray 3D images
 * color and texture features
 * Ray features
 * label histogram

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging

import numpy as np
from scipy import ndimage, interpolate, optimize, spatial
from scipy.ndimage.filters import (gaussian_filter, gaussian_filter1d,
                                   gaussian_laplace)
from sklearn import preprocessing
from skimage import morphology
# from numba.decorators import jit
# from numba import int32, int64, float32

try:
    import imsegm.features_cython as fts_cython
    # logging.debug('try to load Cython implementation')  # CRASH logger
    USE_CYTHON = True
except Exception:
    # NOTE: in some cases following warning may crash all message logging
    logging.warning('descriptors: using pure python libraries')
    USE_CYTHON = False

DEFAULT_FILTERS_SIGMAS = (np.sqrt(2), 2, 2 * np.sqrt(2), 4)
SHORT_FILTERS_SIGMAS = (np.sqrt(2), 2, 4)
FEATURES_SET_ALL = {'color': ('mean', 'std', 'eng', 'median'),
                    'tLM': ('mean', 'std', 'eng', 'mG')}
FEATURES_SET_COLOR = {'color': ('mean', 'std', 'eng')}
FEATURES_SET_TEXTURE = {'tLM': ('mean', 'std', 'eng')}
FEATURES_SET_TEXTURE_SHORT = {'tLM_s': ('mean', 'std', 'eng')}
HIST_CIRCLE_DIAGONALS = (10, 20, 30, 40, 50)

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
#     logging.info('Computing RGB means for {} segments'.format(uniqueLbs.max()))
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
    if image.shape[:2] != segm.shape:
        raise ValueError('arrays - image and segm do not match %s vs %s'
                         % (repr(image.shape), repr(segm.shape)))
    return True


def _check_gray_image_segm(image, segm):
    if image.shape != segm.shape:
        raise ValueError('arrays - image and segm do not match %s vs %s'
                         % (repr(image.shape), repr(segm.shape)))
    return True


def _check_color_image(image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image is not RGB with dims %s' % repr(image.shape))
    return True


def cython_img2d_color_mean(im, seg):
    """ wrapper for fast implementation of colour features

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

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
    logging.debug('Cython: computing Colour means for image %s & segm %s with '
                  '%i segments', repr(im.shape), repr(seg.shape), np.max(seg))
    _check_color_image_segm(im, seg)

    means = fts_cython.computeColorImage2dMean(np.array(im, dtype=np.float32),
                                               np.array(seg, dtype=np.int32))
    return np.array(means)


def cython_img2d_color_energy(im, seg):
    """  wrapper for fast implementation of colour features

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

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
    logging.debug('Cython: computing Colour energy for image %s & segm %s with'
                  ' %i segments', repr(im.shape), repr(seg.shape), np.max(seg))
    _check_color_image_segm(im, seg)

    energy = fts_cython.computeColorImage2dEnergy(np.array(im, dtype=np.float32),
                                                  np.array(seg, dtype=np.int32))
    return np.array(energy)


def cython_img2d_color_std(im, seg, means=None):
    """ wrapper for fast implementation of colour features

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

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
    logging.debug('Cython: computing Colour STD for image %s & segm %s with '
                  '%i segments', repr(im.shape), repr(seg.shape), np.max(seg))
    _check_color_image_segm(im, seg)

    if means is None:
        means = cython_img2d_color_mean(im, seg)
    var = fts_cython.computeColorImage2dVariance(np.array(im, dtype=np.float32),
                                                 np.array(seg, dtype=np.int32),
                                                 np.array(means, dtype=np.float32))
    std = np.sqrt(var)
    return std


def numpy_img2d_color_mean(im, seg):
    """ compute color means by numpy

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

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
    logging.debug('computing Colour mean for image %s & segm %s with '
                  '%i segments', repr(im.shape), repr(seg.shape), np.max(seg))
    _check_color_image_segm(im, seg)

    nb_labels = np.max(seg) + 1
    means = np.zeros((nb_labels, 3))
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            lb = seg[i, j]
            means[lb, 0] += im[i, j, 0]
            means[lb, 1] += im[i, j, 1]
            means[lb, 2] += im[i, j, 2]
            counts[lb] += 1
    means = (means / np.tile(counts, (3, 1)).T.astype(float))
    return means


def numpy_img2d_color_std(im, seg, means=None):
    """ compute color STD by numpy

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

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
    logging.debug('computing Colour STD for image %s & segm %s with '
                  '%i segments', repr(im.shape), repr(seg.shape), np.max(seg))
    _check_color_image_segm(im, seg)

    if means is None:
        means = numpy_img2d_color_mean(im, seg)

    nb_labels = np.max(seg) + 1
    assert len(means) >= nb_labels
    variations = np.zeros((nb_labels, 3))
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            lb = seg[i, j]
            variations[lb, :] += (im[i, j, :] - means[lb, :]) ** 2
            counts[lb] += 1
    variations = (variations / np.tile(counts, (3, 1)).T.astype(float))
    stds = np.sqrt(variations)
    return stds


def numpy_img2d_color_energy(im, seg):
    """ compute color energy by numpy

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

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
    logging.debug('computing Colour energy for image %s & segm %s with '
                  '%i segments', repr(im.shape), repr(seg.shape), np.max(seg))
    _check_color_image_segm(im, seg)

    nb_labels = np.max(seg) + 1
    energy = np.zeros((nb_labels, 3))
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            lb = seg[i, j]
            energy[lb, 0] += im[i, j, 0] ** 2
            energy[lb, 1] += im[i, j, 1] ** 2
            energy[lb, 2] += im[i, j, 2] ** 2
            counts[lb] += 1
    energy = (energy / np.tile(counts, (3, 1)).T.astype(float))
    return energy


def numpy_img2d_color_median(im, seg):
    """ compute color median by numpy

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

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
    logging.debug('computing Colour median for image %s & segm %s with '
                  '%i segments', repr(im.shape), repr(seg.shape), np.max(seg))
    _check_color_image_segm(im, seg)

    nb_labels = np.max(seg) + 1
    list_values = [([], [], []) for _ in range(nb_labels)]

    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            lb = seg[i, j]
            list_values[lb][0].append(im[i, j, 0])
            list_values[lb][1].append(im[i, j, 1])
            list_values[lb][2].append(im[i, j, 2])

    medians = np.zeros((nb_labels, 3))
    for i in range(nb_labels):
        medians[i, 0] = np.median(list_values[i][0])
        medians[i, 1] = np.median(list_values[i][1])
        medians[i, 2] = np.median(list_values[i][2])
    return medians


def cython_img3d_gray_mean(im, seg):
    """ wrapper for fast implementation of colour features

    WARNING: the Z dimension is parallel and without sync,
    multiple equal labels across Z dim may lead to not mistakes in summing

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 1> vector of mean colour per segment

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
    logging.debug('Cython: computing Gray means for image %s and segm %s with '
                  '%i segments', repr(im.shape), repr(seg.shape), np.max(seg))
    _check_gray_image_segm(im, seg)

    means = fts_cython.computeGrayImage3dMean(np.array(im, dtype=np.float32),
                                              np.array(seg, dtype=np.int32))
    return np.array(means)


def cython_img3d_gray_energy(im, seg):
    """ wrapper for fast implementation of colour features

    WARNING: the Z dimension is parallel and without sync,
    multiple equal labels across Z dim may lead to not mistakes in summing

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return:np.array<nb_lbs, 1> vector of mean colour per segment

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> cython_img3d_gray_energy(image, segm)
    array([ 0.5 ,  0.5 ,  2.25,  6.75])
    """
    logging.debug('Cython: computing Gray energy for image %s and segm %s with'
                  ' %i segments', repr(im.shape), repr(seg.shape), np.max(seg))
    _check_gray_image_segm(im, seg)

    energy = fts_cython.computeGrayImage3dEnergy(np.array(im, dtype=np.float32),
                                                 np.array(seg, dtype=np.int32))
    return np.array(energy)


def cython_img3d_gray_std(im, seg, mean=None):
    """ wrapper for fast implementation of colour features

    WARNING: the Z dimension is parallel and without sync,
    multiple equal labels across Z dim may lead to not mistakes in summing

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return:np.array<nb_lbs, 1> vector of mean colour per segment

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> cython_img3d_gray_std(image, segm)
    array([ 0.5       ,  0.5       ,  1.29903811,  1.29903811])
    """
    logging.debug('Cython: computing Gray STD for image %s and segm %s with '
                  '%i segments', repr(im.shape), repr(seg.shape), np.max(seg))
    _check_gray_image_segm(im, seg)

    if mean is None:
        mean = cython_img3d_gray_mean(im, seg)
    var = fts_cython.computeGrayImage3dVariance(np.array(im, dtype=np.float32),
                                                np.array(seg, dtype=np.int32),
                                                np.array(mean, dtype=np.float32))
    std = np.sqrt(var)
    return std


def numpy_img3d_gray_mean(im, seg):
    """ compute gray (3D) means by numpy

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> numpy_img3d_gray_mean(image, segm)
    array([ 0.5 ,  0.5 ,  0.75,  2.25])
    """
    logging.debug('computing Gray mean for %i segments', np.max(seg))
    _check_gray_image_segm(im, seg)

    nb_labels = np.max(seg) + 1
    means = np.zeros(nb_labels)
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                lb = seg[i, j, k]
                means[lb] += im[i, j, k]
                counts[lb] += 1
    means = (means / counts.astype(float))
    return means


def numpy_img3d_gray_std(im, seg, means=None):
    """ compute gray (3D) STD by numpy

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> numpy_img3d_gray_std(image, segm)
    array([ 0.5       ,  0.5       ,  1.29903811,  1.29903811])
    """
    logging.debug('computing Gray mean for %i segments', np.max(seg))
    _check_gray_image_segm(im, seg)

    if means is None:
        means = numpy_img3d_gray_mean(im, seg)

    nb_labels = np.max(seg) + 1
    assert len(means) >= nb_labels
    variances = np.zeros(nb_labels)
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                lb = seg[i, j, k]
                variances[lb] += (im[i, j, k] - means[lb]) ** 2
                counts[lb] += 1
    variances = (variances / counts.astype(float))
    stds = np.sqrt(variances)
    return stds


def numpy_img3d_gray_energy(im, seg):
    """ compute gray (3D) energy by numpy

    :param im: input RGB image
    :param seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> numpy_img3d_gray_energy(image, segm)
    array([ 0.5 ,  0.5 ,  2.25,  6.75])
    """
    logging.debug('computing Gray energy for %i segments', np.max(seg))
    _check_gray_image_segm(im, seg)

    nb_labels = np.max(seg) + 1
    energy = np.zeros(nb_labels)
    counts = np.zeros(nb_labels)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                lb = seg[i, j, k]
                energy[lb] += im[i, j, k] ** 2
                counts[lb] += 1
    energy = (energy / counts.astype(float))
    return energy


def numpy_img3d_gray_median(im, seg):
    """ compute gray (3D) median by numpy

    :param ndarray im: input RGB image
    :param ndarray seg: segmentation og the image
    :return: np.array<nb_lbs, 3> matrix features per segment

    >>> image = np.zeros((2, 3, 8))
    >>> image[0, :, 2:6] = 1
    >>> image[1, :, 3:7] = 3
    >>> segm = np.array([[[0, 0, 0, 0, 1, 1, 1, 1]] * 3,
    ...                  [[2, 2, 2, 2, 3, 3, 3, 3]] * 3])
    >>> numpy_img3d_gray_median(image, segm)
    array([ 0.5,  0.5,  0. ,  3. ])
    """
    logging.debug('computing Gray median for %i segments', np.max(seg))
    _check_gray_image_segm(im, seg)

    nb_labels = np.max(seg) + 1
    list_values = [[] for _ in range(nb_labels)]

    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                list_values[seg[i, j, k]].append(im[i, j, k])

    medians = np.zeros(nb_labels)
    for i in range(nb_labels):
        medians[i] = np.median(list_values[i])
    return medians


def compute_image3d_gray_statistic(image, segm,
                                   list_feature_flags=('mean', 'std', 'eng',
                                                       'median', 'mG'),
                                   ch_name='gray'):
    """ compute complete descriptors / statistic on gray (3D) images

    :param ndarray image:
    :param ndarray segm:
    :param list_feature_flags:
    :return np.ndarray<nb_samples, nb_features>, [str]:

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

    assert len(list_feature_flags) > 0
    image = np.nan_to_num(image)
    features, names = [], []
    # nb_fts = image.shape[0]
    # ch_names = ['%s-ch%i' % (ch_name, i + 1) for i in range(nb_fts)]

    # MEAN
    mean = None
    if 'mean' in list_feature_flags:
        if USE_CYTHON:
            mean = cython_img3d_gray_mean(image, segm)
        else:
            mean = numpy_img3d_gray_mean(image, segm)
        features.append(mean)
        names += ['%s_mean' % ch_name]
    # Standard Deviation
    if 'std' in list_feature_flags:
        if USE_CYTHON:
            std = cython_img3d_gray_std(image, segm, mean)
        else:
            std = numpy_img3d_gray_std(image, segm, mean)
        features.append(std)
        names += ['%s_std' % ch_name]
    # ENERGY
    if 'eng' in list_feature_flags:
        if USE_CYTHON:
            energy = cython_img3d_gray_energy(image, segm)
        else:
            energy = numpy_img3d_gray_energy(image, segm)
        features.append(energy)
        names += ['%s_energy' % ch_name]
    # MEDIAN
    if 'median' in list_feature_flags:
        median = numpy_img3d_gray_median(image, segm)
        features.append(median)
        names += ['%s_median' % ch_name]
    # mean Gradient
    if 'mG' in list_feature_flags:
        grad_matrix = np.zeros_like(image)
        for i in range(image.shape[0]):
            grad_matrix[i, :, :] = np.sum(np.gradient(image[i]), axis=0)
        if USE_CYTHON:
            grad = cython_img3d_gray_mean(grad_matrix, segm)
        else:
            grad = numpy_img3d_gray_mean(grad_matrix, segm)
        features.append(grad)
        names += ['%s_meanGrad' % ch_name]
    features = np.concatenate(tuple([fts] for fts in features), axis=0)
    features = np.nan_to_num(features).T
    assert features.shape[1] == len(names), \
        'features: %s and names %s' % (features.shape, repr(names))
    return features, names


def compute_image2d_color_statistic(image, segm,
                                    list_feature_flags=('mean', 'std', 'eng',
                                                        'median'),
                                    ch_name='color'):
    """ compute complete descriptors / statistic on color (2D) images

    :param ndarray image:
    :param ndarray segm:
    :param list_feature_flags:
    :return np.ndarray<nb_samples, nb_features>, [str]:

    >>> image = np.zeros((2, 10, 3))
    >>> image[:, 2:6, 0] = 1
    >>> image[:, 3:7, 1] = 3
    >>> image[:, 4:9, 2] = 2
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> features, names = compute_image2d_color_statistic(image, segm,
    ...                                       ['mean', 'std', 'eng', 'median'])
    >>> names  # doctest: +NORMALIZE_WHITESPACE
    ['color-ch1_mean', 'color-ch2_mean', 'color-ch3_mean',
     'color-ch1_std', 'color-ch2_std', 'color-ch3_std',
     'color-ch1_energy', 'color-ch2_energy', 'color-ch3_energy',
     'color-ch1_median', 'color-ch2_median', 'color-ch3_median']
    >>> features.shape
    (2, 12)
    >>> np.round(features, 1).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [[0.6, 1.2, 0.4, 0.5, 1.5, 0.8, 0.6, 3.6, 0.8, 1.0, 0.0, 0.0],
     [0.2, 1.2, 1.6, 0.4, 1.5, 0.8, 0.2, 3.6, 3.2, 0.0, 0.0, 2.0]]
    """
    _check_color_image(image)
    _check_color_image_segm(image, segm)

    image = np.nan_to_num(image)
    features = np.empty((np.max(segm) + 1, 0))
    names = []
    ch_names = ['%s-ch%i' % (ch_name, i + 1) for i in range(3)]

    # MEAN
    mean = None
    if 'mean' in list_feature_flags:
        if USE_CYTHON:
            mean = cython_img2d_color_mean(image, segm)
        else:
            mean = numpy_img2d_color_mean(image, segm)
        features = np.hstack((features, mean))
        names += ['%s_mean' % n for n in ch_names]
    # Standard Deviation
    if 'std' in list_feature_flags:
        if USE_CYTHON:
            std = cython_img2d_color_std(image, segm, mean)
        else:
            std = numpy_img2d_color_std(image, segm, mean)
        features = np.hstack((features, std))
        names += ['%s_std' % n for n in ch_names]
    # ENERGY
    if 'eng' in list_feature_flags:
        if USE_CYTHON:
            energy = cython_img2d_color_energy(image, segm)
        else:
            energy = numpy_img2d_color_energy(image, segm)
        features = np.hstack((features, energy))
        names += ['%s_energy' % n for n in ch_names]
    # Median
    if 'median' in list_feature_flags:
        median = numpy_img2d_color_median(image, segm)
        features = np.hstack((features, median))
        names += ['%s_median' % n for n in ch_names]
    # mean Gradient
    # G = np.zeros_like(image)
    # for i in range(image.shape[0]):
    #     G[i,:,:] = np.sum(np.gradient(image[i]), axis=0)
    # grad = cython_img3d_gray_mean(G, segm)
    features = np.nan_to_num(features)
    assert features.shape[1] == len(names), \
        'features: %s and names %s' % (features.shape, repr(names))
    return features, names


def norm_features(features, scaler=None):
    """ normalise features to be in range(0;1)

    :param ndarray features: vector of features
    :param obj scaler:
    :return [[float]]:
    """
    if scaler is None:
        scaler = preprocessing.StandardScaler()
        scaler.fit(features)
    features = scaler.transform(features)
    return features, scaler


def make_gaussian_filter1d(x, sigma, order=0):
    if order > 2:
        raise ValueError("Only orders up to 2 are supported")
    # compute unnormalized Gaussian response
    response = np.exp(-x ** 2 / (2. * sigma ** 2))
    if order == 1:
        response = - response * x
    elif order == 2:
        response = (response * (x ** 2 - sigma ** 2))
    # normalize
    response /= np.abs(response).sum()
    return response


def make_edge_filter2d(sig, phase, pts, sup):
    gx = make_gaussian_filter1d(pts[0, :], sigma=3 * sig)
    gy = make_gaussian_filter1d(pts[1, :], sigma=sig, order=phase)
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
    :return np.ndarray<nb_samples, nb_features>, [str]:

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


def compute_img_filter_response2d(im, filter_battery):
    """ compute image filter response in 2D

    :param [[float]] im:
    :param [[[float]]] filter_battery:
    :return[[float]] :
    """
    if filter_battery.ndim != 3:
        raise ValueError('wrong batery dim %s' % repr(filter_battery.shape))
    responses = np.array([ndimage.convolve(im, fl) for fl in filter_battery])
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
        img_smooth[i, :, :] = gaussian_filter(
                                            img[i, :, :].astype(float), sigma)
    img = (img - img_smooth)
    return img


def compute_texture_desc_lm_img3d_val(img, seg, list_feature_flags,
                                      bank_type='normal'):
    """ compute texture descriptors as mean / std / ...
    on Lewen-Malik filter bank response

    :param [[[float]]] img: np.array
    :param [[[int]]] seg:
    :param [str] list_feature_flags:
    :param str bank_type: define used LM filter bank ['short', 'normal']
    :return np.ndarray<nb_samples, nb_features>, [str]:
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
        # norm responces
        l_n = np.sqrt(np.sum(np.power(response, 2)))
        response = (response * (np.log(1 + l_n) / 0.03)) / l_n
        fts, n = compute_image3d_gray_statistic(response, seg,
                                                list_feature_flags, fl_name)
        features += [fts]
        names += n
    features = np.concatenate(tuple(features), axis=1)
    names = ['tLM_%s' % n for n in names]
    assert features.shape[1] == len(names), \
        'features: %s and names %s' % (features.shape, repr(names))
    return features, names


def compute_texture_desc_lm_img2d_clr(img, seg, list_feature_flags,
                                      bank_type='normal'):
    """ compute texture descriptors via Lewen-Malik filter response

    :param ndarray img:
    :param ndarray seg:
    :param [str] list_feature_flags:
    :param str bank_type: define used LM filter bank ['short', 'normal']
    :return np.ndarray<nb_samples, nb_features>, [str]:

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
        # norm responses
        norm = np.sqrt(np.sum(response_roll ** 2))
        response_roll = (response_roll * (np.log(1 + norm) / 0.03)) / norm
        response = np.rollaxis(response_roll, 0, 3)
        fts, n = compute_image2d_color_statistic(response, seg,
                                                 list_feature_flags, fl_name)
        features += [fts]
        names += n
    features = np.concatenate(tuple(features), axis=1)
    names = ['tLM_%s' % n for n in names]
    assert features.shape[1] == len(names)
    return features, names


def compute_selected_features_gray3d(img, segments,
                                     dict_feature_flags=FEATURES_SET_COLOR):
    """ compute selected features on gray 3D image

    :param ndarray img:
    :param ndarray segments:
    :param {str: [str]} dict_feature_flags:
    :return np.ndarray<nb_samples, nb_features>, [str]:

    >>> np.random.seed(0)
    >>> img = np.random.random((2, 10, 15))
    >>> slic = np.zeros((2, 10, 15), dtype=int)
    >>> slic[:, :, :7] += 1
    >>> slic[1, :, :] += 2
    >>> fts, names = compute_selected_features_gray3d(img, slic,
    ...                                   {'color': ['mean', 'std', 'median']})
    >>> fts.shape
    (4, 3)
    >>> names  # doctest: +NORMALIZE_WHITESPACE
    ['gray_mean', 'gray_std', 'gray_median']
    >>> _ = compute_selected_features_gray3d(img, slic,
    ...                                      {'tLM': ['median', 'std', 'eng']})
    >>> fts, names = compute_selected_features_gray3d(img, slic,
    ...                                      {'tLM_s': ['mean', 'std', 'eng']})
    >>> fts.shape
    (4, 45)
    >>> names  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['tLM_sigma1.4-edge_mean', ..., 'tLM_sigma4.0-GaussLap2_energy']

    """
    _check_gray_image_segm(img, segments)
    assert len(dict_feature_flags) > 0

    features, names = [], []
    if 'color' in dict_feature_flags:
        fts, n = compute_image3d_gray_statistic(img, segments,
                                                dict_feature_flags['color'])
        features.append(fts)
        names += n
    if 'tLM' in dict_feature_flags:
        fts, n = compute_texture_desc_lm_img3d_val(img, segments,
                                                dict_feature_flags['tLM'])
        features.append(fts)
        names += n
    elif 'tLM_s' in dict_feature_flags:
        fts, n = compute_texture_desc_lm_img3d_val(img, segments,
                                                   dict_feature_flags['tLM_s'],
                                                   'short')
        features.append(fts)
        names += n
    if len(features) == 0:
        logging.error('not supported features: %s', repr(dict_feature_flags))
    features = np.concatenate(tuple(features), axis=1)
    assert features.shape[1] == len(names), \
        'features: %s and names %s' % (features.shape, repr(names))
    return features, names


def compute_selected_features_gray2d(img, segments,
                                     dict_features_flags=FEATURES_SET_ALL):
    """

    :param ndarray img:
    :param ndarray segments:
    :param dict_features_flags:
    :return np.ndarray<nb_samples, nb_features>, [str]:

    >>> image = np.zeros((2, 10))
    >>> image[0, 2:6] = 1
    >>> image[1, 3:7] = 3
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> features, names = compute_selected_features_gray2d(image, segm,
    ...                              {'color': ['mean', 'std', 'median']})
    >>> np.round(features, 3)
    array([[ 0.9  ,  1.136,  0.5  ],
           [ 0.7  ,  1.187,  0.   ]])
    >>> _ = compute_selected_features_gray2d(image, segm,
    ...                                      {'tLM': ['mean', 'std', 'median']})
    >>> features, names = compute_selected_features_gray2d(image, segm,
    ...                                  {'tLM_s': ['mean', 'std', 'eng']})
    >>> features.shape
    (2, 45)
    >>> features, names = compute_selected_features_gray2d(image, segm)
    >>> features.shape
    (2, 84)
    """
    _check_gray_image_segm(img, segments)

    features, names = compute_selected_features_gray3d(img[np.newaxis, ...],
                                                       segments[np.newaxis, ...],
                                                       dict_features_flags)
    assert features.shape[1] == len(names)
    return features, names


def compute_selected_features_color2d(img, segments,
                                      dict_feature_flags=FEATURES_SET_ALL):
    """ compute selected features color2d

    :param ndarray img:
    :param ndarray segments:
    :param dict_feature_flags:
    :return np.ndarray<nb_samples, nb_features>, [str]:

    >>> image = np.zeros((2, 10, 3))
    >>> image[:, 2:6, 0] = 1
    >>> image[:, 3:7, 1] = 3
    >>> image[:, 4:9, 2] = 2
    >>> segm = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> features, names = compute_selected_features_color2d(image, segm,
    ...                                   {'color': ['mean', 'std', 'median']})
    >>> np.round(features, 3)
    array([[ 0.6 ,  1.2 ,  0.4 ,  0.49,  1.47,  0.8 ,  1.  ,  0.  ,  0.  ],
           [ 0.2 ,  1.2 ,  1.6 ,  0.4 ,  1.47,  0.8 ,  0.  ,  0.  ,  2.  ]])
    >>> _ = compute_selected_features_color2d(image, segm,
    ...                                       {'tLM': ['mean', 'std', 'eng']})
    >>> features, names = compute_selected_features_color2d(image, segm,
    ...                                   {'tLM_s': ['mean', 'std', 'eng']})
    >>> features.shape
    (2, 135)
    >>> features, names = compute_selected_features_color2d(image, segm)
    >>> features.shape
    (2, 192)
    """
    _check_color_image(img)
    features = np.empty((np.max(segments) + 1, 0))
    names = []
    if 'color' in dict_feature_flags:
        fts, n = compute_image2d_color_statistic(img, segments,
                                                 dict_feature_flags['color'])
        features = np.concatenate((features, fts), axis=1)
        names += n
    if 'tLM' in dict_feature_flags:
        fts, n = compute_texture_desc_lm_img2d_clr(img, segments,
                                                   dict_feature_flags['tLM'])
        features = np.concatenate((features, fts), axis=1)
        names += n
    elif 'tLM_s' in dict_feature_flags:
        fts, n = compute_texture_desc_lm_img2d_clr(img, segments,
                                                   dict_feature_flags['tLM_s'],
                                                   'short')
        features = np.concatenate((features, fts), axis=1)
        names += n
    if len(features) == 0:
        logging.error('not supported features: %s', repr(dict_feature_flags))
    assert features.shape[1] == len(names)
    return features, names


def compute_selected_features_img2d(image, segm,
                                    dict_features_flags=FEATURES_SET_COLOR):
    if image.ndim == 3 and image.shape[2] == 3:
        return compute_selected_features_color2d(image, segm,
                                                 dict_features_flags)
    elif image.ndim == 2:
        return compute_selected_features_gray2d(image, segm,
                                                dict_features_flags)
    else:
        logging.error('invalid image size - %s', repr(image.shape))


def extend_segm_by_struct_elem(segm, struc_elem):
    """ extend the image by size of the stuctur element

    :param [[int]] segm:
    :param [[int]] struc_elem:
    :return [[int]]:
    """
    assert segm.ndim >= struc_elem.ndim

    shape_new = np.array(segm.shape[:struc_elem.ndim]) \
                 + np.array(struc_elem.shape)
    begin = (np.array(struc_elem.shape) / 2).astype(int)
    if segm.ndim == struc_elem.ndim:
        segm_extend = np.full(shape_new, fill_value=np.NaN)
        segm_extend[begin[0]:begin[0] + segm.shape[0],
                    begin[1]:begin[1] + segm.shape[1]] = segm

    else:
        shape_new = np.hstack((shape_new, segm.shape[struc_elem.ndim:]))
        segm_extend = np.zeros(shape_new)
        segm_extend[begin[0]:begin[0] + segm.shape[0],
                    begin[1]:begin[1] + segm.shape[1], :] = segm
    return segm_extend


def compute_label_histograms_positions(segm, list_positions,
                                       diameters=HIST_CIRCLE_DIAGONALS,
                                       nb_labels=None):
    """ compute the histogram features doe consecutive growing diameter
    of inter circle neighbouring around given points in the segmentation

    :param ndarray segm: np.array<height, width>
    :param list_positions:  [(int, int)]
    :param diameters: [int]
    :param nb_labels: int
    :return: np.array<nb_samples, nb_features>, [str]


    >>> segm = np.zeros((10, 10), dtype=int)
    >>> segm[1:9, 2:8] = 1
    >>> segm[3:7, 4:6] = 2
    >>> points = [[3, 3], [4, 4], [2, 7], [6, 6]]
    >>> hists, names = compute_label_histograms_positions(segm, points,
    ...                                                   [1, 2, 4], 3)
    >>> names  # doctest: +NORMALIZE_WHITESPACE
    ['hist-d_1-lb_0', 'hist-d_1-lb_1', 'hist-d_1-lb_2', \
     'hist-d_2-lb_0', 'hist-d_2-lb_1', 'hist-d_2-lb_2', \
     'hist-d_4-lb_0', 'hist-d_4-lb_1', 'hist-d_4-lb_2']
    >>> hists.shape
    (4, 9)
    >>> np.round(hists, 2)
    array([[ 0.  ,  0.8 ,  0.2 ,  0.12,  0.62,  0.25,  0.42,  0.39,  0.14],
           [ 0.  ,  0.2 ,  0.8 ,  0.  ,  0.62,  0.38,  0.22,  0.75,  0.03],
           [ 0.2 ,  0.8 ,  0.  ,  0.5 ,  0.5 ,  0.  ,  0.31,  0.22,  0.14],
           [ 0.  ,  0.8 ,  0.2 ,  0.12,  0.62,  0.25,  0.42,  0.39,  0.14]])
    """
    pos_dim = np.asarray(list_positions).shape[1]
    assert (segm.ndim - pos_dim) in (0, 1)

    if nb_labels is None:
        if segm.ndim == pos_dim:
            nb_labels = segm.max() + 1
        elif segm.ndim == (pos_dim + 1):
            nb_labels = segm.shape[-1]
        else:
            logging.error('estimate nb labels failed')

    logging.debug('prepare extended segm. and struc. elements')
    list_struct_elems = [morphology.disk(d) for d in diameters]
    list_segm_extend = [extend_segm_by_struct_elem(segm, sel)
                        for sel in list_struct_elems]

    pos_hists = list()
    logging.debug('compute circular histogram')
    # for each postion compute features
    for pos in list_positions:
        hist_pos = list()
        hist_last = np.zeros(nb_labels)
        sel_last = np.zeros(1)
        for segm_extend, sel in zip(list_segm_extend, list_struct_elems):
            norm = np.sum(sel) - np.sum(sel_last)
            assert norm > 0
            # hist_new = segm_convol[diam, :, pos[1], pos[0]]
            if segm_extend.ndim == len(pos):
                hist = compute_label_hist_segm(segm_extend, pos,
                                               sel, nb_labels)
            else:
                hist = compute_label_hist_proba(segm_extend, pos, sel)
            # logging.debug('diff: %s last: %s new: %s',
            # repr((hist - hist_last).tolist()), repr(hist_last.tolist()),
            # repr(hist.tolist()))
            hist_pos += ((hist - hist_last) / norm).tolist()
            hist_last = hist
            sel_last = sel
        pos_hists.append(hist_pos)

    feature_names = ['hist-d_%i-lb_%i' % (d, lb)
                     for d in diameters for lb in range(nb_labels)]
    pos_hists = np.array(pos_hists)
    assert pos_hists.shape[1] == len(feature_names)
    return np.array(pos_hists), feature_names


def compute_label_hist_segm(segm, position, struc_elem, nb_labels):
    """ compute histogram of labels for set of centric annulus

    :param ndarray segm: np.array<height, width>
    :param position: (float, float)
    :param struc_elem: np.array<h, w>
    :param nb_labels: int, total number of labels in the segm.
    :return: [float]

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
    array([ 3.,  6.,  0.])
    >>> compute_label_hist_segm(segm, [4, 4], np.ones((5, 5)), 3)
    array([  5.,  14.,   6.])
    """
    assert segm.ndim == len(position)
    position = [int(p) for p in position]
    # take selection around point with size of struc element
    segm_select = segm[position[0]:position[0] + struc_elem.shape[0],
                       position[1]:position[1] + struc_elem.shape[1]]
    assert segm_select.shape == struc_elem.shape
    hist = np.zeros(nb_labels)
    for lb in range(nb_labels):
        hist[lb] = np.sum(np.logical_and(segm_select == lb, struc_elem == 1))
    return hist


def compute_label_hist_proba(segm, position, struc_elem):
    """ compute histogram of labels for set of centric annulus
    expecting that each label has own layer

    :param ndarray segm: np.array<height, width>
    :param position: (float, float)
    :param ndarray struc_elem: np.array<h, w>
    :return: [float]
    """
    assert segm.ndim == (len(position) + 1)
    position = map(int, position)
    # take selection around point with size of struc element
    segm_select = segm[position[0]:position[0] + struc_elem.shape[0],
                       position[1]:position[1] + struc_elem.shape[1], :]
    assert segm_select.shape[:-1] == struc_elem.shape
    segm_mask = np.rollaxis(segm_select, -1, 0) \
                 * np.tile(struc_elem, (segm_select.shape[-1], 1, 1))
    hist = np.sum(segm_mask, axis=tuple(range(1, segm_mask.ndim)))
    return hist


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


def compute_ray_features_segm_2d_OLD(seg_binary, position, angle_step=5.,
                                     smooth_coef=0, edge='up'):
    """ USES WHOLE IMAGE ROTATION SO IT IS VERY SLOW
    compute ray features vector , shift them to be startig from larges
    and smooth_coef them by gauss filter
    (from fiven point the close distance to boundary)

    :param str edge: pointing to the up of down edge o
    :param int smooth_coef:
    :param ndarray seg_binary: np.array<height, width>
    :param (int, int) position:
    :param float angle_step:
    :return [float]:

    example, see unittests
    >>> from skimage import draw
    >>> seg = np.ones((100, 100), dtype=bool)
    >>> x, y = draw.circle(45, 55, 30, shape=seg.shape)
    >>> seg[x, y] = False
    >>> compute_ray_features_segm_2d_OLD(seg, (50, 50), 45)
    array([35, 29, 25, 23, 24, 29, 34, 36])
    >>> compute_ray_features_segm_2d_OLD(seg, (60, 40), 30, smooth_coef=1)
    array([35, 27, 18, 12, 10,  9, 12, 18, 27, 37, 45, 49])
    >>> compute_ray_features_segm_2d_OLD(seg, (40, 60), 20).tolist()
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


def compute_ray_features_segm_2d(seg_binary, position, angle_step=5.,
                                 smooth_coef=0, edge='up'):
    """ compute ray features vector , shift them to be startig from larges
    and smooth_coef them by gauss filter
    (from fiven point the close distance to boundary)

    :param str edge: pointing to the up of down edge o
    :param int smooth_coef:
    :param ndarray seg_binary: np.array<height, width>
    :param (int, int) position:
    :param float angle_step:
    :return [float]:

    example, see unittests
    >>> seg_empty = np.zeros((100, 150), dtype=bool)
    >>> compute_ray_features_segm_2d(seg_empty, (50, 75), 90)
    array([-1, -1, -1, -1])
    >>> from skimage import draw
    >>> seg = np.ones((100, 150), dtype=bool)
    >>> x, y = draw.circle(50, 75, 40, shape=seg.shape)
    >>> seg[x, y] = False
    >>> compute_ray_features_segm_2d(seg, (50, 75), 45)
    array([40, 41, 40, 41, 40, 41, 40, 41])
    >>> compute_ray_features_segm_2d(seg, (60, 40), 30, smooth_coef=1).tolist()
    [65, 51, 31, 15, 6, 4, 4, 7, 15, 32, 52, 66]
    >>> compute_ray_features_segm_2d(seg, (40, 60), 20).tolist()
    [54, 57, 58, 56, 50, 43, 36, 31, 26, 24, 22, 22, 23, 25, 29, 34, 40, 47]
    """
    seg_binary = seg_binary.astype(bool)
    # nb_steps = 360 / angle_step
    angles = np.arange(0, 360, angle_step)
    ray_dist = np.array([-1] * len(angles))

    # in case the position is inside the border lable
    label_position = seg_binary[int(position[0]), int(position[1])]
    if bool(label_position) and edge == 'up':
        return ray_dist * 0
    rect_diag = int(np.sqrt(seg_binary.shape[0] ** 2 +
                            seg_binary.shape[1] ** 2))

    for i, ang in enumerate(angles):
        pos = np.array(position, dtype=float)
        rad = np.deg2rad(ang)
        grad = np.array([np.sin(rad), np.cos(rad)])
        grad /= np.abs(grad).max()
        last = seg_binary[int(position[0]), int(position[1])]
        for _ in range(rect_diag):
            pos += grad
            if pos[0] < 0 or pos[0] >= seg_binary.shape[0] \
                    or pos[1] < 0 or pos[1] >= seg_binary.shape[1]:
                break
            actual = seg_binary[int(pos[0]), int(pos[1])]
            if (edge == 'up' and actual) \
                    or (edge == 'down' and last and not actual):
                ray_dist[i] = np.sqrt((pos[0] - position[0]) ** 2
                                      + (pos[1] - position[1]) ** 2)
                break
            last = actual

    if smooth_coef is not None and smooth_coef > 0:
        ray_dist = gaussian_filter1d(ray_dist, smooth_coef)

    return np.array(ray_dist)


def shift_ray_features(ray_dist):
    """ shift Ray features ti the global maxim to be rotation invariant

    :param [float] ray_dist:
    :return [float]:

    >>> vec = np.array([43, 46, 46, 39, 28, 18, 12, 10,  9, 12, 17, 22])
    >>> ray, shift = shift_ray_features(vec)
    >>> shift
    30.0
    >>> ray
    array([46, 46, 39, 28, 18, 12, 10,  9, 12, 17, 22, 43])
    >>> ray2, shift = shift_ray_features(ray)
    >>> shift
    0.0
    >>> np.array_equal(ray, ray2)
    True
    """
    angle_step = 360 / len(ray_dist)
    max_loc = np.argmax(ray_dist)
    ray_dist = ray_dist[max_loc:].tolist() + ray_dist[:max_loc].tolist()
    shift = float(max_loc * angle_step)

    return np.array(ray_dist), shift


def compute_ray_features_positions(segm, list_positions, angle_step=5.,
                                   border_labels=None, segm_open=None,
                                   smooth_ray=None, shifting=True, edge='up'):
    """ compute ray features fo multiple points in the segmentation
    with given boundary labels and step angle

    :param segm: np.array<height, width>
    :param list_positions: [(int, int)]
    :param angle_step: float
    :param border_labels: [int] all labels to be set as boundaries
    :param int segm_open:
    :param float smooth_ray:
    :param bool shifting:
    :param str edge: type of edge up/down
    :return:

    example, see unittests
    >>> from skimage import draw
    >>> np.random.seed(0)
    >>> seg = np.zeros((100, 100), dtype=int)
    >>> x, y = draw.circle(45, 55, 30, shape=seg.shape)
    >>> seg[x, y] = 1
    >>> x, y = draw.circle(55, 45, 10, shape=seg.shape)
    >>> seg[x, y] = 2
    >>> points = [(50, 50), (60, 40), (45, 55)]
    >>> ray_dist, shift, _ = compute_ray_features_positions(seg, points, 20)
    >>> shift
    [300.0, 300.0, 40.0]
    >>> ray_dist.tolist()  # doctest: +NORMALIZE_WHITESPACE
    [[38, 37, 36, 35, 32, 30, 27, 25, 24, 23, 23, 24, 25, 26, 28, 31, 33, 35],
     [50, 50, 47, 41, 32, 23, 17, 13, 10, 9, 10, 9, 11, 14, 19, 26, 36, 44],
    [31, 30, 30, 30, 30, 31, 30, 30, 29, 30, 30, 30, 30, 30, 30, 29, 30, 30]]
    >>> noise_pos = np.random.randint(10, 80, (2, 300))
    >>> seg[noise_pos[0], noise_pos[1]] = 0  # add random noise
    >>> ray_dist, shift, names = compute_ray_features_positions(seg, points,
    ...                                                     45, segm_open=10)
    >>> names  # doctest: +NORMALIZE_WHITESPACE
    ['ray-lb_0-agl_0', 'ray-lb_0-agl_45', 'ray-lb_0-agl_90',
     'ray-lb_0-agl_135', 'ray-lb_0-agl_180', 'ray-lb_0-agl_225',
     'ray-lb_0-agl_270', 'ray-lb_0-agl_315']
    >>> shift
    [315.0, 315.0, 45.0]
    >>> ray_dist
    array([[38, 35, 29, 25, 24, 25, 29, 35],
           [52, 41, 21, 11,  9, 11, 21, 41],
           [31, 30, 31, 30, 31, 30, 31, 30]])
    """
    logging.debug('compute Ray features with border label=%s and angle step=%f',
                  repr(border_labels), angle_step)
    pos_dim = np.asarray(list_positions).shape[1]
    assert (segm.ndim - pos_dim) in (0, 1)
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

    pos_rays, pos_shift = list(), list()
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
    assert pos_rays.shape[1] == len(feature_names)
    return pos_rays, pos_shift, feature_names


def interpolate_ray_dist(ray_dists, order='spline'):
    """ interpolate ray distances

    :param [float] ray_dists:
    :param str order: degree of interpolation
    :return [float]:

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
    x_train_ext = np.hstack((x_train - len(x_space),
                             x_train,
                             x_train + len(x_space)))
    y_train = ray_dists[ray_dists != -1]
    y_train_ext = np.array(y_train.tolist() * 3)

    if isinstance(order, int):
        # model = pipeline.make_pipeline(preprocessing.PolynomialFeatures(order),
        #                                linear_model.Ridge())
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
        def fn_cos(x, t):
            return x[0] + x[1] * np.sin(x[2] + x[3] * t)

        def fn_cos_residual(x, t, y):
            return fn_cos(x, t) - y

        x0 = np.array([np.mean(y_train), (y_train.max() - y_train.min()) / 2.,
                       0, len(x_space) / np.pi])
        lsm_res = optimize.least_squares(fn_cos_residual, x0, gtol=1e-1,
                                         # loss='soft_l1', f_scale=0.1,
                                         args=(x_train, y_train))
        ray_dists[missing] = fn_cos(lsm_res.x, x_space[missing])

    return ray_dists


def reconstruct_ray_features_2d(position, ray_features, shift=0):
    """ reconstruct ray features for 2D image

    :param (int, int) position:
    :param [float] ray_features:
    :param float shift:
    :return [[float, float]]:

    example, see unittests
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
    dx = np.cos(angles) * ray_features
    dy = np.sin(angles) * ray_features

    positions = np.tile(position, (len(ray_features), 1))
    points = positions + np.array([dx, dy]).T
    mask = np.logical_and(np.array(ray_features) >= 0,
                          ~ np.isinf(ray_features))
    points = points[mask, :]

    return points


def reduce_close_points(points, dist_thr):
    """ reduce remove points with smaller internal distance then treshold
    assumption, the points are in sequence geometrically ordered)

    :param [[float, float]] points:
    :param float dist_thr:
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
    assert len(points) > 2

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

"""

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging
import warnings

import numpy as np
# from collections import Counter
from scipy import ndimage
# from numba import jit, autojit


def contour_binary_map(seg, label=1, include_boundary=False):
    """ get object boundaries

    :param ndarray seg: integer images, typically a segmentation
    :param int label: selected singe label in segmentation
    :param bool include_boundary: assume that the object end with image boundary
    :return ndarray:

    >>> img = np.zeros((6, 6), dtype=int)
    >>> img[1:5, 2:] = 1
    >>> contour_binary_map(img)
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> contour_binary_map(img, include_boundary=True)
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1],
           [0, 0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0, 1],
           [0, 0, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0]])
    """
    w, h = seg.shape[:2]
    # logger.debug('testing label {}'.format(label))
    res = np.zeros((w, h), dtype=np.int)
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            # just for 4-connected
            if seg[i, j] == label \
                    and (seg[i - 1, j] != label or seg[i, j - 1] != label
                         or seg[i + 1, j] != label or seg[i, j + 1] != label):
                res[i, j] = 1
    if include_boundary:
        for i in range(0, w):
            if seg[i, 0] == label:
                    res[i, 0] = 1
            if seg[i, -1] == label:
                    res[i, -1] = 1
        for j in range(0, h):
            if seg[0, j] == label:
                    res[0, j] = 1
            if seg[-1, j] == label:
                    res[-1, j] = 1
    # logger.debug('matrix seg_pipe \total{}'.format(repr(res)))
    return res


def contour_coords(seg, label=1, include_boundary=False):
    """ get object boundaries

    :param ndarray seg: integer images, typically a segmentation
    :param int label: selected singe label in segmentation
    :param bool include_boundary: assume that the object end with image boundary
    :return [[int, int]]:

    >>> img = np.zeros((6, 6), dtype=int)
    >>> img[1:5, 2:] = 1
    >>> contour_coords(img)
    [[1, 2], [1, 3], [1, 4], [2, 2], [3, 2], [4, 2], [4, 3], [4, 4]]
    >>> contour_coords(img, include_boundary=True)  #doctest: +NORMALIZE_WHITESPACE
    [[1, 2], [1, 3], [1, 4], [2, 2], [3, 2], [4, 2], [4, 3], [4, 4], \
     [1, 5], [2, 5], [3, 5], [4, 5]]
    """
    w, h = seg.shape[:2]
    # logger.debug('testing label {}'.format(label))
    res = []
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            # just for 4-connected
            if seg[i, j] == label \
                    and (seg[i-1, j] != label or seg[i, j-1] != label
                         or seg[i + 1, j] != label or seg[i, j + 1] != label):
                res.append([i, j])
    if include_boundary:
        for i in range(0, w):
            if seg[i, 0] == label:
                    res.append([i, 0])
            if seg[i, -1] == label:
                    res.append([i, h - 1])
        for j in range(0, h):
            if seg[0, j] == label:
                    res.append([0, j])
            if seg[-1, j] == label:
                    res.append([w - 1, j])
    return res


def binary_image_from_coords(coords, size):
    """ create binary image just from point contours

    :param ndarray seg: integer images, typically a segmentation
    :param int label: selected singe label in segmentation
    :return ndarray:

    >>> img = np.zeros((6, 6), dtype=int)
    >>> img[1:5, 2:] = 1
    >>> coords = contour_coords(img)
    >>> binary_image_from_coords(coords, img.shape)
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0]])
    """
    contour_map = np.zeros(size, dtype=np.int)
    w, h = size
    for cd in coords:
        if 0 <= cd[0] < w and 0 <= cd[1] < h:
            contour_map[cd[0], cd[1]] = 1
    return contour_map


def compute_distance_map(seg, label=1):
    """ compute distance from label boundaries

    :param ndarray seg: integer images, typically a segmentation
    :param int label: selected singe label in segmentation
    :return:

    >>> img = np.zeros((6, 6), dtype=int)
    >>> img[1:5, 2:] = 1
    >>> dist = compute_distance_map(img)
    >>> np.round(dist, 2)
    array([[ 2.24,  1.41,  1.  ,  1.  ,  1.  ,  1.41],
           [ 2.  ,  1.  ,  0.  ,  0.  ,  0.  ,  1.  ],
           [ 2.  ,  1.  ,  0.  ,  1.  ,  1.  ,  1.41],
           [ 2.  ,  1.  ,  0.  ,  1.  ,  1.  ,  1.41],
           [ 2.  ,  1.  ,  0.  ,  0.  ,  0.  ,  1.  ],
           [ 2.24,  1.41,  1.  ,  1.  ,  1.  ,  1.41]])
    """
    contour_coord = contour_coords(seg, label)
    # logger.debug('contour coordinates {}'.format(repr(contourCoord)))
    contour_map = 1 - binary_image_from_coords(contour_coord, seg.shape)
    dist = ndimage.distance_transform_edt(contour_map)
    # logger.debug('distance map \total{}'.format(repr(dist)))
    return dist


def segm_labels_assignemet(segm, segm_gt):
    """

    :param seg:
    :param segm_gt:
    :return:

    >>> slic = np.array([[0] * 3 + [1] * 3 + [2] * 3 + [3] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3 + [7] * 3] * 4)
    >>> segm = np.zeros(slic.shape, dtype=int)
    >>> segm[4:, 6:] = 1
    >>> segm_labels_assignemet(slic, segm)  #doctest: +NORMALIZE_WHITESPACE
    {0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     4: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     5: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     6: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     7: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    """
    assert segm_gt.shape == segm.shape
    labels = np.unique(segm)
    # label_hist = {}
    # for lb in labels:
    #     label_hist[lb] = (segm_gt[segm == lb].tolist())
    label_hist = {lb: list() for lb in labels}
    segm_gt_flat = segm_gt.ravel()
    segm_flat = segm.ravel()
    for i, lb in enumerate(segm_flat):
        label_hist[lb].append(segm_gt_flat[i])
    return label_hist


# @autojit
def histogram_regions_labels_counts(slic, segm):
    """

    :param ndarray slic: input superpixel segmenatation
    :param ndarray segm: reference segmentation
    :return:

    >>> slic = np.array([[0] * 3 + [1] * 3 + [2] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3] * 4)
    >>> segm = np.zeros(slic.shape, dtype=int)
    >>> segm[4:, 5:] = 1
    >>> histogram_regions_labels_counts(slic, segm)
    array([[ 12.,   0.],
           [ 12.,   0.],
           [ 12.,   0.],
           [  0.,   0.],
           [ 12.,   0.],
           [  8.,   4.],
           [  0.,  12.]])
    """
    assert slic.shape == segm.shape, 'dimension does not agree'
    segm_flat = slic.ravel()
    annot_flat = segm.ravel()
    idx_max = slic.max()
    label_max = segm.max()
    matrix_hist = np.zeros((idx_max + 1, label_max + 1))

    for i, lb in enumerate(segm_flat):
        matrix_hist[lb, annot_flat[i]] += 1

    return matrix_hist


def histogram_regions_labels_norm(slic, segm):
    """

    :param ndarray slic: input superpixel segmenatation
    :param ndarray segm: reference segmentation
    :return:

    >>> slic = np.array([[0] * 3 + [1] * 3 + [2] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3] * 4)
    >>> segm = np.zeros(slic.shape, dtype=int)
    >>> segm[4:, 5:] = 1
    >>> histogram_regions_labels_norm(slic, segm)  # doctest: +ELLIPSIS
    array([[ 1.        ,  0.        ],
           [ 1.        ,  0.        ],
           [ 1.        ,  0.        ],
           [ 0.        ,  0.        ],
           [ 1.        ,  0.        ],
           [ 0.66666667,  0.33333333],
           [ 0.        ,  1.        ]])
    """
    matrix_hist = histogram_regions_labels_counts(slic, segm)
    region_sums = np.tile(np.sum(matrix_hist, axis=1),
                          (matrix_hist.shape[1], 1)).T
    matrix_hist = (matrix_hist / region_sums)
    matrix_hist = np.nan_to_num(matrix_hist)
    return matrix_hist

# DEPRECATED
# def histogram_regions_labels(slic, seg_pipe):
#     """  compute the histogram matrix for each region with given labels
#
#     :param slic: np.array
#     :param seg_pipe: np.array
#     :return: np.array<max_regions, max_label>
#     """
#     label_hist = segm_labels_assignemet(slic, seg_pipe)
#     idx_max = slic.max()
#     label_max = seg_pipe.max()
#     matrix_hist = np.zeros((idx_max + 1, label_max + 1))
#     for idx in label_hist:
#         counts = dict(Counter(label_hist[idx]))
#         procs = [v / float(len(vals)) for v in counts.itervalues()]
#         matrix_hist[idx, counts.keys()] = procs
#     return matrix_hist


def assigne_label_by_threshold(dict_label_hist, thresh=0.75):
    """

    :param dict_lb_hist:
    :param thresh:
    :return:

    >>> slic = np.array([[0] * 4 + [1] * 3 + [2] * 3 + [3] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3 + [7] * 4] * 4)
    >>> segm = np.zeros(slic.shape, dtype=int)
    >>> segm[4:, 6:] = 1
    >>> lb_hist = segm_labels_assignemet(slic, segm)
    >>> assigne_label_by_threshold(lb_hist)
    array([0, 0, 0, 0, 0, 0, 1, 1])
    """
    lut = np.zeros(max(dict_label_hist.keys()) + 1, dtype=int) - 1
    for k in dict_label_hist:
        v = dict_label_hist[k]
        # unique, counts = np.unique(v, return_counts=True)
        counts = np.bincount(v) / float(len(v))
        mx = counts.max()
        # logger.debug('#{} hist: {} while max: {}'.format(k, counts, mx))
        if mx > thresh:
            lut[k] = counts.tolist().index(mx)
    return lut


def assigne_label_by_max(dict_label_hist):
    """

    :param label_hist:
    :return:

    >>> slic = np.array([[0] * 4 + [1] * 3 + [2] * 3 + [3] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3 + [7] * 4] * 4)
    >>> segm = np.zeros(slic.shape, dtype=int)
    >>> segm[4:, 6:] = 1
    >>> lb_hist = segm_labels_assignemet(slic, segm)
    >>> assigne_label_by_max(lb_hist)
    array([0, 0, 0, 0, 0, 0, 1, 1])
    """
    lut = np.zeros(max(dict_label_hist.keys()) + 1, dtype=int) - 1
    for k in dict_label_hist:
        v = dict_label_hist[k]
        counts = np.bincount(v) / float(len(v))
        lut[k] = np.argmax(counts)
    return lut


def convert_segms_2_list(segms):
    """

    :param ndarray segms:
    :return [int]:

    >>> seg_pipe = np.ones((2, 3), dtype=int)
    >>> convert_segms_2_list([seg_pipe, seg_pipe * 0, seg_pipe * 2])
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2]
    """
    y = np.concatenate(tuple(seg.ravel() for seg in segms), axis=0).tolist()
    return y


def mask_segm_labels(im_labeling, labels, mask_init=None):
    """ with given labels image and list of desired labels it create mask finding
    all labels in the list (perform logical or on image with a list of labels)

    :param im_labeling: np.array<height, width> input labeling
    :param labels: [int] list of wanted labels to be detected in image
    :param mask_init: np.array<height, width> initial bool mask on the beginning
    :return: np.array<height, width> bool mask
    """
    if mask_init is None:
        mask = np.full(im_labeling.shape, False, dtype=bool)
    else:
        mask = mask_init.copy()
    for l in labels:
        mask = np.logical_or(mask, (im_labeling == l))
    return mask


def sequence_labels_merge(labels_stack, dict_colors, labels_free, change_label=-1):
    """ the input is time series of labeled images and output id labeled image
    with labels that was constant for all the time
    the special case is using free labels which can be assumed as any labeled

    Example if labels series, {0, 1, 2} and 0 is free label:
    - 11111111 -> 1
    - 11211211 -> CHANGE_LABEL
    - 10111100 -> 1
    - 00000000 -> CHANGE_LABEL

    :param ndarray labels_stack: np.array<height, width, date> input stack of labeled images
    :param {int: (int, int, int)} dict_colors: dictionary of labels-colors
    :param [int] labels_free: list of free labels
    :param int change_label: label that is set for non constant time series
    :return ndarray: np.array<height, width>
    """
    im_labels = np.full(labels_stack.shape[:-1], change_label, dtype=np.int)
    labels_used = [lb for lb in dict_colors if lb not in labels_free]
    # generate mask of free labels
    mask_free = mask_segm_labels(labels_stack, labels_free)
    for lb in labels_used:
        mask = mask_segm_labels(labels_stack, [lb], mask_free)
        im_labels[np.all(mask, axis=2)] = lb
    return im_labels


def relabel_by_dict(labels, dict_labels):
    """ relabel according given dictionary of new - old labels

    :param ndarray labels:
    :param {int: [int]} dict_labels:
    :return ndarray:

    >>> labels = np.array([2, 1, 0, 3, 3, 0, 2, 3, 0, 0])
    >>> relabel_by_dict(labels, {0: [1, 2], 1: [0, 3]}).tolist()
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    """
    assert dict_labels is not None
    labels_new = np.zeros_like(labels)
    for lb_new in dict_labels:
        for lb_old in dict_labels[lb_new]:
            labels_new[labels == lb_old] = lb_new
    return labels_new


def merge_probab_labeling_2d(proba, dict_labels):
    """

    :param proba:
    :param {int: [int]} dict_labels:
    :return:

    >>> p = np.ones((5, 5))
    >>> proba = np.array([p * 0.3, p * 0.4, p * 0.2])
    >>> proba = np.rollaxis(proba, 0, 3)
    >>> proba.shape
    (5, 5, 3)
    >>> proba_new = merge_probab_labeling_2d(proba, {0: [1, 2], 1: [0]})
    >>> proba_new.shape
    (5, 5, 2)
    >>> proba_new[0, 0]
    array([ 0.6,  0.3])
    """
    assert proba.ndim == 3
    assert dict_labels is not None
    max_label = max(dict_labels.keys()) + 1
    size = proba.shape[:-1] + (max_label,)
    proba_new = np.zeros(size)
    for lb_new in dict_labels:
        lbs_old = dict_labels[lb_new]
        proba_new[:, :, lb_new] = np.sum(proba[:, :, lbs_old], axis=-1)
    return proba_new

"""
Framework for labeling

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging

import numpy as np
from scipy import ndimage
import skimage.segmentation as sk_segm

import imsegm.utils.data_io as tl_data


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
    [[1, 2], [1, 3], [1, 4], [2, 2], [3, 2], [4, 2], [4, 3], [4, 4],
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

    :param ndarray coords:
    :param (int, int) size:
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
    :return ndarray:

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


def segm_labels_assignment(segm, segm_gt):
    """ create labels assign to the particular regions

    :param ndarray segm: input segmentation
    :param ndarray segm_gt: true segmentation
    :return:

    >>> slic = np.array([[0] * 3 + [1] * 3 + [2] * 3 + [3] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3 + [7] * 3] * 4)
    >>> segm = np.zeros(slic.shape, dtype=int)
    >>> segm[4:, 6:] = 1
    >>> segm_labels_assignment(slic, segm)  #doctest: +NORMALIZE_WHITESPACE
    {0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     4: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     5: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     6: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     7: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    """
    assert segm_gt.shape == segm.shape, 'segm %s and annot %s should match' \
                                        % (repr(segm.shape), repr(segm_gt.shape))
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
    """ histogram or overlaping region between two segmentations,
    the typical usage is label superpixel from annotation

    :param ndarray slic: input superpixel segmenatation
    :param ndarray segm: reference segmentation
    :return ndarray:

    >>> slic = np.array([[0] * 3 + [1] * 3 + [2] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3] * 4)
    >>> segm = np.zeros(slic.shape, dtype=int)
    >>> segm[4:, 5:] = 2
    >>> histogram_regions_labels_counts(slic, segm)
    array([[ 12.,   0.,   0.],
           [ 12.,   0.,   0.],
           [ 12.,   0.,   0.],
           [  0.,   0.,   0.],
           [ 12.,   0.,   0.],
           [  8.,   0.,   4.],
           [  0.,   0.,  12.]])
    """
    assert slic.shape == segm.shape, 'dimension does not agree'
    assert np.sum(np.unique(segm) < 0) == 0, 'only positive labels are allowed'
    segm_flat = slic.ravel()
    annot_flat = segm.ravel()
    idx_max = slic.max()
    label_max = segm.max()
    matrix_hist = np.zeros((idx_max + 1, label_max + 1))

    for i, lb in enumerate(segm_flat):
        matrix_hist[lb, annot_flat[i]] += 1

    return matrix_hist


def histogram_regions_labels_norm(slic, segm):
    """ normalised histogram or overlapping region between two segmentation,
    the typical usage is label superpixel from annotation - relative overlap

    :param ndarray slic: input superpixel segmentation
    :param ndarray segm: reference segmentation
    :return ndarray:

    >>> slic = np.array([[0] * 3 + [1] * 3 + [2] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3] * 4)
    >>> segm = np.zeros(slic.shape, dtype=int)
    >>> segm[4:, 5:] = 2
    >>> histogram_regions_labels_norm(slic, segm)  # doctest: +ELLIPSIS
    array([[ 1.        ,  0.        ,  0.        ],
           [ 1.        ,  0.        ,  0.        ],
           [ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ],
           [ 1.        ,  0.        ,  0.        ],
           [ 0.66666667,  0.        ,  0.33333333],
           [ 0.        ,  0.        ,  1.        ]])
    """
    assert slic.shape == segm.shape, 'dimension of SLIC %s and segm %s should match' \
                                     % (repr(slic.shape), repr(segm.shape))
    assert np.sum(np.unique(segm) < 0) == 0, 'only positive labels are allowed'
    matrix_hist = histogram_regions_labels_counts(slic, segm)
    region_sums = np.tile(np.sum(matrix_hist, axis=1),
                          (matrix_hist.shape[1], 1)).T
    # prevent dividing by 0
    region_sums[region_sums == 0] = -1.
    matrix_hist = (matrix_hist / region_sums)
    matrix_hist = np.nan_to_num(matrix_hist)
    # preventing negative zeros
    matrix_hist[matrix_hist == 0] = 0
    return matrix_hist

# DEPRECATED
# def histogram_regions_labels(slic, seg_pipe):
#     """  compute the histogram matrix for each region with given labels
#
#     :param slic: np.array
#     :param seg_pipe: np.array
#     :return: np.array<max_regions, max_label>
#     """
#     label_hist = segm_labels_assignment(slic, seg_pipe)
#     idx_max = slic.max()
#     label_max = seg_pipe.max()
#     matrix_hist = np.zeros((idx_max + 1, label_max + 1))
#     for idx in label_hist:
#         counts = dict(Counter(label_hist[idx]))
#         procs = [v / float(len(vals)) for v in counts.itervalues()]
#         matrix_hist[idx, counts.keys()] = procs
#     return matrix_hist


def assign_label_by_threshold(dict_label_hist, thresh=0.75):
    """ assign label if the purity reach certain threshold

    :param {int: [int]} dict_lb_hist: dictionary of label histogram
    :param float thresh: threshold for region purity
    :return [int]: resulting LookUpTable

    >>> slic = np.array([[0] * 4 + [1] * 3 + [2] * 3 + [3] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3 + [7] * 4] * 4)
    >>> segm = np.zeros(slic.shape, dtype=int)
    >>> segm[4:, 6:] = 1
    >>> lb_hist = segm_labels_assignment(slic, segm)
    >>> assign_label_by_threshold(lb_hist)
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


def assign_label_by_max(dict_label_hist):
    """ assign label according maximal label count in particular region

    :param {int: [int]} dict_lb_hist: dictionary of label histogram
    :return [int]: resulting LookUpTable

    >>> slic = np.array([[0] * 4 + [1] * 3 + [2] * 3 + [3] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3 + [7] * 4] * 4)
    >>> segm = np.zeros(slic.shape, dtype=int)
    >>> segm[4:, 6:] = 1
    >>> lb_hist = segm_labels_assignment(slic, segm)
    >>> assign_label_by_max(lb_hist)
    array([0, 0, 0, 0, 0, 0, 1, 1])
    """
    lut = np.zeros(max(dict_label_hist.keys()) + 1, dtype=int) - 1
    for k in dict_label_hist:
        v = dict_label_hist[k]
        counts = np.bincount(v) / float(len(v))
        lut[k] = np.argmax(counts)
    return lut


def convert_segms_2_list(segms):
    """ convert segmentation to a list tha can be simpy user for standard
    evaluation (classification or clustering metrics)

    :param [ndarray] segms: list of segmentations
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

    :param ndarray im_labeling: np.array<height, width> input labeling
    :param [int] labels: list of wanted labels to be detected in image
    :param ndarray mask_init: np.array<height, width> initial bool mask on the beginning
    :return ndarray: np.array<height, width> bool mask

    >>> img = np.zeros((4, 6))
    >>> img[:-1, 1:] = 1
    >>> img[1:2, 2:4] = 2
    >>> mask_segm_labels(img, [1])
    array([[False,  True,  True,  True,  True,  True],
           [False,  True, False, False,  True,  True],
           [False,  True,  True,  True,  True,  True],
           [False, False, False, False, False, False]], dtype=bool)
    >>> mask_segm_labels(img, [2], np.full(img.shape, True, dtype=bool))
    array([[ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True]], dtype=bool)
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

    :param ndarray labels_stack: np.array<date, height, width> input stack of labeled images
    :param {int: (int, int, int)} dict_colors: dictionary of labels-colors
    :param [int] labels_free: list of free labels
    :param int change_label: label that is set for non constant time series
    :return ndarray: np.array<height, width>

    >>> dict_colors = {0: [], 1: [], 2: []}
    >>> sequence_labels_merge(np.zeros((8, 1, 1)), dict_colors, [0])
    array([[-1]])
    >>> sequence_labels_merge(np.ones((8, 1, 1)), dict_colors, [0])
    array([[1]])
    >>> sequence_labels_merge(np.array([[1], [1], [2], [1], [1], [1], [2], [1]]), dict_colors, [0])
    array([-1])
    >>> sequence_labels_merge(np.array([[1], [0], [1], [1], [1], [1], [0], [0]]), dict_colors, [0])
    array([1])
    """
    labels_stack = np.array(labels_stack)
    im_labels = np.full(labels_stack.shape[1:], change_label, dtype=np.int)
    labels_used = [lb for lb in dict_colors if lb not in labels_free]
    lb_all = labels_used + labels_free + [change_label]
    assert all(l in lb_all for l in np.unique(labels_stack)), 'some extra labels in image stack'
    # generate mask of free labels
    mask_free = mask_segm_labels(labels_stack, labels_free)
    for lb in labels_used:
        mask1 = mask_segm_labels(labels_stack, [lb], mask_free)
        mask2 = mask_segm_labels(labels_stack, [lb])
        mask = np.logical_and(np.all(mask1, axis=0), np.any(mask2, axis=0))
        im_labels[mask] = lb
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
    assert dict_labels is not None, '"dict_labels" is required'
    labels_new = np.zeros_like(labels)
    for lb_new in dict_labels:
        for lb_old in dict_labels[lb_new]:
            labels_new[labels == lb_old] = lb_new
    return labels_new


def merge_probab_labeling_2d(proba, dict_labels):
    """ merging probability labeling

    :param ndarray proba: probabilities
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
    assert dict_labels is not None, '"dict_labels" is required'
    max_label = max(dict_labels.keys()) + 1
    size = proba.shape[:-1] + (max_label,)
    proba_new = np.zeros(size)
    for lb_new in dict_labels:
        lbs_old = dict_labels[lb_new]
        proba_new[:, :, lb_new] = np.sum(proba[:, :, lbs_old], axis=-1)
    return proba_new


def compute_labels_overlap_matrix(seg1, seg2):
    """ compute overlap between tho segmentation atlasess) with same sizes

    :param ndarray seg1: np.array<height, width>
    :param ndarray seg2: np.array<height, width>
    :return ndarray: np.array<height, width>

    >>> seg1 = np.zeros((7, 15), dtype=int)
    >>> seg1[1:4, 5:10] = 3
    >>> seg1[5:7, 6:13] = 2
    >>> seg2 = np.zeros((7, 15), dtype=int)
    >>> seg2[2:5, 7:12] = 1
    >>> seg2[4:7, 7:14] = 3
    >>> compute_labels_overlap_matrix(seg1, seg1)
    array([[76,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, 14,  0],
           [ 0,  0,  0, 15]])
    >>> compute_labels_overlap_matrix(seg1, seg2)
    array([[63,  4,  0,  9],
           [ 0,  0,  0,  0],
           [ 2,  0,  0, 12],
           [ 9,  6,  0,  0]])
    """
    logging.debug('computing overlap of two seg_pipe of shapes %s <-> %s',
                  repr(seg1.shape), repr(seg2.shape))
    assert seg1.shape == seg2.shape, 'segm (%s) and segm (%s) should match' \
                                     % (repr(seg1.shape), repr(seg2.shape))
    maxims = [np.max(seg1) + 1, np.max(seg2) + 1]
    overlap = np.zeros(maxims, dtype=int)
    for lb1, lb2 in zip(seg1.ravel(), seg2.ravel()):
        if lb1 >= 0 and lb2 >= 0:
            overlap[lb1, lb2] += 1
    # logging.debug(res)
    return overlap


def relabel_max_overlap_unique(seg_ref, seg_relabel, keep_bg=False):
    """ relabel the second segmentation cu that maximise relative overlap
    for each pattern (object), the relation among patterns is 1-1
    NOTE: it skips background class - 0

    :param ndarray seg_ref: np.array<height, width>
    :param ndarray seg_relabel: np.array<height, width>
    :return ndarray: np.array<height, width>

    >>> atlas1 = np.zeros((7, 15), dtype=int)
    >>> atlas1[1:4, 5:10] = 1
    >>> atlas1[5:7, 3:13] = 2
    >>> atlas2 = np.zeros((7, 15), dtype=int)
    >>> atlas2[0:3, 7:12] = 1
    >>> atlas2[3:7, 1:7] = 2
    >>> atlas2[4:7, 7:14] = 3
    >>> atlas2[:2, :3] = 5
    >>> relabel_max_overlap_unique(atlas1, atlas2, keep_bg=True)
    array([[5, 5, 5, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0]])
    >>> relabel_max_overlap_unique(atlas2, atlas1, keep_bg=True)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0]])
    >>> relabel_max_overlap_unique(atlas1, atlas2, keep_bg=False)
    array([[5, 5, 5, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0]])
    >>> atlas2[0, 0] = -1
    >>> relabel_max_overlap_unique(atlas1, atlas2, keep_bg=True)
    array([[-1,  5,  5,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0],
           [ 5,  5,  5,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0],
           [ 0,  3,  3,  3,  3,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  0],
           [ 0,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  0],
           [ 0,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  0]])
    """
    assert seg_ref.shape == seg_relabel.shape, \
        'Reference segm (%s) and input segm (%s) should match' \
        % (repr(seg_ref.shape), repr(seg_relabel.shape))
    overlap = compute_labels_overlap_matrix(seg_ref, seg_relabel)

    lut = [-1] * (np.max(seg_relabel) + 1)
    if keep_bg:  # keep the background label
        lut[0] = 0
        overlap[0, :] = 0
        overlap[:, 0] = 0
    # select always the maximal value and reset it
    for i in range(max(overlap.shape) + 1):
        if np.sum(overlap) == 0:
            break
        lb_ref, lb_est = np.argwhere(overlap.max() == overlap)[0]
        lut[lb_est] = lb_ref
        overlap[lb_ref, :] = 0
        overlap[:, lb_est] = 0
    # fill all not used by its equal id it is not used yet
    for i, lb in enumerate(lut):
        if lb == -1 and i not in lut:
            lut[i] = i
    # fill by any unused yet
    for i, lb in enumerate(lut):
        if lb > -1:
            continue
        for j in range(len(lut)):
            if j not in lut:
                lut[i] = j
    # lut[lut == -1] = 0

    seg_new = np.array(lut)[seg_relabel].astype(int)
    # hold all negative labels
    seg_new[seg_relabel < 0] = seg_relabel[seg_relabel < 0]
    return seg_new


def relabel_max_overlap_merge(seg_ref, seg_relabel, keep_bg=False):
    """ relabel the second segmentation cu that maximise relative overlap
    for each pattern (object), if one pattern in reference atlas is likely
    composed from multiple patterns in relabel atlas, it merge them
    NOTE: it skips background class - 0

    :param ndarray seg_ref: np.array<height, width>
    :param ndarray seg_relabel: np.array<height, width>
    :param bool keep_bg: the label 0 holds
    :return ndarray: np.array<height, width>

    >>> atlas1 = np.zeros((7, 15), dtype=int)
    >>> atlas1[1:4, 5:10] = 1
    >>> atlas1[5:7, 3:13] = 2
    >>> atlas2 = np.zeros((7, 15), dtype=int)
    >>> atlas2[0:3, 7:12] = 1
    >>> atlas2[3:7, 1:7] = 2
    >>> atlas2[4:7, 7:14] = 3
    >>> atlas2[:2, :3] = 5
    >>> relabel_max_overlap_merge(atlas1, atlas2, keep_bg=True)
    array([[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]])
    >>> relabel_max_overlap_merge(atlas2, atlas1, keep_bg=True)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0]])
    >>> relabel_max_overlap_merge(atlas1, atlas2, keep_bg=False)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0]])
    """
    assert seg_ref.shape == seg_relabel.shape, 'Ref segm (%s) and segm (%s) should match' \
                                               % (repr(seg_ref.shape), repr(seg_relabel.shape))
    overlap = compute_labels_overlap_matrix(seg_ref, seg_relabel)
    # ref_ptn_size = np.bincount(seg_ref.ravel())
    # overlap = overlap.astype(float) / np.tile(ref_ptn_size, (overlap.shape[1], 1)).T
    # overlap = np.nan_to_num(overlap)
    max_axis = 1 if overlap.shape[0] > overlap.shape[1] else 0
    if keep_bg:
        id_max = np.argmax(overlap[1:, 1:], axis=max_axis) + 1
        lut = np.array([0] + id_max.tolist())
    else:
        lut = np.argmax(overlap, axis=max_axis)
    # in case there is no overlap
    ptn_sum = np.sum(overlap, axis=0)
    if 0 in ptn_sum:
        lut[ptn_sum == 0] = np.arange(len(lut))[ptn_sum == 0]
    seg_new = lut[seg_relabel].astype(int)
    # hold all negative labels
    seg_new[seg_relabel < 0] = seg_relabel[seg_relabel < 0]
    return seg_new


def compute_boundary_distances(segm_ref, segm):
    """ compute distances among boundaries of two segmentation

    :param ndarray segm_ref:
    :param ndarray segm:
    :return ndarray:

    >>> segm_ref = np.zeros((6, 10), dtype=int)
    >>> segm_ref[3:4, 4:5] = 1
    >>> segm = np.zeros((6, 10), dtype=int)
    >>> segm[:, 2:9] = 1
    >>> pts, dist = compute_boundary_distances(segm_ref, segm)
    >>> pts
    array([[2, 4],
           [3, 3],
           [3, 4],
           [3, 5],
           [4, 4]])
    >>> dist.tolist()
    [2.0, 1.0, 2.0, 3.0, 2.0]
    """
    assert segm_ref.shape == segm.shape, 'Ref segm %s and segm %s should match'\
                                         % (repr(segm_ref.shape), repr(segm.shape))
    grid_y, grid_x = np.meshgrid(range(segm_ref.shape[1]),
                                 range(segm_ref.shape[0]))
    segr_boundary = sk_segm.find_boundaries(segm_ref, mode='thick')
    points = np.array([grid_x[segr_boundary].ravel(),
                            grid_y[segr_boundary].ravel()]).T
    segm_boundary = sk_segm.find_boundaries(segm, mode='thick')
    segm_distance = ndimage.distance_transform_edt(~segm_boundary)
    dist = segm_distance[segr_boundary].ravel()

    assert len(points) == len(dist), \
        'number of points and disntances should be equal'
    return points, dist


def assume_bg_on_boundary(segm, bg_label=0, boundary_size=1):
    """ swap labels such that the bacround label will be mostly on image boundary

    :param ndarray segm:
    :param int bg_label:
    :return:

    >>> segm = np.zeros((6, 12), dtype=int)
    >>> segm[1:4, 4:] = 2
    >>> assume_bg_on_boundary(segm, boundary_size=1)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> segm[segm == 0] = 1
    >>> assume_bg_on_boundary(segm, boundary_size=1)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    boundary_lb = tl_data.get_image2d_boundary_color(segm, size=boundary_size)
    used_lbs = np.unique(segm)
    if boundary_lb not in used_lbs:
        segm[segm == boundary_lb] = bg_label
    else:
        lut = list(range(used_lbs.max() + 1))
        lut[boundary_lb] = bg_label
        lut[bg_label] = boundary_lb
        segm = np.array(lut)[segm]
    return segm


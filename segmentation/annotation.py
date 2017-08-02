"""
Quantize annotation and so remove some noise

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging

import numpy as np
from PIL import Image
from skimage import io
from scipy import interpolate


def unique_image_colors(img):
    """ find all unique color in image and return its list

    :param ndarray img: np.array<height, width, 3>
    :return: [(int, int, int)]

    >>> np.random.seed(0)
    >>> img = np.random.randint(0, 2, (50, 50, 3))
    >>> unique_image_colors(img)  # doctest: +NORMALIZE_WHITESPACE
    [(1, 0, 0), (1, 1, 0), (0, 1, 0), (1, 1, 1),
     (0, 1, 1), (0, 0, 1), (1, 0, 1), (0, 0, 0)]
    """
    image = Image.fromarray(np.asarray(img, dtype=np.uint8))
    uq_colors = [clr for nb, clr in image.getcolors()]
    return uq_colors


def convert_img_colors_to_labels(img_rgb, dict_label_color):
    """ take a RGB image and dictionary of labels and apply this dictionary
    it returns relabels image according given dictionary

    :param ndarray img_rgb: np.array<height, width, 3> input RGB image
    :param {int: (int, int, int)} dict_label_color:
    :return ndarray: np.array<height, width> labeling

    >>> np.random.seed(0)
    >>> seg = np.random.randint(0, 2, (5, 7))
    >>> img = np.array([(0.2, 0.2, 0.2), (0.9, 0.9, 0.9)])[seg]
    >>> d_lb_clr = {0: (0.2, 0.2, 0.2), 1: (0.9, 0.9, 0.9)}
    >>> convert_img_colors_to_labels(img, d_lb_clr)
    array([[0, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 0, 0, 1],
           [0, 0, 0, 0, 0, 1, 0],
           [1, 1, 0, 0, 1, 1, 1],
           [1, 0, 1, 0, 1, 0, 1]])
    """
    dict_color_label = dict((dict_label_color[k], k) for k in dict_label_color)
    return convert_img_colors_to_labels_reverted(img_rgb, dict_color_label)


def convert_img_colors_to_labels_reverted(img_rgb, dict_color_label):
    """ take a RGB image and dictionary of labels and apply this dictionary
    it returns relabels image according given dictionary

    :param ndarray img_rgb: np.array<height, width, 3> input RGB image
    :param {(int, int, int): int} dict_color_label:
    :return ndarray: np.array<height, width> labeling

    >>> np.random.seed(0)
    >>> seg = np.random.randint(0, 2, (5, 7))
    >>> img = np.array([(0.2, 0.2, 0.2), (0.9, 0.9, 0.9)])[seg]
    >>> d_clr_lb = {(0.2, 0.2, 0.2): 0, (0.9, 0.9, 0.9): 1}
    >>> convert_img_colors_to_labels_reverted(img, d_clr_lb)
    array([[0, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 0, 0, 1],
           [0, 0, 0, 0, 0, 1, 0],
           [1, 1, 0, 0, 1, 1, 1],
           [1, 0, 1, 0, 1, 0, 1]])
    """
    img_labels = np.zeros(img_rgb.shape[:-1])
    converted_labels = 0
    for color in dict_color_label:
        class_number = dict_color_label[color]
        m_lb = np.all(img_rgb == color, axis=2)
        img_labels[m_lb] = class_number
        changed = np.bincount(m_lb.flatten())
        if len(changed) == 2:
            converted_labels += np.bincount(m_lb.flatten())[1]
    assert converted_labels == np.prod(img_labels.shape), \
        'There is different number of pixels than number of converted labels.'
    img_labels = img_labels.astype(np.int, copy=False)
    return img_labels


def convert_img_labels_to_colors(segm, dict_label_colors):
    """ convert labeling according given dictionary of colors

    :param ndarray segm: np.array<height, width>
    :param {int: (int, int, int)} dict_label_colors:
    :return ndarray: np.array<height, width, 3>

    >>> np.random.seed(0)
    >>> seg = np.random.randint(0, 2, (5, 7))
    >>> d_lb_clr = {0: (0.2, 0.2, 0.2), 1: (0.9, 0.9, 0.9)}
    >>> img = convert_img_labels_to_colors(seg, d_lb_clr)
    >>> img[:, :, 0]
    array([[ 0.2,  0.9,  0.9,  0.2,  0.9,  0.9,  0.9],
           [ 0.9,  0.9,  0.9,  0.9,  0.2,  0.2,  0.9],
           [ 0.2,  0.2,  0.2,  0.2,  0.2,  0.9,  0.2],
           [ 0.9,  0.9,  0.2,  0.2,  0.9,  0.9,  0.9],
           [ 0.9,  0.2,  0.9,  0.2,  0.9,  0.2,  0.9]])
    """
    assert all(lb in dict_label_colors.keys() for lb in np.unique(segm)), \
        'some labels %s are missing in dictionary %s' \
        % (repr(np.unique(segm)), repr(dict_label_colors.keys()))
    # init Look-Up-Table
    min_label = np.min(segm)
    nb_labels = np.max(segm) - min_label + 1
    lut = [None] * nb_labels
    # convert Look-Up-Table
    for i in range(len(lut)):
        label = i + min_label
        if label in dict_label_colors:
            lut[i] = dict_label_colors[label]
    # replace labels by colours back
    im_labels_shift = np.asarray(segm - min_label, dtype=np.int)
    im_rgb = np.array(lut)[im_labels_shift]
    return im_rgb


def image_frequent_colors(img, ratio_treshold=1e-3):
    """ look  all images and estimate most frequent colours

    :param imgs: np.array<h, w, 3>
    :param float pixel_treshold: percentage of nb clr pixels to be assumed as important
    :return:

    >>> np.random.seed(0)
    >>> img = np.random.randint(0, 2, (50, 50, 3)).astype(np.uint8)
    >>> d = image_frequent_colors(img)
    >>> sorted(d.keys()) # doctest: +NORMALIZE_WHITESPACE
    [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1),
     (1, 1, 0), (1, 1, 1)]
    >>> sorted(d.values()) # doctest: +NORMALIZE_WHITESPACE
    [271, 289, 295, 317, 318, 330, 335, 345]
    """
    if img.ndim == 3:
        img = img[:, :, :3]
    nb_pixels = np.product(img.shape[:2])
    nb_px_min = nb_pixels * ratio_treshold
    image = Image.fromarray(img)
    img_colors = image.getcolors(maxcolors=nb_pixels)
    if img_colors is None:
        return dict()
    dict_clrs = dict([(clr, nb) for nb, clr in img_colors if nb >= nb_px_min])
    ration_main_colors = sum(dict_clrs.values()) / float(nb_pixels)
    logging.debug('image main colors=%f and other=%f with colours: \n%s',
                  ration_main_colors, 1. - ration_main_colors, repr(dict_clrs))
    return dict_clrs


def dir_images_frequent_colors(paths_img, raratio_treshold=1e-3):
    """ look  all images and estimate most frequent colours

    :param imgs: [np.array<h, w, 3>]
    :param pixel_treshold: float, percentage of nb clr pixels to be assumed as important
    :return:
    """
    logging.debug('passing %i images', len(paths_img))
    dict_clrs = dict()
    for path_im in paths_img:
        img = io.imread(path_im)
        local_dict_colors = image_frequent_colors(img, raratio_treshold)
        for clr in local_dict_colors:
            if clr not in dict_clrs:
                dict_clrs[clr] = 0
            dict_clrs[clr] += local_dict_colors[clr]
    logging.info('img folder colours: %s', repr(dict_clrs))
    return dict_clrs


def image_color_2_labels(img, list_colors=None):
    """ quantize input image according given list of possible colours

    :param img: np.array<h, w, 3>, input image
    :param [(int, int, int)] list_colors: list of possible colours
    :return: np.array<h, w>

    >>> np.random.seed(0)
    >>> rand = np.random.randint(0, 2, (5, 7)).astype(np.uint8)
    >>> img = np.rollaxis(np.array([rand] * 3), 0, 3)
    >>> image_color_2_labels(img)
    array([[1, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0],
           [1, 1, 1, 1, 1, 0, 1],
           [0, 0, 1, 1, 0, 0, 0],
           [0, 1, 0, 1, 0, 1, 0]])
    """
    if list_colors is None:
        list_colors = image_frequent_colors(img).keys()
    pixels = img.reshape(-1, 3)
    dist = [np.sum(np.abs(np.subtract(pixels, clr)), axis=1) for clr in list_colors]
    lut = np.argmin(np.asarray(dist), axis=0)
    seg = lut.reshape(img.shape[:2])
    return seg


def quantize_image_nearest_color(img, list_colors):
    """ quantize input image according given list of possible colours

    :param img: np.array<h, w, 3>, input image
    :param [(int, int, int)] list_colors: list of possible colours
    :return: np.array<h, w, 3>

    >>> np.random.seed(0)
    >>> img = np.random.randint(0, 2, (5, 7, 3)).astype(np.uint8)
    >>> im = quantize_image_nearest_color(img, [(0, 0, 0), (1, 1, 1)])
    >>> im[:, :, 0]
    array([[1, 1, 1, 1, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 0],
           [1, 1, 0, 1, 1, 0, 1],
           [0, 0, 1, 0, 1, 0, 1],
           [1, 1, 1, 0, 1, 0, 0]], dtype=uint8)
    >>> [np.array_equal(im[:, :, 0], im[:, :, i]) for i in [1, 2]]
    [True, True]
    """
    pixels = img.reshape(-1, 3)
    dist = [np.sum(np.abs(np.subtract(pixels, clr)), axis=1)
            for clr in list_colors]
    lut = np.argmin(np.asarray(dist), axis=0)
    pixels = np.asarray(list_colors)[lut]
    img_q = np.asarray(pixels, dtype=img.dtype).reshape(img.shape)
    return img_q


def image_inpaint_pixels(img, valid_mask):
    assert img.shape == valid_mask.shape
    coords = np.array(np.nonzero(valid_mask)).T
    values = img[valid_mask]
    it = interpolate.NearestNDInterpolator(coords, values)
    img_paint = it(list(np.ndindex(img.shape))).reshape(img.shape)
    return img_paint


def quantize_image_nearest_pixel(img, list_colors):
    """ quantize input image according given list of possible colours

    :param img: np.array<h, w, 3>, input image
    :param [(int, int, int)] list_colors: list of possible colours
    :return: np.array<h, w, 3>

    >>> np.random.seed(0)
    >>> img = np.random.randint(0, 2, (5, 7, 3)).astype(np.uint8)
    >>> im = quantize_image_nearest_pixel(img, [(0, 0, 0), (1, 1, 1)])
    >>> im[:, :, 0]
    array([[1, 1, 1, 1, 0, 0, 0],
           [1, 1, 1, 1, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0]])
    >>> [np.array_equal(im[:, :, 0], im[:, :, i]) for i in [1, 2]]
    [True, True]
    """
    labels = np.empty(img.shape[:-1])
    labels.fill(np.nan)

    for i, clr in enumerate(list_colors):
        # male hogenious images of this single color
        color = np.tile(clr, labels.shape + (1,))
        # find different pixels
        diff = np.sum(abs(img - color), axis=-1)
        labels[diff == 0] = i

    valid_mask = ~np.isnan(labels)
    labels_inpaint = image_inpaint_pixels(labels, valid_mask).astype(int)
    img_inpaint = np.asarray(list_colors)[labels_inpaint]
    return img_inpaint


def compute_labels_overlap_matrix(seg1, seg2):
    """ compute overlap between tho segmentation atlasess) with same sizes

    :param seg1: np.array<height, width>
    :param seg2: np.array<height, width>
    :return: np.array<height, width>

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
    assert np.array_equal(seg1.shape, seg2.shape)
    maxims = [np.max(seg1) + 1, np.max(seg2) + 1]
    overlap = np.zeros(maxims, dtype=int)
    for i in range(seg1.shape[0]):
        for j in range(seg1.shape[1]):
            lb1, lb2 = seg1[i, j], seg2[i, j]
            overlap[lb1, lb2] += 1
    # logging.debug(res)
    return overlap


def relabel_max_overlap_unique(seg_ref, seg_relabel, keep_bg=False):
    """ relabel the second segmentation cu that maximise relative overlap
    for each pattern (object), the relation among patterns is 1-1
    NOTE: it skips background class - 0

    :param seg1:
    :param seg2:
    :return:

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
    """
    assert seg_ref.shape == seg_relabel.shape
    overlap = compute_labels_overlap_matrix(seg_ref, seg_relabel)

    lut = [-1] * (np.max(seg_relabel) + 1)
    if keep_bg:  # keep the background label
        lut[0] = 0
        overlap[0, :] = 0
        overlap[:, 0] = 0
    for i in range(max(overlap.shape) + 1):
        if np.sum(overlap) == 0: break
        lb_ref, lb_est = np.argwhere(overlap.max() == overlap)[0]
        lut[lb_est] = lb_ref
        overlap[lb_ref, :] = 0
        overlap[:, lb_est] = 0

    for i, lb in enumerate(lut):
        if lb == -1 and i not in lut:
            lut[i] = i
    for i, lb in enumerate(lut):
        if lb > -1: continue
        for j in range(len(lut)):
            if j not in lut:
                lut[i] = j

    seg_new = np.array(lut)[seg_relabel]
    return seg_new


def relabel_max_overlap_merge(seg_ref, seg_relabel, keep_bg=False):
    """ relabel the second segmentation cu that maximise relative overlap
    for each pattern (object), if one pattern in reference atlas is likely
    composed from multiple patterns in relabel atlas, it merge them
    NOTE: it skips background class - 0

    :param seg1:
    :param seg2:
    :return:

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
    assert seg_ref.shape == seg_relabel.shape
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
    seg_new = lut[seg_relabel]
    return seg_new

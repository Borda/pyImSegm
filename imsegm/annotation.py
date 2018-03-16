"""
Framework for handling annotations

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os, sys
import logging

import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from scipy import interpolate

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utils.data_io as tl_data
COLUMNS_POSITION = ('ant_x', 'ant_y', 'post_x', 'post_y', 'lat_x', 'lat_y')
SLICE_NAME_GROUPING = 'stack_path'
# set distance in Z axis whether near sliuce may still belong to the same egg
ANNOT_SLICE_DIST_TOL = {  # stage : distance in z-axis
    1: 1,
    2: 2,
    3: 2,
    4: 3,
    5: 3,
    6: 0,
}
# default colors for particular label
DICT_COLOURS = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (255, 229, 0),  # yellow
    4: (142, 68, 173),  # purple
    5: (127, 140, 141),  # gray
    6: (0, 212, 255),  # blue
    7: (128, 0, 0),  # brown
}


def unique_image_colors(img):
    """ find all unique color in image and return its list

    :param ndarray img: np.array<height, width, 3>
    :return: [(int, int, int)]

    >>> np.random.seed(0)
    >>> img = np.random.randint(0, 2, (50, 50, 3))
    >>> unique_image_colors(img)  # doctest: +NORMALIZE_WHITESPACE
    [(1, 0, 0), (1, 1, 0), (0, 1, 0), (1, 1, 1),
     (0, 1, 1), (0, 0, 1), (1, 0, 1), (0, 0, 0)]
    >>> img = np.random.randint(0, 256, (150, 150, 3))
    >>> unique_image_colors(img)  # doctest: +ELLIPSIS
    [...]
    """
    image = Image.fromarray(np.asarray(img, dtype=np.uint8))
    colors = image.convert('RGB').getcolors()
    if colors is None:
        logging.warning('selected image contains more then 256 colors')
        nb_pixels = np.prod(img.shape[:2])
        colors = image.convert('RGB').getcolors(maxcolors=nb_pixels)
    uq_colors = [clr for nb, clr in colors]
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
    for i, _ in enumerate(lut):
        label = i + min_label
        if label in dict_label_colors:
            lut[i] = dict_label_colors[label]
    # replace labels by colours back
    im_labels_shift = np.asarray(segm - min_label, dtype=np.int)
    im_rgb = np.array(lut)[im_labels_shift]
    return im_rgb


def image_frequent_colors(img, ratio_treshold=1e-3):
    """ look  all images and estimate most frequent colours

    :param ndarray img: np.array<h, w, 3>
    :param float ratio_treshold: percentage of nb color pixels to be assumed
        as important
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


def dir_images_frequent_colors(paths_img, ratio_treshold=1e-3):
    """ look  all images and estimate most frequent colours

    :param paths_img: [np.array<h, w, 3>]
    :param ratio_treshold: float, percentage of nb clr pixels to be assumed
        as important
    :return:
    """
    logging.debug('passing %i images', len(paths_img))
    dict_colors = dict()
    for path_im in paths_img:
        img = tl_data.io_imread(path_im)
        local_dict_colors = image_frequent_colors(img, ratio_treshold)
        for clr in local_dict_colors:
            if clr not in dict_colors:
                dict_colors[clr] = 0
            dict_colors[clr] += local_dict_colors[clr]
    logging.info('img folder colours: %s', repr(dict_colors))
    return dict_colors


def image_color_2_labels(img, list_colors=None):
    """ quantize input image according given list of possible colours

    :param ndarray img: np.array<h, w, 3>, input image
    :param [(int, int, int)] list_colors: list of possible colours
    :return ndarray: np.array<h, w>

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
    dist = [np.sum(np.abs(np.subtract(pixels, clr)), axis=1)
            for clr in list_colors]
    lut = np.argmin(np.asarray(dist), axis=0)
    seg = lut.reshape(img.shape[:2])
    return seg


def quantize_image_nearest_color(img, list_colors):
    """ quantize input image according given list of possible colours

    :param ndarray img: np.array<h, w, 3>, input image
    :param [(int, int, int)] list_colors: list of possible colours
    :return ndarray: np.array<h, w, 3>

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
    assert img.shape == valid_mask.shape, \
        'image size %s and mask size %s should be equal' \
        % (repr(img.shape), repr(valid_mask.shape))
    coords = np.array(np.nonzero(valid_mask)).T
    values = img[valid_mask]
    it = interpolate.NearestNDInterpolator(coords, values)
    img_paint = it(list(np.ndindex(img.shape))).reshape(img.shape)
    return img_paint


def quantize_image_nearest_pixel(img, list_colors):
    """ quantize input image according given list of possible colours

    :param ndarray img: np.array<h, w, 3>, input image
    :param [(int, int, int)] list_colors: list of possible colours
    :return ndarray: np.array<h, w, 3>

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
        # male homogenious images of this single color
        color = np.tile(clr, labels.shape + (1,))
        # find different pixels
        diff = np.sum(abs(img - color), axis=-1)
        labels[diff == 0] = i

    valid_mask = ~np.isnan(labels)
    labels_inpaint = image_inpaint_pixels(labels, valid_mask).astype(int)
    img_inpaint = np.asarray(list_colors)[labels_inpaint]
    return img_inpaint


def load_info_group_by_slices(path_txt, stages, pos_columns=COLUMNS_POSITION,
                              dict_slice_tol=ANNOT_SLICE_DIST_TOL):
    """ load all info and group position info according name if stack

    :param str path_txt:
    :param [str] pos_columns:
    :return: DF

    >>> path_txt = os.path.join(tl_data.update_path('images'),
    ...                 'drosophila_ovary_slice', 'info_ovary_images.txt')
    >>> load_info_group_by_slices(path_txt, [4]) # doctest: +NORMALIZE_WHITESPACE
                ant_x  ant_y  lat_x  lat_y post_x post_y
    image
    insitu7569  [298]  [327]  [673]  [411]  [986]  [155]
    """
    logging.info('loading info file and filter stages...')
    df = pd.DataFrame().from_csv(path_txt, sep='\t')
    logging.debug('loaded %i records', len(df))
    df = df[df['stage'].isin(list(stages))]
    logging.debug('filtered %i records', len(df))

    # solving issue with different pandas versiona
    if hasattr(df, 'sort_values'):
        df.sort_values(['stage'], ascending=False, inplace=True)
    elif hasattr(df, 'sort'):
        df.sort(['stage'], ascending=False, inplace=True)

    df_marked = pd.DataFrame()
    logging.info('grouping info by stacks...')
    tqdm_bar = tqdm.tqdm(total=len(df[SLICE_NAME_GROUPING].unique()))
    for _, df_group in df.groupby(SLICE_NAME_GROUPING):
        slice_idxs = df_group['slice_index'].values
        slice_tols = np.array([dict_slice_tol[i]
                               for i in df_group['stage'].values])
        for _, row in df_group.iterrows():
            sl_idx = row['slice_index']
            diff = abs(slice_idxs - sl_idx)
            filter_slice = (diff <= slice_tols)
            dict_slice = {col: df_group[col].values[filter_slice]
                          for col in pos_columns}
            dict_slice['image'] = os.path.splitext(row['image_path'])[0]
            df_marked = df_marked.append(dict_slice, ignore_index=True)
        tqdm_bar.update()
    if len(df_marked) > 0:
        df_marked.set_index('image', inplace=True)
    return df_marked

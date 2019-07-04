"""
Framework for visualisations

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import logging

import matplotlib
if os.environ.get('DISPLAY', '') == '' and matplotlib.rcParams['backend'] != 'agg':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import draw, color, segmentation
from planar import line as pl_line

SIZE_CHESS_FIELD = 50
# columns from description files which marks the egg annotation by expert
COLUMNS_POSITION_EGG_ANNOT = ['ant_x', 'ant_y',
                              'post_x', 'post_y',
                              'lat_x', 'lat_y']
# http://matplotlib.org/examples/color/colormaps_reference.html
# http://htmlcolorcodes.com/
COLOR_ORANGE = '#FF5733'
COLOR_GRAY = '#7E7E7E'
COLOR_GREEN = '#1FFF00'
COLOR_YELLOW = '#FFFB00'
COLOR_PINK = '#FF00FF'
COLOR_BLUE = '#00AAFF'
COLORS = 'bgrmyck'

DICT_LABEL_MARKER = {
    -1: ('.', COLOR_GRAY),
    0: ('x', COLOR_GRAY),
    1: ('.', COLOR_YELLOW),
}
DICT_LABEL_MARKER_FN_FP = {
    -2: ('.', COLOR_PINK),
    -1: ('.', COLOR_BLUE),
    0: ('x', 'w'),
    1: ('.', COLOR_YELLOW),
}


def _ellipse(r, c, r_radius, c_radius, orientation=0., shape=None):
    """ temporary wrapper until release New version scikit-image v0.13

    :param int r: center position in rows
    :param int c: center position in columns
    :param int r_radius: ellipse diam in rows
    :param int c_radius: ellipse diam in columns
    :param float orientation: ellipse orientation
    :param tuple(int,int) shape: size of output mask
    :return tuple(list(int),list(int)): indexes of filled positions

    >>> img = np.zeros((10, 12), dtype=int)
    >>> rr, cc = _ellipse(5, 6, 3, 5, orientation=np.deg2rad(30))
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    center = np.array([r, c])
    radii = np.array([r_radius, c_radius])
    # allow just rotation with in range +/- 180 degree
    orientation %= np.pi

    sin_alpha, cos_alpha = np.sin(orientation), np.cos(orientation)
    # compute rotated radii by given rotation
    r_radius_rot = abs(r_radius * cos_alpha) + c_radius * sin_alpha
    c_radius_rot = r_radius * sin_alpha + abs(c_radius * cos_alpha)
    # The upper_left and lower_right corners of the smallest rectangle
    # containing the ellipse.
    radii_rot = np.array([r_radius_rot, c_radius_rot])
    upper_left = np.ceil(center - radii_rot).astype(int)
    lower_right = np.floor(center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:2]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    r_lim, c_lim = np.ogrid[0:int(bounding_shape[0]),
                            0:int(bounding_shape[1])]
    r_org, c_org = shifted_center
    r_rad, c_rad = radii
    r, c = (r_lim - r_org), (c_lim - c_org)
    dist_1 = ((r * cos_alpha + c * sin_alpha) / r_rad) ** 2
    dist_2 = ((r * sin_alpha - c * cos_alpha) / c_rad) ** 2
    rr, cc = np.nonzero((dist_1 + dist_2) <= 1)

    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc


def ellipse(r, c, r_radius, c_radius, orientation=0., shape=None):
    """ temporary wrapper until release New version scikit-image v0.13

    .. note:: Should be solved in skimage v0.13

    :param int r: center position in rows
    :param int c: center position in columns
    :param int r_radius: ellipse diam in rows
    :param int c_radius: ellipse diam in columns
    :param float orientation: ellipse orientation
    :param tuple(int,int) shape: size of output mask
    :return tuple(list(int),list(int)): indexes of filled positions

    >>> img = np.zeros((14, 20), dtype=int)
    >>> rr, cc = ellipse(7, 10, 3, 9, np.deg2rad(30), img.shape)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    rr, cc = draw.ellipse(r, c, r_radius, c_radius,
                          rotation=orientation, shape=shape)
    # alternative version
    # rr, cc = _ellipse(r, c, r_radius, c_radius, orientation, shape)
    return rr, cc


def ellipse_perimeter(r, c, r_radius, c_radius, orientation=0., shape=None):
    """ see New version scikit-image v0.14

    .. note:: Should be solved in skimage v0.14

    :param int r: center position in rows
    :param int c: center position in columns
    :param int r_radius: ellipse diam in rows
    :param int c_radius: ellipse diam in columns
    :param float orientation: ellipse orientation
    :param tuple(int,int) shape: size of output mask
    :return tuple(list(int),list(int)): indexes of filled positions

    >>> img = np.zeros((14, 20), dtype=int)
    >>> rr, cc = ellipse_perimeter(7, 10, 3, 9, np.deg2rad(30), img.shape)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    rr, cc = draw.ellipse_perimeter(r, c, r_radius, c_radius,
                                    orientation=-orientation, shape=shape)
    return rr, cc


def norm_aplha(alpha):
    """ normalise alpha in range (0, 1)

    :param float alpha:
    :return float:

    >>> norm_aplha(0.5)
    0.5
    >>> norm_aplha(255)
    1.0
    >>> norm_aplha(-1)
    0
    """
    alpha = alpha / 255. if alpha > 1. else alpha
    alpha = 0 if alpha < 0. else alpha
    alpha = 1. if alpha > 1. else alpha
    return alpha


def figure_image_adjustment(fig, img_size):
    """ adjust figure as nice image without axis

    :param fig: Figure
    :param tuple(int,int) img_size: image size
    :return Figure:

    >>> fig = figure_image_adjustment(plt.figure(), (150, 200))
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    ax = fig.gca()
    ax.set(xlim=[0, img_size[1]], ylim=[img_size[0], 0])
    ax.axis('off')
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def figure_image_segm_results(img, seg, subfig_size=9, mid_labels_alpha=0.2,
                              mid_image_gray=True):
    """ creating subfigure with original image, overlapped segmentation contours
    and clean result segmentation...
    it turns the sequence in vertical / horizontal according major image dim

    :param ndarray img: image as background
    :param ndarray seg: segmentation
    :param int subfig_size: max image size
    :param fool mid_image_gray: used color image as bacround in middele
    :param float mid_labels_alpha: alpha for middle segmentation overlap
    :return Figure:

    >>> img = np.random.random((100, 150, 3))
    >>> seg = np.random.randint(0, 2, (100, 150))
    >>> fig = figure_image_segm_results(img, seg)
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    assert img.shape[:2] == seg.shape[:2], \
        'different image %r & seg_pipe %r sizes' % (img.shape, seg.shape)
    if img.ndim == 2:  # for gray images of ovary
        # img = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)
        img = color.gray2rgb(img)

    fig, axarr = create_figure_by_image(img.shape[:2], subfig_size,
                                        nb_subfigs=3)
    axarr[0].set_title('original image')
    axarr[0].imshow(img)

    # visualise the 3th label
    axarr[1].set_title('original image w. segment overlap')
    img_bg = color.rgb2gray(img) if mid_image_gray else img
    axarr[1].imshow(img_bg, cmap=plt.cm.Greys_r)
    axarr[1].imshow(seg, alpha=mid_labels_alpha, cmap=plt.cm.jet)
    axarr[1].contour(seg, levels=np.unique(seg), linewidths=2, cmap=plt.cm.jet)

    axarr[2].set_title('segmentation - all labels')
    axarr[2].imshow(seg, cmap=plt.cm.jet)

    for ax in axarr:
        ax.axis('off')
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])

    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.tight_layout()
    return fig


def figure_overlap_annot_segm_image(annot, segm, img=None, subfig_size=9,
                                    drop_labels=None, segm_alpha=0.2):
    """ figure showing overlap annotation - segmentation - image

    :param ndarray annot: user annotation
    :param ndarray segm: segmentation
    :param ndarray img: original image
    :param int subfig_size: maximal sub-figure size
    :param float segm_alpha: use transparency
    :param list(int) drop_labels: labels to be ignored
    :return Figure:

    >>> img = np.random.random((100, 150, 3))
    >>> seg = np.random.randint(0, 2, (100, 150))
    >>> fig = figure_overlap_annot_segm_image(seg, seg, img, drop_labels=[5])
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    norm_size = np.array(annot.shape) / float(np.max(annot.shape))
    fig_size = norm_size[::-1] * subfig_size * np.array([3, 1])
    fig, axarr = plt.subplots(ncols=3, figsize=fig_size)

    if img is None:
        img = np.ones(annot.shape)
    if img.ndim == 2:  # for gray images of ovary
        img = color.gray2rgb(img)

    axarr[0].set_title('Annotation')
    axarr[0].imshow(img)
    axarr[0].imshow(annot, alpha=segm_alpha)
    axarr[0].contour(annot, levels=np.unique(annot), linewidths=2)

    axarr[1].set_title('Segmentation')
    axarr[1].imshow(img)
    axarr[1].imshow(segm, alpha=segm_alpha)
    axarr[1].contour(segm, levels=np.unique(segm), linewidths=2)

    # visualise the 3th label
    axarr[2].set_title('difference: annot. & segment')
    # axarr[2].imshow(~(annot == segm), cmap=plt.cm.Reds)
    max_val = np.max(annot.astype(int))
    diff = annot - segm
    if drop_labels is not None:
        for lb in drop_labels:
            diff[annot == lb] = 0
    cax = axarr[2].imshow(diff, vmin=-max_val, vmax=max_val, alpha=0.5,
                          cmap=plt.cm.bwr)
    # vals = np.linspace(-max_val, max_val, max_val * 2 + 1)
    plt.colorbar(cax, ticks=np.linspace(-max_val, max_val, max_val * 2 + 1),
                 boundaries=np.linspace(-max_val - 0.5, max_val + 0.5,
                                        max_val * 2 + 2))
    # plt.clim(-max_val - 0.5, max_val - 0.5)
    # axarr[2].contour(annot, levels=np.unique(annot), linewidths=1, colors='g')
    # axarr[2].contour(segm, levels=np.unique(segm), linewidths=1, colors='b')

    for i in range(len(axarr)):
        axarr[i].axis('off')
        axarr[i].axes.get_xaxis().set_ticklabels([])
        axarr[i].axes.get_yaxis().set_ticklabels([])

    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.tight_layout()
    return fig


def figure_segm_graphcut_debug(images, subfig_size=9):
    """ creating subfigure with slic, graph edges and results in the first row
    and individual class unary terms in the second row

    :param dict images: dictionary composed from name and image array
    :param int subfig_size: maximal sub-figure size
    :return Figure:

    >>> images = {
    ...     'image': np.random.random((100, 150, 3)),
    ...     'slic': np.random.randint(0, 2, (100, 150)),
    ...     'slic_mean': np.random.random((100, 150, 3)),
    ...     'img_graph_edges': np.random.random((100, 150, 3)),
    ...     'img_graph_segm': np.random.random((100, 150, 3)),
    ...     'imgs_unary_cost': [np.random.random((100, 150, 3))],
    ... }
    >>> fig = figure_segm_graphcut_debug(images)
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    assert all(n in images for n in [
        'image', 'slic', 'slic_mean', 'img_graph_edges', 'img_graph_segm', 'imgs_unary_cost'
    ]), 'missing keys in debug structure %r' % tuple(images.keys())
    nb_cols = max(3, len(images['imgs_unary_cost']))
    img = images['image']
    if img.ndim == 2:  # for gray images of ovary
        img = color.gray2rgb(img)
    norm_size = np.array(img.shape[:2]) / float(np.max(img.shape))

    fig_size = norm_size[::-1] * subfig_size * np.array([nb_cols, 2])
    fig, axarr = plt.subplots(2, nb_cols, figsize=fig_size)

    img_slic = segmentation.mark_boundaries(img, images['slic'],
                                            mode='subpixel')
    axarr[0, 0].set_title('SLIC')
    axarr[0, 0].imshow(img_slic)
    for i, k in enumerate(['img_graph_edges', 'img_graph_segm']):
        axarr[0, i + 1].set_title(k)
        axarr[0, i + 1].imshow(images[k])
    for i, im_uc in enumerate(images['imgs_unary_cost']):
        axarr[1, i].set_title('unary cost #%i' % i)
        axarr[1, i].imshow(im_uc)

    for j in range(2):
        for i in range(nb_cols):
            axarr[j, i].axis('off')
            axarr[j, i].axes.get_xaxis().set_ticklabels([])
            axarr[j, i].axes.get_yaxis().set_ticklabels([])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0.05, hspace=0.05)
    return fig


def create_figure_by_image(img_size, subfig_size, nb_subfigs=1, extend=0.):
    """ crearting image according backround_image

    :param tuple(int,int) img_size: image size
    :param float subfig_size: maximal sub-figure size
    :param int nb_subfigs: number of sub-figure
    :param float extend: extension
    :return tuple(Figure,list):
    """
    norm_size = np.array(img_size) / float(np.max(img_size))
    # reverse dimensions and scale by fig size
    if norm_size[0] >= norm_size[1]:  # horizontal
        fig_size = norm_size[::-1] * subfig_size * np.array([nb_subfigs, 1])
        fig_size[0] += extend * fig_size[0]
        fig, axarr = plt.subplots(ncols=nb_subfigs, figsize=fig_size)
    else:  # vertical
        fig_size = norm_size[::-1] * subfig_size * np.array([1, nb_subfigs])
        fig_size[0] += extend * fig_size[0]
        fig, axarr = plt.subplots(nrows=nb_subfigs, figsize=fig_size)
    return fig, axarr


def figure_ellipse_fitting(img, seg, ellipses, centers, crits, fig_size=9):
    """ show figure with result of the ellipse fitting

    :param ndarray img: image
    :param ndarray seg: segmentation
    :param list(tuple(int,int,int,int,float)) ellipses: collection of ellipse parameters
        ell. parameters: (x, y, height, width, orientation)
    :param list(tuple(int,int)) centers: points
    :param list(float) crits:
    :param float fig_size: maximal figure size
    :return Figure:

    >>> img = np.random.random((100, 150, 3))
    >>> seg = np.random.randint(0, 2, (100, 150))
    >>> ells = np.random.random((3, 5)) * 25
    >>> centers = np.random.random((3, 2)) * 25
    >>> crits = np.random.random(3)
    >>> fig = figure_ellipse_fitting(img[:, :, 0], seg, ells, centers, crits)
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    assert len(ellipses) == len(centers) == len(crits), \
        'number of ellipses (%i) and centers (%i) and criteria (%i) ' \
        'should match' % (len(ellipses), len(centers), len(crits))

    fig, ax = create_figure_by_image(img.shape[:2], fig_size)
    assert img.ndim == 2, \
        'required image dimension is 2 to instead %r' % img.shape
    ax.imshow(img, cmap=plt.cm.Greys_r)

    for i, params in enumerate(ellipses):
        c1, c2, h, w, phi = params
        rr, cc = ellipse_perimeter(int(c1), int(c2), int(h), int(w), phi)
        ax.plot(cc, rr, '.', color=COLORS[i % len(COLORS)],
                label='#%i with crit=%d' % ((i + 1), int(crits[i])))
    ax.legend(loc='lower right')

    # plt.plot(centers[:, 1], centers[:, 0], 'ow')
    for i in range(len(centers)):
        ax.plot(centers[i, 1], centers[i, 0], 'o',
                color=COLORS[i % len(COLORS)])
    ax.set(xlim=[0, seg.shape[1]], ylim=[seg.shape[0], 0])
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def figure_annot_slic_histogram_labels(dict_label_hist, slic_size=-1, slic_regul=-1):
    """ plot ration of labels  assigned to each superpixel

    :param dict_label_hist: dictionary of label name and histogram
    :param int slic_size: used for figure title
    :param float slic_regul: used for figure title
    :return Figure:

    >>> np.random.seed(0)
    >>> dict_label_hist = {'a': np.tile([1, 0, 0, 0, 1], (25, 1)),
    ...                    'b': np.tile([0, 1, 0, 0, 1], (30, 1))}
    >>> fig = figure_annot_slic_histogram_labels(dict_label_hist)
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    matrix_hist_all = np.concatenate(tuple(dict_label_hist.values()), axis=0)
    lb_sums = np.sum(matrix_hist_all, axis=0)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    for i, nb in enumerate(lb_sums):
        if nb == 0:
            continue
        patches, bin_edges = np.histogram(matrix_hist_all[:, i], bins=50,
                                          density=True)
        bins = [(a + b) / 2. for a, b in zip(bin_edges[:-1], bin_edges[1:])]
        # ax.plot(bins, patches, label='label: %i' % i)
        ax.semilogy(bins, patches, label='label: %i' % i)
    ax.set_title('Histogram of labels density in each segments '
                 'over all annotated images\n (superpixels: size=%i, regul=%f)'
                 % (slic_size, slic_regul))
    ax.set(xlabel='region densities', ylabel='[%]')
    ax.legend()
    ax.grid()
    return fig


def figure_ray_feature(segm, points, ray_dist_raw=None, ray_dist=None,
                       points_reconst=None, title=''):
    """ visualise the segmentation with specific point and estimated ray dist.

    :param ndarray segm: segmentation
    :param [(float, float)] points: collection of points
    :param list(float) ray_dist_raw:
    :param list(float) ray_dist: Ray feature distances
    :param ndarray points_reconst: collection of reconstructed points
    :param str title: figure title
    :return Figure:

    .. note:: for more examples, see unittests
    """
    ray_dist_raw = ray_dist_raw if ray_dist_raw is not None else []
    ray_dist = ray_dist if ray_dist is not None else []

    fig, axarr = plt.subplots(nrows=2, ncols=1)
    if title:
        axarr[0].set_title(title)
    axarr[0].imshow(1 - segm, cmap='gray', interpolation='nearest')
    axarr[0].plot(points[1], points[0], 'bo')
    axarr[0].set(xlim=[0, segm.shape[1]], ylim=[segm.shape[0], 0])
    if points_reconst is not None:
        axarr[0].plot(points_reconst[:, 1], points_reconst[:, 0], 'g.')
    axarr[1].plot(np.linspace(0, 360, len(ray_dist_raw)).tolist(),
                  ray_dist_raw, 'b', label='original')
    axarr[1].plot(np.linspace(0, 360, len(ray_dist)).tolist(),
                  ray_dist, 'r', label='final')
    axarr[1].set(xlabel='angles [deg]', xlim=[0, 360])
    axarr[1].legend(loc=0)
    axarr[1].grid()
    return fig


def figure_used_samples(img, labels, slic, used_samples, fig_size=12):
    """ draw used examples (superpixels)

    :param ndarray img: input image for background
    :param list(int) labels: labels associated for superpixels
    :param ndarray slic: superpixel segmentation
    :param list(bool) used_samples: used samples for training
    :param int fig_size: figure size
    :return Figure:

    >>> img = np.random.random((50, 75, 3))
    >>> labels = [-1, 0, 2]
    >>> used = [1, 0, 0]
    >>> seg = np.random.randint(0, 3, img.shape[:2])
    >>> fig = figure_used_samples(img, labels, seg, used)
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    w_samples = np.asarray(used_samples)[slic]
    img = color.gray2rgb(img) if img.ndim == 2 else img

    fig, axarr = create_figure_by_image(img.shape[:2], fig_size,
                                        nb_subfigs=2, extend=0.15)
    axarr[0].imshow(np.asarray(labels)[slic], cmap=plt.cm.jet)
    axarr[0].contour(slic, levels=np.unique(slic), colors='w', linewidths=0.5)
    axarr[0].axis('off')

    axarr[1].imshow(img)
    axarr[1].contour(slic, levels=np.unique(slic), colors='w', linewidths=0.5)
    cax = axarr[1].imshow(w_samples, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, alpha=0.5)
    cbar = plt.colorbar(cax, ticks=[0, 1], boundaries=[-0.5, 0.5, 1.5])
    cbar.ax.set_yticklabels(['drop', 'used'])
    axarr[1].axis('off')

    fig.tight_layout()
    return fig


def draw_color_labeling(segments, lut_labels):
    """ visualise the graph cut results

    :param ndarray segments: np.array<height, width>
    :param list(int) lut_labels: look-up-table
    :return ndarray: np.array<height, width, 3>
    """
    seg = np.asarray(lut_labels)[segments]
    clrs = plt.get_cmap('jet')
    lbs = np.arange(np.max(seg) + 1)
    lut = clrs(lbs / float(lbs.max()))[:, :3]
    img = lut[seg]
    return img


def draw_graphcut_unary_cost_segments(segments, unary_cost):
    """ visualise the unary cost for each class

    :param ndarray segments: np.array<height, width>
    :param ndarray unary_cost: np.array<nb_spx, nb_classes>
    :return []: [np.array<height, width, 3>] * nb_cls

    >>> seg = np.random.randint(0, 100, (100, 150))
    >>> u_cost = np.random.random((100, 3))
    >>> imgs = draw_graphcut_unary_cost_segments(seg, u_cost)
    >>> len(imgs)
    3
    >>> [img.shape for img in imgs]
    [(100, 150, 3), (100, 150, 3), (100, 150, 3)]
    """
    clrs = plt.get_cmap('Greens')
    imgs_u_cost = [None] * unary_cost.shape[-1]
    for i in range(unary_cost.shape[-1]):
        pw_c_norm = 1 - (unary_cost[:, i] / unary_cost.max())
        lut = np.asarray([clrs(p) for p in pw_c_norm])[:, :3]
        imgs_u_cost[i] = lut[segments]
    return imgs_u_cost


def closest_point_on_line(start, end, point):
    """ projection of the point to the line

    :param list(int) start: line starting point
    :param list(int) end: line ending point
    :param list(int) point: point for extimation
    :return list(int): point on the line

    >>> closest_point_on_line([0, 0], [1, 2], [0, 2])
    array([ 0.8,  1.6])
    """
    start, end, point = [np.array(a) for a in [start, end, point]]
    line = pl_line.Line(start, (end - start))
    proj = np.array(line.project(point))
    return proj


def draw_eggs_ellipse(mask_shape, pos_ant, pos_lat, pos_post,
                      threshold_overlap=0.6):
    """ from given 3 point estimate the ellipse

    :param tuple(int,int) mask_shape:
    :param [tuple(int,int)] pos_ant: anterior
    :param [tuple(int,int)] pos_lat: latitude
    :param [tuple(int,int)] pos_post: postlude
    :param float threshold_overlap:
    :return ndarray:

    >>> pos_ant, pos_lat, pos_post = [10, 10], [20, 20], [35, 20]
    >>> points = np.array([pos_ant, pos_lat, pos_post])
    >>> _= plt.plot(points[:, 0], points[:, 1], 'og')
    >>> mask = draw_eggs_ellipse([30, 50], [pos_ant], [pos_lat], [pos_post])
    >>> mask.shape
    (30, 50)
    >>> _= plt.imshow(mask, alpha=0.5, interpolation='nearest')
    >>> _= plt.xlim([0, mask.shape[1]]), plt.ylim([0, mask.shape[0]]), plt.grid()
    >>> # plt.show()
    """
    mask_eggs = np.zeros(mask_shape)
    for i, (ant, lat, post) in enumerate(zip(pos_ant, pos_lat, pos_post)):
        ant, lat, post = map(np.array, [ant, lat, post])
        center = ant + (post - ant) / 2.
        lat_proj = closest_point_on_line(ant, post, lat)
        # http://stackoverflow.com/questions/433371/ellipse-bounding-a-rectangle
        radius_a = (np.linalg.norm(post - ant) / 2. / np.sqrt(2)) * 1.
        radius_b = (np.linalg.norm(lat - lat_proj) / np.sqrt(2)) * 1.
        angle = np.arctan2(*(post - ant))
        rr, cc = ellipse(int(center[1]), int(center[0]),
                         int(radius_a), int(radius_b),
                         orientation=angle, shape=mask_eggs.shape)
        mask = np.zeros(mask_shape)
        mask[rr, cc] = True

        # mask = ndimage.morphology.binary_fill_holes(mask)
        # distance = ndimage.distance_transform_edt(mask)
        # probab = distance / np.max(distance)
        # mask = probab >= threshold_dist

        m_overlap = np.sum(np.logical_and(mask > 0, mask_eggs > 0)) / float(np.sum(mask))
        if m_overlap > threshold_overlap:
            logging.debug('skip egg drawing while it overlap by %f', m_overlap)
            continue
        mask_eggs[mask.astype(bool)] = i + 1

    return mask_eggs


def parse_annot_rectangles(rows_slice):
    """ parse annotation fromDF to lists

    :param rows_slice: a row from a table
    :return tuple: the three points

    >>> import pandas as pd
    >>> dict_row = dict(ant_x=1, ant_y=2, lat_x=3, lat_y=4, post_x=5, post_y=6)
    >>> row = pd.DataFrame([dict_row])
    >>> parse_annot_rectangles(row)
    ([(1, 2)], [(3, 4)], [(5, 6)])
    >>> rows = pd.DataFrame([dict_row, {n: dict_row[n] + 10 for n in dict_row}])
    >>> rows
       ant_x  ant_y  lat_x  lat_y  post_x  post_y
    0      1      2      3      4       5       6
    1     11     12     13     14      15      16
    >>> parse_annot_rectangles(rows)
    ([(1, 2), (11, 12)], [(3, 4), (13, 14)], [(5, 6), (15, 16)])
    """
    dict_eggs = {col: rows_slice[col] for col in COLUMNS_POSITION_EGG_ANNOT}
    if all(isinstance(dict_eggs[col], str) for col in dict_eggs):
        dict_eggs = {col: map(int, dict_eggs[col][1:-1].lstrip().split())
                     for col in dict_eggs}

    pos_ant = list(zip(dict_eggs['ant_x'], dict_eggs['ant_y']))
    pos_lat = list(zip(dict_eggs['lat_x'], dict_eggs['lat_y']))
    pos_post = list(zip(dict_eggs['post_x'], dict_eggs['post_y']))

    return pos_ant, pos_lat, pos_post


def draw_eggs_rectangle(mask_shape, pos_ant, pos_lat, pos_post):
    """ from given 3 point estimate the ellipse

    :param tuple(int,int) mask_shape: segmentation size
    :param [tuple(int,int)] pos_ant: points
    :param [tuple(int,int)] pos_lat: points
    :param [tuple(int,int)] pos_post: points
    :return [ndarray]:

    >>> pos_ant, pos_lat, pos_post = [10, 10], [20, 20], [35, 20]
    >>> points = np.array([pos_ant, pos_lat, pos_post])
    >>> _= plt.plot(points[:, 0], points[:, 1], 'og')
    >>> masks = draw_eggs_rectangle([30, 50], [pos_ant], [pos_lat], [pos_post])
    >>> [m.shape for m in masks]
    [(30, 50)]
    >>> for mask in masks:
    ...     _= plt.imshow(mask, alpha=0.5, interpolation='nearest')
    >>> _= plt.xlim([0, mask.shape[1]]), plt.ylim([0, mask.shape[0]]), plt.grid()
    >>> # plt.show()
    """
    list_masks = []
    pos_ant, pos_lat, pos_post = list(pos_ant), list(pos_lat), list(pos_post)
    for ant, lat, post in zip(pos_ant, pos_lat, pos_post):
        ant, lat, post = map(np.array, [ant, lat, post])
        lat_proj = closest_point_on_line(ant, post, lat)
        shift = lat - lat_proj
        # center = ant + (post - ant) / 2.
        # dist = np.linalg.norm(shift)
        # angle = np.arctan2(*(post - ant))
        points = np.array([ant + shift, ant - shift,
                           post - shift, post + shift,
                           ant + shift])
        rr, cc = draw.polygon(points[:, 1], points[:, 0], shape=mask_shape)
        mask = np.zeros(mask_shape)
        mask[rr, cc] = True
        list_masks.append(mask)

    return list_masks


def merge_object_masks(masks, overlap_thr=0.7):
    """ merge several mask into one multi-class segmentation

    :param [ndarray] masks: collection of masks
    :param float overlap_thr: threshold for overlap
    :return ndarray:

    >>> m1 = np.zeros((5, 6), dtype=int)
    >>> m1[:4, :4] = 1
    >>> m2 = np.zeros((5, 6), dtype=int)
    >>> m2[2:, 2:] = 1
    >>> merge_object_masks([m1, m1])
    array([[1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> merge_object_masks([m1, m2])
    array([[1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0, 0],
           [1, 1, 2, 2, 2, 2],
           [1, 1, 2, 2, 2, 2],
           [0, 0, 2, 2, 2, 2]])
    """
    assert len(masks) > 0, 'no masks are given'
    mask = np.array(masks[0])

    for i in range(1, len(masks)):
        overlap_ratios = []
        for j in range(1, int(np.max(mask) + 1)):
            overlap = np.sum(np.logical_and(mask == j, masks[i] == 1))
            union = np.sum(np.logical_or(mask == j, masks[i] == 1))
            overlap_ratios.append(float(overlap) / float(union))
        if any(r > overlap_thr for r in overlap_ratios):
            logging.debug('skip egg drawing while it overlap by %r', overlap_ratios)
            continue
        mask[masks[i] == 1] = np.max(mask) + 1

    return mask


def draw_image_segm_points(ax, img, points, labels=None, slic=None,
                           color_slic='w', lut_label_marker=DICT_LABEL_MARKER,
                           seg_contour=None):
    """ on plane draw background image or segmentation, overlap with slic
    contours, add contour of adative segmentation like annot. for centers
    plot point with specific property (shape and colour) according label

    :param ax: figure axis
    :param ndarray img: image
    :param list(tuple(int,int)) points:collection of points
    :param list(int) labels: LUT labels for superpixels
    :param ndarray slic: superpixel segmentation
    :param str color_slic: color dor superpixels
    :param dict lut_label_marker: dictionary {int: (str, str)} of label and markers
    :param ndarray seg_contour: segmentation contour

    >>> img = np.random.randint(0, 256, (100, 100))
    >>> points = np.random.randint(0, 100, (25, 2))
    >>> labels = np.random.randint(0, 5, len(points))
    >>> slic = np.random.randint(0, 256, (100, 100))
    >>> draw_image_segm_points(plt.Figure().gca(), img, points, labels, slic)
    """
    # background image or segmentation
    if img.ndim == 2:
        ax.imshow(img, alpha=0.3, cmap=plt.cm.gist_earth)
    else:
        ax.imshow(img)

    if slic is not None:
        ax.contour(slic, levels=np.unique(slic), alpha=0.5, colors=color_slic,
                   linewidths=0.5)
    # fig.gca().imshow(mark_boundaries(img, slic))
    if seg_contour is not None and isinstance(seg_contour, np.ndarray):
        assert img.shape[:2] == seg_contour.shape[:2], \
            'image size %r and segm. %r should match' % (img.shape, seg_contour.shape)
        ax.contour(seg_contour, linewidths=3, levels=np.unique(seg_contour))
    if labels is not None:
        assert len(points) == len(labels), \
            'number of points (%i) and labels (%i) should match' \
            % (len(points), len(labels))
        for lb in lut_label_marker:
            marker, clr = lut_label_marker[lb]
            ax.plot(points[(labels == lb), 1], points[(labels == lb), 0],
                    marker, color=clr)
    else:
        ax.plot(points[:, 1], points[:, 0], 'o', color=COLOR_ORANGE)
    ax.set(xlim=[0, img.shape[1]], ylim=[img.shape[0], 0])


def figure_image_segm_centres(img, segm, centers=None, cmap_contour=plt.cm.Blues):
    """ visualise the input image and segmentation in common frame

    :param ndarray img: image
    :param ndarray segm: segmentation
    :param [tuple(int,int)]|ndarray centers: or np.array
    :param obj cmap_contour:
    :return Figure:

    >>> img = np.random.random((100, 150, 3))
    >>> seg = np.random.randint(0, 2, (100, 150))
    >>> centre = [[55, 60]]
    >>> fig = figure_image_segm_centres(img, seg, centre)
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    fig, ax = plt.subplots()

    ax.imshow(img)
    if np.sum(segm) > 0:
        segm_show = segm
        if segm.ndim > 2:
            segm_show = np.argmax(segm, axis=2)
        ax.contour(segm_show, cmap=cmap_contour, linewidths=0.5)
    if isinstance(centers, list):
        ax.plot(np.array(centers)[:, 1], np.array(centers)[:, 0], 'o',
                color=COLOR_ORANGE)
    elif isinstance(centers, np.ndarray):
        assert img.shape[:2] == centers.shape[:2], \
            'image size %r and centers %r should match' % (img.shape, centers.shape)
        ax.contour(centers, levels=np.unique(centers), cmap=plt.cm.YlOrRd)

    ax.set(xlim=[0, img.shape[1]], ylim=[img.shape[0], 0])
    fig.tight_layout()

    return fig


def draw_graphcut_weighted_edges(segments, centers, edges, edge_weights,
                                 img_bg=None, img_alpha=0.5):
    """ visualise the edges on the overlapping a background image

    :param [tuple(int,int)] centers: list of centers
    :param ndarray segments: np.array<height, width>
    :param ndarray edges: list of edges of shape <nb_edges, 2>
    :param ndarray edge_weights: weight per edge <nb_edges, 1>
    :param ndarray img_bg: image background
    :param float img_alpha: transparency
    :return ndarray: np.array<height, width, 3>

    >>> slic = np.array([[0] * 3 + [1] * 3 + [2] * 3+ [3] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3 + [7] * 3] * 4)
    >>> centres = [[1, 1], [1, 4], [1, 7], [1, 10],
    ...            [5, 1], [5, 4], [5, 7], [5, 10]]
    >>> edges = [[0, 1], [1, 2], [2, 3], [0, 4], [1, 5],
    ...          [4, 5], [2, 6], [5, 6], [3, 7], [6, 7]]
    >>> img = np.random.randint(0, 256, slic.shape + (3,))
    >>> edge_weights = np.ones(len(edges))
    >>> edge_weights[0] = 0
    >>> img = draw_graphcut_weighted_edges(slic, centres, edges, edge_weights, img_bg=img)
    >>> img.shape
    (8, 12, 3)
    """
    if img_bg is not None:
        if img_bg.ndim == 2:
            # duplicate channels to be like RGB
            img_bg = np.rollaxis(np.tile(img_bg, (3, 1, 1)), 0, 3)
        # convert to range 0,1 so the drawing is correct
        max_val = 1.
        if img_bg.dtype != np.float:
            max_val = max(255., img_bg.max())
        img = img_bg.astype(np.float) / max_val
        # make it partialy transparent
        img = (1. - img_alpha) + img * img_alpha
    else:
        img = np.zeros(segments.shape + (3,))
    clrs = plt.get_cmap('Greens')
    diff = (edge_weights.max() - edge_weights.min())
    if diff > 0:
        edge_ratio = (edge_weights - edge_weights.min()) / diff
    else:
        edge_ratio = np.zeros(edge_weights.shape)
    for i, edge in enumerate(edges):
        n1, n2 = edge
        y1, x1 = map(int, centers[n1])
        y2, x2 = map(int, centers[n2])

        # line = draw.line(y1, x1, y2, x2)  # , shape=img.shape[:2]
        # img[line] = clrs(edge_ratio[i])[:3]

        # using anti-aliasing
        rr, cc, val = draw.line_aa(y1, x1, y2, x2)  # , shape=img.shape[:2]
        color_w = np.tile(val, (3, 1)).T
        img[rr, cc, :] = color_w * clrs(edge_ratio[i])[:3] + (1 - color_w) * img[rr, cc, :]

        circle = draw.circle(y1, x1, radius=2, shape=img.shape[:2])
        img[circle] = 1., 1., 0.
    return img


def draw_rg2sp_results(ax, seg, slic, debug_rg2sp, iter_index=-1):
    """ drawing Region Growing with shape prior

    :param ax: figure axis
    :param ndarray seg: segmentation
    :param ndarray slic: superpixels
    :param dict debug_rg2sp: dictionary with debug results
    :param int iter_index: iteration index
    :return: ax
    """
    ax.imshow(debug_rg2sp['labels'][iter_index][slic], cmap=plt.cm.jet)
    ax.contour(seg, levels=np.unique(seg), colors='#bfbfbf')
    for centre, shift in zip(debug_rg2sp['centres'][iter_index],
                             debug_rg2sp['shifts'][iter_index]):
        rot = np.deg2rad(shift)
        ax.plot(centre[1], centre[0], 'ow')
        ax.arrow(centre[1], centre[0], np.cos(rot) * 50., np.sin(rot) * 50.,
                 fc='w', ec='w', head_width=20., head_length=30.)
    ax.set(xlim=[0, seg.shape[1]], ylim=[seg.shape[0], 0],
           title='Iteration #%i with E=%.0f'
                 % (iter_index, round(debug_rg2sp['criteria'][iter_index])))
    return ax


def figure_rg2sp_debug_complete(seg, slic, debug_rg2sp, iter_index=-1, max_size=5):
    """ draw figure with all debug (intermediate) segmentation steps

    :param ndarray seg: segmentation
    :param ndarray slic: superpixels
    :param debug_rg2sp: dictionary with some debug parameters
    :param int iter_index: iteration index
    :param int max_size: max figure size
    :return Figure:

    >>> seg = np.random.randint(0, 4, (100, 150))
    >>> slic = np.random.randint(0, 80, (100, 150))
    >>> dict_debug = {
    ...     'lut_data_cost': np.random.random((80, 3)),
    ...     'lut_shape_cost': np.random.random((15, 80, 3)),
    ...     'labels': np.random.randint(0, 4, (15, 80)),
    ...     'centres': [np.array([np.random.randint(0, 100, 80),
    ...                           np.random.randint(0, 150, 80)]).T] * 15,
    ...     'shifts': np.random.random((15, 3)),
    ...     'criteria': np.random.random(15),
    ... }
    >>> fig = figure_rg2sp_debug_complete(seg, slic, dict_debug)
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    nb_objects = debug_rg2sp['lut_data_cost'].shape[1] - 1
    nb_subfigs = max(3, nb_objects)
    norm_zise = np.array(seg.shape[:2]) / float(np.max(seg.shape))
    fig_size = np.array(norm_zise)[::-1] * np.array([nb_subfigs, 2]) * max_size
    fig, axarr = plt.subplots(2, nb_subfigs, figsize=fig_size)

    draw_rg2sp_results(axarr[0, 0], seg, slic, debug_rg2sp, iter_index)

    axarr[0, 1].plot(debug_rg2sp['criteria'])
    axarr[0, 1].plot(iter_index, debug_rg2sp['criteria'][iter_index], 'og')
    axarr[0, 1].set(ylabel='Energy', xlabel='iteration')
    axarr[0, 1].grid()

    axarr[0, 2].set_title('Data cost')
    img_shape_cost = debug_rg2sp['lut_shape_cost'][iter_index][:, 0][slic]
    im = axarr[0, 2].imshow(img_shape_cost, cmap=plt.cm.jet)
    fig.colorbar(im, ax=axarr[0, 2])

    for j in range(3):
        axarr[0, j].axis('off')

    for i in range(nb_objects):
        axarr[1, i].set_title('Shape cost for object #%i' % i)
        lut = debug_rg2sp['lut_shape_cost'][iter_index][:, i + 1]
        im = axarr[1, i].imshow(lut[slic], cmap=plt.cm.bone)
        fig.colorbar(im, ax=axarr[1, i])
        axarr[1, i].contour(seg, levels=np.unique(seg), cmap=plt.cm.jet)
        axarr[1, i].plot(debug_rg2sp['centres'][iter_index][i, 1],
                         debug_rg2sp['centres'][iter_index][i, 0], 'or')
        axarr[0, i].axis('off')

    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.96)
    # fig.tight_layout()
    return fig


def make_overlap_images_optical(images):
    """ overlap images and show them

    :param [ndarray] images: collection of images
    :return ndarray: combined image

    >>> im1 = np.zeros((5, 8), dtype=float)
    >>> im2 = np.ones((5, 8), dtype=float)
    >>> make_overlap_images_optical([im1, im2])
    array([[ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5]])
    """
    logging.info(' make_overlap_images_optical: overlap images')
    # get max dimension of the images
    max_size = np.max(np.vstack(tuple([im.shape for im in images])), 0)
    logging.debug('compute maximal image size: %r', max_size)
    imgs_w = []
    for im in images:
        imgs_w.append(np.zeros(max_size, dtype=im.dtype))
    # copy images to the maximal image
    for i, im in enumerate(images):
        imgs_w[i][:im.shape[0], :im.shape[1]] = im
    # put images as backgrounds
    img = imgs_w[0] / len(images)
    for i in range(1, len(images)):
        img = img + imgs_w[i] / len(images)
    return img


def make_overlap_images_chess(images, chess_field=SIZE_CHESS_FIELD):
    """ overlap images and show them

    :param [ndarray] images: collection of images
    :param int chess_field: size of chess field size
    :return ndarray: combined image

    >>> im1 = np.zeros((5, 10), dtype=int)
    >>> im2 = np.ones((5, 10), dtype=int)
    >>> make_overlap_images_chess([im1, im2], chess_field=2)
    array([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
           [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
           [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    logging.info(' make_overlap_images_chess: overlap images')
    # get max dimension of the images
    max_size = np.max(np.vstack(tuple([im.shape for im in images])), 0)
    logging.debug('compute maximal image size: %r', max_size)
    imgs_w = []
    for im in images:
        imgs_w.append(np.zeros(max_size, dtype=im.dtype))
    # copy images to the maximal image
    for i, im in enumerate(images):
        imgs_w[i][:im.shape[0], :im.shape[1]] = im
    img = np.zeros(max_size, dtype=images[0].dtype)
    idx_row = 0
    for i in range(int(max_size[0] / chess_field)):
        idx = idx_row
        for j in range(int(max_size[1] / chess_field)):
            w_b = i * chess_field
            if (w_b + chess_field) < max_size[0]:
                w_e = w_b + chess_field
            else:
                w_e = max_size[0]
            h_b = j * chess_field
            if (h_b + chess_field) < max_size[1]:
                h_e = h_b + chess_field
            else:
                h_e = max_size[1]
            img[w_b:w_e, h_b:h_e] = imgs_w[idx][w_b:w_e, h_b:h_e]
            idx = (idx + 1) % len(images)
        idx_row = (idx_row + 1) % len(images)
    return img


def draw_image_clusters_centers(ax, img, centres, points=None, labels_centre=None, segm=None):
    """ draw imageas bacround and clusters centers

    :param ax: figure axis
    :param ndarray img: image
    :param ndarray centres: points
    :param ndarray points: optional list of all points
    :param list(int) labels_centre: optional list of labels for points
    :param ndarray segm: optional segmentation

    >>> img = np.random.randint(0, 256, (100, 100, 3))
    >>> seg = np.random.randint(0, 3, (100, 100))
    >>> centres = np.random.randint(0, 100, (3, 2))
    >>> points = np.random.randint(0, 100, (25, 2))
    >>> labels = np.random.randint(0, 4, 25)
    >>> draw_image_clusters_centers(plt.Figure().gca(), img[:, :, 0], centres,
    ...                             points, labels, seg)
    """
    if img is not None:
        img = (img / float(np.max(img)))
        assert img.ndim == 2, \
            'required image dimension is 2 to instead %r' % img.shape
        ax.imshow(img, cmap=plt.cm.Greys_r)
        ax.set(xlim=[0, img.shape[1]], ylim=[img.shape[0], 0])
    if segm is not None:
        ax.imshow(segm, alpha=0.1)
        ax.contour(segm)
    if points is not None and len(points) > 0 \
            and labels_centre is not None:
        points = np.array(points)
        for i in range(max(labels_centre) + 1):
            select = points[np.asarray(labels_centre) == i]
            ax.plot(select[:, 1], select[:, 0], '.')
    # ax.plot(np.asarray(centres)[:, 1], np.asarray(centres)[:, 0], 'oy')
    # ax.plot(np.asarray(centres)[:, 1], np.asarray(centres)[:, 0], 'xr')
    if len(centres) == 0:
        return
    centres = np.asarray(centres)
    for s, clr in [(3e3, '#ccff33'), (1e3, '#ff3333'), (1e2, '#00ffff'), ]:
        ax.scatter(centres[:, 1], centres[:, 0], s=s, c=clr)
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])


def figure_segm_boundary_dist(segm_ref, segm, subfig_size=9):
    """ visualise the boundary distances between two segmentation

    :param ndarray segm_ref: reference segmentation
    :param ndarray segm: estimated segmentation
    :param int subfig_size: maximal sub-figure size
    :return Figure:

    >>> seg = np.zeros((100, 100))
    >>> seg[35:80, 10:65] = 1
    >>> fig = figure_segm_boundary_dist(seg, seg.T)
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    assert segm_ref.shape == segm.shape, \
        'ref segm %r and segm %r should match' % (segm_ref.shape, segm.shape)
    segr_boundary = segmentation.find_boundaries(segm_ref, mode='thick')
    segm_boundary = segmentation.find_boundaries(segm, mode='thick')
    segm_distance = ndimage.distance_transform_edt(~segm_boundary)

    norm_size = np.array(segm_ref.shape[:2]) / float(np.max(segm_ref.shape))
    fig_size = norm_size[::-1] * subfig_size * np.array([2, 1])
    fig, axarr = plt.subplots(ncols=2, figsize=fig_size)

    axarr[0].set_title('boundary distances with reference contour')
    im = axarr[0].imshow(segm_distance, cmap=plt.cm.Greys)
    plt.colorbar(im, ax=axarr[0])
    axarr[0].contour(segm_ref, cmap=plt.cm.jet)

    segm_distance[~segr_boundary] = 0
    axarr[1].set_title('distance projected to ref. boundary')
    im = axarr[1].imshow(segm_distance, cmap=plt.cm.Reds)
    plt.colorbar(im, ax=axarr[1])

    return fig

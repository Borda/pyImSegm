"""
Framework for visualisations

Copyright (C) 2016-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import logging

import matplotlib
if os.environ.get('DISPLAY','') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
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
    :param (int, int) shape: size of output mask
    :return ([int], [int]): indexes of filled positions

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

    # compute rotated radii by given rotation
    r_radius_rot = abs(r_radius * np.cos(orientation)) \
                   + c_radius * np.sin(orientation)
    c_radius_rot = r_radius * np.sin(orientation) \
                   + abs(c_radius * np.cos(orientation))
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
    sin_alpha, cos_alpha = np.sin(orientation), np.cos(orientation)
    r, c = (r_lim - r_org), (c_lim - c_org)
    distances = ((r * cos_alpha + c * sin_alpha) / r_rad) ** 2 \
                + ((r * sin_alpha - c * cos_alpha) / c_rad) ** 2
    rr, cc = np.nonzero(distances <= 1)

    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc


# Should be solved in skimage v0.13
def ellipse(r, c, r_radius, c_radius, orientation=0., shape=None):
    """ temporary wrapper until release New version scikit-image v0.13

    :param int r: center position in rows
    :param int c: center position in columns
    :param int r_radius: ellipse diam in rows
    :param int c_radius: ellipse diam in columns
    :param float orientation: ellipse orientation
    :param (int, int) shape: size of output mask
    :return ([int], [int]): indexes of filled positions

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


# Should be solved in skimage v0.14
def ellipse_perimeter(r, c, r_radius, c_radius, orientation=0., shape=None):
    """ see New version scikit-image v0.14


    :param int r: center position in rows
    :param int c: center position in columns
    :param int r_radius: ellipse diam in rows
    :param int c_radius: ellipse diam in columns
    :param float orientation: ellipse orientation
    :param (int, int) shape: size of output mask
    :return ([int], [int]): indexes of filled positions

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


def figure_image_adjustment(fig, img_size):
    """ adjust figure as nice image without axis

    :param fig:
    :param (int, int) img_size: 
    :return:

    >>> figure_image_adjustment(plt.figure(), (150, 200))  # doctest: +ELLIPSIS
    <matplotlib.figure.Figure object at ...>
    """
    ax = fig.gca()
    ax.set_xlim([0, img_size[1]])
    ax.set_ylim([img_size[0], 0])
    ax.axis('off')
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def figure_image_segm_results(img, seg, subfig_size=9):
    """ creating subfigure with original image, overlapped segmentation contours
    and clean result segmentation...
    it turns the sequence in vertical / horizontal according major image dim

    :param ndarray img:
    :param ndarray seg:
    :param int subfig_size:
    :return Figure:

    >>> img = np.random.random((100, 150, 3))
    >>> seg = np.random.randint(0, 2, (100, 150))
    >>> figure_image_segm_results(img, seg)  # doctest: +ELLIPSIS
    <matplotlib.figure.Figure object at ...>
    """
    assert img.shape[:2] == seg.shape[:2], 'different image & seg_pipe sizes'
    if img.ndim == 2:  # for gray images of ovary
        # img = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)
        img = color.gray2rgb(img)

    norm_size = np.array(img.shape[:2]) / float(np.max(img.shape))
    # reverse dimensions and scale by fig size
    if norm_size[0] >= norm_size[1]:  # horizontal
        fig_size = norm_size[::-1] * subfig_size * np.array([3, 1])
        fig, axarr = plt.subplots(ncols=3, figsize=fig_size)
    else:  # vertical
        fig_size = norm_size[::-1] * subfig_size * np.array([1, 3])
        fig, axarr = plt.subplots(nrows=3, figsize=fig_size)

    axarr[0].set_title('original image')
    axarr[0].imshow(img)

    # visualise the 3th label
    axarr[1].set_title('original image w. segment overlap')
    axarr[1].imshow(color.rgb2gray(img), cmap=plt.cm.Greys_r)
    axarr[1].imshow(seg, alpha=0.2, cmap=plt.cm.jet)
    axarr[1].contour(seg, levels=np.unique(seg), linewidth=2, cmap=plt.cm.jet)

    axarr[2].set_title('segmentation of all labels')
    axarr[2].imshow(seg, cmap=plt.cm.jet)

    for i in range(len(axarr)):
        axarr[i].axis('off')
        axarr[i].axes.get_xaxis().set_ticklabels([])
        axarr[i].axes.get_yaxis().set_ticklabels([])

    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.tight_layout()
    return fig


def figure_overlap_annot_segm_image(annot, segm, img=None, subfig_size=9):
    """ figure showing overlap annotation - segmentation - image

    :param ndarray annot:
    :param ndarray segm:
    :param ndarray img:
    :param int subfig_size:
    :return Figure:

    >>> img = np.random.random((100, 150, 3))
    >>> seg = np.random.randint(0, 2, (100, 150))
    >>> figure_overlap_annot_segm_image(seg, seg, img)  # doctest: +ELLIPSIS
    <matplotlib.figure.Figure object at ...>
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
    axarr[0].imshow(annot, alpha=0.2)
    axarr[0].contour(annot, levels=np.unique(annot), linewidth=2)

    axarr[1].set_title('Segmentation')
    axarr[1].imshow(img)
    axarr[1].imshow(segm, alpha=0.2)
    axarr[1].contour(segm, levels=np.unique(segm), linewidth=2)

    # visualise the 3th label
    axarr[2].set_title('difference annot & segment')
    # axarr[2].imshow(~(annot == segm), cmap=plt.cm.Reds)
    max_val = np.max(annot.astype(int))
    cax = axarr[2].imshow(annot - segm, alpha=0.5,
                          vmin=-max_val, vmax=max_val, cmap=plt.cm.bwr)
    # vals = np.linspace(-max_val, max_val, max_val * 2 + 1)
    plt.colorbar(cax, ticks=np.linspace(-max_val, max_val, max_val * 2 + 1),
                 boundaries=np.linspace(-max_val - 0.5, max_val + 0.5,
                                        max_val * 2 + 2))
    # plt.clim(-max_val - 0.5, max_val - 0.5)
    # axarr[2].contour(annot, levels=np.unique(annot), linewidth=1, colors='g')
    # axarr[2].contour(segm, levels=np.unique(segm), linewidth=1, colors='b')

    for i in range(len(axarr)):
        axarr[i].axis('off')
        axarr[i].axes.get_xaxis().set_ticklabels([])
        axarr[i].axes.get_yaxis().set_ticklabels([])

    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.tight_layout()
    return fig


def figure_segm_graphcut_debug(dict_imgs, subfig_size=9):
    """ creating subfigure with slic, graph edges and results in the first row
    and individual class unary terms in the second row

    :param dict_imgs:
    :param int subfig_size:
    :return Figure:

    >>> dict_imgs = {
    ...     'image': np.random.random((100, 150, 3)),
    ...     'slic': np.random.randint(0, 2, (100, 150)),
    ...     'slic_mean': np.random.random((100, 150, 3)),
    ...     'img_graph_edges': np.random.random((100, 150, 3)),
    ...     'img_graph_segm': np.random.random((100, 150, 3)),
    ...     'imgs_unary_cost': [np.random.random((100, 150, 3))],
    ... }
    >>> figure_segm_graphcut_debug(dict_imgs)  # doctest: +ELLIPSIS
    <matplotlib.figure.Figure object at ...>
    """
    assert all(n in dict_imgs for n in ['image', 'slic', 'slic_mean',
                                        'img_graph_edges', 'img_graph_segm',
                                        'imgs_unary_cost'])
    nb_cols = max(3, len(dict_imgs['imgs_unary_cost']))
    img = dict_imgs['image']
    if img.ndim == 2:  # for gray images of ovary
        img = color.gray2rgb(img)
    norm_size = np.array(img.shape[:2]) / float(np.max(img.shape))

    fig_size = norm_size[::-1] * subfig_size * np.array([nb_cols, 2])
    fig, axarr = plt.subplots(2, nb_cols, figsize=fig_size)

    img_slic = segmentation.mark_boundaries(img, dict_imgs['slic'],
                                            mode='subpixel')
    axarr[0, 0].set_title('SLIC')
    axarr[0, 0].imshow(img_slic)
    for i, k in enumerate(['img_graph_edges', 'img_graph_segm']):
        axarr[0, i + 1].set_title(k)
        axarr[0, i + 1].imshow(dict_imgs[k])
    for i, im_uc in enumerate(dict_imgs['imgs_unary_cost']):
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


def figure_ellipse_fitting(img, seg, ellipses, centers, crits, fig_size=9):
    """ show figure with result of the ellipse fitting

    :param ndarray img:
    :param ndarray seg:
    :param [(int, int, int, int, float)] ellipses:
    :param [(int, int)] centers:
    :param [float] crits:
    :param float fig_size:
    :return:

    >>> img = np.random.random((100, 150, 3))
    >>> seg = np.random.randint(0, 2, (100, 150))
    >>> ells = np.random.random((3, 5)) * 25
    >>> centers = np.random.random((3, 2)) * 25
    >>> crits = np.random.random(3)
    >>> figure_ellipse_fitting(img[:, :, 0], seg, ells, centers, crits)  # doctest: +ELLIPSIS
    <matplotlib.figure.Figure object at ...>
    """
    assert len(ellipses) == len(centers)
    assert len(centers) == len(crits)

    fig_size = (fig_size * np.array(img.shape[:2]) / np.max(img.shape))[::-1]
    fig, ax = plt.subplots(figsize=fig_size)
    assert img.ndim == 2
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
    ax.set_xlim([0, seg.shape[1]])
    ax.set_ylim([seg.shape[0], 0])
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def figure_annot_slic_histogram_labels(dict_label_hist, slic_size=-1,
                                       slic_regul=-1):
    """ plot ration of labels  assigned to each superpixel

    :param dict_label_hist:
    :param int slic_size:
    :param float slic_regul:
    :return Figure:
    """
    matrix_hist_all = np.concatenate(tuple(dict_label_hist.values()), axis=0)
    nb_labels = matrix_hist_all.shape[1]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    for i in range(nb_labels):
        patches, bin_edges = np.histogram(matrix_hist_all[:, i], bins=50,
                                          density=True)
        bins = [(a + b) / 2. for a, b in zip(bin_edges[:-1], bin_edges[1:])]
        # ax.plot(bins, patches, label='label: %i' % i)
        ax.semilogy(bins, patches, label='label: %i' % i)
    ax.set_title('Histogram of labels density in each segments '
                 'over all annotated images\n (superpixels: size=%i, regul=%f)'
                 % (slic_size, slic_regul))
    ax.legend()
    ax.grid()
    ax.set_xlabel('region densities')
    ax.set_ylabel('[%]')
    return fig


def figure_ray_feature(segm, points, ray_dist_raw=None, ray_dist=None,
                       points_reconst=None):
    """ visualise the segmentation with specific point and estimated ray dist.

    :param segm:
    :param points:
    :param ray_dist_raw:
    :param ray_dist:
    :return Figure:

    example, see unittests
    """
    ray_dist_raw = ray_dist_raw if ray_dist_raw is not None else []
    ray_dist = ray_dist if ray_dist is not None else []

    fig, axarr = plt.subplots(2, 1)
    axarr[0].imshow(1 - segm, cmap='gray', interpolation='nearest')
    axarr[0].plot(points[1], points[0], 'bo')
    axarr[0].set_xlim([0, segm.shape[1]])
    axarr[0].set_ylim([segm.shape[0], 0])
    if points_reconst is not None:
        axarr[0].plot(points_reconst[:, 1], points_reconst[:, 0], 'g.')
    axarr[1].plot(np.linspace(0, 360, len(ray_dist_raw)).tolist(),
                  ray_dist_raw, 'b', label='original')
    axarr[1].plot(np.linspace(0, 360, len(ray_dist)).tolist(),
                  ray_dist, 'r', label='final')
    axarr[1].set_xlabel('angles [deg]')
    axarr[1].set_xlim([0, 360])
    axarr[1].legend(loc=0)
    axarr[1].grid()
    return fig


def draw_color_labeling(segments, lut_labels):
    """ visualise the graph cut results

    :param ndarray segments: np.array<h, w>
    :param [int] lut_labels:
    :return ndarray: np.array<h, w, 3>
    """
    seg = np.asarray(lut_labels)[segments]
    clrs = plt.get_cmap('jet')
    lbs = np.arange(np.max(seg) + 1)
    lut = clrs(lbs / float(lbs.max()))[:, :3]
    img = lut[seg]
    return img


def draw_graphcut_unary_cost_segments(segments, unary_cost):
    """ visualise the unary cost for each class

    :param ndarray segments: np.array<h, w>
    :param ndarray unary_cost: np.array<nb_spx, nb_cls>
    :return []: [np.array<h, w, 3>] * nb_cls

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

    :param [int] start:
    :param [int] end:
    :param [int] point:
    :return [int]:

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

    :param (int, int) mask_shape:
    :param [[int, int]] pos_ant:
    :param [[int, int]] pos_lat:
    :param [[int, int]] pos_post:
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

        m_overlap = np.sum(np.logical_and(mask > 0, mask_eggs > 0)) \
                       / float(np.sum(mask))
        if m_overlap > threshold_overlap:
            logging.debug('skip egg drawing while it overlap by %f', m_overlap)
            continue
        mask_eggs[mask.astype(bool)] = i + 1

    return mask_eggs


def parse_annot_rectangles(rows_slice):
    """ parse annotation fromDF to lists

    :param row_slice:
    :return:

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

    :param (int, int) mask_shape:
    :param [[int, int]] pos_ant:
    :param [[int, int]] pos_lat:
    :param [[int, int]] pos_post:
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


def merge_object_masks(list_masks, thr_overlap=0.7):
    """ merge several mask into one multi-class segmentation

    :param [ndarray] list_masks:
    :param float thr_overlap:
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
    assert len(list_masks) > 0
    mask = np.array(list_masks[0])

    for i in range(1, len(list_masks)):
        overlap_ratios = []
        for j in range(1, int(np.max(mask) + 1)):
            overlap = np.sum(np.logical_and(mask == j, list_masks[i] == 1))
            union = np.sum(np.logical_or(mask == j, list_masks[i] == 1))
            overlap_ratios.append(float(overlap) / float(union))
        if any(r > thr_overlap for r in overlap_ratios):
             logging.debug('skip egg drawing while it overlap by %s',
                           repr(overlap_ratios))
             continue
        mask[list_masks[i] == 1] = np.max(mask) + 1

    return mask


def draw_image_segm_points(ax, img, points, labels=None, slic=None,
                           clr_slic='w', dict_label_marker=DICT_LABEL_MARKER,
                           seg_contour=None):
    """ on plane draw background image or segmentation, overlap with slic
    contours, add contour of aditive segmentation like annot. for centers
    plot point with specific property (shape and colour) according label

    :param ax:
    :param ndarray img:
    :param [(int, int)] points:
    :param [int] labels:
    :param ndarray slic:
    :param str clr_slic:
    :param {int: (str, str)} dict_label_marker:
    :param seg_contour: np.array

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
        ax.contour(slic, levels=np.unique(slic), alpha=0.5, colors=clr_slic,
                   linewidth=0.5)
    # fig.gca().imshow(mark_boundaries(img, slic))
    if seg_contour is not None and isinstance(seg_contour, np.ndarray):
        assert img.shape[:2] == seg_contour.shape[:2]
        ax.contour(seg_contour, linewidth=3, levels=np.unique(seg_contour))
    if labels is not None:
        assert len(points) == len(labels)
        for lb in dict_label_marker:
            marker, clr = dict_label_marker[lb]
            ax.plot(points[(labels == lb), 1], points[(labels == lb), 0],
                    marker, color=clr)
    else:
        ax.plot(points[:, 1], points[:, 0], 'o', color=COLOR_ORANGE)
    ax.set_xlim([0, img.shape[1]])
    ax.set_ylim([img.shape[0], 0])


def figure_image_segm_centres(img, segm, centers=None,
                              cmap_contour=plt.cm.Blues):
    """ visualise the input image and segmentation in common frame

    :param ndarray img: np.array
    :param ndarray segm: np.array
    :param [(int, int)] centers: or np.array
    :param cmap_contour:
    :return Figure:

    >>> img = np.random.random((100, 150, 3))
    >>> seg = np.random.randint(0, 2, (100, 150))
    >>> centre = [[55, 60]]
    >>> figure_image_segm_centres(img, seg, centre)  # doctest: +ELLIPSIS
    <matplotlib.figure.Figure object at ...>
    """
    fig, ax = plt.subplots()

    ax.imshow(img)
    if np.sum(segm) > 0:
        segm_show = segm
        if segm.ndim > 2:
            segm_show = np.argmax(segm, axis=2)
        ax.contour(segm_show, cmap=cmap_contour, linewidth=0.5)
    if isinstance(centers, list):
        ax.plot(np.array(centers)[:, 1], np.array(centers)[:, 0], 'o',
                color=COLOR_ORANGE)
    elif isinstance(centers, np.ndarray):
        assert img.shape[:2] == centers.shape[:2]
        ax.contour(centers, levels=np.unique(centers), cmap=plt.cm.YlOrRd)

    ax.set_xlim([0, img.shape[1]])
    ax.set_ylim([img.shape[0], 0])
    fig.tight_layout()

    return fig


def draw_graphcut_weighted_edges(segments, list_centers, edges, edge_weights,
                                 img_bg=None, img_alpha=0.5):
    """ visualise the edges on the overlapping a background image

    :param ndarray segments: np.array<h, w>
    :param ndarray edges: np.array<nb_edges, 2>
    :param ndarray edge_weights: np.array<nb_edges, 1>
    :param ndarray img_bg: np.array<h, w, 3>
    :return ndarray: np.array<h, w, 3>

    >>> slic = np.array([[0] * 3 + [1] * 3 + [2] * 3+ [3] * 3] * 4 +
    ...                 [[4] * 3 + [5] * 3 + [6] * 3 + [7] * 3] * 4)
    >>> centres = [[1, 1], [1, 4], [1, 7], [1, 10],
    ...            [5, 1], [5, 4], [5, 7], [5, 10]]
    >>> edges = [[0, 1], [1, 2], [2, 3], [0, 4], [1, 5],
    ...          [4, 5], [2, 6], [5, 6], [3, 7], [6, 7]]
    >>> img = np.random.randint(0, 256, slic.shape + (3,))
    >>> edge_weights = np.ones(len(edges))
    >>> edge_weights[0] = 0
    >>> img = draw_graphcut_weighted_edges(slic, centres, edges, edge_weights,
    ...                                    img_bg=img)
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
    edge_ratio = (edge_weights - edge_weights.min()) / diff
    for i, edge in enumerate(edges):
        n1, n2 = edge
        y1, x1 = map(int, list_centers[n1])
        y2, x2 = map(int, list_centers[n2])

        # line = draw.line(y1, x1, y2, x2)  # , shape=img.shape[:2]
        # img[line] = clrs(edge_ratio[i])[:3]

        # using anti-aliasing
        rr, cc, val = draw.line_aa(y1, x1, y2, x2)  # , shape=img.shape[:2]
        color_w = np.tile(val, (3, 1)).T
        img[rr, cc, :] = color_w * clrs(edge_ratio[i])[:3] + \
                         (1 - color_w) * img[rr, cc, :]

        circle = draw.circle(y1, x1, radius=2, shape=img.shape[:2])
        img[circle] = 1., 1., 0.
    return img


def draw_rg2sp_results(ax, seg, slic, dict_rg2sp_debug, iter_index=-1):
    ax.set_title('Iteration #%i with E=%.0f' %
                 (iter_index, round(dict_rg2sp_debug['energy'][iter_index])))
    ax.imshow(dict_rg2sp_debug['labels'][iter_index][slic], cmap=plt.cm.jet)
    ax.contour(seg, levels=np.unique(seg), colors='#bfbfbf')
    for centre, shift in zip(dict_rg2sp_debug['centres'][iter_index],
                             dict_rg2sp_debug['shifts'][iter_index]):
        rot = np.deg2rad(shift)
        ax.plot(centre[1], centre[0], 'ow')
        ax.arrow(centre[1], centre[0], np.cos(rot) * 50., np.sin(rot) * 50.,
                 fc='w', ec='w', head_width=20., head_length=30.)
    ax.set_xlim([0, seg.shape[1]])
    ax.set_ylim([seg.shape[0], 0])
    return ax


def figure_rg2sp_debug_complete(seg, slic, dict_rg2sp_debug, iter_index=-1,
                                max_size=5):
    """ draw figure with all debug (intermediate) segmenatation steps

    :param ndarray seg:
    :param ndarray slic:
    :param dict_rg2sp_debug:
    :param int iter:
    :param int max_size:
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
    ...     'energy': np.random.random(15),
    ... }
    >>> figure_rg2sp_debug_complete(seg, slic, dict_debug) # doctest: +ELLIPSIS
    <matplotlib.figure.Figure object at ...>
    """
    nb_objects = dict_rg2sp_debug['lut_data_cost'].shape[1] - 1
    nb_subfigs = max(3, nb_objects)
    norm_zise = np.array(seg.shape[:2]) / float(np.max(seg.shape))
    fig_size = np.array(norm_zise)[::-1] * np.array([nb_subfigs, 2]) * max_size
    fig, axarr = plt.subplots(2, nb_subfigs, figsize=fig_size)

    draw_rg2sp_results(axarr[0, 0], seg, slic, dict_rg2sp_debug, iter_index)

    axarr[0, 1].plot(dict_rg2sp_debug['energy'])
    axarr[0, 1].plot(iter_index, dict_rg2sp_debug['energy'][iter_index], 'og')
    axarr[0, 1].set_ylabel('Energy')
    axarr[0, 1].set_xlabel('iteration')
    axarr[0, 1].grid()

    axarr[0, 2].set_title('Data cost')
    img_shape_cost = dict_rg2sp_debug['lut_shape_cost'][iter_index][:, 0][slic]
    im = axarr[0, 2].imshow(img_shape_cost, cmap=plt.cm.jet)
    fig.colorbar(im, ax=axarr[0, 2])

    for j in range(3):
        axarr[0, j].axis('off')

    for i in range(nb_objects):
        axarr[1, i].set_title('Shape cost for object #%i' % i)
        lut = dict_rg2sp_debug['lut_shape_cost'][iter_index][:, i + 1]
        im = axarr[1, i].imshow(lut[slic], cmap=plt.cm.bone)
        fig.colorbar(im, ax=axarr[1, i])
        axarr[1, i].contour(seg, levels=np.unique(seg), cmap=plt.cm.jet)
        axarr[1, i].plot(dict_rg2sp_debug['centres'][iter_index][i, 1],
                         dict_rg2sp_debug['centres'][iter_index][i, 0], 'or')
        axarr[0, i].axis('off')

    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.96)
    # fig.tight_layout()
    return fig


def make_overlap_images_optical(imgs):
    """ overlap images and show them

    :param [ndarray] imgs:
    :return ndarray:

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
    max_size = np.max(np.vstack(tuple([im.shape for im in imgs])), 0)
    logging.debug('compute maximal image size: ' + repr(max_size))
    imgs_w = []
    for im in imgs:
        imgs_w.append(np.zeros(max_size, dtype=im.dtype))
    # copy images to the maximal image
    for i, im in enumerate(imgs):
        imgs_w[i][:im.shape[0], :im.shape[1]] = im
    # put images as backgrounds
    img = imgs_w[0] / len(imgs)
    for i in range(1, len(imgs)):
        img = img + imgs_w[i] / len(imgs)
    return img


def make_overlap_images_chess(imgs, chess_field=SIZE_CHESS_FIELD):
    """ overlap images and show them

    :param ndarray imgs:
    :param int chess_field:
    :return ndarray:

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
    max_size = np.max(np.vstack(tuple([im.shape for im in imgs])), 0)
    logging.debug('compute maximal image size: ' + repr(max_size))
    imgs_w = []
    for im in imgs:
        imgs_w.append(np.zeros(max_size, dtype=im.dtype))
    # copy images to the maximal image
    for i in range(len(imgs)):
        imgs_w[i][:imgs[i].shape[0], :imgs[i].shape[1]]=imgs[i]
    img = np.zeros(max_size, dtype=im.dtype)
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
            img[w_b:w_e, h_b:h_e]=imgs_w[idx][w_b:w_e, h_b:h_e]
            idx = (idx+1) % len(imgs)
        idx_row = (idx_row+1) % len(imgs)
    return img


def draw_image_clusters_centers(ax, img, centres, points=None,
                                labels_centre=None, segm=None):
    """ draw imageas bacround and clusters centers

    :param ax:
    :param ndarray img:
    :param ndarray centres:
    :param ndarray points: optional list of all points
    :param [int] labels_centre: optional list of labels for points
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
        assert img.ndim == 2
        ax.imshow(img, cmap=plt.cm.Greys_r)
        ax.set_xlim([0, img.shape[1]])
        ax.set_ylim([img.shape[0], 0])
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
    """ visualise the boundary distances bteween two segmentations

    :param ndarray segm_ref:
    :param ndarray segm:
    :param int subfig_size:
    :return Figure:

    >>> seg = np.zeros((100, 100))
    >>> seg[35:80, 10:65] = 1
    >>> figure_segm_boundary_dist(seg, seg.T) # doctest: +ELLIPSIS
    <matplotlib.figure.Figure object at ...>
    """
    assert segm_ref.shape == segm.shape
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
    axarr[0].set_title('distance projected to ref. boundary')
    im = axarr[1].imshow(segm_distance, cmap=plt.cm.Reds)
    plt.colorbar(im, ax=axarr[1])

    return fig

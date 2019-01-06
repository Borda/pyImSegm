"""
Framework for superpixels
 * wrapper over skimage.SLIC
 * other related functions

SEE:
* http://scikit-image.org/docs/dev/auto_examples/plot_segmentations.html

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import logging

import numpy as np
import skimage.segmentation as ski_segm
from skimage import measure

IMAGE_SPACING = (1, 1, 1)


def segment_slic_img2d(img, sp_size=50, relative_compact=0.1, slico=False):
    """ segmentation by SLIC superpixels using original SLIC implementation

    :param ndarray img: input color image
    :param int sp_size: superpixel initial size
    :param float relative_compact: relative regularisation in range (0, 1)
        where 0 is for free form and 1 for nearly rectangular superpixels
    :param bool slico: whether use parameter free version ASLIC/SLICO
    :return ndarray: segmentation

    >>> np.random.seed(0)
    >>> img = np.random.random((100, 150, 3))
    >>> slic = segment_slic_img2d(img, 20, 0.2)
    >>> slic.shape
    (100, 150)
    >>> img = np.random.random((150, 100))
    >>> slic = segment_slic_img2d(img, 20, 0.2)
    >>> slic.shape
    (150, 100)
    """
    logging.debug('Init SLIC superpixels 2d RGB clustering with params'
                  ' size=%i and regul=%f for image dims %s',
                  sp_size, relative_compact, repr(img.shape))
    nb_pixels = np.prod(img.shape[:2])

    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if img.ndim == 2:  # duplicate channels to be like RGB
        img = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)
    # scale image values
    if img.min() != 0. or img.max() != 1.:
        img = (img - img.min()) / float(img.max() - img.min())

    # set native SLIC parameters
    slic_nb_spx = int(nb_pixels / (sp_size ** 2))
    slic_compact = (sp_size * relative_compact) ** 1.5
    logging.debug('Starting SLIC with params NB=%i & compat=%f for image %s',
                  slic_nb_spx, slic_compact, repr(img.shape))
    # run SLIC segmentation
    slic_segments = ski_segm.slic(img, n_segments=slic_nb_spx,
                                  compactness=slic_compact,
                                  sigma=1, enforce_connectivity=True,
                                  slic_zero=slico)
    logging.debug('SLIC finished')
    # slic_segments, _, _ = ski_segm.relabel_sequential(slic_segments)
    # fix: unconnected segments - [ndimage.label(slic==i)[1]
    #                              for i in range(slic.max() + 1)]
    # slic_segments = measure.label(slic_segments, neighbors=4)
    return np.array(slic_segments)


def segment_slic_img3d_gray(im, sp_size=50, relative_compact=0.1,
                            space=IMAGE_SPACING):
    """ segmentation by SLIC superpixels using originla SLIC implementation

    :param ndarray im: input 3D grascale image
    :param int sp_size: superpixel initial size
    :param float relative_compact: relative regularisation in range (0, 1)
        where 0 is for free form and 1 for nearly rectangular superpixels
    :param (int, int, int) space: spacing in 3d image may not be equal
    :return ndarray:

    >>> np.random.seed(0)
    >>> img = np.random.random((100, 100, 10))
    >>> slic = segment_slic_img3d_gray(img, 20, 0.2, (1, 1, 5))
    >>> slic.shape
    (100, 100, 10)
    """
    logging.debug('Init SLIC superpixels 3d Gray clustering with params'
                  ' size=%i and regul=%f for image dims %s',
                  sp_size, relative_compact, repr(im.shape))
    nb_pixels = np.prod(im.shape)
    sp_size = np.prod(sp_size / np.asarray(space, dtype=np.float32) * min(space))
    # set native SLIC parameters
    slic_nb_sp = int(nb_pixels / sp_size)
    # slic_compact = int((sp_size * relative_compact) ** 1.5)
    slic_compact = int((sp_size * relative_compact) ** 1.5)
    logging.debug('Starting SLIC superpixels clustering with params NB=%i and '
                  'compat=%f and spacing=%s', slic_nb_sp, slic_compact, repr(space))
    # run SLIC segmentation
    # slic_segments = SLIC.slic_n(np.array(im), slic_nb_sp, slic_compact)
    slic_segments = ski_segm.slic(np.array(im), n_segments=slic_nb_sp,
                                  compactness=slic_compact, multichannel=False,
                                  spacing=space, sigma=1)
    logging.debug('SLIC superpixels estimated.')
    # slic_segments, _, _ = ski_segm.relabel_sequential(slic_segments)
    # fix: unconnected segments - [ndimage.label(slic==i)[1]
    #                              for i in range(slic.max() + 1)]
    slic_segments = measure.label(slic_segments)
    return np.array(slic_segments)


def make_graph_segment_connect_edges(vertices, all_edges):
    """ make graph of connencted components
    SEE: http://peekaboo-vision.blogspot.cz/2011/08/region-connectivity-graphs-in-python.html

    :param ndarray vertices:
    :param ndarray all_edges:
    :return (ndarray, ndarray):
    """
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges, axis=1)
    nb_vertices = len(vertices)
    edge_hash = all_edges[:, 0] + nb_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[int(edge % nb_vertices)],
              vertices[int(edge / nb_vertices)]] for edge in edges]
    return vertices, edges


def get_segment_diffs_2d_conn4(grid):
    """ wrapper for getting 4-connected in 2D image plane

    :param ndarray grid: segmentation
    :return [(int, int)]:
    """
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    return np.vstack([right, down])


def get_segment_diffs_3d_conn6(grid):
    """ wrapper for getting 6-connected in 3D image plane

    :param ndarray grid: segmentation
    :return [(int, int, int)]:
    """
    bellow = np.c_[grid[:-1, :, :].ravel(), grid[1:, :, :].ravel()]
    down = np.c_[grid[:, :-1, :].ravel(), grid[:, 1:, :].ravel()]
    right = np.c_[grid[:, :, :-1].ravel(), grid[:, :, 1:].ravel()]
    return np.vstack([bellow, right, down])


def make_graph_segm_connect_grid2d_conn4(grid):
    """ construct graph of connected components

    :param ndarray grid: segmentation
    :return [int], [(int, int)]:

    >>> grid = np.array([[0] * 5 + [1] * 5, [2] * 5 + [3] * 5])
    >>> v, edges = make_graph_segm_connect_grid2d_conn4(grid)
    >>> v
    array([0, 1, 2, 3])
    >>> edges
    [[0, 1], [0, 2], [1, 3], [2, 3]]
    """
    # get unique labels
    logging.debug('make graph segment connect edges - 2d conn4')
    vertices = np.unique(grid)
    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices, np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)
    all_edges = get_segment_diffs_2d_conn4(grid)
    return make_graph_segment_connect_edges(vertices, all_edges)


def make_graph_segm_connect_grid3d_conn6(grid):
    """ construct graph of connected components

    :param ndarray grid: segmentation
    :return [int], [(int, int)]:

    >>> grid_2d = np.array([[0] * 5 + [1] * 5, [2] * 5 + [3] * 5])
    >>> grid = np.array([grid_2d, grid_2d + 4])
    >>> v, edges = make_graph_segm_connect_grid3d_conn6(grid)
    >>> v
    array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> edges  # doctest: +NORMALIZE_WHITESPACE
    [[0, 1], [0, 2], [1, 3], [2, 3], [0, 4], [1, 5], [4, 5], [2, 6], [4, 6],
    [3, 7], [5, 7], [6, 7]]
    """
    # get unique labels
    logging.debug('make graph segment connect edges - 3d conn6')
    vertices = np.unique(grid)
    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices, np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)
    all_edges = get_segment_diffs_3d_conn6(grid)
    return make_graph_segment_connect_edges(vertices, all_edges)


def superpixel_centers(segments):
    """ estimate centers of each superpixel

    :param ndarray segments: segmentation np.array<h, w>
    :return [(float, float)]:

    >>> segm = np.array([[0] * 6 + [1] * 5, [0] * 6 + [2] * 5])
    >>> superpixel_centers(segm)
    [(0.5, 2.5), (0.0, 8.0), (1.0, 8.0)]
    >>> superpixel_centers(np.array([segm, segm, segm]))
    [[1.0, 0.5, 2.5], [1.0, 0.0, 8.0], [1.0, 1.0, 8.0]]
    """
    logging.debug('compute centers for %d superpixels', segments.max())
    centers = [list() for _ in range(np.max(segments) + 1)]

    if segments.ndim <= 2:
        # regionprops works for labels from 1
        regions = measure.regionprops(segments + 1)
        for region in regions:
            centers[region['label'] - 1] = region['centroid']
    elif segments.ndim == 3:
        # http://peekaboo-vision.blogspot.cz/2011/08/region-connectivity-graphs-in-python.html
        grids = np.mgrid[:segments.shape[0], :segments.shape[1], :segments.shape[2]]
        # for v in range(len(centers)):
        #     centers[v] = [grids[g][segments == v].mean() for g in range(3)]
        segm_flat = segments.ravel()
        grids_flat = [g.ravel() for g in grids]
        for i, lb in enumerate(segm_flat):
            vals = [grids_flat[g][i] for g in range(3)]
            centers[lb].append(vals)
        for lb, vals in enumerate(centers):
            centers[lb] = np.mean(vals, axis=0).tolist()
    else:
        logging.error('not supported image dim: %s', repr(segments.shape))
    return centers


def get_neighboring_segments(edges):
    """ get the indexes of neighboring superpixels for each superpixel
    the input is list edges of all neighboring segments

    :param [[int, int]] edges:
    :return [[int]]:

    >>> get_neighboring_segments([[0, 1], [1, 2], [1, 3], [2, 3]])
    [[1], [0, 2, 3], [1, 3], [1, 2]]
    """
    list_neighbours = np.zeros((np.max(edges) + 1, 0)).tolist()
    for e1, e2 in edges:
        list_neighbours[e1].append(e2)
        list_neighbours[e2].append(e1)
    return list_neighbours

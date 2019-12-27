"""
Framework for region growing
 * general GraphCut segmentation with and without shape model
 * region growing with shape prior - greedy & GraphCut

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging

import numpy as np
from scipy import stats, ndimage, interpolate
from sklearn import cluster, mixture
from skimage import morphology
try:
    from gco import cut_general_graph, cut_grid_graph
except Exception:
    print('WARNING: Missing Grah-Cut (GCO) library,'
          ' please install it from https://github.com/Borda/pyGCO.')

from imsegm.graph_cuts import MAX_PAIRWISE_COST, get_vertexes_edges, compute_spatial_dist
from imsegm.labeling import histogram_regions_labels_norm
from imsegm.descriptors import (
    compute_ray_features_segm_2d, interpolate_ray_dist, shift_ray_features)
from imsegm.superpixels import (
    superpixel_centers, get_neighboring_segments, make_graph_segm_connect_grid2d_conn4)

#: all infinty values in Grah-Cut terms replace by this value
GC_REPLACE_INF = 1e5
#: define minimal value for any vodel of shape prior term
MIN_SHAPE_PROB = 0.01
#: define maximal value of unary (being a class) term in Graph-Cut
MAX_UNARY_PROB = 1 - 0.01
#: define thresholds parameters for iterative Region Growing
RG2SP_THRESHOLDS = {
    'centre': 30,  # min center displacement since last iteration
    'shift': 15,  # min rotation change since last iteration
    'volume': 0.1,  # min volume change since last iteration
    'centre_init': 50,  # maximal move from original estimate
}


def object_segmentation_graphcut_slic(slic, segm, centres,
                                      labels_fg_prob=(0.1, 0.9),
                                      gc_regul=1, edge_coef=0.5,
                                      edge_type='model',
                                      coef_shape=0., shape_mean_std=(50., 10.),
                                      add_neighbours=False,
                                      debug_visual=None):
    """ object segmentation using Graph Cut directly on super-pixel level

    :param ndarray slic: superpixel pre-segmentation
    :param ndarray segm: input structure segmentation
    :param [(int, int)] centres: superpixel centres
    :param list(float) labels_fg_prob: weight for particular label belongs to FG
    :param float gc_regul: regularisation for GC
    :param float edge_coef: weight og edges on GC
    :param str edge_type: select the egde weights on graph
    :param float coef_shape: set the weight of shape prior
    :param shape_mean_std: mean and STD for shape prior
    :param bool add_neighbours: add also neighboring supepixels to the center
    :param dict debug_visual: dictionary with some intermediate results
    :return list(list(int)):

    >>> slic = np.array([[0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3,
    ...                  [5] * 3 + [6] * 3 + [7] * 3 + [8] * 3 + [9] * 3])
    >>> segm = np.array([[0] * 15, [1] * 12 + [0] * 3])
    >>> object_segmentation_graphcut_slic(slic, segm, [(1, 7)],
    ...                              gc_regul=0., edge_coef=1., coef_shape=1.)
    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=int32)
    >>> object_segmentation_graphcut_slic(slic, segm, [(1, 7)],
    ...                              gc_regul=1., edge_coef=1., debug_visual={})
    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=int32)
    """
    assert np.min(labels_fg_prob) < 1, 'non label can ce strictly 1'
    label_hist = histogram_regions_labels_norm(slic, segm)
    labels = np.argmax(label_hist, axis=1)

    assert segm.max() <= len(labels_fg_prob), \
        'table of label proba is shorter then the nb of labels in segmentation'
    labels_fg_prob = np.array(labels_fg_prob)
    labels_bg_prob = 1. - labels_fg_prob

    assert list(centres), 'at least one center has to be given'
    centres = [np.round(c).astype(int) for c in centres]
    slic_points = superpixel_centers(slic)

    proba = np.ones((len(labels), len(centres) + 1))
    proba[:, 0] = labels_bg_prob[labels]
    for i, centre in enumerate(centres):
        proba[:, i + 1] = labels_fg_prob[labels]

    shape = np.ones((len(labels), len(centres) + 1))
    if coef_shape > 0:
        shape_mean, shape_std = shape_mean_std
        shape[:, 0] = labels_bg_prob[labels]
        for i, centre in enumerate(centres):
            diff = slic_points - np.tile(centre, (len(slic_points), 1))
            dist = np.sqrt(np.sum(diff ** 2, axis=1))
            cdf = stats.norm.cdf(range(int(np.max(dist) + 1)),
                                 shape_mean, shape_std)
            cum = 1. - cdf + 1e-9
            shape[:, i + 1] = cum[dist.astype(int)]

    _, edges = get_vertexes_edges(slic)
    edges = np.array(edges)

    unary_cost = - np.log(proba) - coef_shape * np.log(shape)
    for i, pos in enumerate(centres):
        vertex = slic.item(tuple(pos))
        unary_cost[vertex, i + 1] = 0
        # unary[pos[0], pos[1], 0] = np.Inf
        if add_neighbours:
            mask = np.logical_or(edges[:, 0] == vertex, edges[:, 1] == vertex)
            near = edges[mask]
            for v in near.ravel():
                unary_cost[v, i + 1] = 0
            edges[mask] = 0

    # remove too small unary terms
    min_unary = -np.log(MAX_UNARY_PROB)
    unary_cost[unary_cost < min_unary] = min_unary

    # compute edge weight as difference in prob
    if edge_type == 'model':
        proba_fg = labels_fg_prob[labels]
        vertex_1 = proba_fg[edges[:, 0]]
        vertex_2 = proba_fg[edges[:, 1]]
        dist = np.abs(vertex_1 - vertex_2)
        edge_weights = np.exp(- dist / (2 * np.std(dist) ** 2))
        slic_centres = superpixel_centers(slic)
        spatial_dist = compute_spatial_dist(slic_centres, edges, relative=True)
        edge_weights /= spatial_dist
    else:
        edge_weights = np.ones(len(edges))

    edge_weights *= edge_coef

    pairwise_cost = (1 - np.eye(proba.shape[-1])) * gc_regul

    # run GraphCut
    logging.debug('perform GraphCut')
    # labels = np.argmax(proba, axis=1)
    graph_labels = cut_general_graph(edges, edge_weights, unary_cost,
                                     pairwise_cost, n_iter=999)

    if debug_visual is not None:
        list_unary_imgs = []
        for i in range(unary_cost.shape[-1]):
            list_unary_imgs.append(unary_cost[:, i][slic])
        debug_visual['unary_imgs'] = list_unary_imgs

    return graph_labels


def object_segmentation_graphcut_pixels(segm, centres,
                                        labels_fg_prob=(0.1, 0.9),
                                        gc_regul=1, seed_size=0, coef_shape=0.,
                                        shape_mean_std=(50., 10.),
                                        debug_visual=None):
    """ object segmentation using Graph Cut directly on pixel level

    :param ndarray centres:
    :param ndarray segm: input structure segmentation
    :param [(int, int)] centres: superpixel centres
    :param list(float) labels_fg_prob: set how much particular label belongs to foreground
    :param float gc_regul: regularisation for GC
    :param int seed_size: create circular neighborhood around initial centre
    :param float coef_shape: set the weight of shape prior
    :param shape_mean_std: mean and STD for shape prior
    :param dict debug_visual: dictionary with some intermediate results
    :return list(list(int)):

    >>> segm = np.array([[0] * 10,
    ...                 [1] * 5 + [0] * 5, [1] * 4 + [0] * 6,
    ...                 [0] * 6 + [1] * 4, [0] * 5 + [1] * 5,
    ...                 [0] * 10])
    >>> centres = [(1, 2), (4, 8)]
    >>> object_segmentation_graphcut_pixels(segm, centres, gc_regul=0., coef_shape=0.5)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [2, 2, 1, 2, 2, 0, 0, 0, 0, 0],
           [2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
    >>> object_segmentation_graphcut_pixels(segm, centres, gc_regul=.5, seed_size=1)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
    """
    assert np.min(labels_fg_prob) < 1, 'non label can ce strictly 1'
    assert segm.max() <= len(labels_fg_prob), \
        'table of label proba is shorter then the nb of labels in segmentation'
    height, width = segm.shape
    labels_fg_prob = np.array(labels_fg_prob)
    labels_bg_prob = 1. - labels_fg_prob

    assert list(centres), 'at least one center has to be given'
    centres = [np.round(c).astype(int) for c in centres]

    proba = np.ones((height, width, len(centres) + 1))
    proba[:, :, 0] = labels_bg_prob[segm]
    for i in range(len(centres)):
        proba[:, :, i + 1] = labels_fg_prob[segm]

    shape = np.ones((height, width, len(centres) + 1))
    if coef_shape > 0:
        shape_mean, shape_std = shape_mean_std
        shape[:, :, 0] = labels_bg_prob[segm]
        grid_y, grid_x = np.meshgrid(range(width), range(height))
        for i, centre in enumerate(centres):
            diff_x2 = (grid_x - centre[0]) ** 2
            diff_y2 = (grid_y - centre[1]) ** 2
            dist = np.sqrt(diff_x2 + diff_y2)
            cdf = stats.norm.cdf(range(int(np.max(dist) + 1)),
                                 shape_mean, shape_std)
            cum = 1. - cdf + 1e-9
            shape[:, :, i + 1] = cum[dist.astype(int)]

    unary = - np.log(proba) - coef_shape * np.log(shape)
    for i, pos in enumerate(centres):
        if seed_size > 0:
            mask = np.zeros(segm.shape, dtype=bool)
            selem = morphology.disk(seed_size)
            mask[pos[0] - seed_size:pos[0] + seed_size + 1,
                 pos[1] - seed_size:pos[1] + seed_size + 1] = selem
            mask = np.logical_and(mask, segm > 0)
            unary[mask.astype(bool), i + 1] = 0
        else:
            unary[pos[0], pos[1], i + 1] = 0
        # unary[pos[0], pos[1], 0] = np.Inf

    pairwise = (1 - np.eye(proba.shape[-1])) * gc_regul

    cost_v = np.ones((height - 1, width)) * 1.
    cost_h = np.ones((height, width - 1)) * 1.
    labels = cut_grid_graph(unary, pairwise, cost_v, cost_h, n_iter=999)
    segm_obj = labels.reshape(*segm.shape)

    if debug_visual is not None:
        list_unary_imgs = []
        for i in range(unary.shape[-1]):
            list_unary_imgs.append(unary[:, :, i])
        debug_visual['unary_imgs'] = list_unary_imgs
    return segm_obj


def compute_segm_object_shape(img_object, ray_step=5, interp_order=3,
                              smooth_coef=0, shift_method='phase'):
    """ assuming single object in image and compute gravity centre and for
    this point compute Ray features and optionally:
    - interpolate missing values
    - smooth the Ray features

    :param ndarray img_object: binary segmentation of single object
    :param int ray_step: select the angular step    for Ray features
    :param int interp_order: if None, no interpolation is performed
    :param float smooth_coef: smoothing the ray features
    :param str shift_method: use method for estimate shift maxima (phase or max)
    :return tuple(list(int), int):

    >>> img = np.zeros((100, 100))
    >>> img[20:70, 30:80] = 1
    >>> rays, shift = compute_segm_object_shape(img, ray_step=45)
    >>> rays  # doctest: +ELLIPSIS
    [36.7..., 26.0..., 35.3..., 25.0..., 35.3..., 25.0..., 35.3..., 26.0...]
    """
    centre = ndimage.measurements.center_of_mass(img_object)
    centre = [int(round(c)) for c in centre]
    ray_dist = compute_ray_features_segm_2d(img_object, centre, ray_step, 0, edge='down')
    if interp_order is not None and -1 in ray_dist:
        ray_dist = interpolate_ray_dist(ray_dist, interp_order)
    if smooth_coef > 0:
        ray_dist = ndimage.filters.gaussian_filter1d(ray_dist, smooth_coef)
    ray_dist, shift = shift_ray_features(ray_dist, shift_method)
    return ray_dist.tolist(), shift


def compute_object_shapes(list_img_objects, ray_step=5, interp_order=3,
                          smooth_coef=0, shift_method='phase'):
    """ for all object in all images compute gravity center and Ray beatures
    (if object are not split already by different label is made here)

    :param [nadarray] list_img_objects: list of binary segmentation
    :param int ray_step: select the angular step for Ray features
    :param int interp_order: if None, no interpolation is performed
    :param float smooth_coef: smoothing the ray features
    :param str shift_method: use method for estimate shift maxima (phase or max)
    :return tuple(list(list(int)),list(int)):

    >>> img1 = np.zeros((100, 100))
    >>> img1[20:50, 30:60] = 1
    >>> img1[40:80, 50:90] = 2
    >>> img2 = np.zeros((100, 100))
    >>> img2[10:40, 20:50] = 1
    >>> img2[50:80, 20:50] = 1
    >>> img2[50:80, 60:90] = 1
    >>> list_imgs = [img1, img2]
    >>> list_rays, list_shifts = compute_object_shapes(list_imgs, ray_step=45)
    >>> np.array(list_rays).astype(int) # doctest: +NORMALIZE_WHITESPACE
    array([[19, 17,  9, 17, 19, 14, 19, 14],
           [29, 21, 28, 20, 28, 20, 28, 21],
           [22, 16, 21, 15, 21, 15, 21, 16],
           [22, 16, 21, 15, 21, 15, 21, 16],
           [22, 16, 21, 15, 21, 15, 21, 16]])
    >>> np.array(list_shifts) % 180
    array([ 135.,   45.,   45.,   45.,   45.])
    """
    list_rays, list_shifts = [], []
    for img_objects in list_img_objects:
        uq_labels = np.unique(img_objects)
        if len(uq_labels) <= 2:
            # selects individual object
            img_objects, _ = ndimage.measurements.label(img_objects)
            uq_labels = np.unique(img_objects)
        for label in uq_labels[1:]:
            img_object = (img_objects == label)
            rays, shift = compute_segm_object_shape(img_object, ray_step,
                                                    interp_order, smooth_coef,
                                                    shift_method)
            list_rays.append(rays)
            list_shifts.append(shift)

    return list_rays, list_shifts


def compute_cumulative_distrib(means, stds, weights, max_dist):
    """ compute invers cumulative distribution based given means,
    covariance and weights for each segment

    :param [[float]] means: mean values for each model and ray direction
    :param [[float]] stds: STD for each model and ray direction
    :param [float] weights: model wights
    :param float max_dist: maxim distance for shape model
    :return [[float]]:

    >>> cdist = compute_cumulative_distrib(np.array([[1, 2]]),
    ...                 np.array([[1.5, 0.5], [0.5, 1]]), np.array([0.5]), 6)
    >>> np.round(cdist, 2)
    array([[ 1.  ,  0.67,  0.34,  0.12,  0.03,  0.  ,  0.  ],
           [ 1.  ,  0.98,  0.5 ,  0.02,  0.  ,  0.  ,  0.  ]])
    """
    list_cdist = []
    samples = range(int(max_dist) + 1)
    for i in range(means.shape[1]):
        cdf = np.zeros(int(max_dist + 1))
        for j, w in enumerate(weights):
            cdf += stats.norm.cdf(samples, means[j, i], stds[j, i]) * w
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
        cum = 1. - cdf + 1e-9
        list_cdist.append(cum.tolist())
    cdist = np.array(list_cdist)
    # cdist = cdist[:, (np.sum(cdist, axis=0) >= 1e-3)]
    return cdist


def transform_rays_model_cdf_mixture(list_rays, coef_components=1):
    """ compute the mixture model and transform it into cumulative distribution

    :param list(list(int)) list_rays: list ray features (distances)
    :param int coef_components: multiplication for number of components
    :return any, list(list(int)): mixture model, cumulative distribution

    >>> np.random.seed(0)
    >>> list_rays = [[9, 4, 9], [4, 9, 7], [9, 7, 11], [10, 8, 10],
    ...              [9, 11, 8], [4, 8, 5], [8, 10, 6], [9, 7, 11]]
    >>> mm, cdist = transform_rays_model_cdf_mixture(list_rays)
    >>> # the rounding variate a bit according GMM estimated model
    >>> np.round(np.array(cdist) * 4) / 4.  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[ 1. , 1. , 1. , 1. , 1. , 1. , 0.75, 0.75, 0.5 , 0.25, 0. ],
           [ 1. , 1. , 1. , 1. , 1. , 1. , 1.  , 0.75, 0.5 , 0.25, 0. ],
           [ 1. , 1. , 1. , 1. , 1. , 1. , ...,  0.75, 0.5 , 0.25, 0. ]])
    """
    rays = np.array(list_rays)
    ms = cluster.MeanShift()
    ms.fit(rays)
    logging.debug('MeanShift found: %r', np.bincount(ms.labels_))

    nb_components = int(len(np.unique(ms.labels_)) * coef_components)
    mm = mixture.BayesianGaussianMixture(n_components=nb_components)
    # gmm.fit(np.array(list_rays))
    mm.fit(rays, ms.labels_)
    logging.debug('Mixture model found % components with weights: %r',
                  len(mm.weights_), mm.weights_)

    # compute the fairest mean + sigma over all components and ray angles
    max_dist = np.max([[m[i] + np.sqrt(c[i, i]) for i in range(len(m))]
                       for m, c in zip(mm.means_, mm.covariances_)])
    # max_dist = np.max(rays)

    # fixing, AttributeError: 'BayesianGaussianMixture' object has no attribute 'covariances'
    covs = mm.covariances if hasattr(mm, 'covariances') else mm.covariances_
    stds = np.sqrt(abs(covs))[:, np.eye(mm.means_.shape[1], dtype=bool)]
    # stds = np.sum(mm.covariances_, axis=-1)
    cdist = compute_cumulative_distrib(mm.means_, stds, mm.weights_, max_dist)
    return mm, cdist.tolist()


def transform_rays_model_sets_mean_cdf_mixture(list_rays, nb_components=5, slic_size=15):
    """ compute the mixture model and transform it into cumulative distribution

    :param list(list(int)) list_rays: list ray features (distances)
    :param int nb_components: number components in mixture model
    :param int slic_size: superpixel size
    :return tuple(any,list(list(int))):  mixture model, list of stat/param of models

    >>> np.random.seed(0)
    >>> list_rays = [[9, 4, 9], [4, 9, 7], [9, 7, 11], [10, 8, 10],
    ...              [9, 11, 8], [4, 8, 5], [8, 10, 6], [9, 7, 11]]
    >>> mm, mean_cdf = transform_rays_model_sets_mean_cdf_mixture(list_rays, 2)
    >>> len(mean_cdf)
    2
    """
    rays = np.array(list_rays)
    # mm = mixture.GaussianMixture(n_components=nb_components,
    #                                      covariance_type='diag')
    mm = mixture.BayesianGaussianMixture(n_components=nb_components,
                                         covariance_type='diag')
    mm.fit(rays)
    logging.debug('Mixture model found % components with weights: %r',
                  len(mm.weights_), mm.weights_)

    list_mean_cdf = []
    # stds = mm.covariances_[:, np.eye(mm.means_.shape[1], dtype=bool)]
    # stds = mm.covariances_  # for covariance_type='diag'
    # diff_means = np.max(mm.means_, axis=0) - np.min(mm.means_, axis=0)
    for mean, covar in zip(mm.means_, mm.covariances_):
        std = np.sqrt(covar + 1) * 2 + slic_size
        mean = ndimage.gaussian_filter1d(mean, 1)
        std = ndimage.gaussian_filter1d(std, 1)
        max_dist = np.max(mean + 2 * std)
        cdist = compute_cumulative_distrib(np.array([mean]), np.array([std]),
                                           np.array([1]), max_dist)
        list_mean_cdf.append((mean.tolist(), cdist))

    return mm, list_mean_cdf


def transform_rays_model_sets_mean_cdf_kmeans(list_rays, nb_components=5):
    """ compute the mixture model and transform it into cumulative distribution

    :param list(list(int)) list_rays: list ray features (distances)
    :param int nb_components: number components in mixture model
    :return tuple(any,list(list(int))):  mixture model, list of stat/param of models

    >>> np.random.seed(0)
    >>> list_rays = [[9, 4, 9], [4, 9, 7], [9, 7, 11], [10, 8, 10],
    ...              [9, 11, 8], [4, 8, 5], [8, 10, 6], [9, 7, 11]]
    >>> mm, mean_cdf = transform_rays_model_sets_mean_cdf_kmeans(list_rays, 2)
    >>> len(mean_cdf)
    2
    """
    rays = np.array(list_rays)
    kmeans = cluster.KMeans(nb_components)
    kmeans.fit(rays)

    list_mean_cdf = []
    means = kmeans.cluster_centers_
    for lb, mean in enumerate(means):
        std = np.std(np.asarray(list_rays)[kmeans.labels_ == lb], axis=0)
        mean = ndimage.gaussian_filter1d(mean, 1)
        std = ndimage.gaussian_filter1d(std, 1)
        std = (std + 1) * 5.
        max_dist = np.max(mean + 2 * std)
        cdist = compute_cumulative_distrib(np.array([mean]), np.array([std]),
                                           np.array([1]), max_dist)
        list_mean_cdf.append((mean.tolist(), cdist))

    return kmeans, list_mean_cdf


def transform_rays_model_cdf_spectral(list_rays, nb_components=5):
    """ compute the mixture model and transform it into cumulative distribution

    :param list(list(int)) list_rays: list ray features (distances)
    :param int nb_components: number components in mixture model
    :return tuple(any,list(list(int))):  mixture model, list of stat/param of models

    >>> np.random.seed(0)
    >>> list_rays = [[9, 4, 9], [4, 9, 7], [9, 7, 11], [10, 8, 10],
    ...              [9, 11, 8], [4, 8, 5], [8, 10, 6], [9, 7, 11]]
    >>> mm, cdist = transform_rays_model_cdf_spectral(list_rays)
    >>> np.round(cdist, 1).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [[1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.5, 0.2, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.7, 0.5, 0.2, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]]
    """
    rays = np.array(list_rays)
    sc = cluster.SpectralClustering(nb_components)
    sc.fit(rays)
    logging.debug('SpectralClustering found % components with counts: %r',
                  len(np.unique(sc.labels_)), np.bincount(sc.labels_))

    labels = sc.labels_
    means = np.zeros((len(np.unique(labels)), rays.shape[1]))
    stds = np.zeros((len(means), rays.shape[1]))
    for i, lb in enumerate(np.unique(labels)):
        means[i, :] = np.mean(np.asarray(list_rays)[labels == lb], axis=0)
        means[i, :] = ndimage.filters.gaussian_filter1d(means[i, :], 1)
        stds[i, :] = np.std(np.asarray(list_rays)[labels == lb], axis=0)
    stds += 1
    weights = np.bincount(sc.labels_) / float(len(sc.labels_))

    # compute the fairest mean + sigma over all components and ray angles
    max_dist = np.max([[m[i] + c[i] for i in range(len(m))]
                       for m, c in zip(means, stds)])

    cdist = compute_cumulative_distrib(means, stds, weights, max_dist)
    return sc, cdist.tolist()


def transform_rays_model_cdf_kmeans(list_rays, nb_components=None):
    """ compute the mixture model and transform it into cumulative distribution

    :param list(list(int)) list_rays: list ray features (distances)
    :param int nb_components: number components in mixture model
    :return any, list(list(int)):  mixture model, list of stat/param of models

    >>> np.random.seed(0)
    >>> list_rays = [[9, 4, 9], [4, 9, 7], [9, 7, 11], [10, 8, 10],
    ...              [9, 11, 8], [4, 8, 5], [8, 10, 6], [9, 7, 11]]
    >>> mm, cdist = transform_rays_model_cdf_kmeans(list_rays)
    >>> np.round(cdist, 1).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [[1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.7, 0.6, 0.4, 0.2, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.2, 0.1, 0.0]]
    >>> mm, cdist = transform_rays_model_cdf_kmeans(list_rays, nb_components=2)
    """
    rays = np.array(list_rays)
    if not nb_components:
        ms = cluster.MeanShift()
        ms.fit(rays)
        logging.debug('MeanShift found: %r', np.bincount(ms.labels_))
        nb_components = len(np.unique(ms.labels_))
        kmeans = cluster.KMeans(nb_components)
        kmeans.fit(rays, ms.labels_)
    else:
        kmeans = cluster.KMeans(nb_components)
        kmeans.fit(rays)

    labels = kmeans.labels_
    means = kmeans.cluster_centers_
    stds = np.zeros((len(means), rays.shape[1]))
    for i, lb in enumerate(np.unique(labels)):
        stds[i, :] = np.std(np.asarray(list_rays)[labels == lb], axis=0)
    stds += 1
    weights = np.bincount(kmeans.labels_) / float(len(kmeans.labels_))

    # compute the fairest mean + sigma over all components and ray angles
    max_dist = np.max([[m[i] + c[i] for i in range(len(m))]
                       for m, c in zip(means, stds)])

    cdist = compute_cumulative_distrib(means, stds, weights, max_dist)
    return kmeans, cdist.tolist()


def transform_rays_model_cdf_histograms(list_rays, nb_bins=10):
    """ from list of all measured rays create cumulative histogram for each ray

    :param list(list(int)) list_rays: list ray features (distances)
    :param int nb_bins: binarise histogram
    :return:

    >>> list_rays = [[9, 4, 9], [4, 9, 7], [9, 7, 11], [10, 8, 10],
    ...              [9, 11, 8], [4, 8, 5], [8, 10, 6], [9, 7, 11]]
    >>> chist = transform_rays_model_cdf_histograms(list_rays, nb_bins=5)
    >>> chist  # doctest: +NORMALIZE_WHITESPACE
    [[1.0, 1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.625, 0.625, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 0.875, 0.875, 0.875, 0.375, 0.25, 0.25, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.625, 0.5, 0.375, 0.375, 0.0, 0.0]]
    """
    rays = np.array(list_rays)
    max_dist = np.max(rays)
    logging.debug('computing cumulative histogram od size %f for %i bins',
                  max_dist, nb_bins)
    list_chist = []
    for i in range(rays.shape[1]):
        cum = np.zeros(max_dist + 1)
        hist, bin_edges = np.histogram(rays[:, i], nb_bins)
        hist = hist.astype(float) / np.sum(hist)
        bin_edges = bin_edges.astype(int)
        bins = (bin_edges[1:] + bin_edges[:-1]) / 2
        bins = bins.astype(int)
        cum[:bins[0]] = 1
        for j, edge in enumerate(bins):
            val = cum[edge - 1] - hist[j]
            cum[edge:] = val
        list_chist.append(cum.tolist())
    return list_chist


def compute_shape_prior_table_cdf(point, cum_distribution, centre, angle_shift=0):
    """ compute shape prior for a point based on centre, rotation shift
    and cumulative histogram

    :param tuple(int,int) point: single points
    :param tuple(int,int) centre: center of model
    :param [[float]] cum_distribution: cumulative histogram
    :param float angle_shift:
    :return float:

    >>> chist = [[1.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0, 0.0],
    ...          [1.0, 1.0, 0.9, 0.8, 0.7, 0.3, 0.2, 0.2, 0.0],
    ...          [1.0, 1.0, 1.0, 0.7, 0.6, 0.5, 0.3, 0.1, 0.1],
    ...          [1.0, 1.0, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0, 0.0]]
    >>> centre = (1, 1)
    >>> compute_cdf = compute_shape_prior_table_cdf
    >>> compute_cdf([1, 1], chist, centre)
    1.0
    >>> compute_cdf([10, 10], chist, centre)
    0.0
    >>> compute_cdf([10, -10], chist, centre) # doctest: +ELLIPSIS
    0.100...
    >>> compute_cdf([2, 3], chist, centre) # doctest: +ELLIPSIS
    0.805...
    >>> compute_cdf([-3, -2], chist, centre) # doctest: +ELLIPSIS
    0.381...
    >>> compute_cdf([3, -2], chist, centre) # doctest: +ELLIPSIS
    0.676...
    >>> compute_cdf([2, 3], chist, centre, angle_shift=270) # doctest: +ELLIPSIS
    0.891...
    """
    if not isinstance(cum_distribution, np.ndarray):
        cum_distribution = np.array(cum_distribution)
    angle_step = 360. / cum_distribution.shape[0]
    cum_distribution = np.vstack((cum_distribution, cum_distribution[0]))

    dx = point[0] - centre[0]
    dy = point[1] - centre[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    angle = np.rad2deg(np.arctan2(dy, dx))
    angle = ((2 * 360) + 90 - angle - angle_shift) % 360
    angle_norm = angle / angle_step

    if dist >= (cum_distribution.shape[1] - 1):
        return cum_distribution[int(round(angle_norm)), -1]

    a0 = int(np.floor(angle_norm))
    assert a0 < (cum_distribution.shape[0] - 1), \
        'angle %i is larger then size %i' % (a0, cum_distribution.shape[0])
    d0 = int(np.floor(dist))
    assert d0 < (cum_distribution.shape[1] - 1), \
        'distance %i is larger then size %i' % (d0, cum_distribution.shape[1])
    interp = interpolate.interp2d(np.array([[a0, a0 + 1], [a0, a0 + 1]]).T,
                                  np.array([[d0, d0 + 1], [d0, d0 + 1]]),
                                  cum_distribution[a0:a0 + 2, d0:d0 + 2],
                                  kind='linear')
    prior = interp(angle_norm, dist)[0]
    # prior = interp(a0, a0)[0]
    return prior


# def compute_shape_priors_table_cdfs(points, cum_hist, centre, angle_shift=0):
#     """ compute shape prior for a point based on centre, rotation shift
#     and cumulative histogram
#
#     :param tuple(int,int) point:
#     :param tuple(int,int) centre:
#     :param [[float]] cum_hist:
#     :param float shift:
#     :return float:
#
#     >>> chist = [[1.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0, 0.0],
#     ...          [1.0, 1.0, 0.9, 0.8, 0.7, 0.3, 0.2, 0.2, 0.0],
#     ...          [1.0, 1.0, 1.0, 0.7, 0.6, 0.5, 0.3, 0.1, 0.0],
#     ...          [1.0, 1.0, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0, 0.0]]
#     >>> centre = (1, 1)
#     >>> points = [[1, 1], [10, 10], [2, 3], [-3, -2], [3, -2]]
#     >>> priors = compute_shape_priors_table_cdfs(points, centre, chist)
#     >>> np.round(priors, 3)
#     [1.0, 0.0, 0.847, 0.418, 0.514]
#     """
#     raise Exception('This function "compute_shape_priors_table_cdfs" require '
#                     'fix in scipy interpolation part, return strange values.')
#     if not isinstance(points, np.ndarray):
#         points = np.array(points)
#     if not isinstance(cum_hist, np.ndarray):
#         cum_hist = np.array(cum_hist)
#     angle_step = 360. / cum_hist.shape[0]
#     cum_hist = np.vstack((cum_hist, cum_hist[0]))
#     priors = np.zeros(len(points))
#
#     dx = points[:, 0] - centre[0]
#     dy = points[:, 1] - centre[1]
#     dist = np.sqrt(dx ** 2 + dy ** 2)
#     in_range = (dist < cum_hist.shape[1])
#
#     angle = np.rad2deg(np.arctan2(dy, dx))
#     angle = ((2 * 360) + 90 - angle - angle_shift) % 360
#     angle_norm = angle / angle_step
#
#     x, y = np.meshgrid(range(cum_hist.shape[0]), range(cum_hist.shape[1]))
#
#     grid_points = np.array((x.flatten(), y.flatten())).T
#     values = cum_hist.flatten()
#     # FIX: do not return correct values eve for the "input points"
#     priors[in_range] = interpolate.griddata(grid_points, values,
#                                     (angle_norm[in_range], dist[in_range]))
#     return priors


def compute_centre_moment_points(points):
    """ compute centre and moment from set of points

    :param [(float, float)] points:
    :return:

    >>> points = list(zip([0] * 10, np.arange(10))) + [(0, 0)] * 5
    >>> compute_centre_moment_points(points)
    (array([ 0.,  3.]), 0.0)
    >>> points = list(zip(np.arange(10), [0] * 10)) + [(10, 0)]
    >>> compute_centre_moment_points(points)
    (array([ 5.,  0.]), 90.0)
    >>> points = list(zip(-np.arange(10), -np.arange(10))) + [(0, 0)] * 5
    >>> compute_centre_moment_points(points)
    (array([-3., -3.]), 45.0)
    >>> points = list(zip(-np.arange(10), np.arange(10))) + [(-10, 10)]
    >>> compute_centre_moment_points(points)
    (array([-5.,  5.]), 135.0)
    """
    centre = np.mean(points, axis=0)
    diff = np.array(points) - np.tile(centre, (len(points), 1))
    # dist = np.sqrt(np.sum(diff ** 2, axis=1))
    # idx = np.argmax(dist)
    # theta = np.arctan2(diff[idx, 0], diff[idx, 1])

    # # https: // en.wikipedia.org / wiki / Image_moment
    # nb_points = float(len(points))
    # mu_11 = np.sum(np.prod(diff, axis=1)) / nb_points
    # mu_20 = np.sum(diff[:, 0] ** 2) / nb_points
    # mu_02 = np.sum(diff[:, 1] ** 2) / nb_points
    # eps = 1e-9 if (mu_20 - mu_02) == 0 else 0
    # theta = 0.5 * np.arctan(2 * mu_11 / (mu_20 - mu_02 + eps))

    # https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
    if len(points) > 1:
        cov = np.cov(diff.T)
        evals, evecs = np.linalg.eig(cov)
        evec1 = evecs[:, np.argmax(evals)]
        theta = np.arctan2(evec1[0], evec1[1])
    else:
        theta = 0

    theta = (360 + round(np.rad2deg(theta))) % 360
    return centre, theta


def compute_update_shape_costs_points_table_cdf(lut_shape_cost, points, labels,
                                                init_centres, centres, shifts,
                                                volumes, shape_chist,
                                                selected_idx=None,
                                                swap_shift=False,
                                                dict_thresholds=None):
    """ update the shape prior for given segmentation (new centre is computed),
    set of points and cumulative histogram representing the shape model

    :param lut_shape_cost: look-up-table for shape cost for GC
    :param [[int, int]] points: subsample space, points = superpixel centres
    :param list(int) labels: labels for points to be assigned to an object
    :param [[int, int]] init_centres: initial centre position for compute
        center shift during the iteretions
    :param [[int, int]] centres: actual centre postion
    :param list(int) shifts: orientation for each region / object
    :param list(int) volumes: size / volume for each region
    :param shape_chist: represent the shape prior and histograms
    :param list(int) selected_idx: selected object for update
    :param bool swap_shift: allow swapping orientation by 90 degree,
        try to get out from local optimal
    :param dict dict_thresholds: configuration with thresholds
    :param dict|None dict_thresholds: set some threshold updating shape prior
    :return tuple(list(float),list(int)):

    >>> cdf = np.zeros((8, 20))
    >>> cdf[:10] = 0.5
    >>> cdf[:4] = 1.0
    >>> points = np.array([[13, 16], [1, 5], [10, 15], [15, 25], [10, 5]])
    >>> labels = np.ones(len(points))
    >>> s_costs = np.zeros((len(points), 2))
    >>> s_costs, centres, shifts, _ = compute_update_shape_costs_points_table_cdf(
    ...     s_costs, points, labels, [(0, 0)], [(np.Inf, np.Inf)], [0], [0], (None, cdf))
    >>> centres
    array([[10, 13]])
    >>> shifts
    array([ 209.])
    >>> np.round(s_costs, 3)
    array([[ 0.   ,  0.673],
           [ 0.   , -0.01 ],
           [ 0.   ,  0.184],
           [ 0.   ,  0.543],
           [ 0.   ,  0.374]])
    >>> dict_thrs = RG2SP_THRESHOLDS
    >>> dict_thrs['centre_init'] = 1
    >>> _, centres, _, _ = compute_update_shape_costs_points_table_cdf(
    ...     s_costs, points, labels, [(7, 18)], [(np.Inf, np.Inf)], [0], [0], (None, cdf),
    ...     dict_thresholds=dict_thrs)
    >>> np.round(centres, 1)
    array([[  7.5,  17.1]])
    """
    assert len(points) == len(labels), \
        'number of points (%i) and labels (%i) should match' % (len(points), len(labels))
    if selected_idx is None:
        selected_idx = list(range(len(points)))
    thresholds = RG2SP_THRESHOLDS if dict_thresholds is None else dict_thresholds
    _, cdf = shape_chist
    # segm_obj = labels[slic]
    for i, centre in enumerate(centres):
        # segm_binary = (segm_obj == i + 1)
        # centre_new = ndimage.measurements.center_of_mass(segm_binary)
        # ray = seg_fts.compute_ray_features_segm_2d(
        #     segm_binary, centre_new, edge='down', angle_step=10)
        # _, shift = seg_fts.shift_ray_features(ray)
        centre_new, shift = compute_centre_moment_points(points[labels == i + 1])
        centre_new = np.round(centre_new).astype(int)

        if swap_shift:
            shift = (shift + 90) % 360
            shifts[i] = shift

        # shift it to the edge of max init distance
        cdist_init_2 = np.sum((np.array(centre_new) - np.array(init_centres[i])) ** 2)
        if cdist_init_2 > thresholds['centre_init'] ** 2:
            diff = np.asarray(centre_new) - np.asarray(init_centres[i])
            thr = thresholds['centre_init'] / np.sqrt(cdist_init_2)
            centre_new = init_centres[i] + thr * diff

        cdist_act_2 = np.sum((np.array(centre_new) - np.array(centre)) ** 2)
        if cdist_act_2 <= thresholds['centre'] ** 2 and \
                np.abs(shift - shifts[i]) <= thresholds['shift'] and not swap_shift:
            continue
        if cdist_act_2 > thresholds['centre'] ** 2:
            centres[i] = centre_new.tolist()
        if np.abs(shift - shifts[i]) > thresholds['shift']:
            shifts[i] = shift

        shape_proba = np.zeros(len(points))
        for j in selected_idx:
            shape_proba[j] = compute_shape_prior_table_cdf(points[j], cdf,
                                                           centres[i], shifts[i])

        lut_shape_cost[:, i + 1] = - np.log(shape_proba + MIN_SHAPE_PROB)

    lut_shape_cost[np.isinf(lut_shape_cost)] = GC_REPLACE_INF
    return lut_shape_cost, np.array(centres), np.array(shifts), volumes


def compute_update_shape_costs_points_close_mean_cdf(
        lut_shape_cost, slic, points, labels, init_centres, centres, shifts,
        volumes, shape_model_cdfs, selected_idx=None, swap_shift=False,
        dict_thresholds=None):
    """ update the shape prior for given segmentation (new centre is computed),
    set of points and cumulative histogram representing the shape model

    :param lut_shape_cost: look-up-table for shape cost for GC
    :param ndarray slic: superpixel segmentation
    :param [[int, int]] points: subsample space, points = superpixel centres
    :param list(int) labels: labels for points to be assigned to an object
    :param [[int, int]] init_centres: initial centre position for compute
        center shift during the iterations
    :param [[int, int]] centres: actual centre position
    :param list(int) shifts: orientation for each region / object
    :param list(int) volumes: size / volume for each region
    :param shape_model_cdfs: represent the shape prior and histograms
    :param list(int) selected_idx: selected object for update
    :param bool swap_shift: allow swapping orientation by 90 degree,
        try to get out from local optimal
    :param dict dict_thresholds: configuration with thresholds
    :param dict|None dict_thresholds: set some threshold updating shape prior
    :return tuple(list(float),list(int)):

    >>> np.random.seed(0)
    >>> h, w, step = 8, 8, 2
    >>> slic = np.array([[ 0,  0,  1,  1,  2,  2,  3,  3],
    ...                  [ 0,  0,  1,  1,  2,  2,  3,  3],
    ...                  [ 4,  4,  5,  5,  6,  6,  7,  7],
    ...                  [ 4,  4,  5,  5,  6,  6,  7,  7],
    ...                  [ 8,  8,  9,  9, 10, 10, 11, 11],
    ...                  [ 8,  8,  9,  9, 10, 10, 11, 11],
    ...                  [12, 12, 13, 13, 14, 14, 15, 15],
    ...                  [12, 12, 13, 13, 14, 14, 15, 15]])
    >>> points = np.array([(0, 0), (0, 2), (0, 4), (0, 6), (2, 0), (2, 2),
    ...                    (2, 4), (2, 6), (4, 0), (4, 2), (4, 4), (4, 6),
    ...                    (6, 0), (6, 2), (6, 4), (6, 6)])
    >>> labels = np.array([0] * 4 + [0, 1, 1, 0, 0, 1, 1, 0] + [0] * 4)
    >>> cdf1, cdf2 = np.zeros((8, 10)),  np.zeros((8, 7))
    >>> cdf1[:7] = 0.5
    >>> cdf1[:4] = 1.0
    >>> cdf2[:6] = 1.0
    >>> set_m_cdf = [([4] * 8, cdf1), ([5] * 8, cdf2)]
    >>> s_costs = np.zeros((len(points), 2))
    >>> mm = mixture.GaussianMixture(2).fit(np.random.random((100, 8)))
    >>> s_costs, centres, shifts, _ = compute_update_shape_costs_points_close_mean_cdf(
    ...                         s_costs, slic, points, labels, [(0, 0)],
    ...                         [(np.Inf, np.Inf)], [0], [0], (mm, set_m_cdf))
    >>> centres
    array([[3, 3]])
    >>> shifts
    array([ 90.])
    >>> np.round(s_costs, 3)  # doctest: +ELLIPSIS
    array([[ 0.   , -0.01 ],
           [ 0.   , -0.01 ],
           [ 0.   , -0.01 ],
           [ 0.   , -0.01 ],
           [ 0.   , -0.01 ],
           [ 0.   , -0.01 ],
           [ 0.   , -0.01 ],
           [ 0.   ,  0.868],
           [ 0.   , -0.01 ],
           ...
           [ 0.   ,  4.605]])
    """
    assert len(points) == len(labels), \
        'number of points (%i) and labels (%i) should match' \
        % (len(points), len(labels))
    selected_idx = range(len(points)) if selected_idx is None else selected_idx
    thresholds = RG2SP_THRESHOLDS if dict_thresholds is None else dict_thresholds
    segm_obj = labels[slic]
    model, list_mean_cdf = shape_model_cdfs
    _, list_cdfs = zip(*list_mean_cdf)
    angle_step = 360 / len(list_cdfs[0])
    for i, centre in enumerate(centres):
        # aproximate shape
        segm_binary = (segm_obj == i + 1)
        centre_new, shift = compute_centre_moment_points(points[labels == i + 1])
        centre_new = np.round(centre_new).astype(int)
        rays, _ = compute_segm_object_shape(segm_binary, angle_step, smooth_coef=0)
        if swap_shift:
            shift = (shift + 90) % 360
            shifts[i] = shift

        volume = np.sum(labels == (i + 1))
        volume_diff = 0 if volumes[i] == 0 \
            else np.abs(volume - volumes[i]) / float(volumes[i])

        # shift it to the edge of max init distance
        cdist_init_2 = np.sum((np.array(centre_new) - np.array(init_centres[i])) ** 2)
        if cdist_init_2 > thresholds['centre_init'] ** 2:

            diff = np.asarray(centre_new) - np.asarray(init_centres[i])
            thr = thresholds['centre_init'] / np.sqrt(cdist_init_2)
            centre_new = init_centres[i] + thr * diff

        cdist_act_2 = np.sum((np.array(centre_new) - np.array(centre)) ** 2)
        if cdist_act_2 <= thresholds['centre'] ** 2 \
                and np.abs(shift - shifts[i]) <= thresholds['shift'] \
                and volume_diff <= thresholds['volume'] \
                and not swap_shift:
            continue
        if cdist_act_2 > thresholds['centre'] ** 2:
            centres[i] = centre_new.tolist()
        if np.abs(shift - shifts[i]) > thresholds['shift']:
            shifts[i] = shift
        if volume_diff > thresholds['volume']:
            volumes[i] = volume

        # select closest
        # dists = [spatial.distance.euclidean(rays, mean) for mean in model.means_]
        # dists = [np.sum((np.array(rays) - np.array(mean)) ** 2) for mean in model.means_]
        # dists = [np.median((np.array(rays) - np.array(mean)) ** 2) for mean in model.means_]
        # close_idx = np.argmin(dists)

        weights = model.predict_proba([rays]).ravel()
        cdist = np.zeros(np.max([cdf.shape for cdf in list_cdfs], axis=0))
        for j, cdf in enumerate(list_cdfs):
            cdist[:, :cdf.shape[1]] += weights[j] * cdf

        shape_proba = np.zeros(len(points))
        for j in selected_idx:
            shape_proba[j] = compute_shape_prior_table_cdf(points[j], cdist,
                                                           centres[i], shifts[i])
        lut_shape_cost[:, i + 1] = - np.log(shape_proba + MIN_SHAPE_PROB)

    lut_shape_cost[np.isinf(lut_shape_cost)] = GC_REPLACE_INF
    return lut_shape_cost, np.array(centres), np.array(shifts), volumes


def compute_data_costs_points(slic, slic_prob_fg, centres, labels):
    """ compute Look up Table ro date term costs

    :param nadarray slic: superpixel segmentation
    :param list(float) slic_prob_fg: weight for particular pixel belongs to FG
    :param [[int, int]] centres: actual centre position
    :param list(int) labels: labels for points to be assigned to an object
    :return:
    """
    data_proba = np.empty((len(labels), len(centres) + 1))
    data_proba[:, 0] = 1. - slic_prob_fg
    for i, centre in enumerate(centres):
        data_proba[:, i + 1] = slic_prob_fg
        vertex = slic[centre[0], centre[1]]
        labels[vertex] = i + 1
    # use an offset to avoid 0 in logarithm
    lut_data_cost = -np.log(data_proba + 1e-9)
    lut_data_cost[np.isinf(lut_data_cost)] = GC_REPLACE_INF
    return lut_data_cost, labels


def update_shape_costs_points(lut_shape_cost, slic, points, labels, init_centres,
                              centres, shifts, volumes, shape_model, shape_type,
                              selected_idx=None, swap_shift=False,
                              dict_thresholds=None):
    """ update the shape prior for given segmentation (new centre is computed),
    set of points and shape model

    :param lut_shape_cost: look-up-table for shape cost for GC
    :param nadarray slic: superpixel segmentation
    :param [[int, int]] points: subsample space, points = superpixel centres
    :param list(int) labels: labels for points to be assigned to an object
    :param [[int, int]] init_centres: initial centre position for compute
        center shift during the iteretions
    :param [[int, int]] centres: actual centre postion
    :param list(int) shifts: orientation for each region / object
    :param [int] volumes: size / volume for each region
    :param shape_model: represent the shape prior and histograms
    :param str shape_type: type or shape model
    :param [int] selected_idx: selected object for update
    :param bool swap_shift: allow swapping orientation by 90 degree,
        try to get out from local optima
    :param dict dict_thresholds: configuration with thresholds
    :param dict|None dict_thresholds: set some threshold updating shape prior
    :return tuple(list(float),list(int)):
    """
    thresholds = RG2SP_THRESHOLDS if dict_thresholds is None else dict_thresholds
    if shape_type == 'cdf':
        return compute_update_shape_costs_points_table_cdf(
            lut_shape_cost, points, labels, init_centres, centres, shifts,
            volumes, shape_model, selected_idx, swap_shift, thresholds)
    elif shape_type == 'set_cdfs':
        # select closest by distance and use cdf
        return compute_update_shape_costs_points_close_mean_cdf(
            lut_shape_cost, slic, points, labels, init_centres, centres, shifts,
            volumes, shape_model, selected_idx, swap_shift, thresholds)
    else:
        raise NameError('Not supported type of shape model "%s"' % shape_type)


def compute_pairwise_penalty(edges, labels, prob_bg_fg=0.05, prob_fg1_fg2=0.01):
    """ compute cost of neighboring labels pionts

    :param [(int, int)] edges: graph edges, connectivity
    :param [int] labels: labels for vertexes
    :param float prob_bg_fg: penalty between background and foreground
    :param float prob_fg1_fg2: penaly between two different foreground classes
    :return:

    >>> edges = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [2, 4]])
    >>> labels = np.array([0, 0, 1, 2, 1])
    >>> compute_pairwise_penalty(edges, labels, 0.05, 0.01)
    array([ 0.        ,  2.99573227,  2.99573227,  4.60517019,  0.        ])
    """
    edges_labeled = labels[edges]
    is_diff = (edges_labeled[:, 0] != edges_labeled[:, 1])
    is_bg = np.logical_or(edges_labeled[:, 0] == 0, edges_labeled[:, 1] == 0)
    is_bg = np.logical_and(is_diff, is_bg)
    costs = - np.log(prob_fg1_fg2) * is_diff
    costs[is_bg] = - np.log(prob_bg_fg)
    return costs


def get_neighboring_candidates(slic_neighbours, labels, object_idx,
                               use_other_obj=True):
    """ get neighboring candidates from background
    and optionally also from foreground if it is allowed

    :param [[int]] slic_neighbours: list of neighboring superpixel for each one
    :param [int] labels: labels for each superpixel
    :param int object_idx:
    :param bool use_other_obj: allowing use another foreground object
    :return [int]:

    >>> neighbours = [[1], [0, 2, 3], [1, 3], [1, 2]]
    >>> labels = np.array([0, 0, 1, 1])
    >>> get_neighboring_candidates(neighbours, labels, 1)
    [1]
    """
    neighbours = []
    for l_idx in np.array(slic_neighbours)[labels == object_idx]:
        neighbours += l_idx
    neighbours = np.unique(neighbours)
    if use_other_obj:
        neighbours = [lb for lb in neighbours if labels[lb] != object_idx]
    else:
        neighbours = [lb for lb in neighbours if labels[lb] == 0]
    return neighbours


def compute_rg_crit(labels, lut_data_cost, lut_shape_cost, slic_weights, edges,
                    coef_data, coef_shape, coef_pairwise, prob_label_trans):
    all_range = np.arange(len(labels))
    crit_data = coef_data * lut_data_cost[all_range, labels]
    crit_shape = coef_shape * lut_shape_cost[all_range, labels]
    crit = np.sum(slic_weights * (crit_data + crit_shape))
    if coef_pairwise > 0:
        pairwise_costs = compute_pairwise_penalty(edges, labels,
                                                  prob_label_trans[0],
                                                  prob_label_trans[1])
        pairwise_costs[np.isinf(pairwise_costs)] = GC_REPLACE_INF
        crit += coef_pairwise * np.sum(pairwise_costs)
    return crit


def compute_segm_prob_fg(slic, segm, labels_prob):
    """ compute probability being forground from input segmentation

    :param ndarray slic:
    :param ndarray segm:
    :param list(float) labels_prob:
    :return:

    >>> slic = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [2, 2, 2, 2, 3, 3, 3, 3]])
    >>> segm = np.array([0, 1, 1, 0])[slic]
    >>> compute_segm_prob_fg(slic, segm, [0.3, 0.8])
    array([ 0.3,  0.8,  0.8,  0.3])
    """
    label_hist = histogram_regions_labels_norm(slic, segm)
    slic_labels = np.argmax(label_hist, axis=1)
    slic_prob_fg = np.array(labels_prob)[slic_labels]
    return slic_prob_fg


def region_growing_shape_slic_greedy(slic, slic_prob_fg, centres, shape_model,
                                     shape_type='cdf', coef_data=1., coef_shape=1,
                                     coef_pairwise=1, prob_label_trans=(.1, .01),
                                     allow_obj_swap=True, greedy_tol=1e-3,
                                     dict_thresholds=None, nb_iter=999,
                                     debug_history=None):
    """ Region growing method with given shape prior on pre-segmented images
    it uses the Greedy strategy and set some stopping criterion

    :param ndarray slic: superpixel segmentation
    :param list(float) slic_prob_fg: weight for particular superpixel belongs to FG
    :param [(int, int)] centres: list of initial centres
    :param shape_model: represent the shape prior and histograms
    :param str shape_type: identification of used shape model
    :param float coef_data: weight for data prior
    :param float coef_shape: weight for shape prior
    :param float coef_pairwise: setting for pairwise cost
    :param prob_label_trans: probability transition between background (first)
        and objects and among objects (second)
    :param bool allow_obj_swap: allow swapping foreground object labels
    :param float greedy_tol: stopping criterion - energy change between inters
    :param dict dict_thresholds: configuration with thresholds
    :param int nb_iter: maximal number of iterations
    :param dict|None dict_thresholds: set some threshold updating shape prior
    :return:

    >>> np.random.seed(0)
    >>> h, w, step = 15, 20, 2
    >>> segm = np.zeros((h, w), dtype=int)
    >>> segm[3:12, 5:17] = 1
    >>> segm
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> slic = np.zeros((h, w), dtype=int)
    >>> for i in range(int(np.ceil(h / float(step)))):
    ...     for j in range(int(np.ceil(w / float(step)))):
    ...         val = i * (w / step) + j
    ...         i_step, j_step = int(i * step), int(j * step)
    ...         slic[i_step:int(i_step + step), j_step:int(j_step + step)] = val
    >>> centres = [(7.5, 10)]
    >>> chist = [[1.] * 3 + [0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.0],
    ...          [1.] * 3 + [0.9, 0.8, 0.7, 0.3, 0.2, 0.2, 0.1],
    ...          [1.] * 3 + [1.0, 0.7, 0.6, 0.5, 0.3, 0.1, 0.1],
    ...          [1.] * 3 + [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]]
    >>> dict_debug = {}
    >>> slic_prob_fg = compute_segm_prob_fg(slic, segm, [0.1, 0.9])
    >>> labels = region_growing_shape_slic_greedy(slic, slic_prob_fg, centres,
    ...                                           (None, chist), coef_pairwise=0,
    ...                                           debug_history=dict_debug)
    >>> np.round(dict_debug['criteria']).astype(int)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([397, 325, 307, 289, 272, 238, 204, 188, 173, ..., 81,  81])
    >>> labels[slic]
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> labels = region_growing_shape_slic_greedy(slic, slic_prob_fg, centres,
    ...                                           (None, chist), coef_pairwise=1,
    ...                                           debug_history=dict_debug)
    >>> np.round(dict_debug['criteria']).astype(int)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([406, 352, 334, 316, 300, 283, 270, 254, 238, 226, 210, ..., 123, 123])
    >>> labels[slic]
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> segm = np.ones((h, w), dtype=int)
    >>> chist = np.zeros((16, 9))
    >>> chist[:, :5] = 1.
    >>> slic_prob_fg = compute_segm_prob_fg(slic, segm, [0.1, 0.9])
    >>> labels = region_growing_shape_slic_greedy(slic, slic_prob_fg, [(6.5, 9)],
    ...                                           (None, chist), coef_shape=10,
    ...                                           coef_pairwise=1,
    ...                                           debug_history=dict_debug)
    >>> np.round(dict_debug['criteria']).astype(int)  # doctest: +NORMALIZE_WHITESPACE
    array([7506, 7120, 6715, 6328, 5719, 5719])
    >>> labels[slic]
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    """
    assert len(slic_prob_fg) >= np.max(slic), 'dims of probs %s and slic %s not match' \
                                              % (len(slic_prob_fg), np.max(slic))
    thresholds = RG2SP_THRESHOLDS if dict_thresholds is None else dict_thresholds
    slic_points = superpixel_centers(slic)
    slic_points = np.round(slic_points).astype(int)
    slic_weights = np.bincount(slic.ravel())
    init_centres = np.round(centres).astype(int)

    _, edges = make_graph_segm_connect_grid2d_conn4(slic)
    slic_neighbours = get_neighboring_segments(edges)
    labels = np.zeros(len(slic_points), dtype=int)

    lut_data_cost, labels = compute_data_costs_points(slic, slic_prob_fg,
                                                      init_centres, labels)
    # create matrix for cost where in layers are individual objects
    lut_shape_cost = np.empty((len(labels), len(init_centres) + 1))
    # set the background
    lut_shape_cost[:, 0] = - np.log(1 - slic_prob_fg)
    # create other empty variables
    centres = np.ones(np.asarray(init_centres).shape) * np.Inf
    shifts = np.zeros(len(init_centres))
    volumes = [1] * len(shifts)
    list_swap_shift = [False]
    # update variables
    lut_shape_cost, centres, shifts, volumes = update_shape_costs_points(
        lut_shape_cost, slic, slic_points, labels, init_centres, centres, shifts,
        volumes, shape_model, shape_type, None, False, thresholds)

    if debug_history is not None:
        debug_history.update({'criteria': [], 'labels': [],
                              'centres': [], 'shifts': [],
                              'lut_data_cost': lut_data_cost.copy(),
                              'lut_shape_cost': []})

    for _ in range(nb_iter):
        labels = enforce_center_labels(slic, labels, centres)
        crit = compute_rg_crit(labels, lut_data_cost, lut_shape_cost,
                               slic_weights, edges, coef_data, coef_shape,
                               coef_pairwise, prob_label_trans)
        if debug_history is not None:
            debug_history['labels'].append(labels.copy())
            debug_history['criteria'].append(crit)
            debug_history['centres'].append(centres.copy())
            debug_history['shifts'].append(shifts.tolist())
            debug_history['lut_shape_cost'].append(lut_shape_cost.copy())

        # todo, do it as only update
        candidates, objs_idx = [], []
        for i in range(len(centres)):
            near = get_neighboring_candidates(slic_neighbours, labels, i + 1,
                                              allow_obj_swap)
            candidates += near
            objs_idx += [i + 1] * len(near)

        lut_shape_cost, centres, shifts, volumes = update_shape_costs_points(
            lut_shape_cost, slic, slic_points, labels, init_centres, centres,
            shifts, volumes, shape_model, shape_type, None, list_swap_shift[-1],
            thresholds)

        crit = compute_rg_crit(labels, lut_data_cost, lut_shape_cost,
                               slic_weights, edges, coef_data, coef_shape,
                               coef_pairwise, prob_label_trans)

        candidates_scores = []
        for idx, lb in zip(objs_idx, candidates):
            labels_new = labels.copy()
            labels_new[lb] = idx
            crit_new = compute_rg_crit(labels_new, lut_data_cost,
                                       lut_shape_cost, slic_weights, edges,
                                       coef_data, coef_shape, coef_pairwise,
                                       prob_label_trans)
            energy_change = crit - crit_new
            candidates_scores.append((idx, lb, energy_change))
        candidates_scores = sorted(candidates_scores, key=lambda x: x[2],
                                   reverse=True)

        if not candidates_scores or candidates_scores[0][2] < 0:
            # break
            # try the shaking again
            if any(list_swap_shift[-7:]):
                break
            list_swap_shift.append(True)
        else:
            list_swap_shift.append(False)

        best_score = candidates_scores[0][2]
        for lb, idx, score in candidates_scores:
            if (best_score - score) / best_score < greedy_tol and score > 0:
                labels[idx] = lb

    return labels


def prepare_graphcut_variables(candidates, slic_points, slic_neighbours,
                               slic_weights, labels, nb_centres,
                               lut_data_cost, lut_shape_cost,
                               coef_data, coef_shape, coef_pairwise, prob_label_trans):
    """ for boundary get connected points in BG and FG
    construct graph and set potentials and hard connect BG and FG in unary

    :param [int] candidates: list of candidates, neighbours of actual objects
    :param [(int, int)] slic_points:
    :param [[int]] slic_neighbours: list of neighboring superpixel for each one
    :param list(float) slic_weights: weight for each superpixel
    :param [int] labels: labels for each superpixel
    :param int nb_centres: number of centres - classes
    :param ndarray lut_data_cost: look-up-table for data cost for each
        object (class) with superpixel as first index
    :param ndarray lut_shape_cost: look-up-table for shape cost for each
        object (class) with superpixel as first index
    :param float coef_data: weight for data priors
    :param float coef_shape: weight for shape priors
    :param float coef_pairwise: CG pairwise coeficient
    :param prob_label_trans: probability transition between background (first)
        and objects and among objects (second)
    :return:
    """
    assert np.max(candidates) < len(slic_points), \
        'max candidate idx: %d for %d centres' \
        % (np.max(candidates), len(slic_points))
    max_slic_neighbours = max(max(l) for l in slic_neighbours)
    assert max_slic_neighbours < len(slic_points), \
        'max slic neighbours idx: %d for %d centres' \
        % (max_slic_neighbours, len(slic_points))
    unary = np.zeros((len(candidates), nb_centres + 1))
    vertexes, edges = list(candidates), []
    for i, idx in enumerate(candidates):
        near_idx = slic_neighbours[idx]
        near_labels = labels[near_idx]
        cost = coef_data * lut_data_cost[idx] + coef_shape * lut_shape_cost[idx]
        unary[i, :] = slic_weights[idx] * cost
        for lb in range(unary.shape[-1]):
            if lb not in near_labels:
                unary[i, lb] = GC_REPLACE_INF
        for n_idx in near_idx:
            if n_idx not in vertexes:
                vertexes.append(n_idx)
                u = np.ones(unary.shape[-1]) * GC_REPLACE_INF
                u[labels[n_idx]] = 0
                unary = np.vstack((unary, u))
            j = vertexes.index(n_idx)
            edges.append((i, j))

    # remove too small unary terms
    min_unary = -np.log(MAX_UNARY_PROB)
    unary[unary < min_unary] = min_unary

    spatial_dist = compute_spatial_dist(slic_points[vertexes], edges, relative=True)
    edge_weights = np.ones(len(edges)) / spatial_dist

    pairwise = np.empty((unary.shape[-1], unary.shape[-1]))
    pairwise[:, :] = - np.log(prob_label_trans[0])
    pairwise[1:, 1:] = - np.log(prob_label_trans[1])
    pairwise[np.eye(unary.shape[-1], dtype=bool)] = 0
    pairwise *= coef_pairwise

    # limit the maximal value
    pairwise[pairwise > MAX_PAIRWISE_COST] = MAX_PAIRWISE_COST
    return vertexes, np.array(edges), edge_weights, unary, pairwise


def enforce_center_labels(slic, labels, centres):
    """ force the labels to hold label of the center,
    prevention of desepearing labels of any center in list

    :param slic:
    :param labels:
    :param centres:
    :return:
    """
    for i, center in enumerate(centres):
        idx = slic[int(center[0]), int(center[1])]
        labels[idx] = i + 1
    return labels


def region_growing_shape_slic_graphcut(slic, slic_prob_fg, centres, shape_model,
                                       shape_type='cdf', coef_data=1., coef_shape=1,
                                       coef_pairwise=2, prob_label_trans=(0.1, 0.03),
                                       optim_global=True, allow_obj_swap=True,
                                       dict_thresholds=None, nb_iter=999,
                                       debug_history=None):
    """ Region growing method with given shape prior on pre-segmented images
    it uses the GraphCut strategy on neigbouring superpixels

    :param ndarray slic: superpixel segmentation
    :param list(float) slic_prob_fg: weight for particular superpixel belongs to FG
    :param [(int, int)] centres: list of initial centres
    :param shape_model: represent the shape prior and histograms
    :param str shape_type: identification of used shape model
    :param float coef_data: weight for data prior
    :param float coef_shape: weight for shape prior
    :param float coef_pairwise: setting for pairwise cost
    :param prob_label_trans: probability transition between background (first)
        and objects and among objects (second)
    :param bool optim_global: optimise the GC as global or per object
    :param bool allow_obj_swap: allow swapping foreground object labels
    :param dict dict_thresholds: configuration with thresholds
    :param int nb_iter: maximal number of iterations
    :param dict|None dict_thresholds: set some threshold updating shape prior

    >>> h, w, step = 15, 20, 2
    >>> segm = np.zeros((h, w), dtype=int)
    >>> segm[3:12, 5:17] = 1
    >>> segm
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> slic = np.zeros((h, w), dtype=int)
    >>> for i in range(int(np.ceil(h / float(step)))):
    ...     for j in range(int(np.ceil(w / float(step)))):
    ...         val = i * (w / step) + j
    ...         i_step, j_step = int(i * step), int(j * step)
    ...         slic[i_step:int(i_step + step), j_step:int(j_step + step)] = val
    >>> centres = [(7.5, 10)]
    >>> chist = [[1.] * 3 + [0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.0],
    ...          [1.] * 3 + [0.9, 0.8, 0.7, 0.3, 0.2, 0.2, 0.1],
    ...          [1.] * 3 + [1.0, 0.7, 0.6, 0.5, 0.3, 0.1, 0.1],
    ...          [1.] * 3 + [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]]
    >>> dict_debug = {}
    >>> slic_prob_fg = compute_segm_prob_fg(slic, segm, [0.1, 0.9])
    >>> labels = region_growing_shape_slic_graphcut(slic, slic_prob_fg, centres,
    ...                                             (None, chist), coef_pairwise=0,
    ...                                             debug_history=dict_debug)
    >>> np.round(dict_debug['criteria']).astype(int)
    array([397, 325, 206, 111,  81,  81])
    >>> labels[slic]
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> labels = region_growing_shape_slic_graphcut(slic, slic_prob_fg, centres,
    ...                                             (None, chist), coef_pairwise=2,
    ...                                             debug_history=dict_debug)
    >>> np.round(dict_debug['criteria']).astype(int)
    array([415, 380, 289, 193, 164, 164])
    >>> labels[slic]
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> segm = np.ones((h, w), dtype=int)
    >>> chist = np.zeros((16, 9))
    >>> chist[:, :5] = 1.
    >>> dict_debug = {}
    >>> slic_prob_fg = compute_segm_prob_fg(slic, segm, [0.1, 0.9])
    >>> labels = region_growing_shape_slic_graphcut(slic, slic_prob_fg, [(6.5, 9)],
    ...                                             (None, chist), coef_shape=10.,
    ...                                             coef_pairwise=1,
    ...                                             debug_history=dict_debug)
    >>> np.round(dict_debug['criteria']).astype(int)
    array([7506, 7120, 6328, 5719, 5719])
    >>> labels[slic]
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    assert len(slic_prob_fg) >= np.max(slic), \
        'dims of probs %s and slic %s not match' \
        % (len(slic_prob_fg), np.max(slic))
    thresholds = RG2SP_THRESHOLDS if dict_thresholds is None else dict_thresholds
    slic_points = superpixel_centers(slic)
    slic_points = np.round(slic_points).astype(int)
    slic_weights = np.bincount(slic.ravel())
    init_centres = np.round(centres).astype(int)

    _, edges = make_graph_segm_connect_grid2d_conn4(slic)
    slic_neighbours = get_neighboring_segments(edges)
    labels = np.zeros(len(slic_points), dtype=int)
    labels_history = [labels.copy()]

    lut_data_cost, labels = compute_data_costs_points(slic, slic_prob_fg,
                                                      init_centres, labels)

    lut_shape_cost = np.empty((len(labels), len(init_centres) + 1))
    # use an offset to avoid 0 in logarithm
    lut_shape_cost[:, 0] = - np.log(1 - slic_prob_fg + 1e-9)
    centres = np.ones(np.asarray(init_centres).shape) * np.Inf
    shifts = np.zeros(len(init_centres))
    volumes = [1] * len(shifts)
    list_swap_shift = [False]
    lut_shape_cost, centres, shifts, volumes = update_shape_costs_points(
        lut_shape_cost, slic, slic_points, labels, init_centres, centres, shifts,
        volumes, shape_model, shape_type, None, False, thresholds)

    if debug_history is not None:
        debug_history.update({'criteria': [], 'labels': [],
                              'centres': [], 'shifts': [],
                              'lut_data_cost': lut_data_cost.copy(),
                              'lut_shape_cost': []})

    for _ in range(nb_iter):
        labels = enforce_center_labels(slic, labels, centres)
        crit = compute_rg_crit(labels, lut_data_cost, lut_shape_cost,
                               slic_weights, edges, coef_data, coef_shape,
                               coef_pairwise, prob_label_trans)
        if debug_history is not None:
            debug_history['labels'].append(labels.copy())
            debug_history['criteria'].append(crit)
            debug_history['centres'].append(centres.copy())
            debug_history['shifts'].append(shifts.tolist())
            debug_history['lut_shape_cost'].append(lut_shape_cost.copy())

        labels_gc = labels.copy()

        if optim_global:
            candidates, labels_gc = [], labels.copy()
            for i in range(len(centres)):
                candidates += get_neighboring_candidates(slic_neighbours, labels,
                                                         i + 1, allow_obj_swap)

            lut_shape_cost, centres, shifts, volumes = update_shape_costs_points(
                lut_shape_cost, slic, slic_points, labels, init_centres, centres,
                shifts, volumes, shape_model, shape_type, None, list_swap_shift[-1],
                thresholds)

            gc_vestexes, gc_edges, edge_weights, unary, pairwise = \
                prepare_graphcut_variables(candidates, slic_points, slic_neighbours,
                                           slic_weights, labels, len(centres),
                                           lut_data_cost, lut_shape_cost, coef_data,
                                           coef_shape, coef_pairwise, prob_label_trans)
            # run GraphCut
            if len(gc_edges) > 0:
                graph_labels = cut_general_graph(np.array(gc_edges), edge_weights,
                                                 unary, pairwise, n_iter=999)
                labels_gc[gc_vestexes] = graph_labels

        else:
            for i in range(len(centres)):
                candidates = get_neighboring_candidates(slic_neighbours, labels,
                                                        i + 1, allow_obj_swap)

                lut_shape_cost, centres, shifts, volumes = update_shape_costs_points(
                    lut_shape_cost, slic, slic_points, labels, init_centres, centres,
                    shifts, volumes, shape_model, shape_type, None, list_swap_shift[-1],
                    thresholds)

                gc_vestexes, gc_edges, edge_weights, unary, pairwise = \
                    prepare_graphcut_variables(candidates, slic_points, slic_neighbours,
                                               slic_weights, labels, len(centres),
                                               lut_data_cost, lut_shape_cost, coef_data,
                                               coef_shape, coef_pairwise, prob_label_trans)
                # run GraphCut
                graph_labels = cut_general_graph(np.array(gc_edges), edge_weights,
                                                 unary, pairwise, n_iter=999)
                labels_gc[gc_vestexes] = graph_labels

        if np.array_equal(labels, labels_gc):  # and energy == energy_last
            # try the shaking again
            existed = any(np.array_equal(labels_gc, labels_history[i])
                          for i in range(len(labels_history) - 1))
            if any(list_swap_shift[-2:]) or existed:
                break
            list_swap_shift.append(True)
        else:
            list_swap_shift.append(False)

        labels = labels_gc
        labels_history.append(labels.copy())

    return labels

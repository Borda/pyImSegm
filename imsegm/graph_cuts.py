"""
Framework for GraphCut

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging
from warnings import warn

import numpy as np

try:
    from gco import cut_general_graph
except ImportError:
    warn('Missing Grah-Cut (GCO) library,' ' please install it from https://github.com/Borda/pyGCO.')
from skimage import filters
from sklearn import cluster, decomposition, metrics, mixture, pipeline, preprocessing

from imsegm.descriptors import compute_selected_features_img2d
from imsegm.superpixels import (
    make_graph_segm_connect_grid2d_conn4,
    make_graph_segm_connect_grid3d_conn6,
    superpixel_centers,
)
from imsegm.utilities.drawing import (
    draw_color_labeling,
    draw_graphcut_unary_cost_segments,
    draw_graphcut_weighted_edges,
)

#: define munber of iteration in Grap-Cut optimization
DEFAULT_GC_ITERATIONS = 25
# COEF_INT_CONVERSION = 1e6
# DEBUG_NB_SHOW_SAMPLES = 15
#: define minimal value of unary (being a class) term in Graph-Cut
MIN_UNARY_PROB = 0.01
#: define maximal value of parwise (smoothness) term in Graph-Cut
MAX_PAIRWISE_COST = 1e5
#: max is this value and min is inverse (1 / val)
MIN_MAX_EDGE_WEIGHT = 1e3


def estim_gmm_params(features, prob):
    """ with given soft labeling (take the maxim) get the GMM parameters

    :param ndarray features:
    :param ndarray prob:
    :return:

    >>> np.random.seed(0)
    >>> prob = np.array([[1, 0]] * 30 + [[0, 1]] * 40)
    >>> fts = prob + np.random.random(prob.shape)
    >>> mm = estim_gmm_params(fts, prob)
    >>> mm['weights']
    [0.42857142857142855, 0.5714285714285714]
    >>> mm['means']
    array([[ 1.49537196,  0.53745455],
           [ 0.54199936,  1.42606497]])
    """
    nb_samples, nb_classes = prob.shape
    labels = np.argmax(prob, axis=1)
    gmm_params = {'weights': [], 'means': [], 'covars': []}
    for lb in range(nb_classes):
        labels_sel = (labels == lb)
        gmm_params['weights'].append(np.sum(labels_sel) / float(nb_samples))
        gmm_params['means'].append(np.mean(features[labels_sel], axis=0))
        gmm_params['covars'].append(np.cov(features[labels_sel]))
    for n in ['means', 'covars']:
        gmm_params[n] = np.array([m.tolist() for m in gmm_params[n]])
    return gmm_params


def estim_class_model(features, nb_classes, estim_model='GMM', pca_coef=None, use_scaler=True, max_iter=99):
    """ create pipeline (scaler, PCA, model) over several options how
    to cluster samples and fit it on data

    :param ndarray features:
    :param int nb_classes: number of expected classes
    :param float pca_coef: range (0, 1) or None
    :param bool use_scaler: whether use a scaler
    :param str estim_model: used model
    :param int max_iter:
    :return:

    >>> np.random.seed(0)
    >>> fts = np.row_stack([np.random.random((50, 3)) - 1,
    ...                     np.random.random((50, 3)) + 1])
    >>> mm = estim_class_model(fts, 2)
    >>> mm.predict_proba(fts).shape
    (100, 2)
    >>> mm = estim_class_model(fts, 2, estim_model='GMM_kmeans',
    ...                         pca_coef=0.95, max_iter=3)
    >>> mm.predict_proba(fts).shape
    (100, 2)
    >>> mm = estim_class_model(fts, 2, estim_model='GMM_Otsu', max_iter=3)
    >>> mm.predict_proba(fts).shape
    (100, 2)
    >>> mm = estim_class_model(fts, 2, estim_model='kmeans_quantiles',
    ...                         use_scaler=False, max_iter=3)
    >>> mm.predict_proba(fts).shape
    (100, 2)
    >>> mm = estim_class_model(fts, 2, estim_model='BGM', max_iter=3)
    >>> mm.predict_proba(fts).shape
    (100, 2)
    >>> mm = estim_class_model(fts, 2, estim_model='Otsu', max_iter=3)
    >>> mm.predict_proba(fts).shape
    (100, 2)
    """
    components = []
    if use_scaler:
        components += [('std_scaler', preprocessing.StandardScaler())]
    if pca_coef is not None:
        components += [('reduce_dim', decomposition.PCA(pca_coef))]

    nb_inits = max(1, int(np.sqrt(max_iter)))
    # http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html
    mm = mixture.GaussianMixture(n_components=nb_classes, covariance_type='full', n_init=nb_inits, max_iter=max_iter)

    # split the model and used initilaisation
    if '_' in estim_model:
        init_type = estim_model.split('_')[-1]
        estim_model = estim_model.split('_')[0]
    else:
        init_type = ''

    y = None
    if estim_model == 'GMM':
        # model = estim_class_model_gmm(features, nb_classes)
        if init_type == 'kmeans':
            mm.set_params(n_init=1)
            # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
            kmeans = cluster.KMeans(n_clusters=nb_classes, init='k-means++', n_jobs=-1)
            y = kmeans.fit_predict(features)
        elif init_type == 'Otsu':
            mm.set_params(n_init=1)
            y = compute_multivarian_otsu(features)

    elif estim_model == 'kmeans':
        # http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html
        mm.set_params(max_iter=1)
        init_type = 'quantiles' if init_type == 'quantiles' else 'k-means++'
        _, y = estim_class_model_kmeans(features, nb_classes, init_type=init_type, max_iter=max_iter)

        logging.info('compute probability of each feature to all component')

    elif estim_model == 'BGM':
        mm = mixture.BayesianGaussianMixture(
            n_components=nb_classes, covariance_type='full', n_init=nb_inits, max_iter=max_iter
        )

    elif estim_model == 'Otsu' and nb_classes == 2:
        mm.set_params(max_iter=1, n_init=1)
        y = compute_multivarian_otsu(features)

    components += [('model', mm)]
    # compose the pipeline
    model = pipeline.Pipeline(components)

    if y is not None:
        # fit with examples
        model.fit(features, y)
    else:
        # fit from scrach
        model.fit(features)
    return model


def compute_multivarian_otsu(features):
    """ compute otsu individually over each sample dimension
    WARNING: this compute only localy  and since it does compare all
    combinations of orienting the asign for tight cases it may not decide

    :param ndarray features:
    :return list(bool):

    >>> np.random.seed(0)
    >>> fts = np.row_stack([np.random.random((5, 3)) - 1,
    ...                     np.random.random((5, 3)) + 1])
    >>> fts[:, 1] = - fts[:, 1]
    >>> compute_multivarian_otsu(fts).astype(int)
    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    """
    ys = np.zeros(features.shape)
    for i in range(features.shape[-1]):
        thr = filters.threshold_otsu(features[:, i])
        asign = features[:, i] > thr
        if i > 0:
            m = np.mean(ys[:, :i], axis=1)
            d1 = np.mean(np.abs(asign - m))
            d2 = np.mean(np.abs(~asign - m))
            # check if for this dimension it wount be better to swap it
            if d2 < d1:
                asign = ~asign
        ys[:, i] = asign
    y = np.mean(ys, axis=1) > 0.5
    return y


# def estim_class_model_gmm(features, nb_classes, init='kmeans'):
#     """ from all features estimate Gaussian Mixture Model and assuming
#     each cluster is a single class compute probability that each feature
#     belongs to each class
#
#     :param [[float]] features: list of features per segment
#     :param int nb_classes: number of classes
#     :return [[float]]: probabilities that each feature belongs to each class
#     """
#     logging.debug('estimate GMM for all given features %s and %i component',
#                   repr(features.shape), nb_classes)
#     # http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html
#     mm = mixture.GMM(n_components=nb_classes, covariance_type='full', n_iter=999)
#     if init == 'kmeans':
#         # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#         kmeans = cluster.KMeans(n_clusters=nb_classes, init='k-means++', n_jobs=-1)
#         y = kmeans.fit_predict(features)
#         mm.fit(features, y)
#     else:
#         mm.fit(features)
#     logging.info('compute probability of each feature to all component')
#     return mm


def estim_class_model_gmm(features, nb_classes, init='kmeans'):
    """ from all features estimate Gaussian Mixture Model and assuming
    each cluster is a single class compute probability that each feature
    belongs to each class

    :param [[float]] features: list of features per segment
    :param int nb_classes: number of classes
    :param int init: initialisation
    :return [[float]]: probabilities that each feature belongs to each class

    >>> np.random.seed(0)
    >>> fts = np.row_stack([np.random.random((50, 3)) - 1,
    ...                     np.random.random((50, 3)) + 1])
    >>> mm = estim_class_model_gmm(fts, 2)
    >>> mm.predict_proba(fts).shape
    (100, 2)
    """
    logging.debug('estimate GMM for all given features %r and %i component', features.shape, nb_classes)
    # http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html
    gmm = mixture.GaussianMixture(n_components=nb_classes, covariance_type='full', max_iter=99)
    if init == 'kmeans':
        # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        kmeans = cluster.KMeans(n_clusters=nb_classes, init='k-means++', n_jobs=-1)
        y = kmeans.fit_predict(features)
        gmm.fit(features, y)
    else:
        gmm.fit(features)
    logging.info('compute probability of each feature to all component')
    return gmm


def estim_class_model_kmeans(features, nb_classes, init_type='k-means++', max_iter=99):
    """ from all features estimate Gaussian from k-means clustering

    :param [[float]] features: list of features per segment
    :param int nb_classes: number of classes
    :param str init_type: initialization
    :param int max_iter: maximal number of iterations
    :return [[float]]: probabilities that each feature belongs to each class

    >>> np.random.seed(0)
    >>> fts = np.row_stack([np.random.random((50, 3)) - 1,
    ...                     np.random.random((50, 3)) + 1])
    >>> mm, y = estim_class_model_kmeans(fts, 2, max_iter=9)
    >>> y.shape
    (100,)
    >>> mm.predict_proba(fts).shape
    (100, 2)
    """
    logging.debug(
        'estimate Gaussian from k-means clustering for all given features %r and %i components', features.shape,
        nb_classes
    )
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    if init_type == 'quantiles':
        quntiles = np.linspace(5, 95, nb_classes).tolist()
        init_perc = np.array(np.percentile(features, quntiles, axis=0))
        kmeans = cluster.KMeans(nb_classes, init=init_perc, max_iter=2, n_jobs=-1)
    else:
        nb_inits = max(1, int(np.sqrt(max_iter)))
        kmeans = cluster.KMeans(nb_classes, init=init_type, max_iter=max_iter, n_init=nb_inits, n_jobs=-1)
    y = kmeans.fit_predict(features)
    gmm = mixture.GaussianMixture(n_components=nb_classes, covariance_type='full', max_iter=1)
    gmm.fit(features, y)
    return gmm, y


def get_vertexes_edges(segments):
    """ wrapper - get list of vertexes edges for 2D / 3D images

    :param ndarray segments:
    :return:
    """
    if segments.ndim == 3:
        vertices, edges = make_graph_segm_connect_grid3d_conn6(segments)
    elif segments.ndim == 2:
        vertices, edges = make_graph_segm_connect_grid2d_conn4(segments)
    else:
        return None, None
    return vertices, edges


def compute_spatial_dist(centres, edges, relative=False):
    """ compute spatial distance between all neighbouring segments

    :param [[int, int]] centres: superpixel centres
    :param [[int, int]] edges:
    :param bool relative: normalise the distances to mean distance
    :return:

    >>> from imsegm.superpixels import superpixel_centers
    >>> segments = np.array([[0] * 3 + [1] * 2 + [2] * 5,
    ...                      [4] * 4 + [5] * 2 + [6] * 4])
    >>> centres = superpixel_centers(segments)
    >>> edges = [[0, 1], [1, 2], [4, 5], [5, 6], [0, 4], [1, 5], [2, 6]]
    >>> np.round(compute_spatial_dist(centres, edges), 2)
    array([ 2.5 ,  3.5 ,  3.  ,  3.  ,  1.12,  1.41,  1.12])
    >>> np.round(compute_spatial_dist(centres, edges, relative=True), 2)
    array([ 1.12,  1.57,  1.34,  1.34,  0.5 ,  0.63,  0.5 ])
    """
    assert np.max(edges) < len(centres), \
        'max vertex %i exceed size of centres %i' % (np.max(edges), len(centres))
    ndim = np.max([len(c) for c in centres if c is not None])
    # replace empy segments by a empty vector
    for i, c in enumerate(centres):
        if c is None or len(c) == 0:
            centres[i] = [np.NaN] * ndim
    centres = np.nan_to_num(centres)

    vertex_1 = centres[np.asarray(edges)[:, 0]]
    vertex_2 = centres[np.asarray(edges)[:, 1]]

    dist = metrics.pairwise.paired_euclidean_distances(vertex_1, vertex_2)
    if relative:
        dist = dist / np.mean(dist)
    return dist


# def segment_graph_cut_int_vals(segments, probs, gc_regul):
#     """ segment the image segmented via superpixels and estimated features
#
#     :param segments: segmentation mapping each pixel into a class
#     :param probs: probabilities that each feature belongs to each class
#     :param gc_regul: regularisation for GrphCut
#     :return:int[] LUT that maps superpixels into resulting classes
#     """
#     # source: https://github.com/Borda/gco-python
#     from libs.GCO_python.pygco import cut_from_graph
#
#     nbCls = probs.shape[1]
#     logging.info('extraction segment connectivity...')
#     vertices, edges = get_vertexes_edges(segments)
#     # logging.debug('graph connectivity edges: '
#     #               '\n{}'.format(edges[:DEBUG_NB_SHOW_SAMPLES]))
#
#     pairwise = gc_regul * (np.ones(nbCls)
#                            - np.eye(nbCls)) * COEF_INT_CONVERSION
#     # tmp_examples option with one D topology
#     #x, y = np.ogrid[:nbCls,:nbCls]
#     #pairwise = np.abs(x - y)
#     logging.info('convert variables and run GraphCut on created graph.')
#     # convert variables
#     edges = np.array(edges, dtype=np.int32)
#     # logging.debug("graph edges: \total" + repr(edges))
#
#     # fix: change the COEF_INT_CONVERSION
#     unaries = - COEF_INT_CONVERSION * probs
#     unaries = np.array(unaries , dtype=np.int32)
#     #unaries = np.array(probs , dtype=np.int32)
#     logging.debug("graph unaries potentials: \n" + repr(unaries))
#
#     # original and the right way...
#     pairwise = np.array(pairwise , dtype=np.int32)
#     logging.debug("graph pairwise coefs: \n" + repr(pairwise))
#     # run GraphCut
#     #resultGraph = cut_from_graph(edges, unaries, pairwise,
#                                   n_iter=DEFAULT_GC_ITERATIONS)
#     resultGraph = cut_from_graph(edges, unaries, pairwise)
#     logging.debug("resulting graph: \n" + repr(resultGraph))
#     return resultGraph


def compute_edge_model(edges, proba, metric='l_T'):
    """ compute the edge weight from the feature space

    small differences are large weights, diff close 0 appears to be 1
     setting min weight ~ max difference in proba as weight
     meaning if two vertexes have same proba to all classes the diff is 0
     and weights are 1 on the other hand if there is [0.7, 0.1, 0.2]
     and [0.2, 0.7, 0.1] gives large diff [0.5, 0.6, 0.1] in 1.
     and 2. diff and zero in 3 leading to weights [0.5, 0.4, 0.9]
     and so we take the min values

    :param [(int, int)] edges: edges
    :param [[float]] proba: probablilitirs
    :param str metric: define metric
    :return list(float):


    >>> segments = np.array([[0] * 3 + [1] * 5 + [2] * 4,
    ...                      [4] * 4 + [5] * 5 + [6] * 3])
    >>> edges = np.array(get_vertexes_edges(segments)[1], dtype=int)
    >>> np.random.seed(0)
    >>> img = np.random.random(segments.shape + (3,)) * 255
    >>> proba = np.random.random((segments.max() + 1, 2))
    >>> weights = compute_edge_model(edges, proba, metric='l1')
    >>> np.round(weights, 3).tolist()
    [0.002, 0.015, 0.001, 0.002, 0.0, 0.002, 0.015, 0.034, 0.001]
    >>> weights = compute_edge_model(edges, proba, metric='l2')
    >>> np.round(weights, 3).tolist()
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002, 0.005, 0.0]
    >>> weights = compute_edge_model(edges, proba, metric='lT')
    >>> np.round(weights, 3).tolist()
    [0.0, 0.002, 0.0, 0.005, 0.0, 0.0, 0.101, 0.092, 0.001]
    """
    assert np.max(edges) < len(proba), 'max vertex %i exceed size of proba %r' % (np.max(edges), proba.shape)
    vertex_1 = proba[edges[:, 0]]
    vertex_2 = proba[edges[:, 1]]
    # pp 32, http://www.coe.utah.edu/~cs7640/readings/graph_cuts_intro.pdf

    if metric == 'l1':
        dist = metrics.pairwise.paired_manhattan_distances(vertex_1, vertex_2)
        edge_weights = np.exp(-dist / (2 * np.std(dist)**2))
    elif metric == 'l2':
        dist = metrics.pairwise.paired_euclidean_distances(vertex_1, vertex_2)
        edge_weights = np.exp(-dist / (2 * np.std(dist)**2))
    elif metric == 'lT':
        # exp(- norm value diff) * (geom dist vertex)**-1
        diff = (vertex_1 - vertex_2)**2
        # small differences are large weights, diff close 0 appears to be 1
        # setting min weight ~ max difference in proba as weight
        dist = np.max(diff, axis=1)
        edge_weights = np.exp(-dist / (2 * np.std(dist)**2))
    else:
        logging.error('not implemented for: %s', metric)
        edge_weights = np.ones(len(edges))
    return edge_weights


def create_pairwise_matrix_uniform(gc_reg, nb_classes):
    """ create GC pairwise matrix - uniform with zeros on diagonal

    :param float gc_reg:
    :param int nb_classes:
    :return ndarray:

    >>> create_pairwise_matrix_uniform(0.2, 3)
    array([[ 0. ,  0.2,  0.2],
           [ 0.2,  0. ,  0.2],
           [ 0.2,  0.2,  0. ]])
    """
    pairwise = np.ones(nb_classes) - np.eye(nb_classes)
    pairwise = (pairwise * gc_reg)
    return pairwise


def create_pairwise_matrix_specif(pos_weights, nb_classes=None):
    """ create GC pairwise matrix wih specific values on particular positions

    :param [((int, int), float)] pos_weights: pair of coord in matrix and values
    :param int nb_classes: initialise as empty matrix
    :return: np.array<nb_classes, nb_classes>

    >>> create_pairwise_matrix_specif([((1, 2), 0.5), ((1, 0), 0.7)], 4)
    array([[ 0. ,  0.7,  1. ,  1. ],
           [ 0.7,  0. ,  0.5,  1. ],
           [ 1. ,  0.5,  0. ,  1. ],
           [ 1. ,  1. ,  1. ,  0. ]])
    >>> create_pairwise_matrix_specif([((1, 2), 0.5), ((0, 2), 0.7)])
    array([[ 0. ,  1. ,  0.7],
           [ 1. ,  0. ,  0.5],
           [ 0.7,  0.5,  0. ]])
    """
    if not nb_classes:
        nb_classes = np.max([list(c) for c, w in pos_weights]) + 1
    pairwise = (np.ones(nb_classes) - np.eye(nb_classes))
    for c, w in pos_weights:
        pairwise[c[0], c[1]] = w
        pairwise[c[1], c[0]] = w
    return pairwise


def create_pairwise_matrix(gc_regul, nb_classes):
    """ wrapper for create pairwise matrix - uniform or specific

    :param gc_regul:
    :param int nb_classes:
    :return: np.array<nb_classes, nb_classes>

    >>> create_pairwise_matrix(0.6, 3)
    array([[ 0. ,  0.6,  0.6],
           [ 0.6,  0. ,  0.6],
           [ 0.6,  0.6,  0. ]])
    >>> create_pairwise_matrix([((1, 2), 0.5), ((0, 2), 0.7)], 3)
    array([[ 0. ,  1. ,  0.7],
           [ 1. ,  0. ,  0.5],
           [ 0.7,  0.5,  0. ]])
    >>> trans = np.array([[ 341.,   31.,   22.],
    ...                   [  31.,   12.,   21.],
    ...                   [  22.,   21.,   44.]])
    >>> gc_regul = compute_pairwise_cost_from_transitions(trans)
    >>> np.round(create_pairwise_matrix(gc_regul, len(gc_regul)), 2)
    array([[ 0.  ,  0.58,  1.23],
           [ 0.58,  1.53,  0.97],
           [ 1.23,  0.97,  0.54]])
    """
    if isinstance(gc_regul, np.ndarray):
        assert gc_regul.shape[0] == gc_regul.shape[1] == nb_classes, \
            'GC regul matrix %r should match match number of classes (%i)' % (gc_regul.shape, nb_classes)
        # sub_min = np.tile(np.min(gc_regul, axis=0), (gc_regul.shape[0], 1))
        pairwise = gc_regul - np.min(gc_regul)
    elif isinstance(gc_regul, list):
        pairwise = create_pairwise_matrix_specif(gc_regul, nb_classes)
    else:
        pairwise = create_pairwise_matrix_uniform(gc_regul, nb_classes)
    return pairwise


def compute_unary_cost(proba, min_prob=MIN_UNARY_PROB):
    """ compute the GC unary cost with some threshold on minimal values

    :param ndarray proba:
    :param float min_prob:
    :return ndarray:

    >>> compute_unary_cost(np.random.random((50, 2))).shape
    (50, 2)
    """
    proba = proba.copy()
    # constrain that each class should have at least 1.%
    max_prob = 1 - min_prob
    proba[proba < min_prob] = min_prob
    proba[proba > max_prob] = max_prob
    # unary_cost = np.array(1. / proba , dtype=np.float64)
    unary_cost = np.abs(np.array(-np.log(proba), dtype=np.float64))
    return unary_cost


def compute_pairwise_cost(gc_regul, proba_shape, max_pairwise_cost=MAX_PAIRWISE_COST):
    """ wrapper for creating GC pairwise cost

    :param gc_regul:
    :param tuple(int,int) proba_shape:
    :param float max_pairwise_cost:
    :return ndarray:
    """
    # original and the right way...
    pairwise = create_pairwise_matrix(gc_regul, proba_shape[1])
    pairwise_cost = np.array(pairwise, dtype=np.float64)
    pairwise_cost[pairwise_cost > max_pairwise_cost] = max_pairwise_cost
    return pairwise_cost


def insert_gc_debug_images(debug_visual, segments, graph_labels, unary_cost, edges, edge_weights):
    """ wrapper for placing intermediate variable to a dictionary """
    if debug_visual is None:
        return
    debug_visual['segments'] = segments
    debug_visual['edges'] = edges
    debug_visual['edge_weights'] = edge_weights
    debug_visual['imgs_unary_cost'] = draw_graphcut_unary_cost_segments(segments, unary_cost)
    img = debug_visual.get('slic_mean', None)
    list_centres = superpixel_centers(segments)
    debug_visual['img_graph_edges'] = draw_graphcut_weighted_edges(
        segments, list_centres, edges, edge_weights, img_bg=img
    )
    debug_visual['img_graph_segm'] = draw_color_labeling(segments, graph_labels)


def compute_edge_weights(segments, image=None, features=None, proba=None, edge_type=''):
    """
    pp 32, http://www.coe.utah.edu/~cs7640/readings/graph_cuts_intro.pdf
    exp(- norm value diff) * (geom dist vertex)**-1

    :param ndarry segments: superpixels
    :param ndarry image: input image
    :param ndarry features: features for each segment (superpixel)
    :param ndarry proba: probability of each superpixel and class
    :param str edge_type: contains edge type, if 'model', after '_' you can
        specify the metric, eg. 'model_l2'
    :return [[int, int]], [float]:

    >>> segments = np.array([[0] * 3 + [1] * 5 + [2] * 4,
    ...                      [4] * 4 + [5] * 5 + [6] * 3])
    >>> np.random.seed(0)
    >>> img = np.random.random(segments.shape + (3,)) * 255
    >>> features = np.random.random((segments.max() + 1, 15)) * 10
    >>> proba = np.random.random((segments.max() + 1, 2))
    >>> edges, weights = compute_edge_weights(segments)
    >>> edges.tolist()
    [[0, 1], [1, 2], [0, 4], [1, 4], [1, 5], [2, 5], [4, 5], [2, 6], [5, 6]]
    >>> np.round(weights, 2).tolist()
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    >>> edges, weights = compute_edge_weights(segments, image=img,
    ...                                        edge_type='spatial')
    >>> np.round(weights, 3).tolist()
    [0.776, 0.69, 2.776, 0.853, 2.194, 0.853, 0.69, 2.776, 0.776]
    >>> edges, weights = compute_edge_weights(segments, image=img,
    ...                                        edge_type='color')
    >>> np.round(weights, 3).tolist()
    [0.06, 0.002, 0.001, 0.001, 0.001, 0.009, 0.001, 0.019, 0.044]
    >>> edges, weights = compute_edge_weights(segments, features=features,
    ...                                        edge_type='features')
    >>> np.round(weights, 3).tolist()
    [0.031, 0.005, 0.051, 0.032, 0.096, 0.013, 0.018, 0.033, 0.013]
    >>> edges, weights = compute_edge_weights(segments, proba=proba,
    ...                                        edge_type='model')
    >>> np.round(weights, 3).tolist()
    [0.001, 0.028, 1.122, 0.038, 0.117, 0.688, 0.487, 1.152, 0.282]
    """
    logging.debug('extraction segment connectivity...')
    _, edges = get_vertexes_edges(segments)
    # convert variables
    edges = np.array(edges, dtype=np.int32)
    logging.debug('graph edges %r', edges.shape)

    if edge_type.startswith('model'):
        assert proba is not None, '"proba" is required'
        metric = edge_type.split('_')[-1] if '_' in edge_type else 'lT'
        edge_weights = compute_edge_model(edges, proba, metric)
    elif edge_type == 'color':
        assert image is not None, '"image" is required'
        image_float = np.array(image, dtype=float)
        if np.max(image) > 1:
            image_float /= 255.
        color, _ = compute_selected_features_img2d(image_float, segments, {'color': ['mean']})
        vertex_1 = color[edges[:, 0]]
        vertex_2 = color[edges[:, 1]]
        dist = metrics.pairwise.paired_manhattan_distances(vertex_1, vertex_2)
        weights = dist.astype(float) / (2 * np.std(dist)**2)
        edge_weights = np.exp(-weights)
    elif edge_type == 'features':
        assert features is not None, '"features" is required'
        features_norm = preprocessing.StandardScaler().fit_transform(features)
        vertex_1 = features_norm[edges[:, 0]]
        vertex_2 = features_norm[edges[:, 1]]
        dist = metrics.pairwise.paired_euclidean_distances(vertex_1, vertex_2)
        weights = dist.astype(float) / (2 * np.std(dist)**2)
        edge_weights = np.exp(-weights)
    else:
        edge_weights = np.ones(len(edges))

    edge_weights = np.array(edge_weights, dtype=float)
    if edge_type in ['model', 'features', 'color', 'spatial']:
        centres = superpixel_centers(segments)
        spatial = compute_spatial_dist(centres, edges, relative=True)
        edge_weights /= spatial

    # set the threshold for min edge weight
    min_weight = 1. / MIN_MAX_EDGE_WEIGHT
    max_weight = MIN_MAX_EDGE_WEIGHT
    edge_weights[edge_weights < min_weight] = min_weight
    edge_weights[edge_weights > max_weight] = max_weight
    return edges, edge_weights


def segment_graph_cut_general(
    segments,
    proba,
    image=None,
    features=None,
    gc_regul=1.,
    edge_type='model',
    edge_cost=1.,
    debug_visual=None,
):
    """ segment the image segmented via superpixels and estimated features

    :param ndarray features: features sor each instance
    :param ndarray segments: segmentation mapping each pixel into a class
    :param ndarray image: image
    :param ndarray proba: probabilities that each feature belongs to each class
    :param str edge_type:
    :param str edge_cost:
    :param gc_regul: regularisation for GrphCut
    :param dict debug_visual:
    :return list(int): labelling by resulting classes

    >>> np.random.seed(0)
    >>> segments = np.array([[0] * 3 + [2] * 3 + [4] * 3 + [6] * 3 + [8] * 3,
    ...                      [1] * 3 + [3] * 3 + [5] * 3 + [7] * 3 + [9] * 3])
    >>> proba = np.array([[0.1] * 6 + [0.9] * 4,
    ...                   [0.9] * 6 + [0.1] * 4], dtype=float).T
    >>> proba += (0.5 - np.random.random(proba.shape)) * 0.2
    >>> compute_unary_cost(proba)
    array([[ 2.40531242,  0.15436155],
           [ 2.53266106,  0.11538463],
           [ 2.1604864 ,  0.13831863],
           [ 2.18495711,  0.19644636],
           [ 4.60517019,  0.0797884 ],
           [ 3.17833405,  0.11180231],
           [ 0.12059702,  4.20769207],
           [ 0.0143091 ,  1.70059894],
           [ 0.01005034,  3.39692559],
           [ 0.16916609,  3.64975219]])
    >>> segment_graph_cut_general(segments, proba, gc_regul=0., edge_type='')
    array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=int32)
    >>> labels = segment_graph_cut_general(segments, proba, gc_regul=1.,
    ...                                    edge_type='spatial')
    >>> labels[segments]
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=int32)
    >>> slic = np.array([[0] * 4 + [1] * 6 + [2] * 4,
    ...                  [3] * 5 + [4] * 4 + [5] * 5])
    >>> proba = np.array([[1] * 3 + [0] * 3, [0] * 3 + [1] * 3], dtype=float).T
    >>> proba += np.random.random(proba.shape) / 2.
    >>> np.argmax(proba, axis=1)
    array([0, 0, 0, 1, 1, 1])
    >>> debug_visual = dict()
    >>> segment_graph_cut_general(slic, proba, gc_regul=0., edge_type='',
    ...                           debug_visual=debug_visual)
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> sorted(debug_visual.keys())  #doctest: +NORMALIZE_WHITESPACE
    ['edge_weights', 'edges', 'img_graph_edges', 'img_graph_segm',
     'imgs_unary_cost', 'segments']
    """
    logging.debug('convert variables and run GraphCut on created graph.')

    edges, edge_weights = compute_edge_weights(segments, image, features, proba, edge_type)
    edge_weights *= edge_cost
    logging.debug('graph edges weights %r', edge_weights.shape)

    unary_cost = compute_unary_cost(proba)
    logging.debug('graph unaries potentials: %r', unary_cost.shape)
    pairwise_cost = compute_pairwise_cost(gc_regul, proba.shape)
    logging.debug('graph pairwise coefs: \n%r', pairwise_cost)

    if gc_regul <= 0:
        logging.debug('gc_regul=%f so we use just argmax()', gc_regul)
        graph_labels = np.argmin(unary_cost, axis=-1).astype(np.int32)
    else:
        # run GraphCut
        logging.debug('perform GraphCut')
        graph_labels = cut_general_graph(
            edges,
            edge_weights,
            unary_cost,
            pairwise_cost,
            algorithm='expansion',
            # down_weight_factor=np.abs(unary_cost).max(),
            # init_labels=np.argmax(proba, axis=1),
            n_iter=-1
        )

    insert_gc_debug_images(debug_visual, segments, graph_labels, compute_unary_cost(proba), edges, edge_weights)
    return graph_labels


def count_label_transitions_connected_segments(dict_slics, dict_labels, nb_labels=None):
    """ count transitions among labeled segment in between connected segments

    :param dict(list(list(int))) dict_slics: image name: ndarray
    :param dict(list(int)) dict_labels: image name: ndarray
    :param int nb_labels:
    :return ndarray: matrix of shape nb_labels x nb_labels

    >>> dict_slics = {'a':
    ...        np.array([[0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3,
    ...                  [5] * 3 + [6] * 3 + [7] * 3 + [8] * 3 + [9] * 3])}
    >>> dict_labels = {'a': np.array([0, 0, 1, 1, 2, 0, 1, 1, 0, 2])}
    >>> dict_slics['a']
    array([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
           [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]])
    >>> dict_labels['a'][dict_slics['a']]
    array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2]])
    >>> count_label_transitions_connected_segments(dict_slics, dict_labels)
    array([[ 2.,  5.,  1.],
           [ 5.,  3.,  1.],
           [ 1.,  1.,  1.]])
    """
    if not nb_labels:
        uq_img_labels = [np.unique(lbs).tolist() for lbs in dict_labels.values()]
        uq_labels = np.unique(np.hstack(tuple(uq_img_labels)))
        nb_labels = np.max(uq_labels) + 1
    transitions = np.zeros((nb_labels, nb_labels))
    for name in dict_slics:
        assert (np.max(dict_slics[name]) + 1) == len(dict_labels[name]), \
            'dims are not matching - max slic (%i) and label (%i)' \
            % (np.max(dict_slics[name]), len(dict_labels[name]))
        _, edges = get_vertexes_edges(dict_slics[name])
        label_edges = np.asarray(dict_labels[name])[np.asarray(edges)]
        for lb1, lb2 in label_edges.tolist():
            transitions[lb1, lb2] += 1
            transitions[lb2, lb1] += 1
    for i in range(len(transitions)):
        transitions[i, i] /= 2
    # nb_edges = np.sum(transitions) / 2
    # transitions_norm = transitions / nb_edges
    return transitions


def compute_pairwise_cost_from_transitions(trans, min_prob=1e-9):
    """ compute pairwise cost from segments-label transitions

    :param ndarray trans:
    :param float min_prob: minimal probability
    :return ndarray:

    >>> trans = np.array([[ 25.,   5.,  0.],
    ...                   [  5.,  10.,  8.],
    ...                   [  0.,   8.,  30.]])
    >>> np.round(compute_pairwise_cost_from_transitions(trans), 3)
    array([[  0.182,   1.526,  20.723],
           [  1.526,   0.833,   1.056],
           [ 20.723,   1.056,   0.236]])
    >>> np.round(compute_pairwise_cost_from_transitions(np.ones(3)), 2)
    array([[ 1.1,  1.1,  1.1],
           [ 1.1,  1.1,  1.1],
           [ 1.1,  1.1,  1.1]])
    >>> np.round(compute_pairwise_cost_from_transitions(np.eye(3)), 2)
    array([[  0.  ,  20.72,  20.72],
           [ 20.72,   0.  ,  20.72],
           [ 20.72,  20.72,   0.  ]])
    """
    # e_x = np.exp(trans - np.max(trans))  # softmax
    # softmax = e_x / e_x.sum(axis=0)
    # pw = (1. / softmax) - 1.
    ratio = trans / np.tile(np.sum(trans, axis=0), (len(trans), 1))
    for i in range(1, len(trans)):
        for j in range(i):
            el = max(ratio[i, j], ratio[j, i])
            ratio[i, j] = el
            ratio[j, i] = el
    # prevent dividing by 0, set very small value
    ratio[ratio < min_prob] = min_prob
    pw = np.log(1. / ratio)
    # pw[pw > max_value] = max_value
    return pw

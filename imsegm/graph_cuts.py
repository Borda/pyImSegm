"""
Framework for GraphCut

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging

import numpy as np
from gco import cut_general_graph
from sklearn import metrics, mixture, cluster, preprocessing

import imsegm.utils.drawing as tl_visu
import imsegm.superpixels as seg_spx
import imsegm.descriptors as seg_fts

DEFAULT_GC_ITERATIONS = 25
COEF_INT_CONVERSION = 1e6
DEBUG_NB_SHOW_SAMPLES = 15


def estim_gmm_params(features, prob):
    """ with given soft labeling (take the maxim) get the GMM parameters

    :param ndarray features:
    :param ndarray prob:
    :return:

    >>> np.random.seed(0)
    >>> prob = np.array([[1, 0]] * 30 + [[0, 1]] * 40)
    >>> fts = prob + np.random.random(prob.shape)
    >>> gmm = estim_gmm_params(fts, prob)
    >>> gmm['weights']
    [0.42857142857142855, 0.5714285714285714]
    >>> gmm['means']
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


def estim_class_model(features, nb_classes, proba_type):
    """ wrapper over several options how to cluster samples

    :param ndarray features:
    :param int nb_classes:
    :param str proba_type:
    :return:
    """
    if proba_type == 'GMM':
        model = estim_class_model_gmm(features, nb_classes)
    elif proba_type == 'quantiles':
        model = estim_class_model_kmeans(features, nb_classes,
                                         init_type='quantiles')
    else:
        model = estim_class_model_kmeans(features, nb_classes)
    return model


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
#     gmm = mixture.GMM(n_components=nb_classes, covariance_type='full', n_iter=999)
#     if init == 'kmeans':
#         # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#         kmeans = cluster.KMeans(n_clusters=nb_classes, init='k-means++', n_jobs=-1)
#         y = kmeans.fit_predict(features)
#         gmm.fit(features, y)
#     else:
#         gmm.fit(features)
#     logging.info('compute probability of each feature to all component')
#     return gmm


def estim_class_model_gmm(features, nb_classes, init='kmeans'):
    """ from all features estimate Gaussian Mixture Model and assuming
    each cluster is a single class compute probability that each feature
    belongs to each class

    :param [[float]] features: list of features per segment
    :param int nb_classes: number of classes
    :return [[float]]: probabilities that each feature belongs to each class

    >>> np.random.seed(0)
    >>> fts = np.row_stack([np.random.random((50, 3)) - 1,
    ...                     np.random.random((50, 3)) + 1])
    >>> gmm = estim_class_model_gmm(fts, 2)
    >>> gmm.predict_proba(fts).shape
    (100, 2)
    """
    logging.debug('estimate GMM for all given features %s and %i component',
                  repr(features.shape), nb_classes)
    # http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html
    gmm = mixture.GaussianMixture(n_components=nb_classes,
                                  covariance_type='full', max_iter=99)
    if init == 'kmeans':
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        kmeans = cluster.KMeans(n_clusters=nb_classes, init='k-means++',
                                n_jobs=-1)
        y = kmeans.fit_predict(features)
        gmm.fit(features, y)
    else:
        gmm.fit(features)
    logging.info('compute probability of each feature to all component')
    return gmm


def estim_class_model_kmeans(features, nb_classes, init_type='k-means++'):
    """ from all features estimate Gaussian from k-means clustering

    :param [[float]] features: list of features per segment
    :param int nb_classes:, number of classes
    :return [[float]]: probabilities that each feature belongs to each class

    >>> np.random.seed(0)
    >>> fts = np.row_stack([np.random.random((50, 3)) - 1,
    ...                     np.random.random((50, 3)) + 1])
    >>> gmm = estim_class_model_kmeans(fts, 2)
    >>> gmm.predict_proba(fts).shape
    (100, 2)
    """
    logging.debug('estimate Gaussian from k-means clustering for all given '
                  'features %s and %i component', repr(features.shape), nb_classes)
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    if init_type == 'quantiles':
        quntiles = np.linspace(5, 95, nb_classes).tolist()
        init_perc = np.array(np.percentile(features, quntiles, axis=0))
        kmeans = cluster.KMeans(nb_classes, init=init_perc, max_iter=2, n_jobs=-1)
    else:
        kmeans = cluster.KMeans(nb_classes, init=init_type, n_init=25, n_jobs=-1)
    y = kmeans.fit_predict(features)
    logging.info('compute probability of each feature to all component')
    # http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html
    gmm = mixture.GaussianMixture(n_components=nb_classes,
                                  covariance_type='full', max_iter=1)
    gmm.fit(features, y)
    return gmm


def get_vertexes_edges(segments):
    """ wrapper - get list of vertexes edges for 2D / 3D images

    :param ndarray segments:
    :return:
    """
    if segments.ndim == 3:
        vertices, edges = seg_spx.make_graph_segm_connect3d_conn6(segments)
    elif segments.ndim == 2:
        vertices, edges = seg_spx.make_graph_segm_connect2d_conn4(segments)
    else:
        return None, None
    return vertices, edges


def compute_spatial_dist(centres, edges, relative=False):
    """ compute spatial distance between all neighbouring segments

    :param [[int, int]] centres: superpixel centres
    :param [[int, int]] edges:
    :param bool relative: normalise the distances to mean distance
    :return:

    >>> segments = np.array([[0] * 3 + [1] * 2 + [2] * 5,
    ...                      [4] * 4 + [5] * 2 + [6] * 4])
    >>> centres = seg_spx.superpixel_centers(segments)
    >>> edges = [[0, 1], [1, 2], [4, 5], [5, 6], [0, 4], [1, 5], [2, 6]]
    >>> np.round(compute_spatial_dist(centres, edges), 2)
    array([ 2.5 ,  3.5 ,  3.  ,  3.  ,  1.12,  1.41,  1.12])
    >>> np.round(compute_spatial_dist(centres, edges, relative=True), 2)
    array([ 1.12,  1.57,  1.34,  1.34,  0.5 ,  0.63,  0.5 ])
    """
    assert np.max(edges) < len(centres), \
        'max vertex %i exceed size of centres %i'\
        % (np.max(edges), len(centres))
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
     and so we take the min valus

    :param [(int, int)] edges:
    :param [[float]] features:
    :return [float]:


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
    assert np.max(edges) < len(proba), \
        'max vertex %i exceed size of proba %s' % (np.max(edges), repr(proba.shape))
    vertex_1 = proba[edges[:, 0]]
    vertex_2 = proba[edges[:, 1]]
    # pp 32, http://www.coe.utah.edu/~cs7640/readings/graph_cuts_intro.pdf

    if metric == 'l1':
        dist = metrics.pairwise.paired_manhattan_distances(vertex_1, vertex_2)
        edge_weights = np.exp(- dist / (2 * np.std(dist) ** 2))
    elif metric == 'l2':
        dist = metrics.pairwise.paired_euclidean_distances(vertex_1, vertex_2)
        edge_weights = np.exp(- dist / (2 * np.std(dist) ** 2))
    elif metric == 'lT':
        # exp(- norm value diff) * (geom dist vertex)**-1
        diff = (vertex_1 - vertex_2) ** 2
        # small differences are large weights, diff close 0 appears to be 1
        # setting min weight ~ max difference in proba as weight
        dist = np.max(diff, axis=1)
        edge_weights = np.exp(- dist / (2 * np.std(dist) ** 2))
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
    if nb_classes is None:
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
            'GC regul matrix %s should match match number o lasses (%i)' \
            % (repr(gc_regul.shape), nb_classes)
        # sub_min = np.tile(np.min(gc_regul, axis=0), (gc_regul.shape[0], 1))
        pairwise = gc_regul - np.min(gc_regul)
    elif isinstance(gc_regul, list):
        pairwise = create_pairwise_matrix_specif(gc_regul, nb_classes)
    else:
        pairwise = create_pairwise_matrix_uniform(gc_regul, nb_classes)
    return pairwise


def compute_unary_cost(proba):
    """ compute the GC unary cost with some threshold on minimal values

    :param ndarray proba:
    :return ndarray:

    >>> compute_unary_cost(np.random.random((50, 2))).shape
    (50, 2)
    """
    proba = proba.copy()
    proba[proba < 1e-99] = 1e-99
    # unary_cost = np.array(1. / proba , dtype=np.float64)
    unary_cost = np.abs(np.array(-np.log(proba), dtype=np.float64))
    return unary_cost


def compute_pairwise_cost(gc_regul, proba_shape):
    """ wrapper for creating GC pairwise cost

    :param gc_regul:
    :param (int, int) proba_shape:
    :return ndarray:
    """
    # original and the right way...
    pairwise = create_pairwise_matrix(gc_regul, proba_shape[1])
    pairwise_cost = np.array(pairwise, dtype=np.float64)
    return pairwise_cost


def insert_gc_debug_images(dict_debug_imgs, segments, graph_labels, unary_cost,
                           edges, edge_weights):
    """ wrapper for placing intermediate variable to a dictionary """
    if dict_debug_imgs is None:
        return
    dict_debug_imgs['segments'] = segments
    dict_debug_imgs['edges'] = edges
    dict_debug_imgs['edge_weights'] = edge_weights
    dict_debug_imgs['imgs_unary_cost'] = \
        tl_visu.draw_graphcut_unary_cost_segments(segments, unary_cost)
    img = dict_debug_imgs.get('slic_mean', None)
    list_centres = seg_spx.superpixel_centers(segments)
    dict_debug_imgs['img_graph_edges'] = \
        tl_visu.draw_graphcut_weighted_edges(segments, list_centres, edges,
                                             edge_weights, img_bg=img)
    dict_debug_imgs['img_graph_segm'] = \
        tl_visu.draw_color_labeling(segments, graph_labels)


def compute_edge_weights(segments, image=None, features=None, proba=None,
                         edge_type=''):
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
    [0.06, 0.002, 0.0, 0.0, 0.0, 0.009, 0.0, 0.019, 0.044]
    >>> edges, weights = compute_edge_weights(segments, features=features,
    ...                                        edge_type='features')
    >>> np.round(weights, 3).tolist()
    [0.031, 0.005, 0.051, 0.032, 0.096, 0.013, 0.018, 0.033, 0.013]
    >>> edges, weights = compute_edge_weights(segments, proba=proba,
    ...                                        edge_type='model')
    >>> np.round(weights, 3).tolist()
    [0.0, 0.028, 1.122, 0.038, 0.117, 0.688, 0.487, 1.152, 0.282]
    """
    logging.debug('extraction segment connectivity...')
    _, edges = get_vertexes_edges(segments)
    # convert variables
    edges = np.array(edges, dtype=np.int32)
    logging.debug('graph edges %s', repr(edges.shape))

    if edge_type.startswith('model'):
        assert proba is not None, '"proba" is requuired'
        metric = edge_type.split('_')[-1] if '_' in edge_type else 'lT'
        edge_weights = compute_edge_model(edges, proba, metric)
    elif edge_type == 'color':
        assert image is not None, '"image" is required'
        # {'color': ['mean', 'median']}
        image_float = np.array(image, dtype=float)
        if np.max(image) > 1:
            image_float /= 255.
        color, _ = seg_fts.compute_selected_features_img2d(image_float, segments,
                                                           {'color': ['mean']})
        vertex_1 = color[edges[:, 0]]
        vertex_2 = color[edges[:, 1]]
        dist = metrics.pairwise.paired_manhattan_distances(vertex_1, vertex_2)
        weights = dist.astype(float) / (2 * np.std(dist) ** 2)
        edge_weights = np.exp(- weights)
    elif edge_type == 'features':
        assert features is not None, '"features" is required'
        features_norm = preprocessing.StandardScaler().fit_transform(features)
        vertex_1 = features_norm[edges[:, 0]]
        vertex_2 = features_norm[edges[:, 1]]
        dist = metrics.pairwise.paired_euclidean_distances(vertex_1, vertex_2)
        weights = dist.astype(float) / (2 * np.std(dist) ** 2)
        edge_weights = np.exp(- weights)
    else:
        edge_weights = np.ones(len(edges))

    edge_weights = np.array(edge_weights, dtype=float)
    if edge_type in ['model', 'features', 'color', 'spatial']:
        centres = seg_spx.superpixel_centers(segments)
        spatial = compute_spatial_dist(centres, edges, relative=True)
        edge_weights /= spatial
    return edges, edge_weights


def segment_graph_cut_general(segments, proba, image=None, features=None,
                              gc_regul=1., edge_type='model', edge_cost=1.,
                              dict_debug_imgs=None):
    """ segment the image segmented via superpixels and estimated features

    :param ndarray features: features sor each instance
    :param ndarray segments: segmentation mapping each pixel into a class
    :param ndarray proba: probabilities that each feature belongs to each class
    :param gc_regul: regularisation for GrphCut
    :param {} dict_debug_imgs:
    :return [int]: labelling by resulting classes

    >>> np.random.seed(0)
    >>> segments = np.array([[0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3,
    ...                      [5] * 3 + [6] * 3 + [7] * 3 + [8] * 3 + [9] * 3])
    >>> proba = np.array([[0] * 6 + [1] * 4, [1] * 6 + [0] * 4], dtype=float).T
    >>> proba += np.random.random(proba.shape) / 2.
    >>> compute_unary_cost(proba)
    array([[ 1.29314378,  0.30571452],
           [ 1.19937775,  0.24093757],
           [ 1.55198349,  0.27986187],
           [ 1.51962643,  0.36872263],
           [ 0.73016106,  0.17539828],
           [ 0.9266883 ,  0.23463524],
           [ 0.24999756,  0.77046392],
           [ 0.03490181,  3.13350924],
           [ 0.01005844,  0.87632529],
           [ 0.32864049,  0.83239528]])
    >>> segment_graph_cut_general(segments, proba, gc_regul=0., edge_type='')
    array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=int32)
    >>> labels = segment_graph_cut_general(segments, proba, gc_regul=1.,
    ...                                    edge_type='spatial')
    >>> labels[segments]
    array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
    >>> slic = np.array([[0] * 4 + [1] * 6 + [2] * 4,
    ...                  [3] * 5 + [4] * 4 + [5] * 5])
    >>> proba = np.array([[1] * 3 + [0] * 3, [0] * 3 + [1] * 3], dtype=float).T
    >>> proba += np.random.random(proba.shape) / 2.
    >>> np.argmax(proba, axis=1)
    array([0, 0, 0, 1, 1, 1])
    >>> dict_debug_imgs = dict()
    >>> segment_graph_cut_general(slic, proba, gc_regul=0., edge_type='',
    ...                           dict_debug_imgs=dict_debug_imgs)
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> sorted(dict_debug_imgs.keys())  #doctest: +NORMALIZE_WHITESPACE
    ['edge_weights', 'edges', 'img_graph_edges', 'img_graph_segm',
     'imgs_unary_cost', 'segments']
    """
    logging.debug('convert variables and run GraphCut on created graph.')

    edges, edge_weights = compute_edge_weights(segments, image, features,
                                               proba, edge_type)
    edge_weights *= edge_cost
    logging.debug('graph edges weights %s', repr(edge_weights.shape))

    unary_cost = compute_unary_cost(proba)
    logging.debug('graph unaries potentials: %s', repr(unary_cost.shape))
    pairwise_cost = compute_pairwise_cost(gc_regul, proba.shape)
    logging.debug('graph pairwise coefs: \n%s', repr(pairwise_cost))

    labels = np.argmax(proba, axis=1)
    # run GraphCut
    logging.debug('perform GraphCut')
    graph_labels = cut_general_graph(edges, edge_weights, unary_cost,
                                     pairwise_cost, algorithm='expansion',
                                     # down_weight_factor=np.abs(unary_cost).max()
                                     init_labels=labels, n_iter=9999)

    insert_gc_debug_images(dict_debug_imgs, segments, graph_labels,
                           compute_unary_cost(proba), edges, edge_weights)
    return graph_labels


def count_label_transitions_connected_segments(dict_slics, dict_labels,
                                               nb_labels=None):
    """ count transitions among labeled segment in between connected segments

    :param {str: [[int]]} dict_slics: image name: ndarray
    :param {str: [int]} dict_labels: image name: ndarray
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
    if nb_labels is None:
        uq_img_labels = [np.unique(lbs).tolist()
                         for lbs in dict_labels.values()]
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


def compute_pairwise_cost_from_transitions(trans, max_value=1e3):
    """ compute pairwise cost from segments-label transitions

    :param ndarray trans:
    :return ndarray:

    >>> trans = np.array([[ 25.,   5.,  0.],
    ...                   [  5.,  10.,  8.],
    ...                   [  0.,   8.,  30.]])
    >>> np.round(compute_pairwise_cost_from_transitions(trans), 3)
    array([[  1.82000000e-01,   1.52600000e+00,   1.00000000e+03],
           [  1.52600000e+00,   8.33000000e-01,   1.05600000e+00],
           [  1.00000000e+03,   1.05600000e+00,   2.36000000e-01]])
    >>> np.round(compute_pairwise_cost_from_transitions(np.ones(3)), 2)
    array([[ 1.1,  1.1,  1.1],
           [ 1.1,  1.1,  1.1],
           [ 1.1,  1.1,  1.1]])
    >>> np.round(compute_pairwise_cost_from_transitions(np.eye(3)), 2)
    array([[    0.,  1000.,  1000.],
           [ 1000.,     0.,  1000.],
           [ 1000.,  1000.,     0.]])
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
    pw = np.log(1. / ratio)
    pw[pw > max_value] = max_value
    return pw

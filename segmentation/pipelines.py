"""

Copyright (C) 2014-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging

import numpy as np
import skimage.color as sk_color
from sklearn import preprocessing, mixture, decomposition

import segmentation.graph_cuts as tl_gc
import segmentation.superpixels as tl_sp
import segmentation.descriptors as tl_fts
import segmentation.labeling as tl_lbs
import segmentation.classification as tl_clf

CLASSIF_PARAMS = {'method': 'kNN', 'nb': 10}
FTS_SET_SIMPLE = tl_fts.FEATURES_SET_COLOR
CLUST_METHOD = tl_clf.DEFAULT_CLUSTERING
CROSS_VAL_LEAVE_OUT = 2


def convert_img_clr_space(img, clr_space):
    """ convert image colour space from RGB to xxx

    :param im: rgb image
    :param clr_space: str
    :return: image
    """
    DICT_CONVERT_COLOR = {
        'hsv': sk_color.rgb2hsv,
        'luv': sk_color.rgb2luv,
        'lab': sk_color.rgb2lab,
        'hed': sk_color.rgb2hed,
        'xyz': sk_color.rgb2xyz
    }
    if img.ndim == 3 and img.shape[2] == 3 and clr_space in DICT_CONVERT_COLOR:
        img = DICT_CONVERT_COLOR[clr_space](img)
    return img


def pipe_color2d_slic_features_gmm_graphcut(img, nb_classes=3, clr_space='rgb',
                                            sp_size=30, sp_regul=0.2, gc_regul=1.,
                                            dict_features=FTS_SET_SIMPLE,
                                            proba_type='GMM', gc_edge_type='model',
                                            pca_coef=None, dict_debug_imgs=None):
    """ complete pipe-line for segmentation using superpixels, extracting features
    and graphCut segmentation

    :param img: input RGB image
    :param int sp_size: initial size of a superpixel(meaning edge lenght)
    :param float sp_regul: regularisation in range(0;1) where "0" gives elastic
                   and "1" nearly square slic
    :param int nb_classes: number of classes to be segmented(indexing from 0)
    :param dict_features: {clr: [str], ...}
    :param str clr_space:
    :param str gc_edge_type:
    :param float pca_coef: range (0, 1) or None
    :param dict_debug_imgs: {str: ...}
    :return [[int]]: segmentation matrix maping each pixel into a class

    >>> np.random.seed(0)
    >>> img = np.random.random((125, 150, 3)) / 2.
    >>> img[:, :75] += 0.5
    >>> segm = pipe_color2d_slic_features_gmm_graphcut(img, nb_classes=2)
    >>> segm.shape
    (125, 150)
    """
    logging.info('PIPELINE Superpixels-Features-GMM-GraphCut')
    slic, features = compute_color2d_superpixels_features(img, sp_size, sp_regul,
                                                          dict_features, clr_space)

    if dict_debug_imgs is not None:
        if img.ndim == 2:  # duplicate channels to be like RGB
            img = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)
        dict_debug_imgs['img'] = img
        dict_debug_imgs['slic'] = slic
        dict_debug_imgs['slic_mean'] = sk_color.label2rgb(slic, img, kind='avg')

    if pca_coef is not None:
        pca = decomposition.PCA(pca_coef)
        features = pca.fit_transform(features)

    model = tl_gc.estim_class_model(features, nb_classes, proba_type)
    proba = model.predict_proba(features)
    logging.debug('list of probabilities: %s', repr(proba.shape))

    gmm = mixture.GaussianMixture(n_components=nb_classes,
                                  covariance_type='full', max_iter=1)
    gmm.fit(features, np.argmax(proba, axis=1))
    proba = gmm.predict_proba(features)

    graph_labels = tl_gc.segment_graph_cut_general(slic, proba, img, features, gc_regul,
                                                   gc_edge_type, dict_debug_imgs=dict_debug_imgs)
    segm = graph_labels[slic]
    return segm


def estim_model_classes_group(list_images, nb_classes=4, clr_space='rgb', sp_size=30,
                              sp_regul=0.2, dict_features=FTS_SET_SIMPLE,
                              pca_coef=None, proba_type='GMM'):
    list_slic, list_features = list(), list()
    for img in list_images:
        slic, features = compute_color2d_superpixels_features(img, sp_size,
                          sp_regul, dict_features, clr_space, fts_norm=False)
        list_slic.append(slic)
        list_features.append(features)

    features = np.concatenate(tuple(list_features), axis=0)
    features = np.nan_to_num(features)

    # scaling
    scaler = preprocessing.StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    pca = None
    if pca_coef is not None:
        pca = decomposition.PCA(pca_coef)
        features = pca.fit_transform(features)

    model = tl_gc.estim_class_model(features, nb_classes, proba_type)
    return scaler, pca, model


def segment_color2d_slic_features_model_graphcut(img, scaler, pca, model,
                                                 clr_space='rgb',
                                                 sp_size=30, sp_regul=0.2,
                                                 gc_regul=1.,
                                                 dict_features=FTS_SET_SIMPLE,
                                                 gc_edge_type='model',
                                                 dict_debug_imgs=None):
    """ complete pipe-line for segmentation using superpixels, extracting features
    and graphCut segmentation

    :param img: input RGB image
    :param int sp_size: initial size of a superpixel(meaning edge lenght)
    :param float sp_regul: regularisation in range(0;1) where "0" gives elastic
                   and "1" nearly square slic
    :param int nb_classes: number of classes to be segmented(indexing from 0)
    :param dict_features: {clr: [str], ...}
    :param str clr_space:
    :param str gc_edge_type:
    :param float pca_coef: range (0, 1) or None
    :param dict_debug_imgs: {str: ...}
    :return [[int]]: segmentation matrix maping each pixel into a class

    >>> np.random.seed(0)
    >>> img = np.random.random((125, 150, 3)) / 2.
    >>> img[:, :75] += 0.5
    >>> scaler, pca, model = estim_model_classes_group([img], nb_classes=2)
    >>> segm = segment_color2d_slic_features_model_graphcut(img, scaler, pca, model)
    >>> segm.shape
    (125, 150)
    """
    logging.info('PIPELINE Superpixels-Features-Model-GraphCut')
    slic, features = compute_color2d_superpixels_features(img, sp_size, sp_regul,
                                      dict_features, clr_space, fts_norm=False)

    if dict_debug_imgs is not None:
        if img.ndim == 2:  # duplicate channels to be like RGB
            img = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)
        dict_debug_imgs['img'] = img
        dict_debug_imgs['slic'] = slic
        dict_debug_imgs['slic_mean'] = sk_color.label2rgb(slic, img, kind='avg')

    features = scaler.transform(features)
    if pca is not None:
        features = pca.fit_transform(features)

    proba = model.predict_proba(features)
    logging.debug('list of probabilities: %s', repr(proba.shape))

    gmm = mixture.GaussianMixture(n_components=proba.shape[1],
                                  covariance_type='full', max_iter=1)
    gmm.fit(features, np.argmax(proba, axis=1))
    proba = gmm.predict_proba(features)

    graph_labels = tl_gc.segment_graph_cut_general(slic, proba, img, features, gc_regul,
                                                   gc_edge_type, dict_debug_imgs=dict_debug_imgs)
    segm = graph_labels[slic]
    return segm


def compute_color2d_superpixels_features(img, sp_size=30, sp_regul=0.2,
                                         dict_features=FTS_SET_SIMPLE,
                                         clr_space='rgb', fts_norm=True):
    """ segment image into superpixels and estimate features per superpixel

    :param im: input RGB image
    :param int sp_size: initial size of a superpixel(meaning edge lenght)
    :param float sp_regul: regularisation in range(0;1) where "0" gives elastic
           and "1" nearly square segments
    :param dict_features:
    :param str clr_space:
    :return [[int]], [[floats]]: superpixels and related of features
    """
    assert sp_regul > 0., 'slic. regul must be positive'
    img = convert_img_clr_space(img, clr_space)
    logging.debug('run Superpixel clustering.')
    slic = tl_sp.segment_slic_img2d(img, sp_size=sp_size, rltv_compact=sp_regul)
    # plt.figure(), plt.imshow(slic)

    logging.debug('extract slic/superpixels features.')
    features, names = tl_fts.compute_selected_features_img2d(img, slic, dict_features)
    logging.debug('list of features RAW: %s', repr(features.shape))
    features[np.isnan(features)] = 0

    if fts_norm:
        logging.debug('norm all features.')
        features, scaler = tl_fts.norm_features(features)
        logging.debug('list of features NORM: %s', repr(features.shape))
    return slic, features


def train_classif_color2d_slic_features(list_images, list_annots,
                                        clr_space='rgb', sp_size=30, sp_regul=0.2,
                                        dict_features=FTS_SET_SIMPLE,
                                        clf_name=tl_clf.DEFAULT_CLASSIF_NAME,
                                        label_purity=0.9, feature_balance='unique',
                                        pca_coef=None, nb_classif_search=1,
                                        nb_jobs=1):
    logging.info('TRAIN Superpixels-Features-Classifier')
    assert len(list_images) == len(list_annots)

    list_slic, list_features, list_labels = list(), list(), list()
    for img, annot in zip(list_images, list_annots):
        assert img.shape[:2] == annot.shape[:2]
        slic, features = compute_color2d_superpixels_features(img, sp_size,
                          sp_regul, dict_features, clr_space, fts_norm=False)
        list_slic.append(slic)
        list_features.append(features)

        label_hist = tl_lbs.histogram_regions_labels_norm(slic, annot)
        labels = np.argmax(label_hist, axis=1)
        purity = np.max(label_hist, axis=1)
        labels[purity < label_purity] = -1
        list_labels.append(labels)

    logging.debug('prepare features...')
    # concentrate features, labels
    features, labels, sizes = tl_clf.convert_set_features_labels_2_dataset(
        dict(zip(range(len(list_features)), list_features)),
        dict(zip(range(len(list_labels)), list_labels)),
        balance=feature_balance, drop_labels=[-1])
    # drop do not care label whichare -1
    features = np.nan_to_num(features)

    logging.debug('train classifier...')
    # clf_pipeline = tl_clf.create_clf_pipeline(clf_name, pca_coef)
    # clf_pipeline.fit(np.array(features), np.array(labels, dtype=int))

    if len(sizes) > (CROSS_VAL_LEAVE_OUT * 5):
        cv = tl_clf.CrossValidatePSetsOut(sizes, nb_hold_out=CROSS_VAL_LEAVE_OUT)
    # for small nuber of training images this does not make sence
    else:
        cv = 10
    classif, _ = tl_clf.create_classif_train_export(clf_name, features, labels,
                            nb_search_iter=nb_classif_search, cross_val=cv,
                            nb_jobs=nb_jobs, pca_coef=pca_coef)

    return classif, list_slic, list_features, list_labels


def segment_color2d_slic_features_classif_graphcut(img, classif, clr_space='rgb',
                                                   sp_size=30, sp_regul=0.2, gc_regul=1.,
                                                   dict_features=FTS_SET_SIMPLE,
                                                   gc_edge_type='model',
                                                   dict_debug_imgs=None):
    """

    :param img:
    :param classif:
    :param clr_space:
    :param sp_size:
    :param sp_regul:
    :param gc_regul:
    :param dict_features:
    :param gc_edge_type:
    :param dict_debug_imgs:
    :return:

    >>> np.random.seed(0)
    >>> img = np.random.random((125, 150, 3)) / 2.
    >>> img[:, :75] += 0.5
    >>> annot = np.zeros((125, 150), dtype=int)
    >>> annot[:, :75] = 1
    >>> clf, _, _, _ = train_classif_color2d_slic_features([img], [annot])
    >>> segm = segment_color2d_slic_features_classif_graphcut(img, clf)
    >>> segm.shape
    (125, 150)
    """
    logging.info('SEGMENTATION Superpixels-Features-Classifier-GraphCut')
    slic, features = compute_color2d_superpixels_features(img, sp_size, sp_regul,
                                          dict_features, clr_space, fts_norm=False)

    proba = classif.predict_proba(features)

    if dict_debug_imgs is not None:
        if img.ndim == 2:  # duplicate channels to be like RGB
            img = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)
        dict_debug_imgs['img'] = img
        dict_debug_imgs['slic'] = slic
        dict_debug_imgs['slic_mean'] = sk_color.label2rgb(slic, img, kind='avg')

    graph_labels = tl_gc.segment_graph_cut_general(slic, proba, img, features, gc_regul,
                                                   gc_edge_type, dict_debug_imgs=dict_debug_imgs)
    segm = graph_labels[slic]
    # relabel according classif classes
    segm = classif.classes_[segm]
    return segm


def pipe_gray3d_slic_features_gmm_graphcut(img, nb_classes=4, spacing=(12, 1, 1),
                                           sp_size=15, sp_regul=0.2, gc_regul=0.1,
                                           dict_features=FTS_SET_SIMPLE):
    """ complete pipe-line for segmentation using superpixels, extracting features
    and graphCut segmentation

    :param img: input RGB image
    :param int sp_size: initial size of a superpixel(meaning edge lenght)
    :param float sp_regul: regularisation in range(0;1) where "0" gives elastic
           and "1" nearly square segments
    :param int nb_classes: number of classes to be segmented(indexing from 0)
    :param (int, int, int) spacing:
    :param float gc_regul:
    :return [[int]]: segmentation matrix maping each pixel into a class

    >>> np.random.seed(0)
    >>> img = np.random.random((5, 125, 150)) / 2.
    >>> img[:, :, :75] += 0.5
    >>> segm = pipe_gray3d_slic_features_gmm_graphcut(img)
    >>> segm.shape
    (5, 125, 150)
    """
    logging.info('PIPELINE Superpixels-Features-GraphCut')
    slic = tl_sp.segment_slic_img3d_gray(img, space=spacing, sp_size=sp_size,
                                         rltv_compact=sp_regul)
    # plt.imshow(segments)
    logging.info('extract segments/superpixels features.')
    # f = features.computeColourMean(img, segments)
    features, _ = tl_fts.compute_selected_features_gray3d(img, slic, dict_features)
    # merge features together
    logging.debug('list of features RAW: %s', repr(features.shape))
    features[np.isnan(features)] = 0

    logging.info('norm all features.')
    features, scaler = tl_fts.norm_features(features)
    logging.debug('list of features NORM: %s', repr(features.shape))

    model = tl_gc.estim_class_model_gmm(features, nb_classes)
    proba = model.predict_proba(features)
    logging.debug('list of probabilities: %s', repr(proba.shape))

    # resultGraph = graphCut.segment_graph_cut_int_vals(segments, prob, gcReg)
    graph_labels = tl_gc.segment_graph_cut_general(slic, proba, img, features, gc_regul)

    return graph_labels[slic]


# def pipe_color2d_slic_features_classif_graphcut(img, list_images, list_annots,
#                                                 sp_size=30, sp_regul=0.2,
#                                                 clr_space='rgb', gc_regul=1.,
#                                                 dict_features=FTS_SET_SIMPLE,
#                                                 clf_name=tl_clf.DEFAULT_CLASSIF_NAME,
#                                                 pca_coef=None, gc_edge_type='model',
#                                                 dict_debug_imgs=None):
#     logging.info('PIPELINE Superpixels-Features-Classifier-GraphCut')
#     classif, _, _, _ = train_classif_color2d_slic_features(
#         list_images, list_annots, sp_size=sp_size, sp_regul=sp_regul,
#         dict_features=dict_features, clr_space=clr_space, clf_name=clf_name,
#         pca_coef=pca_coef)
#
#     slic, features = compute_color2d_superpixels_features(img, sp_size, sp_regul,
#                                           dict_features, clr_space, fts_norm=False)
#
#     proba = classif.predict_proba(features)
#
#     if dict_debug_imgs is not None:
#         if img.ndim == 2:  # duplicate channels to be like RGB
#             img = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)
#         dict_debug_imgs['img'] = img
#         dict_debug_imgs['slic'] = slic
#         dict_debug_imgs['slic_mean'] = sk_color.label2rgb(slic, img, kind='avg')
#
#     graph_labels = tl_gc.segment_graph_cut_general(slic, proba, img, features, gc_regul,
#                                                    gc_edge_type, dict_debug_imgs=dict_debug_imgs)
#     segm = graph_labels[slic]
#     # relabel according classif classes
#     segm = classif.classes_[segm]
#     return segm, classif


# def estim_model_classes_annot(img, annot, clr_space='rgb', sp_size=30,
#                               sp_regul=0.2, dict_features=FTS_SET_SIMPLE,
#                               pca_coef=None):
#
#     slic, features = compute_color2d_superpixels_features(img, sp_size,
#                       sp_regul, dict_features, clr_space, fts_norm=False)
#     features = np.nan_to_num(features)
#
#     # scaling
#     scaler = preprocessing.StandardScaler()
#     scaler.fit(features)
#     features = scaler.transform(features)
#
#     pca = None
#     if pca_coef is not None:
#         pca = decomposition.PCA(pca_coef)
#         features = pca.fit_transform(features)
#
#     label_hist = tl_lbs.histogram_regions_labels_norm(slic, annot)
#     labels = np.argmax(label_hist, axis=1)
#     nb_classes = np.max(labels) + 1
#     model = mixture.GaussianMixture(n_components=nb_classes, covariance_type='full',
#                         n_iter=0)
#     model.fit(features, labels)
#
#     return scaler, pca, model

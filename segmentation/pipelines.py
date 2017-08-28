"""
Pipelines for supervised and unsupervised segmentation

Copyright (C) 2014-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging
import multiprocessing as mproc
from functools import partial

import numpy as np
import skimage.color as sk_color
from sklearn import preprocessing, mixture, decomposition

import segmentation.graph_cuts as seg_gc
import segmentation.superpixels as seg_sp
import segmentation.descriptors as seg_fts
import segmentation.labeling as seg_lbs
import segmentation.classification as seg_clf

CLASSIF_PARAMS = {'method': 'kNN', 'nb': 10}
FTS_SET_SIMPLE = seg_fts.FEATURES_SET_COLOR
CLASSIF_NAME = seg_clf.DEFAULT_CLASSIF_NAME
CLUSTER_METHOD = seg_clf.DEFAULT_CLUSTERING
CROSS_VAL_LEAVE_OUT = 2
NB_THREADS = max(1, int(mproc.cpu_count() * 0.6))

DICT_CONVERT_COLOR = {
    'hsv': sk_color.rgb2hsv,
    'luv': sk_color.rgb2luv,
    'lab': sk_color.rgb2lab,
    'hed': sk_color.rgb2hed,
    'xyz': sk_color.rgb2xyz
}


def convert_img_color_space(image, clr_space):
    """ convert image colour space from RGB to xxx

    :param image: rgb image
    :param clr_space: str
    :return: image
    """
    if image.ndim == 3 and image.shape[2] == 3 \
            and clr_space in DICT_CONVERT_COLOR:
        image = DICT_CONVERT_COLOR[clr_space](image)
    return image


def pipe_color2d_slic_features_gmm_graphcut(image, nb_classes=3,
                                            clr_space='rgb',
                                            sp_size=30, sp_regul=0.2,
                                            gc_regul=1.,
                                            dict_features=FTS_SET_SIMPLE,
                                            proba_type='GMM',
                                            gc_edge_type='model_lT',
                                            pca_coef=None,
                                            dict_debug_imgs=None):
    """ complete pipe-line for segmentation using superpixels, extracting features
    and graphCut segmentation

    :param float gc_regul: GC regularisation
    :param image: input RGB image
    :param int sp_size: initial size of a superpixel(meaning edge lenght)
    :param float sp_regul: regularisation in range(0;1) where "0" gives elastic
                   and "1" nearly square slic
    :param int nb_classes: number of classes to be segmented(indexing from 0)
    :param dict_features: {clr: [str], ...}
    :param str clr_space: use color space
    :param float gc_regul: GC regularisation
    :param str gc_edge_type:
    :param float pca_coef: range (0, 1) or None
    :param dict_debug_imgs: {str: ...}
    :return [[int]]: segmentation matrix maping each pixel into a class

    >>> np.random.seed(0)
    >>> image = np.random.random((125, 150, 3)) / 2.
    >>> image[:, :75] += 0.5
    >>> segm = pipe_color2d_slic_features_gmm_graphcut(image, nb_classes=2)
    >>> segm.shape
    (125, 150)
    """
    logging.info('PIPELINE Superpixels-Features-GMM-GraphCut')
    slic, features = compute_color2d_superpixels_features(image, clr_space,
                                                          sp_size, sp_regul,
                                                          dict_features)

    if dict_debug_imgs is not None:
        if image.ndim == 2:  # duplicate channels to be like RGB
            image = np.rollaxis(np.tile(image, (3, 1, 1)), 0, 3)
        dict_debug_imgs['image'] = image
        dict_debug_imgs['slic'] = slic
        dict_debug_imgs['slic_mean'] = sk_color.label2rgb(slic, image,
                                                          kind='avg')

    if pca_coef is not None:
        pca = decomposition.PCA(pca_coef)
        features = pca.fit_transform(features)

    model = seg_gc.estim_class_model(features, nb_classes, proba_type)
    proba = model.predict_proba(features)
    logging.debug('list of probabilities: %s', repr(proba.shape))

    gmm = mixture.GaussianMixture(n_components=nb_classes,
                                  covariance_type='full', max_iter=1)
    gmm.fit(features, np.argmax(proba, axis=1))
    proba = gmm.predict_proba(features)

    graph_labels = seg_gc.segment_graph_cut_general(slic, proba, image,
            features, gc_regul, gc_edge_type, dict_debug_imgs=dict_debug_imgs)
    segm = graph_labels[slic]
    return segm


def estim_model_classes_group(list_images, nb_classes=4, clr_space='rgb',
                              sp_size=30, sp_regul=0.2,
                              dict_features=FTS_SET_SIMPLE,
                              pca_coef=None, proba_type='GMM',
                              nb_jobs=NB_THREADS):
    """ estimate a model from sequence of input images and return it as result

    :param [ndarray] list_images:
    :param int nb_classes: number of clasees
    :param str clr_space: chose the color space
    :param int sp_size: initial size of a superpixel(meaning edge lenght)
    :param float sp_regul: regularisation in range(0;1) where "0" gives elastic
                   and "1" nearly square slic
    :param {str: [str]} dict_features: list of features to be extracted
    :param float pca_coef: range (0, 1) or None
    :param str proba_type:
    :param int nb_jobs:
    :return:
    """
    list_slic, list_features = list(), list()

    mproc_pool = mproc.Pool(nb_jobs)
    wrapper_compute = partial(compute_color2d_superpixels_features,
                              sp_size=sp_size, sp_regul=sp_regul,
                              dict_features=dict_features,
                              clr_space=clr_space, fts_norm=False)
    for slic, features in mproc_pool.imap_unordered(wrapper_compute,
                                                    list_images):
        list_slic.append(slic)
        list_features.append(features)
    mproc_pool.close()
    mproc_pool.join()

    # for img in list_images:
    #     slic, features = compute_color2d_superpixels_features(img, sp_size,
    #                     sp_regul, dict_features, clr_space, fts_norm=False)
    #     list_slic.append(slic)
    #     list_features.append(features)

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

    model = seg_gc.estim_class_model(features, nb_classes, proba_type)
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

    :param ndarry img: input RGB image
    :param str clr_space: chose the color space
    :param int sp_size: initial size of a superpixel(meaning edge lenght)
    :param float sp_regul: regularisation in range(0;1) where "0" gives elastic
                   and "1" nearly square slic
    :param {str: [str]} dict_features: list of features to be extracted
    :param float gc_regul: GC regularisation
    :param str gc_edge_type: select the GC edge type
    :param float pca_coef: range (0, 1) or None
    :param dict_debug_imgs: {str: ...}
    :return [[int]]: segmentation matrix mapping each pixel into a class

    >>> np.random.seed(0)
    >>> img = np.random.random((125, 150, 3)) / 2.
    >>> img[:, :75] += 0.5
    >>> sc, pca, model = estim_model_classes_group([img], nb_classes=2)
    >>> segm = segment_color2d_slic_features_model_graphcut(img, sc, pca, model)
    >>> segm.shape
    (125, 150)
    """
    logging.info('PIPELINE Superpixels-Features-Model-GraphCut')
    slic, features = compute_color2d_superpixels_features(img, clr_space,
                                                          sp_size, sp_regul,
                                                          dict_features,
                                                          fts_norm=False)

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

    graph_labels = seg_gc.segment_graph_cut_general(slic, proba, img, features,
                      gc_regul, gc_edge_type, dict_debug_imgs=dict_debug_imgs)
    segm = graph_labels[slic]
    return segm


def compute_color2d_superpixels_features(image, clr_space='rgb',
                                         sp_size=30, sp_regul=0.2,
                                         dict_features=FTS_SET_SIMPLE,
                                         fts_norm=True):
    """ segment image into superpixels and estimate features per superpixel

    :param ndarray image: input RGB image
    :param str clr_space: chose the color space
    :param int sp_size: initial size of a superpixel(meaning edge lenght)
    :param float sp_regul: regularisation in range(0;1) where "0" gives elastic
           and "1" nearly square segments
    :param {str: [str]} dict_features: list of features to be extracted
    :param bool fts_norm: weather nomalise features
    :return [[int]], [[floats]]: superpixels and related of features
    """
    assert sp_regul > 0., 'slic. regularisation must be positive'
    logging.debug('run Superpixel clustering.')
    slic = seg_sp.segment_slic_img2d(image, sp_size=sp_size,
                                     rltv_compact=sp_regul)
    # plt.figure(), plt.imshow(slic)

    logging.debug('extract slic/superpixels features.')
    image = convert_img_color_space(image, clr_space)
    features, _ = seg_fts.compute_selected_features_img2d(image, slic,
                                                          dict_features)
    logging.debug('list of features RAW: %s', repr(features.shape))
    features[np.isnan(features)] = 0

    if fts_norm:
        logging.debug('norm all features.')
        features, _ = seg_fts.norm_features(features)
        logging.debug('list of features NORM: %s', repr(features.shape))
    return slic, features


def wrapper_compute_color2d_slic_features_labels(img_annot, clr_space,
                                                 sp_size, sp_regul,
                                                 dict_features, label_purity):
    img, annot = img_annot
    assert img.shape[:2] == annot.shape[:2]
    slic, features = compute_color2d_superpixels_features(img, clr_space,
                                                          sp_size, sp_regul,
                                                          dict_features,
                                                          fts_norm=False)

    label_hist = seg_lbs.histogram_regions_labels_norm(slic, annot)
    labels = np.argmax(label_hist, axis=1)
    purity = np.max(label_hist, axis=1)
    labels[purity < label_purity] = -1
    return slic, features, labels


def train_classif_color2d_slic_features(list_images, list_annots, clr_space='rgb',
                                        sp_size=30, sp_regul=0.2,
                                        dict_features=FTS_SET_SIMPLE,
                                        clf_name=CLASSIF_NAME, label_purity=0.9,
                                        feature_balance='unique',
                                        pca_coef=None, nb_classif_search=1,
                                        nb_jobs=1):
    """ train classifier on list of annotated images

    :param [ndarray] list_images:
    :param [ndarray] list_annots:
    :param str clr_space: chose the color space
    :param int sp_size: initial size of a superpixel(meaning edge lenght)
    :param float sp_regul: regularisation in range(0;1) where "0" gives elastic
           and "1" nearly square segments
    :param {str: [str]} dict_features: list of features to be extracted
    :param str clf_name: selet udsed classifier
    :param float label_purity: set the sample-labels purity for training
    :param str feature_balance: set how to balance datasets
    :param float pca_coef: select PCA coef or None
    :param int nb_classif_search: number of tries for hyper-parameters seach
    :param int nb_jobs: parallelism
    :return:
    """
    logging.info('TRAIN Superpixels-Features-Classifier')
    assert len(list_images) == len(list_annots)

    list_slic, list_features, list_labels = list(), list(), list()

    mproc_pool = mproc.Pool(nb_jobs)
    wrapper_compute = partial(wrapper_compute_color2d_slic_features_labels,
                              clr_space=clr_space, sp_size=sp_size,
                              sp_regul=sp_regul, dict_features=dict_features,
                              label_purity=label_purity)
    list_imgs_annot = zip(list_images,  list_annots)
    for slic, fts, lbs in mproc_pool.imap_unordered(wrapper_compute,
                                                    list_imgs_annot):
        list_slic.append(slic)
        list_features.append(fts)
        list_labels.append(lbs)
    mproc_pool.close()
    mproc_pool.join()

    # for img, annot in zip(list_images, list_annots):
    #     assert img.shape[:2] == annot.shape[:2]
    #     slic, features = compute_color2d_superpixels_features(img, clr_space,
    #                                                           sp_size, sp_regul,
    #                                                           dict_features,
    #                                                           fts_norm=False)
    #     list_slic.append(slic)
    #     list_features.append(features)
    #
    #     label_hist = seg_lbs.histogram_regions_labels_norm(slic, annot)
    #     labels = np.argmax(label_hist, axis=1)
    #     purity = np.max(label_hist, axis=1)
    #     labels[purity < label_purity] = -1
    #     list_labels.append(labels)

    logging.debug('prepare features...')
    # concentrate features, labels
    features, labels, sizes = seg_clf.convert_set_features_labels_2_dataset(
        dict(zip(range(len(list_features)), list_features)),
        dict(zip(range(len(list_labels)), list_labels)),
        balance=feature_balance, drop_labels=[-1])
    # drop do not care label whichare -1
    features = np.nan_to_num(features)

    logging.debug('train classifier...')
    # clf_pipeline = seg_clf.create_clf_pipeline(clf_name, pca_coef)
    # clf_pipeline.fit(np.array(features), np.array(labels, dtype=int))

    if len(sizes) > (CROSS_VAL_LEAVE_OUT * 5):
        cv = seg_clf.CrossValidatePSetsOut(sizes, nb_hold_out=CROSS_VAL_LEAVE_OUT)
    # for small nuber of training images this does not make sence
    else:
        cv = 10

    classif, _ = seg_clf.create_classif_train_export(clf_name, features, labels,
                                                     nb_search_iter=nb_classif_search,
                                                     cross_val=cv, nb_jobs=nb_jobs,
                                                     pca_coef=pca_coef)

    return classif, list_slic, list_features, list_labels


def segment_color2d_slic_features_classif_graphcut(image, classif,
                                                   clr_space='rgb',
                                                   sp_size=30, sp_regul=0.2,
                                                   gc_regul=1.,
                                                   dict_features=FTS_SET_SIMPLE,
                                                   gc_edge_type='model',
                                                   dict_debug_imgs=None):
    """ take trained classifier and apply it on new images

    :param ndarray image: input image
    :param classif: trained classifier
    :param str clr_space: chose the color space
    :param int sp_size: initial size of a superpixel(meaning edge lenght)
    :param float sp_regul: regularisation in range(0;1) where "0" gives elastic
           and "1" nearly square segments
    :param {str: [str]} dict_features: list of features to be extracted
    :param gc_regul: regularisation for GC
    :param str gc_edge_type: select the GC edge type
    :param dict_debug_imgs:
    :return:

    >>> np.random.seed(0)
    >>> seg_fts.USE_CYTHON = False
    >>> image = np.random.random((125, 150, 3)) / 2.
    >>> image[:, 75:] += 0.5
    >>> annot = np.zeros(image.shape[:2], dtype=int)
    >>> annot[:, 75:] = 1
    >>> clf, _, _, _ = train_classif_color2d_slic_features([image], [annot])
    >>> segm = segment_color2d_slic_features_classif_graphcut(image, clf)
    >>> segm.shape
    (125, 150)
    """
    logging.info('SEGMENTATION Superpixels-Features-Classifier-GraphCut')
    slic, features = compute_color2d_superpixels_features(image, clr_space,
                                                          sp_size, sp_regul,
                                                          dict_features,
                                                          fts_norm=False)

    proba = classif.predict_proba(features)

    if dict_debug_imgs is not None:
        if image.ndim == 2:  # duplicate channels to be like RGB
            image = np.rollaxis(np.tile(image, (3, 1, 1)), 0, 3)
        dict_debug_imgs['image'] = image
        dict_debug_imgs['slic'] = slic
        dict_debug_imgs['slic_mean'] = sk_color.label2rgb(slic, image, kind='avg')

    graph_labels = seg_gc.segment_graph_cut_general(slic, proba, image, features,
                                                    gc_regul, gc_edge_type,
                                                    dict_debug_imgs=dict_debug_imgs)
    segm = graph_labels[slic]
    # relabel according classif classes
    segm = classif.classes_[segm]
    return segm


def pipe_gray3d_slic_features_gmm_graphcut(image, nb_classes=4, spacing=(12, 1, 1),
                                           sp_size=15, sp_regul=0.2, gc_regul=0.1,
                                           dict_features=FTS_SET_SIMPLE):
    """ complete pipe-line for segmentation using superpixels, extracting features
    and graphCut segmentation

    :param ndarray img: input RGB image
    :param int sp_size: initial size of a superpixel(meaning edge lenght)
    :param float sp_regul: regularisation in range(0;1) where "0" gives elastic
           and "1" nearly square segments
    :param int nb_classes: number of classes to be segmented(indexing from 0)
    :param (int, int, int) spacing:
    :param float gc_regul: regularisation for GC
    :return [[int]]: segmentation matrix maping each pixel into a class

    >>> np.random.seed(0)
    >>> image = np.random.random((5, 125, 150)) / 2.
    >>> image[:, :, :75] += 0.5
    >>> segm = pipe_gray3d_slic_features_gmm_graphcut(image)
    >>> segm.shape
    (5, 125, 150)
    """
    logging.info('PIPELINE Superpixels-Features-GraphCut')
    slic = seg_sp.segment_slic_img3d_gray(image, sp_size=sp_size,
                                          rltv_compact=sp_regul, space=spacing)
    # plt.imshow(segments)
    logging.info('extract segments/superpixels features.')
    # f = features.computeColourMean(image, segments)
    features, _ = seg_fts.compute_selected_features_gray3d(image, slic, dict_features)
    # merge features together
    logging.debug('list of features RAW: %s', repr(features.shape))
    features[np.isnan(features)] = 0

    logging.info('norm all features.')
    features, _ = seg_fts.norm_features(features)
    logging.debug('list of features NORM: %s', repr(features.shape))

    model = seg_gc.estim_class_model_gmm(features, nb_classes)
    proba = model.predict_proba(features)
    logging.debug('list of probabilities: %s', repr(proba.shape))

    # resultGraph = graphCut.segment_graph_cut_int_vals(segments, prob, gcReg)
    graph_labels = seg_gc.segment_graph_cut_general(slic, proba, image, features,
                                                    gc_regul)

    return graph_labels[slic]


# def pipe_color2d_slic_features_classif_graphcut(img, list_images, list_annots,
#                                                 sp_size=30, sp_regul=0.2,
#                                                 clr_space='rgb', gc_regul=1.,
#                                                 dict_features=FTS_SET_SIMPLE,
#                                                 clf_name=CLASSIF_NAME,
#                                                 pca_coef=None,
#                                                 gc_edge_type='model',
#                                                 dict_debug_imgs=None):
#     logging.info('PIPELINE Superpixels-Features-Classifier-GraphCut')
#     classif, _, _, _ = train_classif_color2d_slic_features(
#         list_images, list_annots, sp_size=sp_size, sp_regul=sp_regul,
#         dict_features=dict_features, clr_space=clr_space, clf_name=clf_name,
#         pca_coef=pca_coef)
#
#     slic, features = compute_color2d_superpixels_features(img,
#                   sp_size, sp_regul, dict_features, clr_space, fts_norm=False)
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
#     graph_labels = seg_gc.segment_graph_cut_general(slic, proba, img, features,
#                   gc_regul, gc_edge_type, dict_debug_imgs=dict_debug_imgs)
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
#     label_hist = seg_lb.histogram_regions_labels_norm(slic, annot)
#     labels = np.argmax(label_hist, axis=1)
#     nb_classes = np.max(labels) + 1
#     model = mixture.GaussianMixture(n_components=nb_classes,
#                                     covariance_type='full', n_iter=0)
#     model.fit(features, labels)
#
#     return scaler, pca, model

"""
Supporting file to create and set parameters for scikit-learn classifiers
and some prepossessing functions that support classification

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import pickle
import logging
import random
import collections
import traceback
# import gc
# import time
import multiprocessing as mproc

import numpy as np
import pandas as pd
# import itertools
# import matplotlib.pyplot as plt
from scipy import interp
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_random
from sklearn import grid_search, metrics
from sklearn import preprocessing, feature_selection, decomposition
from sklearn import cluster
from sklearn import ensemble, neighbors, svm, tree
from sklearn import pipeline, linear_model, neural_network
from sklearn import model_selection

import segmentation.annotation as tl_annot

# NAME_FILE_RESULTS = 'results.csv'
TEMPLATE_NAME_CLF = 'classifier_{}.pkl'
DEFAULT_CLASSIF_NAME = 'RandForest'
DEFAULT_CLUSTERING = 'kMeans'
# DEFAULT_MIN_NB_SPL = 25
NB_JOBS_CLASSIF_SEARCH = 5
NB_CLASSIF_SEARCH_ITER = 250
NAME_CSV_FEATURES_SELECT = 'feature_selection.csv'
NAME_CSV_CLASSIF_CV_SCORES = 'classif_{}_cross-val_scores-{}.csv'
NAME_CSV_CLASSIF_CV_ROC = 'classif_{}_cross-val_ROC-{}.csv'
NAME_TXT_CLASSIF_CV_AUC = 'classif_{}_cross-val_AUC-{}.txt'
METRIC_AVERAGES = ['macro', 'weighted']


def create_classifiers(nb_jobs=-1):
    """ create all classifiers with default parameters

    :param nb_jobs: int, number of parallel if possible
    :return: {str: clf}

    >>> create_classifiers()  # doctest: +ELLIPSIS
    {...}
    """
    clfs = {
        'RandForest': ensemble.RandomForestClassifier(n_estimators=20,  # oob_score=True,
                                                      min_samples_leaf=2,
                                                      min_samples_split=3,
                                                      n_jobs=nb_jobs),
        'GradBoost': ensemble.GradientBoostingClassifier(subsample=0.25,
                                                         warm_start=False,
                                                         max_depth=6,
                                                         min_samples_leaf=6,
                                                         n_estimators=200,
                                                         min_samples_split=7),
        'LogReg': linear_model.LogisticRegression(solver='sag', n_jobs=nb_jobs),
        'KNN': neighbors.KNeighborsClassifier(n_jobs=nb_jobs),
        'SVM': svm.SVC(kernel='rbf', probability=True),
        'DecTree': tree.DecisionTreeClassifier(),
        # 'RBM': create_pipeline_neuron_net(),
        # 'Adaboost':   ensemble.AdaBoostClassifier(n_estimators=5),
        # 'NuSVM-rbf': svm.NuSVC(kernel='rbf', probability=True),
    }
    return clfs


def create_clf_pipeline(name_classif=DEFAULT_CLASSIF_NAME, pca_coef=0.95):
    """ create complete pipeline with all required steps

    :param name_classif: str, key name of classif
    :return: object

    >>> create_clf_pipeline()  # doctest: +ELLIPSIS
    Pipeline(...)
    """
    # create the pipeline
    components = [('scaler', preprocessing.StandardScaler())]
    if not pca_coef is None:
        components += [('reduce_dim', decomposition.PCA(pca_coef))]
    components += [('classif', create_classifiers()[name_classif])]
    clf_pipeline = pipeline.Pipeline(components)
    return clf_pipeline


def create_clf_param_search_grid(name_classif=DEFAULT_CLASSIF_NAME):
    """ create parameter grid for search

    :param name_classif: str, key name of classif
    :return: {str: ...}

    >>> create_clf_param_search_grid()  # doctest: +ELLIPSIS
    {...}
    """
    clf_params = {
        'RandForest': {'clf__n_estimators': range(2, 25, 1)},
        'KNN': {'clf__n_neighbors': range(5, 20, 3)},
        'SVM': {'clf__kernel': ('poly', 'rbf', 'sigmoid')},
        'DecTree': {'clf__criterion': ('gini', 'entropy')},
        'GradBoost': {'clf__n_estimators': range(10, 250, 20)},
        'LogReg': {'clf__penalty': ('l1', 'l2')}}
        # if this classif is not set use no params
    if name_classif not in clf_params.keys():
        clf_params[name_classif] = {}
    return clf_params[name_classif]


def create_clf_param_search_dist(name_classif=DEFAULT_CLASSIF_NAME):
    """ create parameter distribution for random search
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html
    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    :param name_classif: str, key name of classif
    :return: {str: ...}

    >>> create_clf_param_search_dist()  # doctest: +ELLIPSIS
    {...}
    """
    clf_params = {
    'RandForest': {
        'classif__n_estimators': sp_randint(2, 25),
        'classif__min_samples_split': sp_randint(2, 9),
        'classif__min_samples_leaf': sp_randint(1, 7),
    },
    'KNN': {
        'classif__n_neighbors': sp_randint(5, 25),
        'classif__algorithm': ('ball_tree', 'kd_tree'), # , 'brute'
        'classif__weights': ('uniform', 'distance'),
        # 'clf__p': [1, 2],
    },
    'SVM': {
        'classif__C': sp_random(0., 1.),
        'classif__kernel': ('poly', 'rbf', 'sigmoid'),
        'classif__degree': sp_randint(2, 9),
    },
    'DecTree': {
        'classif__criterion': ('gini', 'entropy'),
        'classif__min_samples_split': sp_randint(2, 9),
        'classif__min_samples_leaf': sp_randint(1, 7),
    },
    'GradBoost': {
        # 'clf__loss': ('deviance', 'exponential'), # only for 2 cls
        'classif__n_estimators': sp_randint(10, 200),
        'classif__max_depth': sp_randint(1, 7),
        'classif__min_samples_split': sp_randint(2, 9),
        'classif__min_samples_leaf': sp_randint(1, 7),
    },
    'LogReg': {
        'classif__C': sp_random(0., 1.),
        # 'classif__penalty': ('l1', 'l2'),
        # 'classif__dual': (False, True),
        'classif__solver': ('newton-cg', 'lbfgs', 'sag'),
        # 'classif__loss': ('deviance', 'exponential'), # only for 2 cls
    }}
    # if this classif is not set use no params
    if name_classif not in clf_params.keys():
        clf_params[name_classif] = {}
    return clf_params[name_classif]


def create_pipeline_neuron_net():
    """ create classifier for simple neuronal network

    :return: clf

    >>> create_pipeline_neuron_net()  # doctest: +ELLIPSIS
    Pipeline(...)
    """
    # Models we will use
    logistic = linear_model.LogisticRegression()
    rbm = neural_network.BernoulliRBM(learning_rate=0.05, n_components=35,
                                      n_iter=299, verbose=False)
    clf = pipeline.Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    return clf


# def append_matrix_vertical(old, new):
#     """ append a matrix after another one in vertical direction
#
#     :param old: np.matrix<total*l>
#     :param new: np.matrix<total*k>
#     :return: np.matrix<total*(k+l)>
#
#     >>> a, b = np.zeros((10, 5)), np.zeros((10, 4))
#     >>> append_matrix_vertical(a, b).shape
#     (10, 9)
#     """
#     if old is None:
#         old = new.copy()
#     else:
#         # logging.debug('append V:{} <- {}'.format(old.shape, new.shape))
#         old = np.hstack((old, new))
#     return old


def compute_classif_metrics(y_true, y_pred, metric_averages=METRIC_AVERAGES):
    """ compute standard metrics for multi-class classification

    :param [int] y_true:
    :param [int] y_pred:
    :return {str: float}:

    >>> np.random.seed(0)
    >>> y_true = np.random.randint(0, 3, 25)
    >>> y_pred = np.random.randint(0, 2, 25)
    >>> d = compute_classif_metrics(y_true, y_true)
    >>> d['accuracy']  # doctest: +ELLIPSIS
    1.0
    >>> d['confusion']
    [[10, 0, 0], [0, 10, 0], [0, 0, 5]]
    >>> d = compute_classif_metrics(y_true, y_pred)
    >>> d['accuracy']  # doctest: +ELLIPSIS
    0.32...
    >>> d['confusion']
    [[3, 7, 0], [5, 5, 0], [1, 4, 0]]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert y_true.shape == y_pred.shape
    logging.debug('unique lbs true: %s, predict %s',
                  repr(np.unique(y_true)), repr(np.unique(y_pred)))

    uq_y_true = np.unique(y_true)
    # in case the are just two classes relabel them as [0, 1] only
    # solving sklearn error:
    #  "ValueError: pos_label=1 is not a valid label: array([  0, 255])"
    if np.array_equal(sorted(uq_y_true), sorted(np.unique(y_pred))) \
            and len(uq_y_true) <= 2:
        logging.debug('relabeling original %s to [0, 1]', repr(uq_y_true))
        lut = np.zeros(uq_y_true.max() + 1)
        if len(uq_y_true) == 2:
            lut[uq_y_true[1]] = 1
        y_true = lut[y_true]
        y_pred = lut[y_pred]

    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    EVAL_STR = 'EVALUATION: {:<2} PRE: {:.3f} REC: {:.3f} F1: {:.3f} S: {:>6}'
    try:
        p, r, f, s = metrics.precision_recall_fscore_support(y_true, y_pred)
        for l in range(len(p)):
            logging.debug(EVAL_STR.format(l, p[l], r[l], f[l], s[l]))
    except:
        logging.error(traceback.format_exc())

    dict_metrics = {
        'ARS': metrics.adjusted_rand_score(y_true, y_pred),
        # 'F1':  metrics.f1_score(y_true, y_pred),
        'accuracy':  metrics.accuracy_score(y_true, y_pred),
        # 'precision':  metrics.precision_score(y_true, y_pred),
        'confusion': metrics.confusion_matrix(y_true, y_pred).tolist(),
        # 'report':    metrics.classification_report(labels, predicted),
    }
    # compute aggregated precision, recall, f-score, support
    names = ['precision', 'recall', 'f1', 'support']
    for avg in metric_averages:
        try:
            mtr = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                          average=avg)
            res = dict(zip(['{}_{}'.format(n, avg) for n in names], mtr))
        except:
            logging.error(traceback.format_exc())
            res = dict(zip(['{}_{}'.format(n, avg) for n in names], [-1] * 4))
        dict_metrics.update(res)
    return dict_metrics


def compute_classif_stat_segm_annot(set_annot_segm_name, relabel=False):
    annot, segm, name = set_annot_segm_name
    assert segm.shape == annot.shape, \
        'dimension do not match: %s - %s' % (repr(segm.shape), repr(annot.shape))
    if relabel:
        segm = tl_annot.relabel_max_overlap_unique(annot, segm, keep_bg=False)
    y_true, y_pred = annot.ravel(), segm.ravel()
    dict_stat = compute_classif_metrics(y_true, y_pred, metric_averages=['macro'])
    dict_stat['name'] = name
    return dict_stat


def compute_stat_per_image(segms, annots, names=None, nb_jobs=1):
    """ compute statistic over multiple segmentations with annotation

    :param [ndarray] segms:
    :param [ndarray] annots:
    :param [str] names:
    :param int nb_jobs:
    :return DF:


    >>> np.random.seed(0)
    >>> img_true = np.random.randint(0, 3, (50, 100))
    >>> img_pred = np.random.randint(0, 2, (50, 100))
    >>> df = compute_stat_per_image([img_true], [img_true])
    >>> df.iloc[0]  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ARS                                                         1
    accuracy                                                    1
    confusion          [[1672, 0, 0], [0, 1682, 0], [0, 0, 1646]]
    f1_macro                                                    1
    precision_macro                                             1
    recall_macro                                                1
    support_macro                                            None
    Name: 0, dtype: object
    >>> df = compute_stat_per_image([img_true], [img_pred])
    >>> df.iloc[0]  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ARS                                                       0.0...
    accuracy                                                  0.3384
    confusion          [[836, 826, 770], [836, 856, 876], [0, 0, 0]]
    f1_macro                                                0.270077
    precision_macro                                         0.336306
    recall_macro                                            0.225694
    support_macro                                               None
    Name: 0, dtype: object
    """
    assert len(segms) == len(annots)
    if names is None:
        names = map(str, range(len(segms)))
    df_stat = pd.DataFrame()
    if nb_jobs > 1:
        mproc_pool = mproc.Pool(nb_jobs)
        for dict_stat in mproc_pool.imap_unordered(compute_classif_stat_segm_annot,
                                                   zip(annots, segms, names)):
            df_stat = df_stat.append(dict_stat, ignore_index=True)
        mproc_pool.close()
        mproc_pool.join()
    else:
        for annot, seg, name in zip(annots, segms, names):
            dict_stat = compute_classif_stat_segm_annot((annot, seg, name))
            df_stat = df_stat.append(dict_stat, ignore_index=True)
    df_stat.set_index('name', inplace=True)
    return df_stat


def feature_scoring_selection(features, labels, names=None, path_out=''):
    """ find the best features and retrun the indexes
    http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_recovery.html
    http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html

    :param features: np.array<nb_spl, nb_fts>
    :param labels: np.array<nb_spl, 1>
    :param names: [str]
    :param path_out: str
    :return:

    >>> from sklearn.datasets import make_classification
    >>> features, labels = make_classification(n_samples=250, n_features=5,
    ...                         n_informative=3, n_redundant=0, n_repeated=0,
    ...                         n_classes=2, random_state=0, shuffle=False)
    >>> indices, df_scoring = feature_scoring_selection(features, labels)
    >>> indices
    array([1, 0, 2, 3, 4])
    >>> df_scoring  # doctest: +NORMALIZE_WHITESPACE
              ExtTree     F-test     k-Best  variance
    feature
    1        0.248465   0.755881   0.755881  2.495970
    2        0.330818  58.944450  58.944450  1.851036
    3        0.221636   2.242583   2.242583  1.541042
    4        0.106441   4.022076   4.022076  0.965971
    5        0.092639   0.022651   0.022651  1.016170
    >>> features[:, 2] = 1
    >>> indices, df_scoring = feature_scoring_selection(features, labels)
    >>> indices
    array([1, 0, 3, 4, 2])
    """
    logging.info('Feature selection for %s', repr(names))
    logging.debug('Features: %s and labels: %s', repr(features.shape), repr(labels.shape))
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    # Build a forest and compute the feature importances
    forest = ensemble.ExtraTreesClassifier(n_estimators=125, random_state=0)
    forest.fit(features, labels)
    f_test, _ = feature_selection.f_regression(features, labels)
    k_best = feature_selection.SelectKBest(feature_selection.f_classif, k='all')
    k_best.fit(features, labels)
    vars = feature_selection.VarianceThreshold().fit(features, labels)
    imp = {
        'ExtTree': forest.feature_importances_,
        # 'Lasso': np.abs(lars_cv.coef_),
        'k-Best': k_best.scores_,
        'variance': vars.variances_,
        'F-test': f_test
    }
    # std = np.std([t.feature_importances_ for t in forest.estimators_], axis=0)
    indices = np.argsort(forest.feature_importances_)[::-1]
    if names is None or len(names) < features.shape[1]:
        names = map(str, range(1, features.shape[1] + 1))

    df_scoring = pd.DataFrame()
    for i, n in enumerate(names):
        dict_scores = {k: imp[k][i] for k in imp}
        dict_scores['feature'] = n
        df_scoring = df_scoring.append(dict_scores, ignore_index=True)
    df_scoring.set_index(['feature'], inplace=True)
    logging.debug(df_scoring)

    if os.path.exists(path_out):
        path_csv = os.path.join(path_out, NAME_CSV_FEATURES_SELECT)
        logging.debug('export Feature scoting to "%s"', path_csv)
        df_scoring.to_csv(path_csv)
    return indices, df_scoring


def save_classifier(path_out, classif, clf_name, params, feature_names=None,
                    label_names=None):
    """ estimate classif for all data and export it

    :param feature_names: [str]
    :param path_out: str
    :param classif: sklearn classif.
    :param clf_name: str, name
    :param fts_train: [np.array<m, k>]
    :param y_train: [int]
    :param label_names: [str] list of string names of label_names
    :return: str

    >>> clf = create_classifiers()['RandForest']
    >>> p_clf = save_classifier('.', clf, 'TESTINNG', {})
    >>> p_clf
    './classifier_TESTINNG.pkl'
    >>> d_clf = load_classifier(p_clf)
    >>> sorted(d_clf.keys())
    ['clf_pipeline', 'features', 'label_names', 'name', 'params']
    >>> d_clf['clf_pipeline']  # doctest: +ELLIPSIS
    RandomForestClassifier(...)
    >>> d_clf['name']
    'TESTINNG'
    >>> os.remove(p_clf)
    """
    assert os.path.exists(path_out)
    dict_classif = {
        'params': params,
        'name': clf_name,
        'clf_pipeline': classif,
        'features': feature_names,
        'label_names': label_names,
    }

    path_clf = os.path.join(path_out, TEMPLATE_NAME_CLF.format(clf_name))
    logging.info('export classif. of %s to "%s"', dict_classif, path_clf)
    with open(path_clf, 'wb') as f:
        pickle.dump(dict_classif, f)
    logging.debug('export finished')
    return path_clf


def load_classifier(path_classif):
    """ estimate classif. for all data and export it

    :param str path_classif:
    :return {str: ...}:
    """
    assert os.path.exists(path_classif), 'missing "%s"' % path_classif
    # path_classif = os.path.join(path_out, TEMPLATE_NAME_CLF.format(classif_name))
    logging.info('import classif from "%s"', path_classif)
    if not os.path.exists(path_classif):
        logging.debug('classif does not exist')
        return None
    with open(path_classif, 'rb') as f:
        dict_clf = pickle.load(f)
    # dict_clf['name'] = classif_name
    logging.debug('load classif: %s', repr(dict_clf.keys()))
    return dict_clf


def export_results_clf_search(path_out, name_clf, clf_search):
    """ do the final testing and save all results

    :param path_out: str
    :param name_clf: str, name
    :param clf_search: object
    """
    assert os.path.exists(path_out)
    fn_path_out = lambda s: os.path.join(path_out, 'classif_%s_%s.txt' % (name_clf, s))

    with open(fn_path_out('search_params_scores'), 'w') as f:
        f.write('\n'.join([repr(gs) for gs in clf_search.grid_scores_]))

    with open(fn_path_out('search_params_best'), 'w') as f:
        params = clf_search.best_params_
        rows = ['{:30s} {}'.format('"{}":'.format(k), params[k]) for k in params]
        f.write('\n'.join(rows))


def create_classif_train_export(clf_name, features, labels, cross_val=10,
                                nb_search_iter=1, search_type='random',
                                nb_jobs=NB_JOBS_CLASSIF_SEARCH,
                                path_out=None, params=None, pca_coef=0.98,
                                feature_names=None, label_names=None):
    """ create classifier and train it once or find best parameters.
    whether tha path out is given export it for later use

    :param str clf_name:
    :param features: np.array<nb_samples, nb_features>
    :param [int] labels:
    :param cross_val:
    :param int nb_search_iter:
    :param str path_out:
    :param dict params: {str: ...}
    :param [str] feature_names:
    :param [str] label_names:
    :return: (obj, str): classif, path

    >>> np.random.seed(0)
    >>> lbs = np.random.randint(0, 3, 150)
    >>> fts = np.random.random((150, 5)) + np.tile(lbs, (5, 1)).T
    >>> clf, p_clf = create_classif_train_export(DEFAULT_CLASSIF_NAME, fts, lbs)
    >>> clf  # doctest: +ELLIPSIS
    Pipeline(...)
    >>> clf, p_clf = create_classif_train_export(DEFAULT_CLASSIF_NAME,
    ...                                          fts, lbs, path_out='.',
    ...                                          nb_search_iter=2)
    Fitting 10 folds for each of 2 candidates, totalling 20 fits
    >>> clf  # doctest: +ELLIPSIS
    Pipeline(...)
    >>> p_clf
    './classifier_RandForest.pkl'
    >>> os.remove(p_clf)
    >>> import glob
    >>> files = glob.glob(os.path.join('.', 'classif_*.txt'))
    >>> sorted(files)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ['./classif_RandForest_search_params_best.txt',
     './classif_RandForest_search_params_scores.txt']
    >>> for p in files: os.remove(p)
    """
    assert len(labels) > 0
    features = np.nan_to_num(features)
    assert features.shape[0] == len(labels)
    assert features.ndim == 2 and features.shape[1] > 0
    logging.debug('training data: %s, labels (%i): %s', repr(features.shape),
                  len(labels), repr(collections.Counter(labels)))
    # gc.collect(), time.sleep(1)
    logging.info('create Classifier: %s', clf_name)
    clf_pipeline = create_clf_pipeline(clf_name, pca_coef)
    logging.debug('pipeline: %s', repr(clf_pipeline.steps))
    if nb_search_iter > 1:
        # find the best params for the classif.
        logging.debug('Performing param search...')
        nb_labels = len(np.unique(labels))
        clf_search = create_classif_search(clf_name, clf_pipeline, nb_labels,
                                           search_type, cross_val, nb_search_iter, nb_jobs)
        clf_search.fit(features, labels)

        logging.info('Best score: %s', repr(clf_search.best_score_))
        clf_pipeline = clf_search.best_estimator_
        best_parameters = clf_pipeline.get_params()

        logging.info('Best parameters set: \n %s', repr(best_parameters))
        if path_out is not None:
            export_results_clf_search(path_out, clf_name, clf_search)
    else:
        # while there is no search, just train the best one
        clf_pipeline.fit(features, labels)

    if path_out is not None:
        path_classif = save_classifier(path_out, clf_pipeline, clf_name,
                                       params, feature_names, label_names)
    else:
        path_classif = path_out

    return clf_pipeline, path_classif


# # fixme, fix failing cros-val for multilabel
# def eval_classif_cross_val_scores(clf_name, classif, features, labels,
#                                   cross_val=10, path_out=None,
#                                   scorings=['f1_macro', 'accuracy', 'precision', 'recall']):
#     """ compute statistic on cross-validation schema
#
#     http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
#
#     :param str clf_name:
#     :param classif:
#     :param features:
#     :param [int] labels:
#     :param object cross_val:
#     :param str path_out:
#     :param [str] scorings:
#     :return:
#
#     >>> labels = np.array([0] * 150 + [1] * 100 + [2] * 50)
#     >>> data = np.tile(labels, (6, 1)).T.astype(float)
#     >>> data += 0.5 - np.random.random(data.shape)
#     >>> data.shape
#     (300, 6)
#     >>> from sklearn.cross_validation import StratifiedKFold
#     >>> cv = StratifiedKFold(labels, n_folds=5, random_state=0)
#     >>> classif = create_classifiers()[DEFAULT_CLASSIF_NAME]
#     >>> eval_classif_cross_val_scores(DEFAULT_CLASSIF_NAME, classif,
#     ...                               data, labels, cv)
#        f1_macro  accuracy  precision  recall
#     0         1         1          1       1
#     1         1         1          1       1
#     2         1         1          1       1
#     3         1         1          1       1
#     4         1         1          1       1
#     >>> labels[labels == 1] = 2
#     >>> cv = StratifiedKFold(labels, n_folds=3, random_state=0)
#     >>> eval_classif_cross_val_scores(DEFAULT_CLASSIF_NAME, classif,
#     ...                               data, labels, cv)
#        f1_macro  accuracy
#     0       1.0       1.0
#     1       1.0       1.0
#     2       1.0       1.0
#     """
#     df_scoring = pd.DataFrame()
#     for scoring in scorings:
#         try:
#             # ValueError: pos_label=1 is not a valid label: array([0, 2])
#             scores = model_selection.cross_val_score(classif, features, labels,
#                                                      cv=cross_val, scoring=scoring)
#             logging.info('Cross-Val score (%s = %f):\n %s',
#                          scoring, np.mean(scores), repr(scores))
#             df_scoring[scoring] = scores
#         except:
#             logging.error(traceback.format_exc())
#     df_stat = df_scoring.describe()
#
#     if path_out is not None:
#         assert os.path.exists(path_out), 'missing "%s"' % path_out
#         name_csv = NAME_CSV_CLASSIF_CV_SCORES.format(clf_name, 'all-folds')
#         path_csv = os.path.join(path_out, name_csv)
#         df_scoring.to_csv(path_csv)
#
#         name_csv = NAME_CSV_CLASSIF_CV_SCORES.format(clf_name, 'statistic')
#         path_csv = os.path.join(path_out, name_csv)
#         df_stat.to_csv(path_csv)
#     logging.info('cross_val scores: \n %s', repr(df_stat))
#     return df_scoring


def eval_classif_cross_val_roc(clf_name, classif, features, labels,
                               cross_val, path_out=None, nb_thr=100):
    """ compute mean ROC curve on cross-validation schema

    http://scikit-learn.org/0.15/auto_examples/plot_roc_crossval.html

    :param ste clf_name:
    :param object classif:
    :param features:
    :param [int] labels:
    :param object cross_val:
    :param str path_out:
    :return:

    >>> np.random.seed(0)
    >>> labels = np.array([0] * 150 + [1] * 100 + [3] * 50)
    >>> data = np.tile(labels, (6, 1)).T.astype(float)
    >>> data += np.random.random(data.shape)
    >>> data.shape
    (300, 6)
    >>> from sklearn.cross_validation import StratifiedKFold
    >>> cv = StratifiedKFold(labels, n_folds=5, random_state=0)
    >>> classif = create_classifiers()[DEFAULT_CLASSIF_NAME]
    >>> fp_tp, auc = eval_classif_cross_val_roc(DEFAULT_CLASSIF_NAME, classif,
    ...                                         data, labels, cv, nb_thr=10)
    >>> fp_tp
             FP   TP
    0  0.000000  0.0
    1  0.111111  1.0
    2  0.222222  1.0
    3  0.333333  1.0
    4  0.444444  1.0
    5  0.555556  1.0
    6  0.666667  1.0
    7  0.777778  1.0
    8  0.888889  1.0
    9  1.000000  1.0
    >>> auc
    0.94444444444444442
    >>> labels[-50:] -= 1
    >>> data[-50:, :] -= 1
    >>> fp_tp, auc = eval_classif_cross_val_roc(DEFAULT_CLASSIF_NAME, classif,
    ...                                         data, labels, cv, nb_thr=5)
    >>> fp_tp
         FP   TP
    0  0.00  0.0
    1  0.25  1.0
    2  0.50  1.0
    3  0.75  1.0
    4  1.00  1.0
    >>> auc
    0.875
    """
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, nb_thr)
    labels_bin = np.zeros((len(labels), np.max(labels) + 1))
    unique_labels = np.unique(labels)
    assert all(unique_labels >= 0), \
        'some labels are negative: %s' % repr(unique_labels)
    for lb in unique_labels:
        labels_bin[:, lb] = (labels == lb)

    count = 0
    for i, (train, test) in enumerate(cross_val):
        features_train = np.copy(features[train], order='C')
        labels_train = np.copy(labels[train], order='C')
        features_test = np.copy(features[test], order='C')
        classif.fit(features_train, labels_train)
        proba = classif.predict_proba(features_test)
        # Compute ROC curve and area the curve
        for i, j in enumerate(unique_labels):
            fpr, tpr, thresholds = metrics.roc_curve(labels_bin[test, j],
                                                     proba[:, i])
            fpr = [0] + fpr.tolist()
            tpr = [0] + tpr.tolist()
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            count += 1.
        # roc_auc = metrics.auc(fpr, tpr)

    mean_tpr /= count
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    df_roc = pd.DataFrame(np.array([mean_fpr, mean_tpr]).T,
                          columns=['FP', 'TP'])

    auc = metrics.auc(mean_fpr, mean_tpr)

    if path_out is not None:
        assert os.path.exists(path_out), 'missing "%s"' % path_out
        name_csv = NAME_CSV_CLASSIF_CV_ROC.format(clf_name, 'mean')
        path_csv = os.path.join(path_out, name_csv)
        df_roc.to_csv(path_csv)

        name_txt = NAME_TXT_CLASSIF_CV_AUC.format(clf_name, 'mean')
        with open(os.path.join(path_out, name_txt), 'w') as fp:
            fp.write(str(auc))
    logging.info('cross_val ROC: \n %s', repr(df_roc))
    return df_roc, auc


def search_params_cut_down_max_nb_iter(clf_parameters, nb_iter):
    """ create parameters list and count number of possible combination
    in case they are they are limited

    :param clf_parameters: {str: ...}
    :param nb_iter: int, nb of random tryes
    :return: int
    """
    param_list = grid_search.ParameterSampler(clf_parameters, n_iter=nb_iter)
    param_grid = grid_search.ParameterGrid(param_list.param_distributions)
    try:  # this works only in case the set of params is finite, otherwise crash
        if len(param_grid) < nb_iter:
            nb_iter = len(param_grid.param_grid)
            logging.debug('nb iter: -> %i', nb_iter)
    except:
        logging.debug('something went wrong in cutting down nb iter')
    return nb_iter


def create_classif_search(name_clf, clf_pipeline, nb_labels, search_type='random',
                          cross_val=10, nb_iter=NB_CLASSIF_SEARCH_ITER,
                          nb_jobs=NB_JOBS_CLASSIF_SEARCH):
    """ create sklearn search depending on spec. random or grid

    :param nb_iter: int, for random number of tries
    :param name_clf: str, name of classif.
    :param clf_pipeline: object
    :param cross_val: obj specific CV for fix train-test
    :param nb_jobs: int, nb jobs running in parallel
    :return:
    """
    scoring = 'weighted' if nb_labels > 2 else 'binary'
    f1_scoring = metrics.make_scorer(metrics.f1_score, average=scoring)

    if search_type == 'grid':
        clf_parameters = create_clf_param_search_grid(name_clf)
        logging.info('init Grid search...')
        clf_search = grid_search.GridSearchCV(clf_pipeline, clf_parameters,
                              scoring=f1_scoring, cv=cross_val, n_jobs=nb_jobs,
                              verbose=1, refit=True)  #
    else:
        clf_parameters = create_clf_param_search_dist(name_clf)
        nb_iter = search_params_cut_down_max_nb_iter(clf_parameters, nb_iter)
        logging.info('init Randomized search...')
        clf_search = grid_search.RandomizedSearchCV(clf_pipeline, clf_parameters,
                            scoring=f1_scoring, cv=cross_val, n_jobs=nb_jobs,
                            n_iter=nb_iter, verbose=1, refit=True)
    return clf_search


def shuffle_features_labels(features, labels):
    """ take the set of features and labels and shuffle them together
    while keeping link between feature and its label

    :param features: np.array<nb_samples, nb_features>
    :param labels: [int]
    :return: np.array<nb_samples, nb_features>, np.array,<nb_samples>

    >>> np.random.seed(0)
    >>> fts = np.random.random((5, 2))
    >>> lbs = np.random.randint(0, 2, 5)
    >>> fts_new, lbs_new = shuffle_features_labels(fts, lbs)
    >>> np.array_equal(fts, fts_new)
    False
    >>> np.array_equal(lbs, lbs_new)
    False
    """
    assert len(features) == len(labels)
    idx = list(range(len(labels)))
    logging.debug('shuffle indexes - %i', len(labels))
    np.random.shuffle(idx)
    features = features[idx, :]
    labels = np.asarray(labels)[idx]
    return features, labels


def convert_dict_label_features_2_vectors(dict_features):
    """ convert dictionary of features where key is the labels
    to vector of all features and related labels

    :param dict_features: {int: [[float] * nb_features] * nb_samples}
    :return: np.array<nb_samples, nb_features>, [int]
    """
    features, labels = [], []
    for k in dict_features:
        features += dict_features[k].tolist()
        labels += [k] * len(dict_features[k])
    return np.array(features), labels


def compose_dict_label_features(features, labels):
    """ convert vector of features and related labels
    to a dictionary of features where key is the lables

    :param features: np.array<nb_samples, nb_features>
    :param labels: [int]
    :return: {int: np.array<nb, nb_features>}
    """
    dict_features = dict()
    features = np.array(features)
    for lb in np.unique(labels):
        dict_features[lb] = features[labels == lb, :]
    return dict_features


def down_sample_dict_features_random(dict_features, nb_samples):
    """ browse all label features and take random subset of features to have
    given nb_samples per class

    :param {} dict_features: {int: [[float] * nb_features] * nb}
    :param int nb_samples:
    :return {}: {int: [[float] * nb_features] * nb_samples}

    >>> np.random.seed(0)
    >>> d_fts = {'a': np.random.random((100, 3))}
    >>> d_fts = down_sample_dict_features_random(d_fts, 5)
    >>> d_fts['a'].shape
    (5, 3)
    """
    dict_features_new = dict()
    for label in dict_features:
        features = dict_features[label]
        if len(features) <= nb_samples:
            dict_features_new[label] = features.copy()
            continue
        idx = list(range(len(features)))
        random.shuffle(idx)
        idx_select = idx[:nb_samples]
        dict_features_new[label] = np.array(features)[idx_select, :]
    return dict_features_new


def down_sample_dict_features_kmean(dict_features, nb_samples):
    """ cluser with kmeans the features with nb cluster == given nb_samples
    and the retirn features which are closer to each cluster center

    :param {} dict_features: {int: [[float] * nb_features] * nb}
    :param int nb_samples:
    :return {}: {int: [[float] * nb_features] * nb_samples}

    >>> np.random.seed(0)
    >>> d_fts = {'a': np.random.random((100, 3))}
    >>> d_fts = down_sample_dict_features_kmean(d_fts, 5)
    >>> d_fts['a'].shape
    (5, 3)
    """
    dict_features_new = dict()
    for label in dict_features:
        features = dict_features[label]
        if len(features) <= nb_samples:
            dict_features_new[label] = features.copy()
            continue
        kmeans = cluster.KMeans(n_clusters=nb_samples, init='random', n_init=3,
                                max_iter=5, n_jobs=-1)
        dist = kmeans.fit_transform(features)
        find_min = np.argmin(dist, axis=0)
        dict_features_new[label] = features[find_min, :]
    return dict_features_new


# def unique_rows(matrix):
#     matrix = np.ascontiguousarray(matrix)
#     unique_matrix = np.unique(matrix.view([('', matrix.dtype)] * matrix.shape[1]))
#     unique_shape = (unique_matrix.shape[0], matrix.shape[1])
#     unique_matrix = unique_matrix.view(matrix.dtype).reshape(unique_shape)
#     return unique_matrix


def unique_rows(data):
    """ with matrix detect unique row and return only them

    :param data: np.array
    :return: np.array
    """
    # preventing: ValueError: new type not compatible with array.
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.view.html
    data = data.copy()
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


def down_sample_dict_features_unique(dict_features):
    """ browse all label features and take unique features

    :param {} dict_features: {int: [[float] * nb_features] * nb_samples}
    :return {}: {int: [[float] * nb_features] * nb}

    >>> np.random.seed(0)
    >>> d_fts = {'a': np.random.random((100, 3))}
    >>> d_fts = down_sample_dict_features_unique(d_fts)
    >>> d_fts['a'].shape
    (100, 3)
    """
    dict_features_new = dict()
    for label in dict_features:
        features = dict_features[label]
        unique_fts = np.array(unique_rows(features))
        assert features.ndim == unique_fts.ndim
        assert features.shape[1] == unique_fts.shape[1]
        dict_features_new[label] = unique_fts
    return dict_features_new


def balance_dataset_by_(features, labels, type='random', min_samples=None):
    """ balance number of training examples per class by several method

    :param features: [[float]]
    :param labels: [int]
    :param type: str
    :param min_samples: int or Nnone, if Noner take the smallest class
    :return:

    >>> np.random.seed(0)
    >>> fts, lbs = balance_dataset_by_(np.random.random((25, 3)),
    ...                                np.random.randint(0, 2, 25))
    >>> fts.shape
    (24, 3)
    >>> lbs
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    """
    logging.debug('balance dataset using "%s"', type)
    hist_labels = collections.Counter(labels)
    if min_samples is None:
        min_samples = min(hist_labels.values())
    dict_features = compose_dict_label_features(features, labels)

    if type == 'random':
        dict_features = down_sample_dict_features_random(dict_features, min_samples)
    elif type == 'kmeans':
        dict_features = down_sample_dict_features_kmean(dict_features, min_samples)
    elif type == 'unique':
        dict_features = down_sample_dict_features_unique(dict_features)
    else:
        logging.warning('not defined balacing method "%s"', type)

    features, labels = convert_dict_label_features_2_vectors(dict_features)
    # features, labels = shuffle_features_labels(features, labels)
    return features, labels


def convert_set_features_labels_2_dataset(imgs_features, imgs_labels,
                                          drop_labels=None, balance=None):
    """ with dictionary for each image we concentrate all features over images
    and labels into simple form

    :param imgs_features:
    :param imgs_labels:
    :param balance: bool, wether balance number of sampler per class
    :return:

    >>> np.random.seed(0)
    >>> d_fts = {'a': np.random.random((25, 3)),
    ...          'b': np.random.random((30, 3)), }
    >>> d_lbs = {'a': np.random.randint(0, 2, 25),
    ...          'b': np.random.randint(0, 2, 30)}
    >>> fts, lbs, sizes = convert_set_features_labels_2_dataset(d_fts, d_lbs)
    >>> fts.shape
    (55, 3)
    >>> lbs.shape
    (55,)
    >>> sizes
    [25, 30]
    """
    logging.debug('convert set of features and labels to single one')
    assert all(k in imgs_labels.keys() for k in imgs_features.keys())
    features_all, labels_all, sizes = list(), list(), list()
    for name in sorted(imgs_features.keys()):
        features = imgs_features[name]
        labels = imgs_labels[name]

        if drop_labels is not None:
            for lb in drop_labels:
                features = features[labels != lb]
                labels = labels[labels != lb]

        if balance is not None:
            # balance dataset to have comparable nb of samples
            features, labels = balance_dataset_by_(features, labels,
                                                   type=balance)
        features_all += features.tolist()
        labels_all += np.asarray(labels).tolist()
        sizes.append(len(labels))

    return np.array(features_all), np.array(labels_all, dtype=int), sizes


class HoldOut:
    """
    Hold-out cross-validator generator. In the hold-out, the
    data is split only once into a train set and a test set.
    Unlike in other cross-validation schemes, the hold-out
    consists of only one iteration.

    Parameters
    ----------
    n : total number of samples
    hold_idx : int index where the test starts
    random_state :  Seed for the random number generator.

    Example
    -------
    >>> ho = HoldOut(10, 7)
    >>> len(ho)
    1
    >>> list(ho)
    [([0, 1, 2, 3, 4, 5, 6], [7, 8, 9])]
    """
    def __init__(self, nb, hold_idx, random_state=0):
        """

        :param int nb: total number of samples
        :param int hold_idx: index where the test starts
        :param obj random_state: Seed for the random number generator.
        """
        self.total = nb
        self.hold_idx = hold_idx
        self.random_state = random_state
        assert self.total > self.hold_idx

    def __iter__(self):
        """ iterate the folds

        :return ([int], [int]):
        """
        ind_train = list(range(self.hold_idx))
        ind_test = list(range(self.hold_idx, self.total))
        yield ind_train, ind_test

    def __len__(self):
        """ number of folds

        :return int:
        """
        return 1


class CrossValidatePOut:
    """
    Hold-out cross-validator generator. In the hold-out, the
    data is split only once into a train set and a test set.
    Unlike in other cross-validation schemes, the hold-out
    consists of only one iteration.

    Parameters
    ----------

    Example 1
    ---------
    >>> cv = CrossValidatePOut(6, 3, rand_seed=False)
    >>> cv.indexes
    [0, 1, 2, 3, 4, 5]
    >>> len(cv)
    2
    >>> list(cv)  # doctest: +NORMALIZE_WHITESPACE
    [([3, 4, 5], [0, 1, 2]), \
     ([0, 1, 2], [3, 4, 5])]

    Example 2
    ---------
    >>> cv = CrossValidatePOut(7, 3, rand_seed=0)
    >>> list(cv)  # doctest: +NORMALIZE_WHITESPACE
    [([3, 0, 5, 4], [6, 2, 1]), \
     ([6, 2, 1, 4], [3, 0, 5]), \
     ([6, 2, 1, 3, 0, 5], [4])]


    >>> len(list(cv))
    3
    >>> cv.indexes
    [6, 2, 1, 3, 0, 5, 4]
    """

    def __init__(self, nb_samples, nb_hold_out, rand_seed=None):
        """

        :param [int] set_sizes:
        :param int nb_hold_out:
        :param obj random_order:
        """
        assert nb_samples > nb_hold_out, \
            'nb of out has to be smaller then total size'
        self.nb_samples = nb_samples
        self.nb_hold_out = nb_hold_out

        self.indexes = list(range(self.nb_samples))

        if not rand_seed is False:
            np.random.seed(rand_seed)
            np.random.shuffle(self.indexes)
        logging.debug('sets ordering: %s', repr(self.indexes))

        self.iter = 0

    def __iter__(self):
        """ iterate the folds

        :return ([int], [int]):
        """
        for iter in range(0, self.nb_samples, self.nb_hold_out):
            inds_test = self.indexes[iter:iter + self.nb_hold_out]
            inds_train = [i for i in self.indexes if i not in inds_test]
            yield inds_train, inds_test

    def __len__(self):
        """ number of folds

        :return int:
        """
        return int(np.ceil(self.nb_samples / float(self.nb_hold_out)))


class CrossValidatePSetsOut:
    """
    Hold-out cross-validator generator. In the hold-out, the
    data is split only once into a train set and a test set.
    Unlike in other cross-validation schemes, the hold-out
    consists of only one iteration.

    Parameters
    ----------

    Example 1
    ---------
    >>> cv = CrossValidatePSetsOut([2, 3, 2, 3], 2, rand_seed=False)
    >>> cv.set_indexes
    [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]
    >>> len(cv)
    2
    >>> list(cv)  # doctest: +NORMALIZE_WHITESPACE
    [([5, 6, 7, 8, 9], [0, 1, 2, 3, 4]), \
     ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])]

    Example 2
    ---------
    >>> cv = CrossValidatePSetsOut([2, 2, 1, 2, 1], 2, rand_seed=0)
    >>> cv.set_indexes
    [[0, 1], [2, 3], [4], [5, 6], [7]]
    >>> list(cv)  # doctest: +NORMALIZE_WHITESPACE
    [([2, 3, 5, 6, 7], [4, 0, 1]), \
     ([4, 0, 1, 7], [2, 3, 5, 6]), \
     ([4, 0, 1, 2, 3, 5, 6], [7])]
    >>> len(cv)
    3
    >>> cv.sets_order
    [2, 0, 1, 3, 4]
    """

    def __init__(self, set_sizes, nb_hold_out, rand_seed=None):
        """

        :param [int] set_sizes:
        :param int nb_hold_out:
        :param obj random_order:
        """
        assert len(set_sizes) > nb_hold_out, \
            'nb of out has to be smaller then total size'
        self.set_sizes = list(set_sizes)
        self.total = np.sum(self.set_sizes)
        self.nb_hold_out = nb_hold_out

        self.set_indexes = []
        for i, size in enumerate(self.set_sizes):
            start = int(np.sum(self.set_sizes[:i]))
            inds = range(start, start + size)
            self.set_indexes.append(list(inds))

        assert np.sum(len(i) for i in self.set_indexes) == self.total

        self.sets_order = list(range(len(self.set_sizes)))

        if not rand_seed is False:
            np.random.seed(rand_seed)
            np.random.shuffle(self.sets_order)
        logging.debug('sets ordering: %s', repr(self.sets_order))

        self.iter = 0

    def __iter__(self):
        """ iterate the folds

        :return ([int], [int]):
        """
        for iter in range(0, len(self.set_sizes), self.nb_hold_out):
            test = self.sets_order[iter:iter + self.nb_hold_out]
            inds_train, inds_test = [], []
            for i in self.sets_order:
                if i in test:
                    inds_test += self.set_indexes[i]
                else:
                    inds_train += self.set_indexes[i]
            yield inds_train, inds_test

    def __len__(self):
        """ number of folds

        :return int:
        """
        nb = len(self.set_sizes) / float(self.nb_hold_out)
        return int(np.ceil(nb))


# DEPRECATED
# ==========

# def check_exist_labels_dataset(dataset, lut):
#     u_lbs = np.unique(lut.values())
#     logger.debug('labels in dataset are {} and LUT contains {}'.format(dataset.keys(), u_lbs))
#     for l in u_lbs:
#         if not l in dataset:
#             logger.debug('add missing label {} in dataset of {}'.format(l, dataset.keys()))
#             dataset[l] = []
#     return dataset


# def extend_dataset(dataset, fts, lut):
#     logger.info('adding new features to training dataset.')
#     dataset = check_exist_labels_dataset(dataset, lut)
#     for k in lut:
#         dataset[lut[k]].append(fts[k])
#     logger.debug(str_dataset_stat(dataset, 'EXTENDED'))
#     return dataset


# def cluster_dataset_label_samples(data, nb_clts, method):
#     data = np.array(data)
#     if method == 'AggCl':
#         clt = cluster.AgglomerativeClustering(nb_clts, linkage='ward',
#                                               memory='tmpMemoryDump')
#         clt.fit(data)
#         words = [np.mean(data[clt.labels_ == l], axis=0) for l in np.unique(clt.labels_)]
#     if method == 'AffPr':
#         clt = cluster.AffinityPropagation(convergence_iter=7)
#         clt.fit(data)
#         words = clt.cluster_centers_
#     elif method == 'Birch':
#         clt = cluster.Birch(n_clusters=nb_clts)
#         clt.fit(data)
#         words = clt.subcluster_centers_
#     elif method == 'kMeans':
#         # http://www.spectralpython.net/class_func_ref.html?highlight=kmeans#spectral.kmeans
#         # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.manhattan_distances.html
#         clt = cluster.KMeans(init='random', n_clusters=nb_clts, n_init=3,
#                              max_iter=35, n_jobs=5)
#         # clt = cluster.KMeans(init='k-means++', n_clusters=nb_clts, n_init=7, n_jobs=-1)
#         clt.fit(data)
#         words = clt.cluster_centers_
#     else: # random
#         words = data[np.random.choice(data.shape[0], nb_clts)]
#     return words


# def segm_features_classif_general(dataset_dict, ft, clf, prob=False):
#     X, y = convert_standard_dataset(dataset_dict)
#     logger.info('training dataset dims: X -> #{}, y -> #{}'.format(X.shape, y.shape))
#     assert len(X) == len(y)
#     clf.fit(X, y)
#     lbs = clf.predict(ft)
#     if prob:
#         probs = clf.predict_proba(ft)
#     else:
#         probs=None
#     logger.debug('results labeling dim: {} and vals: {}'.format(lbs.shape, lbs))
#     return lbs, probs


# def get_labeling_probability(ft, datasetDict, lbs):
#     probs = np.zeros((len(ft),len(datasetDict)))
#     probs[range(len(lbs)), lbs] = 1.
#     return probs

"""
Supporting file to create and set parameters for scikit-learn classifiers
and some prepossessing functions that support classification

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import pickle
import logging
import random
import collections
import itertools
from functools import partial

import numpy as np
import pandas as pd
# import itertools
# import matplotlib.pyplot as plt
from scipy import interp
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_random
from sklearn.base import clone
from sklearn import preprocessing, feature_selection, decomposition
from sklearn import cluster, metrics
from sklearn import ensemble, neighbors, svm, tree
from sklearn import pipeline, linear_model, neural_network
from sklearn import model_selection
try:  # due to some chnages in between versions
    from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
except Exception:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from imsegm.labeling import relabel_max_overlap_unique
from imsegm.utilities.experiments import WrapExecuteSequence, nb_workers

# NAME_FILE_RESULTS = 'results.csv'
#: name template forexporting trained classifier (adding classifier name and version)
TEMPLATE_NAME_CLF = 'classifier_{}.pkl'
#: default (recommended) classifier for supervised segmentation
DEFAULT_CLASSIF_NAME = 'RandForest'
#: default (recommended) clustering for unsupervised segmentation
DEFAULT_CLUSTERING = 'kMeans'
# DEFAULT_MIN_NB_SPL = 25
# nb_workers_CLASSIF_SEARCH = 5
# NB_CLASSIF_SEARCH_ITER = 250
#: file name of exported evaluation on feature quality
NAME_CSV_FEATURES_SELECT = 'feature_selection.csv'
#: exporting partial results about trained classifier
NAME_CSV_CLASSIF_CV_SCORES = 'classif_{}_cross-val_scores-{}.csv'
#: exporting partial results about trained classifier - Receiver Operating Characteristics
NAME_CSV_CLASSIF_CV_ROC = 'classif_{}_cross-val_ROC-{}.csv'
#: exporting partial results about trained classifier - Area Under Curve
NAME_TXT_CLASSIF_CV_AUC = 'classif_{}_cross-val_AUC-{}.txt'
#: default types of computed metrics
METRIC_AVERAGES = ('macro', 'weighted')
#: default computed metrics
METRIC_SCORING = ('f1_macro', 'accuracy', 'precision_macro', 'recall_macro')
#: rounding unique features, in case to detail precision
ROUND_UNIQUE_FTS_DIGITS = 3
#: default number of workers
NB_WORKERS_SERACH = nb_workers(0.5)

#: mapping of metrics names to used functions
DICT_SCORING = {
    'f1': metrics.f1_score,
    'accuracy': metrics.accuracy_score,
    'precision': metrics.precision_score,
    'recall': metrics.recall_score,
}


def create_classifiers(nb_workers=-1):
    """ create all classifiers with default parameters

    :param int nb_workers: number of parallel if possible
    :return dict: {str: clf}

    >>> classifs = create_classifiers()
    >>> classifs  # doctest: +ELLIPSIS
    {...}
    >>> sum([isinstance(create_clf_param_search_grid(k), dict)
    ...      for k in classifs.keys()])
    7
    >>> sum([isinstance(create_clf_param_search_distrib(k), dict)
    ...      for k in classifs.keys()])
    7
    """
    clfs = {
        'RandForest': ensemble.RandomForestClassifier(n_estimators=20,
                                                      # oob_score=True,
                                                      min_samples_leaf=2,
                                                      min_samples_split=3,
                                                      n_jobs=nb_workers),
        'GradBoost': ensemble.GradientBoostingClassifier(subsample=0.25,
                                                         warm_start=False,
                                                         max_depth=6,
                                                         min_samples_leaf=6,
                                                         n_estimators=200,
                                                         min_samples_split=7),
        'LogistRegr': linear_model.LogisticRegression(solver='sag',
                                                      n_jobs=nb_workers),
        'KNN': neighbors.KNeighborsClassifier(n_jobs=nb_workers),
        'SVM': svm.SVC(kernel='rbf', probability=True,
                       tol=2e-3, max_iter=5000),
        'DecTree': tree.DecisionTreeClassifier(),
        # 'RBM': create_pipeline_neuron_net(),
        'AdaBoost': ensemble.AdaBoostClassifier(n_estimators=5),
        # 'NuSVM-rbf': svm.NuSVC(kernel='rbf', probability=True),
    }
    return clfs


def create_clf_pipeline(name_classif=DEFAULT_CLASSIF_NAME, pca_coef=0.95):
    """ create complete pipeline with all required steps

    :param int|float|None pca_coef: sklearn PCA
    :param str name_classif: key name of classif.
    :return: object

    >>> create_clf_pipeline()  # doctest: +ELLIPSIS
    Pipeline(...)
    """
    # create the pipeline
    components = [('scaler', preprocessing.StandardScaler())]
    if pca_coef is not None:
        components += [('reduce_dim', decomposition.PCA(pca_coef))]
    components += [('classif', create_classifiers()[name_classif])]
    clf_pipeline = pipeline.Pipeline(components)
    return clf_pipeline


def create_clf_param_search_grid(name_classif=DEFAULT_CLASSIF_NAME):
    """ create parameter grid for search

    :param str name_classif: key name of selected classifier
    :return: dict

    >>> create_clf_param_search_grid('RandForest') # doctest: +ELLIPSIS
    {'classif__...': ...}
    >>> dict_classif = create_classifiers()
    >>> all(len(create_clf_param_search_grid(k)) > 0 for k in dict_classif)
    True
    >>> create_clf_param_search_grid('none')
    {}
    """
    def _log_space(b, e, n):
        return np.unique(np.logspace(b, e, n).astype(int)).tolist()

    clf_params = {
        'RandForest': {
            'classif__n_estimators': _log_space(0, 2, 40),
            'classif__min_samples_split': [2, 3, 5, 7, 9],
            'classif__min_samples_leaf': [1, 2, 4, 6, 9],
            'classif__criterion': ('gini', 'entropy'),
        },
        'KNN': {
            'classif__n_neighbors': _log_space(0, 2, 20),
            'classif__algorithm': ('ball_tree', 'kd_tree'),  # , 'brute'
            'classif__weights': ('uniform', 'distance'),
            'classif__leaf_size': _log_space(0, 1.5, 10),
        },
        'SVM': {
            'classif__C': np.linspace(0.2, 1., 8).tolist(),
            'classif__kernel': ('poly', 'rbf', 'sigmoid'),
            'classif__degree': [1, 2, 4, 6, 9],
        },
        'DecTree': {
            'classif__criterion': ('gini', 'entropy'),
            'classif__min_samples_split': [2, 3, 5, 7, 9],
            'classif__min_samples_leaf': range(1, 7, 2),
        },
        'GradBoost': {
            # 'clf__loss': ('deviance', 'exponential'), # only for 2 cls
            'classif__n_estimators': _log_space(0, 2, 25),
            'classif__max_depth': range(1, 7, 2),
            'classif__min_samples_split': [2, 3, 5, 7, 9],
            'classif__min_samples_leaf': range(1, 7, 2),
        },
        'LogistRegr': {
            'classif__C': np.linspace(0., 1., 5).tolist(),
            # 'classif__penalty': ('l1', 'l2'),
            # 'classif__dual': (False, True),
            'classif__solver': ('lbfgs', 'sag'),
            # 'classif__loss': ('deviance', 'exponential'), # only for 2 cls
        },
        'AdaBoost': {
            'classif__n_estimators': _log_space(0, 2, 20),
        }
    }
    if name_classif not in clf_params.keys():
        clf_params[name_classif] = {}
        logging.warning('not defined classifier name "%s"', name_classif)
    return clf_params[name_classif]


def create_clf_param_search_distrib(name_classif=DEFAULT_CLASSIF_NAME):
    """ create parameter distribution for random search

    :param str name_classif: key name of classifier
    :return: dict

    >>> create_clf_param_search_distrib()  # doctest: +ELLIPSIS
    {...}
    >>> dict_classif = create_classifiers()
    >>> all(len(create_clf_param_search_distrib(k)) > 0 for k in dict_classif)
    True
    >>> create_clf_param_search_distrib('none')
    {}
    """
    clf_params = {
        'RandForest': {
            'classif__n_estimators': sp_randint(2, 25),
            'classif__min_samples_split': sp_randint(2, 9),
            'classif__min_samples_leaf': sp_randint(1, 7),
        },
        'KNN': {
            'classif__n_neighbors': sp_randint(5, 25),
            'classif__algorithm': ('ball_tree', 'kd_tree'),  # , 'brute'
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
            # 'clf__loss': ('deviance', 'exponential'),  # only for 2 cls
            'classif__n_estimators': sp_randint(10, 200),
            'classif__max_depth': sp_randint(1, 7),
            'classif__min_samples_split': sp_randint(2, 9),
            'classif__min_samples_leaf': sp_randint(1, 7),
        },
        'LogistRegr': {
            'classif__C': sp_random(0., 1.),
            # 'classif__penalty': ('l1', 'l2'),
            # 'classif__dual': (False, True),
            'classif__solver': ('newton-cg', 'lbfgs', 'sag'),
            # 'classif__loss': ('deviance', 'exponential'),  # only for 2 cls
        },
        'AdaBoost': {
            'classif__n_estimators': sp_randint(2, 100),
        }
    }
    # if this classifier is not set use no params
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

    :param list(int) y_true:
    :param list(int) y_pred:
    :param str|list(str) metric_averages:
    :return dict(float):

    >>> np.random.seed(0)
    >>> y_true = np.random.randint(0, 3, 25) * 2
    >>> y_pred = np.random.randint(0, 2, 25) * 2
    >>> d = compute_classif_metrics(y_true, y_true)
    >>> d['accuracy']
    1.0
    >>> d['confusion']
    [[10, 0, 0], [0, 10, 0], [0, 0, 5]]
    >>> d = compute_classif_metrics(y_true, y_pred)
    >>> d['accuracy']  # doctest: +ELLIPSIS
    0.32...
    >>> d['confusion']
    [[3, 7, 0], [5, 5, 0], [1, 4, 0]]
    >>> d = compute_classif_metrics(y_pred, y_pred)
    >>> d['accuracy']
    1.0
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert y_true.shape == y_pred.shape, \
        'prediction (%i) and annotation (%i) should be equal' \
        % (len(y_true), len(y_pred))
    logging.debug('unique lbs true: %r, predict %r',
                  np.unique(y_true), np.unique(y_pred))

    uq_labels = np.unique(np.hstack((y_true, y_pred)))
    # in case there are just two classes, relabel them as [0, 1], sklearn error:
    #  "ValueError: pos_label=1 is not a valid label: array([  0, 255])"
    if len(uq_labels) <= 2:
        # NOTE, this is temporal just for purposes of computing statistic
        y_true = relabel_sequential(y_true, uq_labels)
        y_pred = relabel_sequential(y_pred, uq_labels)

    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    eval_str = 'EVALUATION: {:<2} PRE: {:.3f} REC: {:.3f} F1: {:.3f} S: {:>6}'
    try:
        p, r, f, s = metrics.precision_recall_fscore_support(y_true, y_pred)
        for l, _ in enumerate(p):
            logging.debug(eval_str.format(l, p[l], r[l], f[l], s[l]))
    except Exception:
        logging.exception('metrics.precision_recall_fscore_support')

    dict_metrics = {
        'ARS': metrics.adjusted_rand_score(y_true, y_pred),
        # 'F1':  metrics.f1_score(y_true, y_pred),
        'accuracy': metrics.accuracy_score(y_true, y_pred),
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
        except Exception:
            logging.exception('metrics.precision_recall_fscore_support')
            res = dict(zip(['{}_{}'.format(n, avg) for n in names], [-1] * 4))
        dict_metrics.update(res)
    return dict_metrics


def compute_classif_stat_segm_annot(annot_segm_name, drop_labels=None,
                                    relabel=False):
    """ compute classification statistic between annotation and segmentation

    :param tuple(ndarray,ndarray,str) annot_segm_name:
    :param list(int) drop_labels: labels to be ignored
    :param bool relabel: whether relabel
    :return:

    >>> np.random.seed(0)
    >>> annot = np.random.randint(0, 2, (5, 10))
    >>> segm = np.random.randint(0, 2, (5, 10))
    >>> d = compute_classif_stat_segm_annot((annot, annot, 'ttt'), relabel=True,
    ...                                     drop_labels=[5])
    >>> d['(FP+FN)/(TP+FN)']  # doctest: +ELLIPSIS
    0.0
    >>> d['(TP+FP)/(TP+FN)']  # doctest: +ELLIPSIS
    1.0
    >>> d = compute_classif_stat_segm_annot((annot, segm, 'ttt'), relabel=True,
    ...                                     drop_labels=[5])
    >>> d['(FP+FN)/(TP+FN)']  # doctest: +ELLIPSIS
    0.846...
    >>> d['(TP+FP)/(TP+FN)']  # doctest: +ELLIPSIS
    1.153...
    >>> d = compute_classif_stat_segm_annot((annot, segm + 1, 'ttt'),
    ...                                     relabel=False, drop_labels=[0])
    >>> d['confusion']
    [[13, 17], [0, 0]]
    """
    annot, segm, name = annot_segm_name
    assert segm.shape == annot.shape, 'dimension do not match for segm: %r - annot: %r' \
                                      % (segm.shape, annot.shape)
    y_true, y_pred = annot.ravel(), segm.ravel()
    # filter particular labels
    if drop_labels is not None:
        mask = np.ones(y_true.shape, dtype=bool)
        for lb in drop_labels:
            mask[y_true == lb] = 0
            mask[y_pred == lb] = 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    # relabel such that the classes maximaly match
    if relabel:
        y_pred = relabel_max_overlap_unique(y_true, y_pred, keep_bg=False)
    dict_stat = compute_classif_metrics(y_true, y_pred,
                                        metric_averages=['macro'])
    # add binary metric
    if len(np.unique(y_pred)) == 2:
        dict_stat['(FP+FN)/(TP+FN)'] = compute_metric_fpfn_tpfn(y_true, y_pred)
        dict_stat['(TP+FP)/(TP+FN)'] = compute_metric_tpfp_tpfn(y_true, y_pred)
    # set the image name
    dict_stat['name'] = name
    return dict_stat


def compute_stat_per_image(segms, annots, names=None, nb_workers=2,
                           drop_labels=None, relabel=False):
    """ compute statistic over multiple segmentations with annotation

    :param [ndarray] segms: segmntations
    :param [ndarray] annots: annotations
    :param list(str) names: list of names
    :param list(int) drop_labels: labels to be ignored
    :param bool relabel: whether relabel
    :param int nb_workers: running jobs in parallel
    :return DF:


    >>> np.random.seed(0)
    >>> img_true = np.random.randint(0, 3, (50, 100))
    >>> img_pred = np.random.randint(0, 2, (50, 100))
    >>> df = compute_stat_per_image([img_true], [img_true], nb_workers=2,
    ...                             relabel=True)
    >>> pd.Series(df.iloc[0]).sort_index()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ARS                                                         1
    accuracy                                                    1
    confusion          [[1672, 0, 0], [0, 1682, 0], [0, 0, 1646]]
    f1_macro                                                    1
    precision_macro                                             1
    recall_macro                                                1
    support_macro                                            None
    Name: 0, dtype: object
    >>> df = compute_stat_per_image([img_true], [img_pred], drop_labels=[-1])
    >>> pd.Series(df.iloc[0]).sort_index()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ARS                                                       0.0...
    accuracy                                                  0.3384
    confusion          [[836, 826, 770], [836, 856, 876], [0, 0, 0]]
    f1_macro                                                0.270077
    precision_macro                                         0.336306
    recall_macro                                            0.225694
    support_macro                                               None
    Name: 0, dtype: object
    """
    assert len(segms) == len(annots), \
        'size of segment. (%i) amd annot. (%i) should be equal' \
        % (len(segms), len(annots))
    if not names:
        names = map(str, range(len(segms)))
    _compute_stat = partial(compute_classif_stat_segm_annot,
                            drop_labels=drop_labels, relabel=relabel)
    iterate = WrapExecuteSequence(_compute_stat, zip(annots, segms, names),
                                  nb_workers=nb_workers,
                                  desc='statistic per image')
    list_stat = list(iterate)
    df_stat = pd.DataFrame(list_stat)
    df_stat.set_index('name', inplace=True)
    return df_stat


def feature_scoring_selection(features, labels, names=None, path_out=''):
    """ find the best features and retrun the indexes
    http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_recovery.html
    http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html

    :param ndarray features: np.array<nb_samples, nb_features>
    :param ndarray labels: np.array<nb_samples, 1>
    :param list(str) names:
    :param str path_out:
    :return tuple(list(int),DF): indices, Dataframe with scoring

    >>> from sklearn.datasets import make_classification
    >>> features, labels = make_classification(n_samples=250, n_features=5,
    ...                                        n_informative=3, n_redundant=0,
    ...                                        n_repeated=0, n_classes=2,
    ...                                        random_state=0, shuffle=False)
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
    >>> path_out = 'test_fts-select'
    >>> os.mkdir(path_out)
    >>> indices, df_scoring = feature_scoring_selection(features.tolist(), labels.tolist(),
    ...                                                 path_out=path_out)
    >>> indices
    array([1, 0, 3, 4, 2])
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    logging.info('Feature selection for %r', names)
    features = np.array(features) if not isinstance(features, np.ndarray) else features
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
    logging.debug('Features: %r and labels: %r', features.shape, labels.shape)
    # Build a forest and compute the feature importances
    forest = ensemble.ExtraTreesClassifier(n_estimators=125, random_state=0)
    forest.fit(features, labels)
    f_test, _ = feature_selection.f_regression(features, labels)
    k_best = feature_selection.SelectKBest(feature_selection.f_classif, k='all')
    k_best.fit(features, labels)
    variances = feature_selection.VarianceThreshold().fit(features, labels)
    imp = {
        'ExtTree': forest.feature_importances_,
        # 'Lasso': np.abs(lars_cv.coef_),
        'k-Best': k_best.scores_,
        'variance': variances.variances_,
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

    :param str path_out: path for exporting trained classofier
    :param classif: sklearn classif.
    :param str clf_name: name of selected classifier
    :param list(str) feature_names: list of string names
    :param dict params: extra parameters
    :param list(str) label_names: list of string names of label_names
    :return str:

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
    assert os.path.isdir(path_out), 'missing folder: %r' % path_out
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
    """ estimate classifier for all data and export it

    :param str path_classif: path to the exported classifier
    :return dict:

    >>> load_classifier('none.abc')
    """
    logging.info('import classifier from "%s"', path_classif)
    if not os.path.exists(path_classif):
        logging.debug('classifier does not exist')
        return None
    with open(path_classif, 'rb') as f:
        dict_clf = pickle.load(f)
    # dict_clf['name'] = classif_name
    logging.debug('load classifier: %r', dict_clf.keys())
    return dict_clf


def export_results_clf_search(path_out, clf_name, clf_search):
    """ do the final testing and save all results

    :param str path_out: path to directory for exporting classifier
    :param str clf_name: name of selected classifier
    :param object clf_search:
    """
    assert os.path.isdir(path_out), 'missing folder: %s' % path_out

    def _fn_path_out(s):
        return os.path.join(path_out, 'classif_%s_%s.txt' % (clf_name, s))

    with open(_fn_path_out('search_params_scores'), 'w') as fp:
        results = 'no results'
        if hasattr(clf_search, 'grid_scores_'):  # for sklearn < 0.18
            results = '\n'.join([repr(gs) for gs in clf_search.grid_scores_])
        elif hasattr(clf_search, 'cv_results_'):
            results = '\n'.join(['"%s": %r' % (k, clf_search.cv_results_[k])
                                 for k in clf_search.cv_results_])
        fp.write(results)

    with open(_fn_path_out('search_params_best'), 'w') as fp:
        params = clf_search.best_params_
        rows = ['{:30s} {}'.format('"{}":'.format(k), params[k]) for k in params]
        fp.write('\n'.join(rows))


def relabel_sequential(labels, uq_labels=None):
    """ relabel sequential vector staring from 0

    :param list(int) labels: all labels
    :param list(int) uq_labels: unique labels
    :return []:

    >>> relabel_sequential([0, 0, 0, 5, 5, 5, 0, 5])
    [0, 0, 0, 1, 1, 1, 0, 1]
    """
    labels = np.asarray(labels)
    if uq_labels is None:
        uq_labels = np.unique(labels)
    lut = np.zeros(np.max(uq_labels) + 1)
    logging.debug('relabeling original %r to %r', uq_labels, range(len(uq_labels)))
    for i, lb in enumerate(uq_labels):
        lut[lb] = i
    labesl_new = lut[labels].astype(labels.dtype).tolist()
    return labesl_new


def create_classif_search_train_export(clf_name, features, labels, cross_val=10,
                                       nb_search_iter=100, search_type='random',
                                       eval_metric='f1', nb_workers=NB_WORKERS_SERACH,
                                       path_out=None, params=None, pca_coef=0.98,
                                       feature_names=None, label_names=None):
    """ create classifier and train it once or find best parameters.
    whether tha path out is given export it for later use

    :param str clf_name: name of selected classifier
    :param ndarray features: features in dimension nb_samples x nb_features
    :param list(int) labels: annotation for samples
    :param int|obj cross_val: Cross validation
    :param str search_type: search type
    :param str eval_metric: evaluation metric
    :param dict params: extra parameters
    :param float pca_coef: sklearn PCA - int/float/None
    :param int nb_search_iter: number of searcher for hyper-parameters
    :param str path_out: path to directory for exporting classifier
    :param int nb_workers: parallel processes
    :param list(str) feature_names: list of extracted features - names
    :param list(str) label_names: list of label names
    :return: (obj, str): classifier, path to the exported classifier

    >>> np.random.seed(0)
    >>> lbs = np.random.randint(0, 3, 150)
    >>> fts = np.random.random((150, 5)) + np.tile(lbs, (5, 1)).T
    >>> _, _ = create_classif_search_train_export('LogistRegr', fts, lbs, nb_search_iter=0)
    >>> clf, p_clf = create_classif_search_train_export('AdaBoost', fts, lbs,
    ...     nb_search_iter=2, path_out='', search_type='grid')  # doctest: +ELLIPSIS
    Fitting ...
    >>> clf  # doctest: +ELLIPSIS
    Pipeline(...)
    >>> clf, p_clf = create_classif_search_train_export('RandForest', fts, lbs,
    ...     nb_search_iter=2, path_out='.', search_type='random')  # doctest: +ELLIPSIS
    Fitting ...
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
    assert list(labels), 'some labels has to be given'
    features = np.nan_to_num(features)
    assert len(features) == len(labels), \
        'features (%i) and labels (%i) should have equal length' \
        % (len(features), len(labels))
    assert features.ndim == 2 and features.shape[1] > 0, \
        'at least one feature is required'
    logging.debug('training data: %r, labels (%i): %r', features.shape,
                  len(labels), collections.Counter(labels))
    # gc.collect(), time.sleep(1)
    logging.info('create Classifier: %s', clf_name)
    clf_pipeline = create_clf_pipeline(clf_name, pca_coef)
    logging.debug('pipeline: %r', clf_pipeline.steps)
    if nb_search_iter > 1 or search_type == 'grid':
        # find the best params for the classif.
        logging.debug('Performing param search...')
        nb_labels = len(np.unique(labels))
        clf_search = create_classif_search(clf_name, clf_pipeline,
                                           nb_labels=nb_labels,
                                           search_type=search_type,
                                           cross_val=cross_val,
                                           eval_metric=eval_metric,
                                           nb_iter=nb_search_iter,
                                           nb_workers=nb_workers)

        # NOTE, this is temporal just for purposes of computing statistic
        clf_search.fit(features, relabel_sequential(labels))

        logging.info('Best score: %r', clf_search.best_score_)
        clf_pipeline = clf_search.best_estimator_
        best_parameters = clf_pipeline.get_params()

        logging.info('Best parameters set: \n %r', best_parameters)
        if path_out is not None and os.path.isdir(path_out):
            export_results_clf_search(path_out, clf_name, clf_search)

    # while there is no search, just train the best one
    clf_pipeline.fit(features, labels)

    if path_out is not None and os.path.isdir(path_out):
        path_classif = save_classifier(path_out, clf_pipeline, clf_name,
                                       params, feature_names, label_names)
    else:
        path_classif = path_out

    return clf_pipeline, path_classif


def eval_classif_cross_val_scores(clf_name, classif, features, labels,
                                  cross_val=10, path_out=None,
                                  scorings=METRIC_SCORING):
    """ compute statistic on cross-validation schema

    http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

    :param str clf_name: name of selected classifier
    :param obj classif: sklearn classifier
    :param ndarray features: features in dimension nb_samples x nb_features
    :param list(int) labels: annotation for samples
    :param object cross_val:
    :param str path_out: path for exporting statistic
    :param list(str) scorings: list of used scorings
    :return DF:

    >>> labels = np.array([0] * 150 + [1] * 100 + [2] * 50)
    >>> data = np.tile(labels, (6, 1)).T.astype(float)
    >>> data += 0.5 - np.random.random(data.shape)
    >>> data.shape
    (300, 6)
    >>> from sklearn.model_selection import StratifiedKFold
    >>> cv = StratifiedKFold(n_splits=5, random_state=0)
    >>> classif = create_classifiers()[DEFAULT_CLASSIF_NAME]
    >>> eval_classif_cross_val_scores(DEFAULT_CLASSIF_NAME, classif,
    ...                               data, labels, cv)
       f1_macro  accuracy  precision_macro  recall_macro
    0       1.0       1.0              1.0           1.0
    1       1.0       1.0              1.0           1.0
    2       1.0       1.0              1.0           1.0
    3       1.0       1.0              1.0           1.0
    4       1.0       1.0              1.0           1.0
    >>> labels[labels == 1] = 2
    >>> cv = StratifiedKFold(n_splits=3, random_state=0)
    >>> eval_classif_cross_val_scores(DEFAULT_CLASSIF_NAME, classif,
    ...                               data, labels, cv, path_out='.')
       f1_macro  accuracy  precision_macro  recall_macro
    0       1.0       1.0              1.0           1.0
    1       1.0       1.0              1.0           1.0
    2       1.0       1.0              1.0           1.0
    >>> import glob
    >>> p_files = glob.glob(NAME_CSV_CLASSIF_CV_SCORES.replace('{}', '*'))
    >>> sorted(p_files)  # doctest: +NORMALIZE_WHITESPACE
    ['classif_RandForest_cross-val_scores-all-folds.csv',
     'classif_RandForest_cross-val_scores-statistic.csv']
    >>> [os.remove(p) for p in p_files]  # doctest: +ELLIPSIS
    [...]
    """
    df_scoring = pd.DataFrame()
    for scoring in scorings:
        try:
            uq_labels = np.unique(labels)
            # ValueError: pos_label=1 is not a valid label: array([0, 2])
            if len(uq_labels) <= 2:
                # NOTE, this is temporal just for purposes of computing stat.
                labels = relabel_sequential(labels, uq_labels)
            scores = model_selection.cross_val_score(classif, features, labels,
                                                     cv=cross_val,
                                                     scoring=scoring)
            logging.info('Cross-Val score (%s = %f):\n %r',
                         scoring, np.mean(scores), scores)
            df_scoring[scoring] = scores
        except Exception:
            logging.exception('model_selection.cross_val_score')

    if path_out is not None:
        assert os.path.exists(path_out), 'missing: "%s"' % path_out
        name_csv = NAME_CSV_CLASSIF_CV_SCORES.format(clf_name, 'all-folds')
        path_csv = os.path.join(path_out, name_csv)
        df_scoring.to_csv(path_csv)

    if len(df_scoring) > 1:
        df_stat = df_scoring.describe()
        logging.info('cross_val scores: \n %r', df_stat)
        if path_out is not None:
            assert os.path.exists(path_out), 'missing: "%s"' % path_out
            name_csv = NAME_CSV_CLASSIF_CV_SCORES.format(clf_name, 'statistic')
            path_csv = os.path.join(path_out, name_csv)
            df_stat.to_csv(path_csv)
    else:
        logging.warning('no statistic collected')
    return df_scoring


def eval_classif_cross_val_roc(clf_name, classif, features, labels,
                               cross_val, path_out=None, nb_steps=100):
    """ compute mean ROC curve on cross-validation schema

    http://scikit-learn.org/0.15/auto_examples/plot_roc_crossval.html

    :param str clf_name: name of selected classifier
    :param obj classif: sklearn classifier
    :param ndarray features: features in dimension nb_samples x nb_features
    :param list(int) labels: annotation for samples
    :param object cross_val:
    :param str path_out: path for exporting statistic
    :param int nb_steps: number of thresholds
    :return:

    >>> np.random.seed(0)
    >>> labels = np.array([0] * 150 + [1] * 100 + [3] * 50)
    >>> data = np.tile(labels, (6, 1)).T.astype(float)
    >>> data += np.random.random(data.shape)
    >>> data.shape
    (300, 6)
    >>> from sklearn.model_selection import StratifiedKFold
    >>> cv = StratifiedKFold(n_splits=5, random_state=0)
    >>> classif = create_classifiers()[DEFAULT_CLASSIF_NAME]
    >>> fp_tp, auc = eval_classif_cross_val_roc(DEFAULT_CLASSIF_NAME, classif,
    ...                                         data, labels, cv, nb_steps=10)
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
    >>> path_out = 'temp_eval-cv-roc'
    >>> os.mkdir(path_out)
    >>> fp_tp, auc = eval_classif_cross_val_roc(DEFAULT_CLASSIF_NAME, classif,
    ...                           data, labels, cv, nb_steps=5, path_out=path_out)
    >>> fp_tp
         FP   TP
    0  0.00  0.0
    1  0.25  1.0
    2  0.50  1.0
    3  0.75  1.0
    4  1.00  1.0
    >>> auc
    0.875
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, nb_steps)
    labels_bin = np.zeros((len(labels), np.max(labels) + 1))
    unique_labels = np.unique(labels)
    assert all(unique_labels >= 0), \
        'some labels are negative: %r' % unique_labels
    for lb in unique_labels:
        labels_bin[:, lb] = (labels == lb)

    # since version change the CV is not iterable by default
    if not hasattr(cross_val, '__iter__'):
        cross_val = cross_val.split(features, labels)
    count = 0.
    for train, test in cross_val:
        classif_cv = clone(classif)
        classif_cv.fit(np.copy(features[train], order='C'),
                       np.copy(labels[train], order='C'))
        proba = classif_cv.predict_proba(np.copy(features[test], order='C'))
        # Compute ROC curve and area the curve
        for i, lb in enumerate(unique_labels):
            fpr, tpr, _ = metrics.roc_curve(labels_bin[test, lb], proba[:, i])
            fpr = [0.] + fpr.tolist() + [1.]
            tpr = [0.] + tpr.tolist() + [1.]
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            count += 1
        # roc_auc = metrics.auc(fpr, tpr)

    mean_tpr /= count
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    df_roc = pd.DataFrame(np.array([mean_fpr, mean_tpr]).T, columns=['FP', 'TP'])
    auc = metrics.auc(mean_fpr, mean_tpr)

    if path_out is not None:
        assert os.path.exists(path_out), 'missing: "%s"' % path_out
        name_csv = NAME_CSV_CLASSIF_CV_ROC.format(clf_name, 'mean')
        df_roc.to_csv(os.path.join(path_out, name_csv))
        name_txt = NAME_TXT_CLASSIF_CV_AUC.format(clf_name, 'mean')
        with open(os.path.join(path_out, name_txt), 'w') as fp:
            fp.write(str(auc))
    logging.debug('cross_val ROC: \n %r', df_roc)
    return df_roc, auc


def search_params_cut_down_max_nb_iter(clf_parameters, nb_iter):
    """ create parameters list and count number of possible combination
    in case they are they are limited

    :param dict clf_parameters: dictionary with parameters
    :param int nb_iter: nb of random tryes
    :return int:

    >>> clf_params = create_clf_param_search_grid(DEFAULT_CLASSIF_NAME)
    >>> search_params_cut_down_max_nb_iter(clf_params, 100)
    100
    >>> search_params_cut_down_max_nb_iter(clf_params, 1e6)
    1450
    """
    counts = []
    for k in clf_parameters:
        vals = clf_parameters[k]
        if hasattr(vals, '__iter__'):
            counts.append(len(vals))
        else:
            return nb_iter
    count = np.product(counts)
    if count < nb_iter:
        nb_iter = count
    return nb_iter


def create_classif_search(name_clf, clf_pipeline, nb_labels,
                          search_type='random', cross_val=10,
                          eval_metric='f1', nb_iter=250, nb_workers=5):
    """ create sklearn search depending on spec. random or grid

    :param int nb_labels: number of labels
    :param str search_type: hyper-params search type
    :param str eval_metric: evaluation metric
    :param int nb_iter: for random number of tries
    :param str name_clf: name of classif.
    :param obj clf_pipeline: object
    :param obj cross_val: obj specific CV for fix train-test
    :param int nb_workers: number jobs running in parallel
    :return:
    """
    score_weight = 'weighted' if nb_labels > 2 else 'binary'
    scoring = metrics.make_scorer(DICT_SCORING[eval_metric.lower()],
                                  average=score_weight)
    if search_type == 'grid':
        clf_parameters = create_clf_param_search_grid(name_clf)
        logging.info('init Grid search...')
        clf_search = GridSearchCV(
            clf_pipeline, clf_parameters, scoring=scoring, cv=cross_val,
            n_jobs=nb_workers, verbose=1, refit=True)
    else:
        clf_parameters = create_clf_param_search_distrib(name_clf)
        nb_iter = search_params_cut_down_max_nb_iter(clf_parameters, nb_iter)
        logging.info('init Randomized search...')
        clf_search = RandomizedSearchCV(
            clf_pipeline, clf_parameters, scoring=scoring, cv=cross_val,
            n_jobs=nb_workers, n_iter=nb_iter, verbose=1, refit=True)
    return clf_search


def shuffle_features_labels(features, labels):
    """ take the set of features and labels and shuffle them together
    while keeping link between feature and its label

    :param ndarray features: features in dimension nb_samples x nb_features
    :param list(int) labels: annotation for samples
    :return: np.array<nb_samples, nb_features>, np.array<nb_samples>

    >>> np.random.seed(0)
    >>> fts = np.random.random((5, 2))
    >>> lbs = np.random.randint(0, 2, 5)
    >>> fts_new, lbs_new = shuffle_features_labels(fts, lbs)
    >>> np.array_equal(fts, fts_new)
    False
    >>> np.array_equal(lbs, lbs_new)
    False
    """
    assert len(features) == len(labels), \
        'features (%i) and labels (%i) should have equal length' \
        % (len(features), len(labels))
    idx = list(range(len(labels)))
    logging.debug('shuffle indexes - %i', len(labels))
    np.random.shuffle(idx)
    features = features[idx, :]
    labels = np.asarray(labels)[idx]
    return features, labels


def convert_dict_label_features_2_vectors(dict_features):
    """ convert dictionary of features where key is the labels
    to vector of all features and related labels

    :param {int: [list(float)]} dict_features: {int: [list(float) * nb_features] * nb_samples}
    :return tuple(ndarray,list(int)): np.array<nb_samples, nb_features>, list(int)
    """
    features, labels = [], []
    for k in dict_features:
        features += dict_features[k].tolist()
        labels += [k] * len(dict_features[k])
    return np.array(features), labels


def compose_dict_label_features(features, labels):
    """ convert vector of features and related labels
    to a dictionary of features where key is the lables

    :param ndarray features: features in dimension nb_samples x nb_features
    :param list(int) labels: annotation for samples
    :return {int: ndarray}: {int: np.array<nb, nb_features>}
    """
    dict_features = dict()
    features = np.array(features)
    for lb in np.unique(labels):
        dict_features[lb] = features[labels == lb, :]
    return dict_features


def down_sample_dict_features_random(dict_features, nb_samples):
    """ browse all label features and take random subset of features to have
    given nb_samples per class

    :param dict dict_features: {int: [list(float) * nb_features] * nb}
    :param int nb_samples:
    :return dict: {int: [list(float) * nb_features] * nb_samples}

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

    :param dict dict_features: {int: [list(float) * nb_features] * nb}
    :param int nb_samples:
    :return dict: {int: [list(float) * nb_features] * nb_samples}

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
#     unique_matrix = np.unique(matrix.view([('', matrix.dtype)]
#                                           * matrix.shape[1]))
#     unique_shape = (unique_matrix.shape[0], matrix.shape[1])
#     unique_matrix = unique_matrix.view(matrix.dtype).reshape(unique_shape)
#     return unique_matrix


def unique_rows(data):
    """ with matrix detect unique row and return only them

    :param ndarray data: np.array
    :return ndarray: np.array
    """
    # preventing: ValueError: new type not compatible with array.
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.view.html
    data = data.copy()
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


def down_sample_dict_features_unique(dict_features):
    """ browse all label features and take unique features

    :param dict dict_features: {int: [list(float) * nb_features] * nb_samples}
    :return dict: {int: [list(float) * nb_features] * nb}

    >>> np.random.seed(0)
    >>> d_fts = {'a': np.random.random((100, 3))}
    >>> d_fts = down_sample_dict_features_unique(d_fts)
    >>> d_fts['a'].shape
    (100, 3)
    """
    dict_features_new = dict()
    for label in dict_features:
        features = np.round(dict_features[label], ROUND_UNIQUE_FTS_DIGITS)
        unique_fts = np.array(unique_rows(features))
        assert features.ndim == unique_fts.ndim, 'feature dim matching'
        assert features.shape[1] == unique_fts.shape[1], \
            'features: %i <> %i' % (features.shape[1], unique_fts.shape[1])
        dict_features_new[label] = unique_fts
    return dict_features_new


def balance_dataset_by_(features, labels, balance_type='random',
                        min_samples=None):
    """ balance number of training examples per class by several method

    :param ndarray features: features in dimension nb_samples x nb_features
    :param list(int) labels: annotation for samples
    :param str balance_type: type of balancing dataset
    :param int|None min_samples: if None take the smallest class
    :return tuple(ndarray,ndarray):

    >>> np.random.seed(0)
    >>> fts, lbs = balance_dataset_by_(np.random.random((25, 3)),
    ...                                np.random.randint(0, 2, 25))
    >>> fts.shape
    (24, 3)
    >>> lbs
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    """
    logging.debug('balance dataset using "%s"', balance_type)
    hist_labels = collections.Counter(labels)
    if not min_samples:
        min_samples = min(hist_labels.values())
    dict_features = compose_dict_label_features(features, labels)

    if balance_type.lower() == 'random':
        dict_features = down_sample_dict_features_random(dict_features, min_samples)
    elif balance_type.lower() == 'kmeans':
        dict_features = down_sample_dict_features_kmean(dict_features, min_samples)
    elif balance_type.lower() == 'unique':
        dict_features = down_sample_dict_features_unique(dict_features)
    else:
        logging.warning('not defined balancing method "%s"', balance_type)

    features, labels = convert_dict_label_features_2_vectors(dict_features)
    # features, labels = shuffle_features_labels(features, labels)
    return features, labels


def convert_set_features_labels_2_dataset(imgs_features, imgs_labels,
                                          drop_labels=None, balance_type=None):
    """ with dictionary for each image we concentrate all features over images
    and labels into simple form

    :param {str: ndarray} imgs_features: dictionary of name and features
    :param {str: ndarray} imgs_labels: dictionary of name and labels
    :param list(int) drop_labels: labels to be ignored
    :param bool balance_type: whether balance_type number of sampler per class
    :return tuple(ndarray,ndarray,ndarray):

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
    assert all(k in imgs_labels.keys() for k in imgs_features.keys()), \
        'missing some items of %r' % imgs_labels.keys()
    features_all, labels_all, sizes = list(), list(), list()
    for name in sorted(imgs_features.keys()):
        features = np.array(imgs_features[name])
        labels = np.array(imgs_labels[name].astype(int))

        drop_labels = [] if drop_labels is None else drop_labels
        for lb in drop_labels:
            features = features[labels != lb]
            labels = labels[labels != lb]

        if balance_type is not None:
            # balance_type dataset to have comparable nb of samples
            features, labels = balance_dataset_by_(features, labels,
                                                   balance_type=balance_type)
        features_all += features.tolist()
        labels_all += np.asarray(labels).tolist()
        sizes.append(len(labels))

    return np.array(features_all), np.array(labels_all, dtype=int), sizes


def compute_tp_tn_fp_fn(annot, segm, label_positive=None):
    """ compute measure TruePositive, TrueNegative, FalsePositive, FalseNegative

    :param ndarray annot: annotation
    :param ndarray segm: segmentation
    :param int label_positive: indexes of positive labels
    :return tuple(float,float,float,float):

    >>> np.random.seed(0)
    >>> annot = np.random.randint(0, 2, (5, 7)) * 9
    >>> segm = np.random.randint(0, 2, (5, 7)) * 9
    >>> annot - segm
    array([[-9,  9,  0, -9,  9,  9,  0],
           [ 9,  0,  0,  0, -9, -9,  9],
           [-9,  0, -9, -9, -9,  0,  0],
           [ 0,  9,  0, -9,  0,  9,  0],
           [ 9, -9,  9,  0,  9,  0,  9]])
    >>> compute_tp_tn_fp_fn(annot, annot)
    (20, 15, 0, 0)
    >>> compute_tp_tn_fp_fn(annot, segm)
    (9, 5, 11, 10)
    >>> compute_tp_tn_fp_fn(annot, np.ones((5, 7)))
    (nan, nan, nan, nan)
    >>> compute_tp_tn_fp_fn(np.zeros((5, 7)), np.zeros((5, 7)))
    (35, 0, 0, 0)
    """
    y_true = np.asarray(annot).ravel()
    y_pred = np.asarray(segm).ravel()
    uq_labels = np.unique([y_true, y_pred]).tolist()
    if len(uq_labels) > 2:
        logging.debug('too many labels: %r', uq_labels)
        return np.nan, np.nan, np.nan, np.nan
    elif len(uq_labels) < 2:
        logging.debug('only one label: %r', uq_labels)
        return len(y_true), 0, 0, 0

    if label_positive is None or label_positive not in uq_labels:
        label_positive = uq_labels[-1]
    uq_labels.remove(label_positive)
    label_negative = uq_labels[0]

    tp = np.sum(
        np.logical_and(y_true == label_positive, y_pred == label_positive))
    tn = np.sum(
        np.logical_and(y_true == label_negative, y_pred == label_negative))
    fp = np.sum(
        np.logical_and(y_true == label_positive, y_pred == label_negative))
    fn = np.sum(
        np.logical_and(y_true == label_negative, y_pred == label_positive))
    return tp, tn, fp, fn


def compute_metric_fpfn_tpfn(annot, segm, label_positive=None):
    """ compute measure (FP + FN) / (TP + FN)

    :param ndarray annot: annotation
    :param ndarray segm: segmentation
    :param int label_positive: indexes of positive labels
    :return float:

    >>> np.random.seed(0)
    >>> annot = np.random.randint(0, 2, (50, 75)) * 3
    >>> segm = np.random.randint(0, 2, (50, 75)) * 3
    >>> compute_metric_fpfn_tpfn(annot, segm)  # doctest: +ELLIPSIS
    1.02...
    >>> compute_metric_fpfn_tpfn(annot, annot)
    0.0
    >>> compute_metric_fpfn_tpfn(annot, np.ones((50, 75)))
    nan
    """
    tp, _, fp, fn = compute_tp_tn_fp_fn(annot, segm, label_positive)
    if tp == np.nan:
        return np.nan
    elif (fp + fn) == 0:
        return 0.
    measure = float(fp + fn) / float(tp + fn)
    return measure


def compute_metric_tpfp_tpfn(annot, segm, label_positive=None):
    """ compute measure (TP + FP) / (TP + FN)

    :param ndarray annot:
    :param ndarray segm:
    :param int label_positive:
    :return float:

    >>> np.random.seed(0)
    >>> annot = np.random.randint(0, 2, (50, 75)) * 3
    >>> segm = np.random.randint(0, 2, (50, 75)) * 3
    >>> compute_metric_tpfp_tpfn(annot, segm)  # doctest: +ELLIPSIS
    1.03...
    >>> compute_metric_tpfp_tpfn(annot, annot)
    1.0
    >>> compute_metric_tpfp_tpfn(annot, np.ones((50, 75)))
    nan
    >>> compute_metric_tpfp_tpfn(annot, np.zeros((50, 75)))
    0.0
    """
    tp, _, fp, fn = compute_tp_tn_fp_fn(annot, segm, label_positive)
    if tp == np.nan:
        return np.nan
    elif (tp + fn) == 0:
        return 0.
    measure = float(tp + fp) / float(tp + fn)
    return measure


# def stat_weight_by_support(dict_vals, id_val, id_sup):
#     val = [v * s for v, s in zip(dict_vals[id_val], dict_vals[id_sup])]
#     n = np.sum(val) / np.sum(dict_vals[id_sup])
#     return n
#
#
# def format_classif_stat(y_true, y_pred):
#     """ format classification statistic
#
#     :param list(int) y_true: annotation
#     :param list(int) y_pred: predictions
#     :return:
#
#     >>> np.random.seed(0)
#     >>> y_true = np.random.randint(0, 2, 25)
#     >>> y_pred = np.random.randint(0, 2, 25)
#     >>> stat = format_classif_stat(y_true, y_pred)
#     >>> pd.Series(stat)
#     f1_score      0.586667
#     precision     0.605882
#     recall        0.600000
#     support      25.000000
#     dtype: float64
#     """
#     vals = metrics.precision_recall_fscore_support(y_true, y_pred)
#     stat = {'precision':    stat_weight_by_support(vals, 0, 3),
#             'recall':       stat_weight_by_support(vals, 1, 3),
#             'f1_score':     stat_weight_by_support(vals, 2, 3),
#             'support':      np.sum(vals[3])}
#     return stat


class HoldOut(object):
    """
    Hold-out cross-validator generator. In the hold-out, the
    data is split only once into a train set and a test set.
    Unlike in other cross-validation schemes, the hold-out
    consists of only one iteration.

    Parameters
    ----------
    nb_samples : int, total number of samples
    hold_out : int, number where the test starts
    rand_seed :  seed for the random number generator

    Example
    -------
    >>> ho = HoldOut(10, 7, rand_seed=None)
    >>> len(ho)
    1
    >>> list(ho)
    [([0, 1, 2, 3, 4, 5, 6], [7, 8, 9])]
    >>> ho = HoldOut(10, 7, rand_seed=0)
    >>> list(ho)
    [([2, 8, 4, 9, 1, 6, 7], [3, 0, 5])]
    """
    def __init__(self, nb_samples, hold_out, rand_seed=0):
        """ constructor

        :param int nb_samples: total number of samples
        :param int hold_out: index where the test starts
        :param obj rand_seed: Seed for the random number generator.
        """
        assert nb_samples > hold_out, \
            'total %i should be higher than hold Idx %i' % (nb_samples, hold_out)

        self._total = nb_samples
        self.hold_out = hold_out
        self._indexes = list(range(nb_samples))

        if rand_seed is not None and rand_seed is not False:
            np.random.seed(rand_seed)
            np.random.shuffle(self._indexes)

    def __iter__(self):
        """ iterate the folds

        :return tuple(list(int),list(int)):
        """
        ind_train = self._indexes[:self.hold_out]
        ind_test = self._indexes[self.hold_out:]
        yield ind_train, ind_test

    def __len__(self):
        """ number of folds

        :return int:
        """
        return 1


class CrossValidate(object):
    """Cross-validator generator.
    In the hold-out, the data is split only once into a train set and a test set.

    Parameters
    ----------
    nb_samples : integer, total number of samples
    nb_hold_out : integer, number of samples hold out
    rand_seed :  seed for the random number generator
    ignore_overflow: float, tolerance while dividing dataset to folds

    Examples
    --------
    >>> # balanced split
    >>> cv = CrossValidate(6, 3, rand_seed=False)
    >>> cv.indexes
    [0, 1, 2, 3, 4, 5]
    >>> len(cv)
    2
    >>> list(cv)  # doctest: +NORMALIZE_WHITESPACE
    [([3, 4, 5], [0, 1, 2]),
     ([0, 1, 2], [3, 4, 5])]
    >>> [(len(tr), len(ts)) for tr, ts in CrossValidate(340, 0.41)]
    [(201, 139), (201, 139), (201, 139)]

    >>> # not rounded split
    >>> cv = CrossValidate(7, 3, rand_seed=0)
    >>> list(cv)  # doctest: +NORMALIZE_WHITESPACE
    [([3, 0, 5, 4], [6, 2, 1]),
     ([6, 2, 1, 4], [3, 0, 5]),
     ([1, 3, 0, 5], [4, 6, 2])]
    >>> len(cv)
    3
    >>> cv.indexes
    [6, 2, 1, 3, 0, 5, 4]

    >>> # larger test then train
    >>> cv = CrossValidate(7, 5, rand_seed=0)
    >>> list(cv)  # doctest: +NORMALIZE_WHITESPACE
    [([6, 2], [1, 3, 0, 5, 4]),
     ([1, 3], [6, 2, 0, 5, 4]),
     ([0, 5], [6, 2, 1, 3, 4]),
     ([4, 6], [2, 1, 3, 0, 5])]
    >>> [(len(tr), len(ts)) for tr, ts in CrossValidate(340, 0.55)]
    [(153, 187), (153, 187), (153, 187)]

    >>> # impact of tolerance
    >>> len(CrossValidate(340, 0.33, ignore_overflow=0.0))
    4
    >>> len(CrossValidate(340, 0.33, ignore_overflow=0.05))
    3

    >>> [(len(tr), len(ts)) for tr, ts in CrossValidate(4651, 0.25, ignore_overflow=0.)]
    [(3488, 1163), (3488, 1163), (3488, 1163), (3488, 1163)]
    >>> [(len(tr), len(ts)) for tr, ts in CrossValidate(4651, 0.25, ignore_overflow=1e-2)]
    [(3488, 1163), (3488, 1163), (3488, 1163), (3489, 1162)]
    """
    def __init__(self, nb_samples, nb_hold_out, rand_seed=None, ignore_overflow=0.01):
        """ constructor

        :param int nb_samples: list of sizes
        :param int|float nb_hold_out: how much hold out
        :param int|None rand_seed: random seed for shuffling
        :param float ignore_overflow: tolerance while dividing dataset to folds
        """
        assert nb_samples > nb_hold_out, \
            'Number of holdout has to be smaller then total size.'
        assert nb_hold_out > 0, 'Number of holdout has to be positive number.'
        self._nb_samples = nb_samples
        self._nb_hold_out = int(np.round(nb_samples * nb_hold_out)) \
            if nb_hold_out < 1 else nb_hold_out
        ignore_overflow = abs(ignore_overflow)
        self._ignore_overflow = int(np.round(nb_samples * ignore_overflow)) \
            if ignore_overflow < 1 else ignore_overflow
        assert self._nb_hold_out > self._ignore_overflow, \
            'The tolerance of overflowing (%i) the split has to be larger than' \
            ' the number of hold out samples (%i).' % (self._ignore_overflow,
                                                       self._nb_hold_out)

        self._revert = False  # sets the sizes
        if self._nb_hold_out > (self._nb_samples / 2.):
            logging.debug('WARNING: You are running in reverse mode, '
                          'while using all training examples '
                          'there are much more yield test cases.')
            self._nb_hold_out = self._nb_samples - self._nb_hold_out
            self._revert = True

        self.indexes = list(range(self._nb_samples))

        if rand_seed is not None and rand_seed is not False:
            self._shuffle = True
            np.random.seed(rand_seed)
            np.random.shuffle(self.indexes)
        else:
            self._shuffle = False
        logging.debug('sets ordering: %r', np.array(self.indexes))

        self.iter = 0

    def __steps(self):
        """ adjust this iterator, tol_balance

        :return list(int): indexes of steps
        """
        steps = list(range(0, self._nb_samples, self._nb_hold_out))
        steps_wtol = [i for i in steps if (self._nb_samples - i) >= self._ignore_overflow]
        if steps != steps_wtol:
            logging.debug('WARNING: The last fold that would overflow was'
                          ' dropped due to low tolerance `ignore_overflow`.')
        return steps_wtol

    def __iter__(self):
        """ iterate the folds

        :return tuple(list(int),list(int)):
        """
        # iterate over all filtered splits
        for i in self.__steps():
            i_end = i + self._nb_hold_out
            inds_test = self.indexes[i:i_end]
            inds_train = self.indexes[:i] + self.indexes[i_end:]
            # over flow the limited set
            if i_end > self._nb_samples:
                if (i_end - self._nb_samples) > self._ignore_overflow:
                    i_begin = i_end - self._nb_samples
                    inds_test += self.indexes[:i_begin]
                    inds_train = self.indexes[i_begin:i]
                    logging.debug('WARNING: Your demand for last test fold overflow'
                                  ' by %i, to keep the train-test ration we reuse part'
                                  ' of the already tested samples from the %s beginning.',
                                  i_begin, 'shuffled' if self._shuffle else '')
                else:
                    logging.debug('WARNING: To keep the set strictly balances,'
                                  ' you need to decrees tolerance `ignore_overflow`.')
            # reverting the train -test split
            if self._revert:
                inds_train, inds_test = inds_test, inds_train
            yield inds_train, inds_test

    def __len__(self):
        """ number of folds

        :return int:
        """
        return len(self.__steps())


class CrossValidateGroups(CrossValidate):
    """
    Cross-validator generator. In the hold-out, the
    data is split only once into a train set and a test set.

    Parameters
    ----------
    set_sizes : list of integers, number of samples in each set
    nb_hold_out : integer, number of sets hold out
    rand_seed :  seed for the random number generator
    ignore_overflow: float, tolerance while dividing dataset to folds

    Examples
    --------
    >>> # balance split
    >>> cv = CrossValidateGroups([2, 3, 2, 3], 2, rand_seed=False)
    >>> cv.set_indexes
    [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]
    >>> len(cv)
    2
    >>> list(cv)  # doctest: +NORMALIZE_WHITESPACE
    [([5, 6, 7, 8, 9], [0, 1, 2, 3, 4]),
     ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])]
    >>> [(len(tr), len(ts)) for tr, ts in CrossValidateGroups([7] * 340, 0.41)]
    [(1407, 973), (1407, 973), (1407, 973)]

    >>> # unbalanced split
    >>> cv = CrossValidateGroups([2, 2, 1, 2, 1], 2, rand_seed=0)
    >>> cv.set_indexes
    [[0, 1], [2, 3], [4], [5, 6], [7]]
    >>> list(cv)  # doctest: +NORMALIZE_WHITESPACE
    [([2, 3, 5, 6, 7], [4, 0, 1]),
     ([4, 0, 1, 7], [2, 3, 5, 6]),
     ([0, 1, 2, 3, 5, 6], [7, 4])]
    >>> len(cv)
    3
    >>> cv.indexes
    [2, 0, 1, 3, 4]

    >>> # larger test then train
    >>> cv = CrossValidateGroups([2, 2, 1, 2, 1, 1], 4, rand_seed=0)
    >>> list(cv)  # doctest: +NORMALIZE_WHITESPACE
    [([8, 4], [2, 3, 5, 6, 0, 1, 7]),
     ([2, 3, 5, 6], [8, 4, 0, 1, 7]),
     ([0, 1, 7], [8, 4, 2, 3, 5, 6])]
    >>> [(len(tr), len(ts)) for tr, ts in CrossValidateGroups([7] * 340, 0.55)]
    [(1071, 1309), (1071, 1309), (1071, 1309)]
    """

    def __init__(self, set_sizes, nb_hold_out, rand_seed=None, ignore_overflow=0.01):
        """ construct

        :param list(int) set_sizes: list of sizes
        :param int|float nb_hold_out: how much hold out
        :param int|None rand_seed: random seed for shuffling
        :param float ignore_overflow: tolerance while dividing dataset to folds
        """
        super(CrossValidateGroups, self).__init__(
            len(set_sizes), nb_hold_out, rand_seed, ignore_overflow)

        self._set_sizes = list(set_sizes)
        self.set_indexes = []

        start = 0
        for i, size in enumerate(self._set_sizes):
            inds = range(start, start + size)
            self.set_indexes.append(list(inds))
            start += size

        total = np.sum(self._set_sizes)
        assert np.sum(len(i) for i in self.set_indexes) == total, \
            'all indexes should sum to total count %i' % total

    def __iter_indexes(self, sets):
        """ return enrol indexes from sets

        :param list(int) sets: selection of indexes
        :return list(int):
        """
        inds = list(itertools.chain(*[self.set_indexes[i] for i in sets]))
        return inds

    def __iter__(self):
        """ iterate the folds

        :return tuple(list(int),list(int)):
        """
        for train, test in super(CrossValidateGroups, self).__iter__():
            inds_train = self.__iter_indexes(train)
            inds_test = self.__iter_indexes(test)
            yield inds_train, inds_test


# DEPRECATED
# ==========

# def check_exist_labels_dataset(dataset, lut):
#     u_lbs = np.unique(lut.values())
#     for l in u_lbs:
#         if not l in dataset:
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
#         words = [np.mean(data[clt.labels_ == l], axis=0)
#                  for l in np.unique(clt.labels_)]
#     if method == 'AffPr':
#         clt = cluster.AffinityPropagation(convergence_iter=7)
#         clt.fit(data)
#         words = clt.cluster_centers_
#     elif method == 'Birch':
#         clt = cluster.Birch(n_clusters=nb_clts)
#         clt.fit(data)
#         words = clt.subcluster_centers_
#     elif method == 'kMeans':
#         clt = cluster.KMeans(init='random', n_clusters=nb_clts, n_init=3,
#                              max_iter=35, n_jobs=5)
#         # clt = cluster.KMeans(init='k-means++', n_clusters=nb_clts,
#                                n_init=7, n_jobs=-1)
#         clt.fit(data)
#         words = clt.cluster_centers_
#     else: # random
#         words = data[np.random.choice(data.shape[0], nb_clts)]
#     return words


# def segm_features_classif_general(dataset_dict, ft, clf, prob=False):
#     X, y = convert_standard_dataset(dataset_dict)
#     assert len(X) == len(y)
#     clf.fit(X, y)
#     lbs = clf.predict(ft)
#     if prob:
#         probs = clf.predict_proba(ft)
#     else:
#         probs=None
#     return lbs, probs


# def get_labeling_probability(ft, datasetDict, lbs):
#     probs = np.zeros((len(ft),len(datasetDict)))
#     probs[range(len(lbs)), lbs] = 1.
#     return probs

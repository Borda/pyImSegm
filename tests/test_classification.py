"""
Unit testing for particular segmentation module

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import unittest
import logging

import numpy as np
from sklearn import metrics

sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Add path to root
from imsegm.classification import create_classifiers, create_classif_search_train_export

CLASSIFIER_NAMES = create_classifiers().keys()


def generate_data(nb_samples=100, nb_classes=3, dim_features=4):
    """ generating separable features pace with specific number of classes,
    samples per class and feature dimension

    :param int nb_samples: number of samples per class
    :param int dim_features: dimension of feature space
    :param int nb_classes: number of classes
    :return tuple(list(int),ndarray): list(int), np.array<nb_samples, dim_fts>
    """
    labels = range(int(nb_classes))
    labels = list(labels) * nb_samples
    # noise around zero
    noise = np.random.rand(len(labels), dim_features) - 0.5
    base_lbs = np.tile(labels, (dim_features, 1)).T
    # step 10 to have difference in features and labeled areas
    base_dim = np.tile(np.arange(dim_features * 1e2, step=1e2), (len(labels), 1))
    data = base_lbs + base_dim + noise
    assert len(labels) == data.shape[0]
    return data, labels


class TestClassification(unittest.TestCase):

    def classif_eval(self, clf, features_train, labels_train,
                     features_test, labels_test):
        """ train and test classifier with assumption of separable data

        :param clf: classifier object
        :param features_train: np.array<nb_spl, dim_fts>
        :param list(int) labels_train:
        :param features_test: np.array<nb_spl, dim_fts>
        :param list(int) labels_test:
        """
        nb_classes = len(np.unique(labels_train))
        f1_train = metrics.f1_score(labels_train, clf.predict(features_train),
                                    average='weighted')
        logging.debug('f1 metric on training: %f', f1_train)
        self.assertGreaterEqual(f1_train, 1. / nb_classes)
        f1_test = metrics.f1_score(labels_test, clf.predict(features_test),
                                   average='weighted')
        logging.debug('f1 metric on testing: %f', f1_test)
        self.assertGreaterEqual(f1_test, 1. / nb_classes)

    def test_classif_simple(self):
        """ test the training of classif with expected F1 score close to one """
        data_train, labels_train = generate_data()
        data_test, labels_test = generate_data()
        for n in CLASSIFIER_NAMES:
            logging.info('created classifier: %s', n)
            clf = create_classifiers()[n]
            clf.fit(data_train, labels_train)
            self.classif_eval(clf, data_train, labels_train,
                              data_test, labels_test)

    def test_classif_pipeline(self):
        """ test the training of classif with expected F1 score close to one """
        data_train, labels_train = generate_data()
        data_test, labels_test = generate_data()
        for n in CLASSIFIER_NAMES:
            logging.info('created classif.: %s', n)
            clf, _ = create_classif_search_train_export(n, data_train, labels_train,
                                                        nb_search_iter=5)
            self.classif_eval(clf, data_train, labels_train,
                              data_test, labels_test)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

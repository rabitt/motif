# -*- coding: utf-8 -*-
""" Classification using a random forrest
"""
from __future__ import print_function
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import cross_validation
from sklearn.utils import shuffle
import numpy as np

from motif.core import ContourClassifier


class RandomForest(ContourClassifier):

    def __init__(self, n_estimators=100, n_jobs=-1, class_weight='auto',
                 max_features=None, max_param=100, param_step=5):
        '''
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest
        n_jobs : int
            Number of cores to use. -1 uses maximum availalbe
        class_weight : str
            How to set class weights.
        max_features : int or None
            The maximum number of features that can be used in a single branch.
        max_param : int
            Maximum depth value to sweep
        param_step : int
            Step size in parameter sweep

        '''
        ContourClassifier.__init__(self)

        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.max_features = max_features
        self.max_param = max_param
        self.param_step = param_step

        self.clf = None
        self.max_depth = None

    def predict(self, X):
        """ Compute probability predictions.

        Parameters
        ----------
        X : np.array [n_samples, n_features]
            Features.

        Returns
        -------
        p : np.array [n_samples]
            predicted probabilities
        """
        if self.clf is None:
            raise ReferenceError(
                "fit must be called before predict can be called"
            )
        p_train = self.clf.predict_proba(X)[:, 1]
        return p_train

    def fit(self, X, Y):
        """ Train classifier.

        Parameters
        ----------
        X : np.array [n_samples, n_features]
            Training features.
        Y : np.array [n_samples]
            Training labels

        """
        x_shuffle, y_shuffle = shuffle(X, Y)
        self._cross_val_sweep(x_shuffle, y_shuffle)
        clf = RFC(n_estimators=self.n_estimators, max_depth=self.max_depth,
                  n_jobs=self.n_jobs, class_weight=self.class_weight,
                  max_features=self.max_features)
        clf.fit(x_shuffle, y_shuffle)
        self.clf = clf

    @property
    def threshold(self):
        """Positive class is score >= 0.5"""
        return 0.5

    @classmethod
    def get_id(cls):
        """Method to get the id of the extractor type"""
        return 'random_forest'

    def _cross_val_sweep(self, X, Y):
        """ Choose best parameter by performing cross fold validation
        """
        scores = []
        for max_depth in np.arange(5, self.max_param, self.param_step):
            print("training with max_depth={}".format(max_depth))
            clf = RFC(n_estimators=self.n_estimators, max_depth=self.max_depth,
                      n_jobs=self.n_jobs, class_weight=self.class_weight,
                      max_features=self.max_features)
            all_scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
            scores.append([max_depth, np.mean(all_scores), np.std(all_scores)])

        depth = [score[0] for score in scores]
        accuracy = [score[1] for score in scores]
        # std_dev = [score[2] for score in scores]

        best_depth = depth[np.argmax(accuracy)]
        # max_cv_accuracy = np.max(accuracy)
        # plot_data = (depth, accuracy, std_dev)

        self.max_depth = best_depth

# -*- coding: utf-8 -*-
""" Score based on multivariate gaussian as in Meloida
"""
import numpy as np
from scipy.stats import boxcox
from scipy.stats import multivariate_normal

from mira.core import Classifier

EPS = 1.0

class MvGaussian(Classifier):

    def __init__(self):
        Classifier.__init__(self)

        self.rv_pos = None
        self.rv_neg = None
        self.n_feats = None
        self.lmbda = None

    def predict(self, X):
        """ Compute melodiness score.

        Parameters
        ----------
        X : np.array [n_samples, n_features]
            Features.

        Returns
        -------
        p : np.array [n_samples]
            melodiness scores
        """
        if self.rv_pos is None:
            raise ReferenceError(
                "fit must be called before predict can be called"
            )
        transformed_feats = self.transform(X)
        numerator = self.rv_pos.pdf(transformed_feats)
        denominator = self.rv_neg.pdf(transformed_feats)
        return numerator / denominator

    def fit(self, X, Y):
        """ Fit class-dependent multivariate gaussians on the training set.

        Parameters
        ----------
        x_train_boxcox : np.array [n_samples, n_features_trans]
            Transformed training features.
        y_train : np.array [n_samples]
            Training labels.

        Returns
        -------
        rv_pos : multivariate normal
            multivariate normal for melody class
        rv_neg : multivariate normal
            multivariate normal for non-melody class
        """
        X_boxcox = self._fit_boxcox(X, Y)
        pos_idx = np.where(Y == 1)[0]
        mu_pos = np.mean(X_boxcox[pos_idx, :], axis=0)
        cov_pos = np.cov(X_boxcox[pos_idx, :], rowvar=0)

        neg_idx = np.where(Y == 0)[0]
        mu_neg = np.mean(X_boxcox[neg_idx, :], axis=0)
        cov_neg = np.cov(X_boxcox[neg_idx, :], rowvar=0)
        rv_pos = multivariate_normal(
            mean=mu_pos, cov=cov_pos, allow_singular=True
        )
        rv_neg = multivariate_normal(
            mean=mu_neg, cov=cov_neg, allow_singular=True
        )
        self.rv_pos = rv_pos
        self.rv_neg = rv_neg

    @classmethod
    def get_id(cls):
        """Method to get the id of the extractor type"""
        return 'mv_gaussian'

    def _fit_boxcox(self, X, Y):
        """ Transform features using a boxcox transform.

        Parameters
        ----------
        x_train : np.array [n_samples, n_features]
            Untransformed training features.
        x_test : np.array [n_samples, n_features]
            Untransformed testing features.

        Returns
        -------
        x_train_boxcox : np.array [n_samples, n_features_trans]
            Transformed training features.
        x_test_boxcox : np.array [n_samples, n_features_trans]
            Transformed testing features.
        """
        _, self.n_feats = X.shape

        X_boxcox = np.zeros(X.shape)
        lmbda_opt = np.zeros((self.n_feats,))

        for i in range(self.n_feats):
            X_boxcox[:, i], lmbda_opt[i] = boxcox(
                X[:, i] + EPS
            )
        self.lmbda = lmbda_opt
        return X_boxcox

    def transform(self, X):
        X_boxcox = np.zeros(X.shape)
        for i in range(self.n_feats):
            X_boxcox[:, i] = boxcox(
                X[:, i] + EPS, lmbda=self.lmbda
            )
        return X_boxcox

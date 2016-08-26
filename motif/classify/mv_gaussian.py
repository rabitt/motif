# -*- coding: utf-8 -*-
""" Score based on multivariate gaussian as in Meloida
"""
import numpy as np
from scipy.stats import boxcox
from scipy.stats import multivariate_normal
from sklearn import metrics

from mira.core import Classifier

EPS = 1.0

class MvGaussian(ContourExtractor):

    def fit(train_features, train_labels):
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
        train_features_boxcox = self._fit_boxcox(train_features, train_labels)
        pos_idx = np.where(train_labels == 1)[0]
        mu_pos = np.mean(train_features_boxcox[pos_idx, :], axis=0)
        cov_pos = np.cov(train_features_boxcox[pos_idx, :], rowvar=0)

        neg_idx = np.where(train_labels == 0)[0]
        mu_neg = np.mean(train_features_boxcox[neg_idx, :], axis=0)
        cov_neg = np.cov(train_features_boxcox[neg_idx, :], rowvar=0)
        rv_pos = multivariate_normal(
            mean=mu_pos, cov=cov_pos, allow_singular=True
        )
        rv_neg = multivariate_normal(
            mean=mu_neg, cov=cov_neg, allow_singular=True
        )
        self.rv_pos = rv_pos
        self.rv_neg = rv_neg

    def predict(self, X, mvg):
        transformed_feats = mvg.transform(X)
        numerator = self.rv_pos.pdf(transformed_feats)
        denominator = self.rv_neg.pdf(transformed_feats)
        score = numerator/denominator
        return score

    def _fit_boxcox(self, train_features, train_labels):
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
        _, self.n_feats = train_features.shape

        train_features_boxcox = np.zeros(train_features.shape)
        lmbda_opt = np.zeros((self.n_feats,))
        
        for i in range(self.n_feats):
            train_features_boxcox[:, i], lmbda_opt[i] = boxcox(
                train_features[:, i] + EPS
            )
        self.lmbda = lmbda_opt
        return train_features_boxcox

    def transform(self, features):
        feat_transformed = np.zeros(features.shape)
        for i in range(self.n_feats):
            feat_transformed[:, i] = boxcox(
                features[:, i] + EPS, lmbda=self.lmbda
            )
        return feat_transformed

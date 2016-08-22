""" Functions for doing scoring based on multivariate gaussian as in Meloida
"""
import numpy as np
from scipy.stats import boxcox
from scipy.stats import multivariate_normal
from sklearn import metrics

EPS = 1.0


def predict(X, mvg):
    return mvg.predict_mvg(X)


def fit(contour_features, contour_labels):
    mvg = MvGaussian(contour_features, contour_labels)
    mvg.fit_mvg()
    return mvg


class MvGaussian(object):
    def __init__(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.lmbda = None
        self.train_features_boxcox = None
        self.n_feats = None
        self.rv_pos = None
        self.rv_neg = None
        self._fit_boxcox()

    def _fit_boxcox(self):
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
        _, self.n_feats = self.train_features.shape

        train_features_boxcox = np.zeros(self.train_features.shape)
        lmbda_opt = np.zeros((self.n_feats,))
        
        for i in range(self.n_feats):
            train_features_boxcox[:, i], lmbda_opt[i] = boxcox(
                self.train_features[:, i] + EPS
            )
        self.train_features_boxcox = train_features_boxcox
        self.lmbda = lmbda_opt

    def transform(self, features):
        feat_transformed = np.zeros(features.shape)
        for i in range(self.n_feats):
            feat_transformed[:, i] = boxcox(
                features[:, i] + EPS, lmbda=self.lmbda
            )
        return feat_transformed

    def fit_mvg(self):
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
        pos_idx = np.where(self.train_labels == 1)[0]
        mu_pos = np.mean(self.train_features_boxcox[pos_idx, :], axis=0)
        cov_pos = np.cov(self.train_features_boxcox[pos_idx, :], rowvar=0)

        neg_idx = np.where(self.train_labels == 0)[0]
        mu_neg = np.mean(self.train_features_boxcox[neg_idx, :], axis=0)
        cov_neg = np.cov(self.train_features_boxcox[neg_idx, :], rowvar=0)
        rv_pos = multivariate_normal(
            mean=mu_pos, cov=cov_pos, allow_singular=True
        )
        rv_neg = multivariate_normal(
            mean=mu_neg, cov=cov_neg, allow_singular=True
        )
        self.rv_pos = rv_pos
        self.rv_neg = rv_neg

    def predict_mvg(self):
        """ Compute melodiness score for an example given trained distributions.

        Parameters
        ----------
        sample : np.array [n_feats]
            Instance of transformed data.
        rv_pos : multivariate normal
            multivariate normal for melody class
        rv_neg : multivariate normal
            multivariate normal for non-melody class

        Returns
        -------
        melodiness: float
            score between 0 and inf. class cutoff at 1
        """
        transformed_feats = mvg.transform(X)
        return rv_pos.pdf(transformed_feats)/rv_neg.pdf(transformed_feats)


def melodiness_metrics(m_train, m_test, y_train, y_test):
    """ Compute metrics on melodiness score

    Parameters
    ----------
    m_train : np.array [n_samples]
        melodiness scores for training set
    m_test : np.array [n_samples]
        melodiness scores for testing set
    y_train : np.array [n_samples]
        Training labels.
    y_test : np.array [n_samples]
        Testing labels.

    Returns
    -------
    melodiness_scores : dict
        melodiness scores for training set
    """
    m_bin_train = 1*(m_train >= 1)
    m_bin_test = 1*(m_test >= 1)

    train_scores = {}
    test_scores = {}

    train_scores['accuracy'] = metrics.accuracy_score(y_train, m_bin_train)
    test_scores['accuracy'] = metrics.accuracy_score(y_test, m_bin_test)

    train_scores['mcc'] = metrics.matthews_corrcoef(y_train, m_bin_train)
    test_scores['mcc'] = metrics.matthews_corrcoef(y_test, m_bin_test)

    (p, r, f, s) = metrics.precision_recall_fscore_support(y_train,
                                                           m_bin_train)
    train_scores['precision'] = p
    train_scores['recall'] = r
    train_scores['f1'] = f
    train_scores['support'] = s

    (p, r, f, s) = metrics.precision_recall_fscore_support(y_test,
                                                           m_bin_test)
    test_scores['precision'] = p
    test_scores['recall'] = r
    test_scores['f1'] = f
    test_scores['support'] = s

    train_scores['confusion matrix'] = \
        metrics.confusion_matrix(y_train, m_bin_train, labels=[0, 1])
    test_scores['confusion matrix'] = \
        metrics.confusion_matrix(y_test, m_bin_test, labels=[0, 1])

    train_scores['auc score'] = \
        metrics.roc_auc_score(y_train, m_train + 1, average='weighted')
    test_scores['auc score'] = \
        metrics.roc_auc_score(y_test, m_test + 1, average='weighted')

    melodiness_scores = {'train': train_scores, 'test': test_scores}

    return melodiness_scores


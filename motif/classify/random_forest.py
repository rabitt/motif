# -*- coding: utf-8 -*-
""" Classification using a random forrest
"""
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import cross_validation
import numpy as np


def predict(X, clf):
    """ Compute probability predictions for all training and test examples.

    Parameters
    ----------
    x_train : np.array [n_samples, n_features]
        Training features.
    x_test : np.array [n_samples, n_features]
        Testing features.
    clf : classifier
        Trained scikit-learn classifier

    Returns
    -------
    p : np.array [n_samples]
        predicted probabilities
    """
    p_train = clf.predict_proba(X)[:, 1]
    return p_train


def fit(contour_features, contour_labels):
    """ Train classifier.

    Parameters
    ----------
    x_train : np.array [n_samples, n_features]
        Training features.
    y_train : np.array [n_samples]
        Training labels
    best_depth : int
        Optimal max_depth parameter

    Returns
    -------
    clf : classifier
        Trained scikit-learn classifier
    """
    best_depth, _, _ = cross_val_sweep(contour_features, contour_labels)
    clf = RFC(n_estimators=100, max_depth=best_depth, n_jobs=-1,
              class_weight='auto', max_features=None)
    clf = clf.fit(contour_features, contour_labels)
    return clf


def cross_val_sweep(x_train, y_train, max_search=100, step=5):
    """ Choose best parameter by performing cross fold validation

    Parameters
    ----------
    x_train : np.array [n_samples, n_features]
        Training features.
    y_train : np.array [n_samples]
        Training labels
    max_search : int
        Maximum depth value to sweep
    step : int
        Step size in parameter sweep
    plot : bool
        If true, plot error bars and cv accuracy

    Returns
    -------
    best_depth : int
        Optimal max_depth parameter
    max_cv_accuracy : DataFrames
        Best accuracy achieved on hold out set with optimal parameter.
    """
    scores = []
    for max_depth in np.arange(5, max_search, step):
        print "training with max_depth=%s" % max_depth
        clf = RFC(n_estimators=100, max_depth=max_depth, n_jobs=-1,
                  class_weight='auto', max_features=None)
        all_scores = cross_validation.cross_val_score(clf, x_train, y_train,
                                                      cv=5)
        scores.append([max_depth, np.mean(all_scores), np.std(all_scores)])

    depth = [score[0] for score in scores]
    accuracy = [score[1] for score in scores]
    std_dev = [score[2] for score in scores]

    best_depth = depth[np.argmax(accuracy)]
    max_cv_accuracy = np.max(accuracy)
    plot_data = (depth, accuracy, std_dev)

    return best_depth, max_cv_accuracy, plot_data


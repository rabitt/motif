# -*- coding: utf-8 -*-
"""Evaluate the output of a classifier.
"""
from sklearn import metrics

def clf_metrics(p_train, p_test, y_train, y_test):
    """ Compute metrics on classifier predictions

    Parameters
    ----------
    p_train : np.array [n_samples]
        predicted probabilities for training set
    p_test : np.array [n_samples]
        predicted probabilities for testing set
    y_train : np.array [n_samples]
        Training labels.
    y_test : np.array [n_samples]
        Testing labels.

    Returns
    -------
    clf_scores : dict
        classifier scores for training set
    """
    y_pred_train = 1*(p_train >= 0.5)
    y_pred_test = 1*(p_test >= 0.5)

    train_scores = {}
    test_scores = {}

    train_scores['accuracy'] = metrics.accuracy_score(y_train, y_pred_train)
    test_scores['accuracy'] = metrics.accuracy_score(y_test, y_pred_test)

    train_scores['mcc'] = metrics.matthews_corrcoef(y_train, y_pred_train)
    test_scores['mcc'] = metrics.matthews_corrcoef(y_test, y_pred_test)

    (p, r, f, s) = metrics.precision_recall_fscore_support(y_train,
                                                           y_pred_train)
    train_scores['precision'] = p
    train_scores['recall'] = r
    train_scores['f1'] = f
    train_scores['support'] = s

    (p, r, f, s) = metrics.precision_recall_fscore_support(y_test,
                                                           y_pred_test)
    test_scores['precision'] = p
    test_scores['recall'] = r
    test_scores['f1'] = f
    test_scores['support'] = s

    train_scores['confusion matrix'] = \
        metrics.confusion_matrix(y_train, y_pred_train, labels=[0, 1])
    test_scores['confusion matrix'] = \
        metrics.confusion_matrix(y_test, y_pred_test, labels=[0, 1])

    train_scores['auc score'] = \
        metrics.roc_auc_score(y_train, p_train + 1, average='weighted')
    test_scores['auc score'] = \
        metrics.roc_auc_score(y_test, p_test + 1, average='weighted')

    clf_scores = {'train': train_scores, 'test': test_scores}

    return clf_scores



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

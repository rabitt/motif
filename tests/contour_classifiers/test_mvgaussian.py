"""Test for motif.classify.mvgaussian
"""
from __future__ import print_function
import unittest
import numpy as np

from motif.contour_classifiers import mv_gaussian

def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2))


class TestMvGaussian(unittest.TestCase):

    def setUp(self):
        self.clf = mv_gaussian.MvGaussian()

    def test_rv_pos(self):
        expected = None
        actual = self.clf.rv_pos
        self.assertEqual(expected, actual)

    def test_rv_neg(self):
        expected = None
        actual = self.clf.rv_neg
        self.assertEqual(expected, actual)

    def test_n_feats(self):
        expected = None
        actual = self.clf.n_feats
        self.assertEqual(expected, actual)

    def test_lmbda(self):
        expected = None
        actual = self.clf.lmbda
        self.assertEqual(expected, actual)

    def test_predict_error(self):
        with self.assertRaises(ReferenceError):
            self.clf.predict(np.array([0, 0, 0]))

    def test_fit(self):
        X = np.array([[1.0, 2.0], [0.0, 0.0], [0.5, 0.7]])
        Y = np.array([1, 1, 0])
        self.clf.fit(X, Y)
        self.assertIsNotNone(self.clf.rv_pos)
        self.assertIsNotNone(self.clf.rv_neg)
        self.assertEqual(self.clf.n_feats, 2)
        self.assertIsNotNone(self.clf.lmbda)

    def test_predict(self):
        X = np.array([[1.0, 2.0], [1.0, 2.0], [0.5, 0.7]])
        Y = np.array([1, 1, 0])
        self.clf.fit(X, Y)
        actual = self.clf.predict(
            np.array([[1.0, 2.0], [1.0, 2.0], [0.5, 0.7]])
        )
        expected = np.array([np.inf, np.inf, 2.71818876])
        self.assertTrue(array_equal(expected, actual))

    def test_threshold(self):
        expected = 1.0
        actual = self.clf.threshold
        self.assertEqual(expected, actual)

    def test_get_id(self):
        expected = 'mv_gaussian'
        actual = self.clf.get_id()
        self.assertEqual(expected, actual)

    def test_score(self):
        predicted_scores = np.array([0.0, 0.5, np.inf, 1.0, 2.0])
        y_target = np.array([0, 0, 1, 1, 1])
        expected = {
            'accuracy': 1.0,
            'mcc': 1.0,
            'precision': np.array([1.0, 1.0]),
            'recall': np.array([1.0, 1.0]),
            'f1': np.array([1.0, 1.0]),
            'support': np.array([2, 3]),
            'confusion matrix': np.array([[2, 0], [0, 3]]),
            'auc score': 1.0
        }
        actual = self.clf.score(predicted_scores, y_target)
        self.assertEqual(expected['accuracy'], actual['accuracy'])
        self.assertAlmostEqual(expected['mcc'], actual['mcc'], places=1)
        self.assertTrue(array_equal(expected['precision'], actual['precision']))
        self.assertTrue(array_equal(expected['recall'], actual['recall']))
        self.assertTrue(array_equal(expected['f1'], actual['f1']))
        self.assertTrue(array_equal(expected['support'], actual['support']))
        self.assertTrue(array_equal(
            expected['confusion matrix'], actual['confusion matrix']
        ))
        self.assertEqual(expected['auc score'], actual['auc score'])

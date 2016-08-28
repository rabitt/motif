"""Test for motif.classify.mvgaussian
"""
from __future__ import print_function
import unittest
import numpy as np

from motif.classify import mv_gaussian

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





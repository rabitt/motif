"""Test for motif.classify.mvgaussian
"""
from __future__ import print_function
import unittest
import numpy as np

from motif.contour_classifiers import random_forest

def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2))


class TestRandomForest(unittest.TestCase):

    def setUp(self):
        self.clf = random_forest.RandomForest(
            n_estimators=2, max_param=15
        )

    def test_n_estimators(self):
        expected = 2
        actual = self.clf.n_estimators
        self.assertEqual(expected, actual)

    def test_n_jobs(self):
        expected = -1
        actual = self.clf.n_jobs
        self.assertEqual(expected, actual)

    def test_class_weight(self):
        expected = 'auto'
        actual = self.clf.class_weight
        self.assertEqual(expected, actual)

    def test_max_features(self):
        expected = None
        actual = self.clf.max_features
        self.assertEqual(expected, actual)

    def test_max_param(self):
        expected = 15
        actual = self.clf.max_param
        self.assertEqual(expected, actual)

    def test_param_step(self):
        expected = 5
        actual = self.clf.param_step
        self.assertEqual(expected, actual)

    def test_clf(self):
        expected = None
        actual = self.clf.clf
        self.assertEqual(expected, actual)

    def test_max_depth(self):
        expected = None
        actual = self.clf.max_depth
        self.assertEqual(expected, actual)

    def test_predict_error(self):
        with self.assertRaises(ReferenceError):
            self.clf.predict(np.array([0, 0, 0]))

    def test_fit(self):
        X = np.array([
            [1.0, 2.0], [0.0, 0.0], [0.5, 0.7],
            [0.0, 0.0], [1.0, 2.5], [-1.0, 2.1],
            [1.2, 1.2], [1.0, 1.0], [4.0, 0.0],
            [-1.0, -1.0]
        ])
        Y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        self.clf.fit(X, Y)
        self.assertIsNotNone(self.clf.clf)
        self.assertIsNotNone(self.clf.max_depth)

    def test_predict(self):
        X = np.array([
            [1.0, 2.0], [0.0, 0.0], [0.5, 0.7],
            [0.0, 0.0], [1.0, 2.5], [-1.0, 2.1],
            [1.2, 1.2], [1.0, 1.0], [4.0, 0.0],
            [-1.0, -1.0]
        ])
        Y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        self.clf.fit(X, Y)
        self.clf.predict(
            np.array([[1.0, 2.0], [1.0, 3.0], [-2.0, -2.0]])
        )

    def test_threshold(self):
        expected = 0.5
        actual = self.clf.threshold
        self.assertEqual(expected, actual)

    def test_get_id(self):
        expected = 'random_forest'
        actual = self.clf.get_id()
        self.assertEqual(expected, actual)

    def test_score(self):
        predicted_scores = np.array([0.0, 0.25, 1.0, 0.5, 0.9])
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

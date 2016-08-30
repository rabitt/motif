"""Test motif.features.cesium
"""
import unittest
import numpy as np

from motif.feature_extractors import cesium


def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2, atol=1e-7))


class TestCesiumFeatures(unittest.TestCase):

    def setUp(self):
        self.ftr = cesium.CesiumFeatures()

    def test_get_feature_vector(self):
        times = np.linspace(0, 1, 2000)
        freqs_hz = 440.0 * np.ones((2000, ))
        salience = 0.5 * np.ones((2000, ))
        sample_rate = 2000
        with self.assertRaises(NotImplementedError):
            self.ftr.get_feature_vector(
                times, freqs_hz, salience, sample_rate
            )

    def test_feature_names(self):
        expected = range(80)
        actual = self.ftr.feature_names
        self.assertEqual(expected, actual)

    def test_get_id(self):
        expected = 'cesium'
        actual = self.ftr.get_id()
        self.assertEqual(expected, actual)

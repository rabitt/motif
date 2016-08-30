"""Test motif.features.melodia
"""
import unittest
import numpy as np

from motif.feature_extractors import melodia


def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2, atol=1e-7))


class TestMelodiaFeatures(unittest.TestCase):

    def setUp(self):
        self.ftr = melodia.MelodiaFeatures()

    def test_ref_hz(self):
        expected = 55.0
        actual = self.ftr.ref_hz
        self.assertEqual(expected, actual)

    def test_get_feature_vector(self):
        times = np.linspace(0, 1, 2000)
        freqs_hz = 440.0 * np.ones((2000, ))
        salience = 0.5 * np.ones((2000, ))
        sample_rate = 2000
        actual = self.ftr.get_feature_vector(
            times, freqs_hz, salience, sample_rate
        )
        expected = np.array([
            0.0, 1.0, 1.0,
            3600.0, 0.0,
            0.5, 0.0,
            1000,
            0, 0.0, 0.0, 0.0
        ])
        self.assertTrue(array_equal(expected, actual))
        self.assertEqual(len(actual), len(self.ftr.feature_names))

    def test_get_feature_names(self):
        expected = [
            'onset',
            'offset',
            'duration',
            'pitch mean (cents)',
            'pitch stddev (cents)',
            'salience mean',
            'salience stdev',
            'salience total',
            'vibrato',
            'vibrato rate',
            'vibrato extent (cents)',
            'vibrato coverage'
        ]
        actual = self.ftr.feature_names
        self.assertEqual(expected, actual)

    def test_get_id(self):
        expected = 'melodia'
        actual = self.ftr.get_id()
        self.assertEqual(expected, actual)

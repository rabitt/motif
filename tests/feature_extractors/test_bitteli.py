"""Test motif.features.bitteli
"""
import unittest
import numpy as np

from motif.feature_extractors import bitteli


def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2, atol=1e-7))


class TestBitteliFeatures(unittest.TestCase):

    def setUp(self):
        self.ftr = bitteli.BitteliFeatures()

    def test_ref_hz(self):
        expected = 55.0
        actual = self.ftr.ref_hz
        self.assertEqual(expected, actual)

    def test_poly_degree(self):
        expected = 5
        actual = self.ftr.poly_degree
        self.assertEqual(expected, actual)

    def test_min_freq(self):
        expected = 3
        actual = self.ftr.min_freq
        self.assertEqual(expected, actual)

    def test_max_freq(self):
        expected = 30
        actual = self.ftr.max_freq
        self.assertEqual(expected, actual)

    def test_freq_step(self):
        expected = 0.1
        actual = self.ftr.freq_step
        self.assertEqual(expected, actual)

    def test_vibrato_threshold(self):
        expected = 0.25
        actual = self.ftr.vibrato_threshold
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
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            3600.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
            0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
            0.0, 1.0, 1.0,
            3600.0, 0.0, 0.0, 0.0,
            0.5, 0.0, 0.0,
            1000, 0.0
        ])
        self.assertTrue(array_equal(expected, actual))
        self.assertEqual(len(actual), len(self.ftr.feature_names))

    def test_get_feature_names(self):
        expected = [
            'vibrato rate',
            'vibrato extent',
            'vibrato coverage',
            'vibrato coverage - beginning',
            'vibrato coverage - middle',
            'vibrato coverage - end',
            '0th polynomial coeff - freq',
            '1st polynomial coeff - freq',
            '2nd polynomial coeff - freq',
            '3rd polynomial coeff - freq',
            '4th polynomial coeff - freq',
            '5th polynomial coeff - freq',
            'polynomial fit residual - freq',
            'overall model fit residual - freq',
            '0th polynomial coeff - salience',
            '1st polynomial coeff - salience',
            '2nd polynomial coeff - salience',
            '3rd polynomial coeff - salience',
            '4th polynomial coeff - salience',
            '5th polynomial coeff - salience',
            'polynomial fit residual - salience',
            'onset',
            'offset',
            'duration',
            'pitch mean (cents)',
            'pitch stddev (cents)',
            'pitch range (cents)',
            'pitch total variation',
            'salience mean',
            'salience stdev',
            'salience range',
            'salience total',
            'salience total variation'
        ]
        actual = self.ftr.feature_names
        self.assertEqual(expected, actual)

    def test_get_id(self):
        expected = 'bitteli'
        actual = self.ftr.get_id()
        self.assertEqual(expected, actual)

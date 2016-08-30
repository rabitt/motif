"""Test motif.features.melodia
"""
import unittest
import numpy as np
import os

from motif import core
from motif.feature_extractors import melodia


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


AUDIO_FILE = relpath("../data/short.wav")


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

    def test_compute_all(self):
        index = np.array([0, 0, 1, 1, 1, 2])
        times = np.array([0.0, 0.1, 0.0, 0.1, 0.2, 0.5])
        freqs = np.array([440.0, 441.0, 50.0, 52.0, 55.0, 325.2])
        salience = np.array([0.2, 0.4, 0.5, 0.2, 0.4, 0.0])
        sample_rate = 10.0
        ctr = core.Contours(
            index, times, freqs, salience,
            sample_rate, AUDIO_FILE
        )
        expected = np.array([
            [0.0, 0.1, 0.1, 3601.96508, 1.96507922,
             0.6, 0.2, 1.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.2, -87.3694077, 67.7134674,
             0.733333333, .249443826, 2.20000000, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 3076.58848, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        actual = self.ftr.compute_all(ctr)
        self.assertTrue(array_equal(expected, actual))

"""Tests for contour_decoders/viterbi.py
"""
import unittest
import numpy as np

from motif import core
from motif.contour_decoders import maximum


def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2))


class TestMaxDeocder(unittest.TestCase):

    def setUp(self):
        self.dcd = maximum.MaxDecoder()
        self.index = np.array([0, 0, 1, 1, 1, 2])
        self.times = np.array([0.0, 0.1, 0.0, 0.1, 0.2, 0.5])
        self.freqs = np.array([440.0, 441.0, 50.0, 52.0, 55.0, 325.2])
        self.salience = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.sample_rate = 10.0
        self.ctr = core.Contours(
            self.index, self.times, self.freqs, self.salience,
            self.sample_rate, audio_duration=0.5
        )
        self.Y1 = np.array([0.9, 0.7, 0.1])
        self.Y2 = np.array([0.9, 0.1, 0.8])

    def test_decode(self):
        expected_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        expected_freqs = np.array([440.0, 441.0, 55.0, 0.0, 0.0, 325.2])
        actual_times, actual_freqs = self.dcd.decode(self.ctr, self.Y1)
        self.assertTrue(np.allclose(expected_times, actual_times))
        self.assertTrue(np.allclose(expected_freqs, expected_freqs))

    def test_decode2(self):
        expected_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        expected_freqs = np.array([50.0, 52.0, 55.0, 0.0, 0.0, 325.2])
        actual_times, actual_freqs = self.dcd.decode(self.ctr, self.Y2)
        self.assertTrue(np.allclose(expected_times, actual_times))
        self.assertTrue(np.allclose(expected_freqs, expected_freqs))

    def test_get_id(self):
        expected = 'maximum'
        actual = self.dcd.get_id()
        self.assertEqual(expected, actual)

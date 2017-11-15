"""Tests for motif/contour_extractors/peak_stream.py
"""
import unittest
import os
import numpy as np

from motif.contour_extractors import peak_stream
from motif.core import Contours


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


AUDIO_FILE = relpath("../data/short.wav")
SKIP_CONDITION = (not peak_stream.BINARY_AVAILABLE)


class TestPeakStream(unittest.TestCase):

    def setUp(self):
        self.etr = peak_stream.PeakStream(preprocess=False)

    def test_max_freq(self):
        expected = 3000.0
        actual = self.etr.max_freq
        self.assertEqual(expected, actual)

    def test_hop_length(self):
        expected = 128
        actual = self.etr.hop_length
        self.assertEqual(expected, actual)

    def test_win_length(self):
        expected = 2048
        actual = self.etr.win_length
        self.assertEqual(expected, actual)

    def test_n_fft(self):
        expected = 8192
        actual = self.etr.n_fft
        self.assertEqual(expected, actual)

    def test_h_range(self):
        expected = [1, 2, 3, 4, 5]
        actual = self.etr.h_range
        self.assertEqual(expected, actual)

    def test_h_weights(self):
        expected = [1, 0.5, 0.25, 0.25, 0.25]
        actual = self.etr.h_weights
        self.assertEqual(expected, actual)

    def test_pitch_cont(self):
        expected = 80
        actual = self.etr.pitch_cont
        self.assertEqual(expected, actual)

    def test_max_gap(self):
        expected = 0.01
        actual = self.etr.max_gap
        self.assertEqual(expected, actual)

    def test_amp_thresh(self):
        expected = 0.9
        actual = self.etr.amp_thresh
        self.assertEqual(expected, actual)

    def test_dev_thresh(self):
        expected = 0.9
        actual = self.etr.dev_thresh
        self.assertEqual(expected, actual)

    def test_preprocess(self):
        expected = False
        actual = self.etr.preprocess
        self.assertEqual(expected, actual)

    def test_use_salamon_salience(self):
        expected = False
        actual = self.etr.use_salamon_salience
        self.assertEqual(expected, actual)

    def test_n_gap(self):
        expected = 3.4453125
        actual = self.etr.n_gap
        self.assertEqual(expected, actual)

    def test_audio_sample_rate(self):
        expected = 44100.0
        actual = self.etr.audio_samplerate
        self.assertEqual(expected, actual)

    def test_sample_rate(self):
        expected = 344.53125
        actual = self.etr.sample_rate
        self.assertEqual(expected, actual)

    def test_min_contour_len(self):
        expected = 0.1
        actual = self.etr.min_contour_len
        self.assertAlmostEqual(expected, actual)

    def test_get_id(self):
        expected = 'peak_stream'
        actual = self.etr.get_id()
        self.assertEqual(expected, actual)

    def test_compute_contours(self):
        ctr = self.etr.compute_contours(AUDIO_FILE)
        self.assertTrue(isinstance(ctr, Contours))

    def test_failed_compute_contours(self):
        with self.assertRaises(IOError):
            self.etr.compute_contours('does/not/exist.wav')

    def test_compute_salience(self):
        x = np.arange(0, self.etr.audio_samplerate) / self.etr.audio_samplerate
        y = np.sin(2.0 * np.pi * 440.0 * x)
        actual_times, actual_freqs, actual_S = self.etr._compute_salience(
            y, self.etr.audio_samplerate
        )
        expected_n_times = 345
        expected_n_freqs = 557
        self.assertEqual((expected_n_times, ), actual_times.shape)
        self.assertEqual((expected_n_freqs, ), actual_freqs.shape)
        self.assertEqual((expected_n_freqs, expected_n_times), actual_S.shape)

    @unittest.skipIf(SKIP_CONDITION, "salience binary not available")
    def test_compute_salience_salamon(self):
        (actual_times,
         actual_freqs,
         actual_S) = self.etr._compute_salience_salamon(AUDIO_FILE)
        expected_n_times = 1138
        expected_n_freqs = 601
        self.assertEqual((expected_n_times, ), actual_times.shape)
        self.assertEqual((expected_n_freqs, ), actual_freqs.shape)
        self.assertEqual((expected_n_freqs, expected_n_times), actual_S.shape)

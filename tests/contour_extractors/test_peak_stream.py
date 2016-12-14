"""Tests for motif/extract/hll.py
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


class TestPeakStreamHelper(unittest.TestCase):

    def setUp(self):
        self.S = np.array([
            [0, 0, 0],
            [1, 0, 5],
            [0, 0.002, 1],
            [0.1, 0, 0],
            [0, 0, 0]
        ])
        self.times = np.array([0.0, 0.5, 1.0])
        self.freqs = np.array([10., 100., 150., 200., 300.])
        self.amp_thresh = 0.9
        self.dev_thresh = 0.9
        self.n_gap = 3.234
        self.pitch_cont = 80
        self.psh = peak_stream.PeakStreamHelper(
            self.S, self.times, self.freqs, self.amp_thresh, self.dev_thresh,
            self.n_gap, self.pitch_cont
        )

    def test_S(self):
        expected = np.array([
            [0, 0, 0],
            [1, 0, 5],
            [0, 0.002, 1],
            [0.1, 0, 0],
            [0, 0, 0]
        ])
        actual = self.psh.S
        self.assertTrue(np.allclose(expected, actual))

    def test_S_norm(self):
        expected = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 1.0, 0.2],
            [0.1, 0, 0],
            [0, 0, 0]
        ])
        actual = self.psh.S_norm
        self.assertTrue(np.allclose(expected, actual))

    def test_times(self):
        expected = np.array([0.0, 0.5, 1.0])
        actual = self.psh.times
        self.assertTrue(np.allclose(expected, actual))

    def test_freqs(self):
        expected = np.array([
            0., 3986.31371386, 4688.26871473, 5186.31371386, 5888.26871473
        ])
        actual = self.psh.freqs
        self.assertTrue(np.allclose(expected, actual))

    def test_amp_thresh(self):
        expected = 0.9
        actual = self.psh.amp_thresh
        self.assertEqual(expected, actual)

    def test_dev_thresh(self):
        expected = 0.9
        actual = self.psh.dev_thresh
        self.assertEqual(expected, actual)

    def test_n_gap(self):
        expected = 3.234
        actual = self.psh.n_gap
        self.assertEqual(expected, actual)

    def test_pitch_cont(self):
        expected = 80
        actual = self.psh.pitch_cont
        self.assertEqual(expected, actual)

    def test_n_peaks(self):
        expected = 4
        actual = self.psh.n_peaks
        self.assertEqual(expected, actual)

    def test_peak_index(self):
        expected = np.array([0, 1, 2, 3])
        actual = self.psh.peak_index
        self.assertTrue(np.allclose(expected, actual))

    def test_peak_time_index(self):
        expected = np.array([0, 2, 1, 0])
        actual = self.psh.peak_time_idx
        self.assertTrue(np.allclose(expected, actual))

    def test_first_peak_time_idx(self):
        expected = 0
        actual = self.psh.first_peak_time_idx
        self.assertEqual(expected, actual)

    def test_last_peak_time_idx(self):
        expected = 2
        actual = self.psh.last_peak_time_idx
        self.assertEqual(expected, actual)

    def test_frame_dict(self):
        expected = {
            0: [0, 3],
            1: [2],
            2: [1]
        }
        actual = self.psh.frame_dict
        self.assertEqual(expected.keys(), actual.keys())
        for k in actual.keys():
            self.assertTrue(np.allclose(expected[k], actual[k]))

    def test_peak_freqs(self):
        expected = np.array([
            3986.31371386, 3986.31371386, 4688.26871473, 5186.31371386
        ])
        actual = self.psh.peak_freqs
        self.assertTrue(np.allclose(expected, actual))

    def test_peak_amps(self):
        expected = np.array([1., 5., 0.002, 0.1])
        actual = self.psh.peak_amps
        self.assertTrue(np.allclose(expected, actual))

    def test_peak_amps_norm(self):
        expected = np.array([1., 1., 1., 0.1])
        actual = self.psh.peak_amps_norm
        self.assertTrue(np.allclose(expected, actual))

    def test_good_peaks(self):
        expected = set([0, 1])
        actual = self.psh.good_peaks
        self.assertEqual(expected, actual)

    def test_bad_peaks(self):
        expected = set([2, 3])
        actual = self.psh.bad_peaks
        self.assertEqual(expected, actual)

    def test_good_peaks_sorted(self):
        expected = np.array([1, 0])
        actual = self.psh.good_peaks_sorted
        self.assertTrue(np.allclose(expected, actual))

    def test_good_peaks_sorted_index(self):
        expected = {0: 1, 1: 0}
        actual = self.psh.good_peaks_sorted_index
        self.assertEqual(expected, actual)

    def test_good_peaks_sorted_avail(self):
        expected = np.array([True, True])
        actual = self.psh.good_peaks_sorted_avail
        self.assertTrue(np.allclose(expected, actual))

    def test_n_good_peaks(self):
        expected = 2
        actual = self.psh.n_good_peaks
        self.assertTrue(np.allclose(expected, actual))

    def test_smallest_good_peak_idx(self):
        expected = 0
        actual = self.psh.smallest_good_peak_idx

    def test_get_largest_peak(self):
        S = np.array([
            [0, 0, 0, 0],
            [0, 0.002, 0, 0],
            [1, 0, 5, 0],
            [0, 0.3, 0.1, 0],
            [0.1, 0, 0.2, 0],
            [0, 0.5, 0, 0.2],
            [0, 0, 0, 0]
        ])
        times = np.array([0.05, 0.1, 0.15, 0.2])
        freqs = np.array([97.0, 100.0, 103.0, 105.0, 107.0, 109.0, 112.0])
        psh = peak_stream.PeakStreamHelper(S, times, freqs, 0.9, 0.9, 3.456, 80)

        actual = psh.get_largest_peak()
        expected = 2
        self.assertEqual(expected, actual)

    def test_update_largest_peak_list(self):
        S = np.array([
            [0, 0, 0, 0],
            [0, 0.002, 0, 0],
            [1, 0, 5, 0],
            [0, 0.3, 0.1, 0],
            [0.1, 0, 0.2, 0],
            [0, 0.5, 0, 0.2],
            [0, 0, 0, 0]
        ])
        times = np.array([0.05, 0.1, 0.15, 0.2])
        freqs = np.array([97.0, 100.0, 103.0, 105.0, 107.0, 109.0, 112.0])
        psh = peak_stream.PeakStreamHelper(S, times, freqs, 0.9, 0.9, 3.456, 80)

        expected_avail = np.array([True, True, True, True])
        actual_avail = psh.good_peaks_sorted_avail
        self.assertTrue(np.allclose(expected_avail, actual_avail))

        expected_smallest_idx = 0
        actual_smallest_idx = psh.smallest_good_peak_idx
        self.assertEqual(expected_smallest_idx, actual_smallest_idx)

        psh.update_largest_peak_list(1)

        expected_avail = np.array([True, False, True, True])
        actual_avail = psh.good_peaks_sorted_avail
        self.assertTrue(np.allclose(expected_avail, actual_avail))

        expected_smallest_idx = 0
        actual_smallest_idx = psh.smallest_good_peak_idx
        self.assertEqual(expected_smallest_idx, actual_smallest_idx)

        psh.update_largest_peak_list(2)

        expected_avail = np.array([False, False, True, True])
        actual_avail = psh.good_peaks_sorted_avail
        self.assertTrue(np.allclose(expected_avail, actual_avail))

        expected_smallest_idx = 2
        actual_smallest_idx = psh.smallest_good_peak_idx
        self.assertEqual(expected_smallest_idx, actual_smallest_idx)

    def test_get_closest_peak(self):
        S = np.array([
            [0, 0, 0, 0],
            [0, 0.002, 0, 0],
            [1, 0, 5, 0],
            [0, 0.3, 0.1, 0],
            [0.1, 0, 0.2, 0],
            [0, 0.5, 0, 0.2],
            [0, 0, 0, 0]
        ])
        times = np.array([0.05, 0.1, 0.15, 0.2])
        freqs = np.array([97.0, 100.0, 103.0, 105.0, 107.0, 109.0, 112.0])
        psh = peak_stream.PeakStreamHelper(S, times, freqs, 0.9, 0.9, 3.456, 80)

        actual = psh.get_closest_peak(237.2, [2, 4, 5])
        expected = 2
        self.assertEqual(expected, actual)

    def test_get_peak_candidates(self):
        S = np.array([
            [0, 0, 0, 0],
            [0, 0.002, 0, 0],
            [1, 0, 5, 0],
            [0, 0.3, 0.1, 0],
            [0.1, 0, 0.2, 0],
            [0, 0.5, 0, 0.2],
            [0, 0, 0, 0]
        ])
        times = np.array([0.05, 0.1, 0.15, 0.2])
        freqs = np.array([97.0, 100.0, 103.0, 105.0, 107.0, 109.0, 112.0])
        psh = peak_stream.PeakStreamHelper(S, times, freqs, 0.9, 0.9, 3.456, 80)

        frame_idx = 0
        f0_val = 4000.0
        expected_cands = [1]
        expected_from_good = True
        actual_cands, actual_from_good = psh.get_peak_candidates(
            frame_idx, f0_val
        )
        self.assertEqual(expected_cands, actual_cands)
        self.assertEqual(expected_from_good, actual_from_good)

    def test_get_peak_candidates2(self):
        S = np.array([
            [0, 0, 0, 0],
            [0, 0.002, 0, 0],
            [1, 0, 5, 0],
            [0, 0.3, 0.1, 0],
            [0.1, 0, 0.2, 0],
            [0, 0.5, 0, 0.002],
            [0, 0, 0, 0]
        ])
        times = np.array([0.05, 0.1, 0.15, 0.2])
        freqs = np.array([97.0, 100.0, 103.0, 105.0, 107.0, 109.0, 112.0])
        psh = peak_stream.PeakStreamHelper(S, times, freqs, 0.9, 0.9, 3.456, 80)

        frame_idx = 3
        f0_val = 4125.5
        expected_cands = [7]
        expected_from_good = False
        actual_cands, actual_from_good = psh.get_peak_candidates(
            frame_idx, f0_val
        )
        self.assertEqual(expected_cands, actual_cands)
        self.assertEqual(expected_from_good, actual_from_good)

    def test_get_peak_candidates3(self):
        S = np.array([
            [0, 0, 0, 0],
            [0, 0.002, 0, 0],
            [1, 0, 5, 0],
            [0, 0.3, 0.1, 0],
            [0.1, 0, 0.2, 0],
            [0, 0.5, 0, 0.002],
            [0, 0, 0, 0]
        ])
        times = np.array([0.05, 0.1, 0.15, 0.2])
        freqs = np.array([97.0, 100.0, 103.0, 105.0, 107.0, 109.0, 112.0])
        psh = peak_stream.PeakStreamHelper(S, times, freqs, 0.9, 0.9, 3.456, 80)

        frame_idx = 3
        f0_val = 0
        expected_cands = None
        expected_from_good = None
        actual_cands, actual_from_good = psh.get_peak_candidates(
            frame_idx, f0_val
        )
        self.assertEqual(expected_cands, actual_cands)
        self.assertEqual(expected_from_good, actual_from_good)

    def test_get_contour(self):
        S = np.array([
            [0, 0, 0, 0],
            [0, 0.002, 0, 0],
            [1, 0, 5, 0],
            [0, 0.3, 0.1, 0],
            [0.1, 0, 0.2, 0],
            [0, 0.5, 0, 0.002],
            [0, 0, 0, 0]
        ])
        times = np.array([0.05, 0.1, 0.15, 0.2])
        freqs = np.array([97.0, 100.0, 103.0, 105.0, 107.0, 109.0, 112.0])
        psh = peak_stream.PeakStreamHelper(S, times, freqs, 0.9, 0.9, 3.456, 80)

        psh.get_contour()
        actual_contour_idx = psh.contour_idx
        expected_contour_idx = [2, 3, 1]
        self.assertEqual(expected_contour_idx, actual_contour_idx)

        actual_c_len = psh.c_len
        expected_c_len = [3]
        self.assertEqual(expected_c_len, actual_c_len)

        psh.get_contour()
        actual_contour_idx = psh.contour_idx
        expected_contour_idx = [2, 3, 1, 6, 5, 7, 4]
        self.assertEqual(expected_contour_idx, actual_contour_idx)

        actual_c_len = psh.c_len
        expected_c_len = [3, 4]
        self.assertEqual(expected_c_len, actual_c_len)

    def test_peak_streaming(self):
        S = np.array([
            [0, 0, 0, 0],
            [0, 0.002, 0, 0],
            [1, 0, 5, 0],
            [0, 0.3, 0.1, 0],
            [0.1, 0, 0.2, 0],
            [0, 0.5, 0, 0.2],
            [0, 0, 0, 0]
        ])
        times = np.array([0.05, 0.1, 0.15, 0.2])
        freqs = np.array([97.0, 100.0, 103.0, 105.0, 107.0, 109.0, 112.0])

        psh = peak_stream.PeakStreamHelper(S, times, freqs, 0.9, 0.9, 3.456, 80)

        expected_c_numbers = np.array([0, 0, 0, 1, 1, 1, 1])
        expected_c_times = np.array([0.15, 0.1, 0.05, 0.1, 0.15, 0.2, 0.05])
        expected_c_freqs = np.array([103., 105., 103., 109., 107., 109., 107.])
        expected_c_sal = np.array([5, 0.3, 1.0, 0.5, 0.2, 0.2, 0.1])

        (actual_c_numbers,
         actual_c_times,
         actual_c_freqs,
         actual_c_sal) = psh.peak_streaming()

        self.assertTrue(np.allclose(expected_c_numbers, actual_c_numbers))
        self.assertTrue(np.allclose(expected_c_times, actual_c_times))
        self.assertTrue(np.allclose(expected_c_freqs, actual_c_freqs))
        self.assertTrue(np.allclose(expected_c_sal, actual_c_sal))


class TestPeakStreamHelperNoPeaks(unittest.TestCase):

    def setUp(self):
        self.S = np.array([
            [0., 0., 0.],
            [1., 0., 1.],
            [2., 0., 1.],
            [3., 0., 1.],
            [4., 0., 1.]
        ])
        self.times = np.array([0.0, 0.5, 1.0])
        self.freqs = np.array([10., 100., 150., 200., 300.])
        self.amp_thresh = 0.9
        self.dev_thresh = 0.9
        self.n_gap = 3.234
        self.pitch_cont = 80
        self.psh = peak_stream.PeakStreamHelper(
            self.S, self.times, self.freqs, self.amp_thresh, self.dev_thresh,
            self.n_gap, self.pitch_cont
        )

    def test_S(self):
        expected = np.array([
            [0., 0., 0.],
            [1., 0., 1.],
            [2., 0., 1.],
            [3., 0., 1.],
            [4., 0., 1.]
        ])
        actual = self.psh.S
        self.assertTrue(np.allclose(expected, actual))

    def test_S_norm(self):
        expected = np.array([
            [0, 0, 0],
            [0.25, 0, 1],
            [0.5, 0, 1],
            [0.75, 0, 1],
            [1, 0, 1]
        ])
        actual = self.psh.S_norm
        self.assertTrue(np.allclose(expected, actual))

    def test_times(self):
        expected = np.array([0.0, 0.5, 1.0])
        actual = self.psh.times
        self.assertTrue(np.allclose(expected, actual))

    def test_freqs(self):
        expected = np.array([
            0., 3986.31371386, 4688.26871473, 5186.31371386, 5888.26871473
        ])
        actual = self.psh.freqs
        self.assertTrue(np.allclose(expected, actual))

    def test_amp_thresh(self):
        expected = 0.9
        actual = self.psh.amp_thresh
        self.assertEqual(expected, actual)

    def test_dev_thresh(self):
        expected = 0.9
        actual = self.psh.dev_thresh
        self.assertEqual(expected, actual)

    def test_n_gap(self):
        expected = 3.234
        actual = self.psh.n_gap
        self.assertEqual(expected, actual)

    def test_pitch_cont(self):
        expected = 80
        actual = self.psh.pitch_cont
        self.assertEqual(expected, actual)

    def test_n_peaks(self):
        expected = 0
        actual = self.psh.n_peaks
        self.assertEqual(expected, actual)

    def test_peak_index(self):
        expected = np.array([])
        actual = self.psh.peak_index
        self.assertTrue(np.allclose(expected, actual))

    def test_peak_time_index(self):
        expected = np.array([])
        actual = self.psh.peak_time_idx
        self.assertTrue(np.allclose(expected, actual))

    def test_first_peak_time_idx(self):
        expected = None
        actual = self.psh.first_peak_time_idx
        self.assertEqual(expected, actual)

    def test_last_peak_time_idx(self):
        expected = None
        actual = self.psh.last_peak_time_idx
        self.assertEqual(expected, actual)

    def test_frame_dict(self):
        expected = {}
        actual = self.psh.frame_dict
        self.assertEqual(expected, actual)

    def test_peak_freqs(self):
        expected = np.array([])
        actual = self.psh.peak_freqs
        self.assertTrue(np.allclose(expected, actual))

    def test_peak_amps(self):
        expected = np.array([])
        actual = self.psh.peak_amps
        self.assertTrue(np.allclose(expected, actual))

    def test_peak_amps_norm(self):
        expected = np.array([])
        actual = self.psh.peak_amps_norm
        self.assertTrue(np.allclose(expected, actual))

    def test_good_peaks(self):
        expected = set()
        actual = self.psh.good_peaks
        self.assertEqual(expected, actual)

    def test_bad_peaks(self):
        expected = set()
        actual = self.psh.bad_peaks
        self.assertEqual(expected, actual)

    def test_good_peaks_sorted(self):
        expected = np.array([])
        actual = self.psh.good_peaks_sorted
        self.assertTrue(np.allclose(expected, actual))

    def test_good_peaks_sorted_index(self):
        expected = {}
        actual = self.psh.good_peaks_sorted_index
        self.assertEqual(expected, actual)

    def test_good_peaks_sorted_avail(self):
        expected = np.array([])
        actual = self.psh.good_peaks_sorted_avail
        self.assertTrue(np.allclose(expected, actual))

    def test_n_good_peaks(self):
        expected = 0
        actual = self.psh.n_good_peaks
        self.assertTrue(np.allclose(expected, actual))

    def test_smallest_good_peak_idx(self):
        expected = 0
        actual = self.psh.smallest_good_peak_idx
        self.assertEqual(expected, actual)

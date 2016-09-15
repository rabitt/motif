"""Tests for features.utils
"""
from __future__ import print_function
import unittest
import numpy as np

from motif.feature_extractors import utils


def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2))


class TestHzToCents(unittest.TestCase):

    def test_freq_series(self):
        freqs_hz = np.array([32.0, 64.0, 128.0])
        expected = np.array([0.0, 1200.0, 2400.0])
        actual = utils.hz_to_cents(freqs_hz)
        self.assertTrue(array_equal(expected, actual))


class TestGetContourOnset(unittest.TestCase):

    def test_get_contour_onset(self):
        expected = np.array([1.2])
        actual = utils.get_contour_onset(np.array([1.2, 2.2, 3.1]))
        self.assertTrue(array_equal(expected, actual))


class TestGetContourOffset(unittest.TestCase):

    def test_get_contour_offset(self):
        expected = np.array([3.1])
        actual = utils.get_contour_offset(np.array([1.2, 2.2, 3.1]))
        self.assertTrue(array_equal(expected, actual))


class TestGetContourDuration(unittest.TestCase):

    def test_get_contour_duration(self):
        expected = np.array([2.1])
        actual = utils.get_contour_duration(np.array([1.2, 2.2, 3.3]))
        self.assertTrue(array_equal(expected, actual))


class TestGetMean(unittest.TestCase):

    def test_get_mean(self):
        expected = np.array([2.2])
        actual = utils.get_mean(np.array([1.2, 2.2, 3.2]))
        self.assertTrue(array_equal(expected, actual))


class TestGetStd(unittest.TestCase):

    def test_get_std(self):
        expected = np.array([0.70710678])
        actual = utils.get_std(np.array([1.2, 2.2, 2.2, 3.2]))
        self.assertTrue(array_equal(expected, actual))


class TestGetSum(unittest.TestCase):

    def test_get_sum(self):
        expected = np.array([6.6])
        actual = utils.get_sum(np.array([1.2, 2.2, 3.2]))
        self.assertTrue(array_equal(expected, actual))


class TestGetRange(unittest.TestCase):

    def test_get_range(self):
        expected = np.array([2.0])
        actual = utils.get_range(np.array([1.2, 2.2, 3.2]))
        self.assertTrue(array_equal(expected, actual))


class TestTotalVariation(unittest.TestCase):

    def test_flat(self):
        signal = np.array([0.0, 0.0, 0.0, 0.0])
        expected = np.array([0.0])
        actual = utils.get_total_variation(signal)
        self.assertEqual(expected, actual)

    def test_unit_step(self):
        signal = np.array([0.0, 0.0, 1.0, 1.0])
        expected = np.array([1.0])
        actual = utils.get_total_variation(signal)
        self.assertEqual(expected, actual)

    def test_unit_step_reverse(self):
        signal = np.array([1.0, 1.0, 0.0, 0.0])
        expected = np.array([1.0])
        actual = utils.get_total_variation(signal)
        self.assertEqual(expected, actual)


class TestGetPolynomialFitFeatures(unittest.TestCase):

    def test_poly_fit_features(self):
        times = np.array([0.0, 1.0])
        signal = np.array([0.0, 1.0])
        actual = utils.get_polynomial_fit_features(times, signal, n_deg=1)
        expected = np.array([0.0, 1.0, 0.0])
        self.assertTrue(array_equal(expected, actual))

    def test_norm(self):
        times = np.array([0.0, 1.0])
        signal = np.array([0.5, 0.9])
        actual = utils.get_polynomial_fit_features(
            times, signal, n_deg=1, norm=True
        )
        expected = np.array([0.0, 1.0, 0.0])
        self.assertTrue(array_equal(expected, actual))


class TestFitPoly(unittest.TestCase):

    def test_line(self):
        signal = np.array([0.0, 1.0])
        expected_coeffs = np.array([0.0, 1.0])
        expected_diff = 0.0
        expected_approx = signal
        actual_coeffs, actual_approx, actual_diff = utils._fit_poly(1, signal)
        self.assertTrue(array_equal(expected_coeffs, actual_coeffs))
        self.assertTrue(array_equal(expected_approx, actual_approx))
        self.assertTrue(array_equal(expected_diff, actual_diff))

    def test_cubic(self):
        signal = np.array([0, 1.0 / 27.0, 8.0 / 27.0, 1.0])
        expected_coeffs = np.array([0.0, 0.0, 0.0, 1.0])
        expected_diff = 0.0
        expected_approx = np.power(np.linspace(0, 1, 4), 3)
        actual_coeffs, actual_approx, actual_diff = utils._fit_poly(3, signal)
        self.assertTrue(array_equal(expected_coeffs, actual_coeffs))
        self.assertTrue(array_equal(expected_approx, actual_approx))
        self.assertTrue(array_equal(expected_diff, actual_diff))

    def test_grid(self):
        grid = np.array([-2, -1, 0, 1, 2])
        signal = np.array([4, 1, 0, 1, 4])
        expected_coeffs = np.array([0.0, 0.0, 1.0])
        expected_diff = 0.0
        expected_approx = np.power(grid, 2)
        actual_coeffs, actual_approx, actual_diff = utils._fit_poly(
            2, signal, grid=grid
        )
        self.assertTrue(array_equal(expected_coeffs, actual_coeffs))
        self.assertTrue(array_equal(expected_approx, actual_approx))
        self.assertTrue(array_equal(expected_diff, actual_diff))

    def test_too_short(self):
        with self.assertRaises(ValueError):
            utils._fit_poly(5, np.array([0.0, 1.0]))


class TestFitNormalizedCosine(unittest.TestCase):

    def test_perfect_cosine(self):
        x = np.linspace(0, 1, 256)
        y = np.cos(2.0 * np.pi * 8.3 * x - 1.2)
        expected_freq = 8.3
        expected_phase = 1.2
        actual_freq, actual_phase = utils._fit_normalized_cosine(
            x, y, min_freq=3, max_freq=30, step=0.1
        )
        self.assertAlmostEqual(expected_freq, actual_freq)
        self.assertAlmostEqual(expected_phase, actual_phase, places=1)

    def test_perfect_cosine_2(self):
        x = np.linspace(0, 1, 256)
        y = np.cos(2.0 * np.pi * 12.9 * x)
        expected_freq = 12.9
        expected_phase = 0.0
        actual_freq, actual_phase = utils._fit_normalized_cosine(
            x, y, min_freq=10, max_freq=15, step=0.1
        )
        self.assertAlmostEqual(expected_freq, actual_freq)
        self.assertAlmostEqual(expected_phase, actual_phase, places=1)

    def test_line(self):
        x = np.linspace(0, 1, 256)
        y = (0.0 * x) + 0.0
        expected_freq = 0.0
        expected_phase = 0.0
        actual_freq, actual_phase = utils._fit_normalized_cosine(
            x, y, min_freq=10, max_freq=15, step=0.1
        )
        self.assertEqual(expected_freq, actual_freq)
        self.assertEqual(expected_phase, actual_phase)


class TestComputeCoverageArray(unittest.TestCase):

    def test_uniform(self):
        y_sinfit_diff = 0.6 * np.ones((10, ))
        cycle_length = 2
        vibrato_threshold = 0.8
        expected = np.ones((10, )).astype(bool)
        actual = utils._compute_coverage_array(
            y_sinfit_diff, cycle_length, vibrato_threshold
        )
        self.assertTrue(array_equal(expected, actual))

    def test_nonuniform(self):
        y_sinfit_diff = np.array([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.2, 0.4,
            0.0, 0.8
        ])
        cycle_length = 3
        vibrato_threshold = 0.3
        expected = np.array([
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            0, 0
        ]).astype(bool)
        actual = utils._compute_coverage_array(
            y_sinfit_diff, cycle_length, vibrato_threshold
        )
        self.assertTrue(array_equal(expected, actual))

    def test_too_short(self):
        y_sinfit_diff = 0.6 * np.ones((10, ))
        cycle_length = 4
        vibrato_threshold = 0.8
        expected = np.zeros((10, )).astype(bool)
        actual = utils._compute_coverage_array(
            y_sinfit_diff, cycle_length, vibrato_threshold
        )
        self.assertTrue(array_equal(expected, actual))


class TestGetContourShapeFeatures(unittest.TestCase):

    def test_line(self):
        sample_rate = 2000
        times = np.linspace(0, 1, sample_rate)
        freqs = 1.2 * times + 440.0
        expected = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            440.0, 1.2, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ])
        actual = utils.get_contour_shape_features(times, freqs, sample_rate)
        self.assertTrue(array_equal(expected, actual))

    def test_flat_line(self):
        sample_rate = 2000
        times = np.linspace(0, 1, sample_rate)
        freqs = 0.0 * times + 440.0
        expected = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            440.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ])
        actual = utils.get_contour_shape_features(times, freqs, sample_rate)
        self.assertTrue(array_equal(expected, actual))

    def test_zeros(self):
        sample_rate = 2000
        times = np.linspace(0, 1, sample_rate)
        freqs = 0.0 * times
        expected = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ])
        actual = utils.get_contour_shape_features(times, freqs, sample_rate)
        self.assertTrue(array_equal(expected, actual))

    def test_line_with_vib(self):
        sample_rate = 2000
        times = np.linspace(0, 1, sample_rate)
        freqs = (
            (1.2 * times + 440.0) + 7.3 * np.cos(2.0 * np.pi * 12.0 * times)
        )
        expected = np.array([
            12.0, 7.296015, 1.0, 1.0, 1.0, 1.0,
            4.40586339e+02, -9.14578468e+00,
            4.45852838e+01, -6.84789982e+01, 3.42394991e+01,
            1.40143396e-11, 1.15390225e-01, 3.77435044e-03
        ])
        actual = utils.get_contour_shape_features(times, freqs, sample_rate)
        self.assertTrue(array_equal(expected, actual))


class TestVibratoEssentia(unittest.TestCase):

    def test_flat(self):
        freqs_cents = np.array([440.0, 440.0, 440.0, 440.0])
        actual = utils.vibrato_essentia(freqs_cents, 44100)
        expected = np.array([0, 0.0, 0.0, 0.0])
        self.assertTrue(array_equal(expected, actual))

    def test_pure_sine(self):
        sample_rate = 2000
        grid = np.linspace(0, 1, sample_rate)
        freqs_cents = 50.0*np.sin(2.0*np.pi*12.0*grid) + 100.0
        actual = utils.vibrato_essentia(freqs_cents, sample_rate)
        expected = np.array([1, 10.21208791208791, 99.999587722367053, 1.0])
        self.assertTrue(array_equal(expected, actual))

    def test_half_sine(self):
        sample_rate = 2000
        grid = np.linspace(0, 1, sample_rate)
        freqs_cents = 50.0*np.sin(2.0*np.pi*12.0*grid) + 100.0
        freqs_cents[0:1000] = 100.0
        actual = utils.vibrato_essentia(freqs_cents, sample_rate)
        expected = np.array([
            1, 9.34, 94.408366757784748, 0.76769230769230767
        ])
        self.assertAlmostEqual(expected[0], actual[0], places=1)
        self.assertAlmostEqual(expected[1], actual[1], places=1)
        self.assertAlmostEqual(expected[2], actual[2], places=1)
        self.assertAlmostEqual(expected[3], actual[3], places=1)

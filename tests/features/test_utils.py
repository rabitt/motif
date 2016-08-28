import unittest
import os
import numpy as np

from motif.features import utils

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
        expected = 0.0
        actual = utils.get_total_variation(signal)
        self.assertEqual(expected, actual)

    def test_unit_step(self):
        signal = np.array([0.0, 0.0, 1.0, 1.0])
        expected = 1.0
        actual = utils.get_total_variation(signal)
        self.assertEqual(expected, actual)

    def test_unit_step_reverse(self):
        signal = np.array([1.0, 1.0, 0.0, 0.0])
        expected = 1.0
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
        signal = np.array([0, 1.0/27.0, 8.0/27.0, 1.0])
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


class TestComputeCoverageArray(unittest.TestCase):

    def test_uniform(self):
        y_sinfit_diff = 0.6 * np.ones((10, ))
        n_intervals = 4
        n_points = 10
        vibrato_threshold = 0.8
        expected = np.ones((10, )).astype(bool)
        actual = utils._compute_coverage_array(
            y_sinfit_diff, n_intervals, n_points, vibrato_threshold
        )
        print expected
        print actual
        self.assertTrue(array_equal(expected, actual))

    def test_nonuniform(self):
        y_sinfit_diff = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.0, 0.8])
        n_intervals = 3
        n_points = 8
        vibrato_threshold = 0.2
        expected = np.array([1, 1, 1, 1, 1, 1, 0, 0]).astype(bool)
        actual = utils._compute_coverage_array(
            y_sinfit_diff, n_intervals, n_points, vibrato_threshold
        )
        print expected
        print actual
        self.assertTrue(array_equal(expected, actual))

# class TestContourFeatures(unittest.TestCase):

#     def setUp(self):
#         self.times = np.array([0.0, 0.1, 0.2, 0.3])
#         self.freqs_hz = np.array([440.0, 440.0, 440.0, 440.0])
#         self.salience = np.array([0.5, 0.5, 0.5, 0.5])

#     def test_init(self):
#         cf = utils.ContourFeatures(
#             self.times, self.freqs_hz, self.salience
#         )
#         actual_times = cf.times
#         expected_times = np.array([0.0, 0.1, 0.2, 0.3])
#         self.assertTrue(array_equal(expected_times, actual_times))

#         actual_sample_rate = cf.sample_rate
#         expected_sample_rate = 10
#         self.assertEqual(expected_sample_rate, actual_sample_rate)

#         actual_freqs_hz = cf.freqs_hz
#         expected_freqs_hz = np.array([440.0, 440.0, 440.0, 440.0])
#         self.assertTrue(array_equal(expected_freqs_hz, actual_freqs_hz))

#         actual_freqs_cents = cf.freqs_cents
#         expected_freqs_cents = np.array([
#             4537.6316562295915, 4537.6316562295915,
#             4537.6316562295915, 4537.6316562295915
#         ])
#         self.assertTrue(array_equal(expected_freqs_cents, actual_freqs_cents))

#         actual_salience = cf.salience
#         expected_salience = np.array([0.5, 0.5, 0.5, 0.5])
#         self.assertTrue(array_equal(expected_salience, actual_salience))

#     def test_get_freq_polynomial_coeffs(self):
#         cf = utils.ContourFeatures(
#             self.times, self.freqs_hz, self.salience
#         )
#         actual = cf.get_freq_polynomial_coeffs(
#             n_poly_degrees=1
#         )
#         expected = np.array([1.0, 0.0, 0.0])
#         self.assertTrue(array_equal(expected, actual))

#     def test_get_salience_polynomail_coeffs(self):
#         cf = utils.ContourFeatures(
#             self.times, self.freqs_hz, self.salience
#         )
#         actual = cf.get_salience_polynomial_coeffs(
#             n_poly_degrees=1
#         )
#         expected = np.array([1.0, 0.0, 0.0])
#         self.assertTrue(array_equal(expected, actual))

#     def test_get_vibrato_features(self):
#         cf = utils.ContourFeatures(
#             self.times, self.freqs_hz, self.salience
#         )
#         actual = cf.get_vibrato_features()
#         expected = np.array([0.0, 0.0, 0.0])
#         self.assertTrue(array_equal(expected, actual))








# class TestVibratoFeatures(unittest.TestCase):

#     def test_flat(self):
#         freqs_cents = np.array([440.0, 440.0, 440.0, 440.0])
#         expected_rate = 0.0
#         expected_extent = 0.0
#         expected_coverate = 0.0
#         actual = utils.vibrato_features(freqs_cents, 44100)
#         actual_rate = actual[0]
#         actual_extent = actual[1]
#         actual_coverage = actual[2]
#         self.assertEqual(expected_rate, actual_rate)
#         self.assertEqual(expected_extent, actual_extent)
#         self.assertEqual(expected_coverate, actual_coverage)

#     def test_pure_sine(self):
#         sample_rate = 2000
#         grid = np.linspace(0, 1, sample_rate)
#         freqs_cents = 50.0*np.sin(2.0*np.pi*12.0*grid) + 100.0
#         expected_rate = 10.21208791208791 #12.0
#         expected_extent = 99.999587722367053 #100.0
#         expected_coverage = 1.0
#         actual = utils.vibrato_features(freqs_cents, sample_rate)
#         actual_rate = actual[0]
#         actual_extent = actual[1]
#         actual_coverage = actual[2]
#         self.assertEqual(expected_rate, actual_rate)
#         self.assertEqual(expected_extent, actual_extent)
#         self.assertEqual(expected_coverage, actual_coverage)

#     def test_half_sine(self):
#         sample_rate = 2000
#         grid = np.linspace(0, 1, sample_rate)
#         freqs_cents = 50.0*np.sin(2.0*np.pi*12.0*grid) + 100.0
#         freqs_cents[0:sample_rate / 2] = 100.0
#         expected_rate = 9.3293730317778376 #12.0
#         expected_extent = 94.408366757784748 #100.0
#         expected_coverage = 0.76769230769230767 # 0.5
#         actual = utils.vibrato_features(freqs_cents, sample_rate)
#         actual_rate = actual[0]
#         actual_extent = actual[1]
#         actual_coverage = actual[2]
#         self.assertEqual(expected_rate, actual_rate)
#         self.assertEqual(expected_extent, actual_extent)
#         self.assertEqual(expected_coverage, actual_coverage)



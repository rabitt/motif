import unittest
import os
import numpy as np

from motif import features

def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2))


class TestContourFeatures(unittest.TestCase):

    def setUp(self):
        self.times = np.array([0.0, 0.1, 0.2, 0.3])
        self.freqs_hz = np.array([440.0, 440.0, 440.0, 440.0])
        self.salience = np.array([0.5, 0.5, 0.5, 0.5])

    def test_init(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        actual_times = cf.times
        expected_times = np.array([0.0, 0.1, 0.2, 0.3])
        self.assertTrue(array_equal(expected_times, actual_times))

        actual_sample_rate = cf.sample_rate
        expected_sample_rate = 10
        self.assertEqual(expected_sample_rate, actual_sample_rate)

        actual_freqs_hz = cf.freqs_hz
        expected_freqs_hz = np.array([440.0, 440.0, 440.0, 440.0])
        self.assertTrue(array_equal(expected_freqs_hz, actual_freqs_hz))

        actual_freqs_cents = cf.freqs_cents
        expected_freqs_cents = np.array([
            4537.6316562295915, 4537.6316562295915,
            4537.6316562295915, 4537.6316562295915
        ])
        self.assertTrue(array_equal(expected_freqs_cents, actual_freqs_cents))

        actual_salience = cf.salience
        expected_salience = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertTrue(array_equal(expected_salience, actual_salience))

    def test_get_freq_polynomial_coeffs(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        actual = cf.get_freq_polynomial_coeffs(
            n_poly_degrees=1
        )
        expected = np.array([1.0, 0.0, 0.0])
        self.assertTrue(array_equal(expected, actual))

    def test_get_salience_polynomail_coeffs(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        actual = cf.get_salience_polynomial_coeffs(
            n_poly_degrees=1
        )
        expected = np.array([1.0, 0.0, 0.0])
        self.assertTrue(array_equal(expected, actual))

    def test_get_vibrato_features(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        actual = cf.get_vibrato_features()
        expected = np.array([0.0, 0.0, 0.0])
        self.assertTrue(array_equal(expected, actual))

    def test_get_time_series_features(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        cf.get_time_series_features()

    def test_get_duration(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        actual = cf.get_duration()
        expected = 0.3
        self.assertEqual(expected, actual)

    def test_get_pitch_mean(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        actual = cf.get_pitch_mean()
        expected = 440.0
        self.assertEqual(expected, actual)

    def test_get_pitch_std(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        actual = cf.get_pitch_std()
        expected = 0.0
        self.assertEqual(expected, actual)

    def test_get_pitch_range(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        actual = cf.get_pitch_range()
        expected = 0.0
        self.assertEqual(expected, actual)

    def test_get_freq_total_variation(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        actual = cf.get_freq_total_variation()
        expected = 0.0
        self.assertEqual(expected, actual)

    def test_get_sal_total_variation(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        actual = cf.get_sal_total_variation()
        expected = 0.0
        self.assertEqual(expected, actual)

    def test_get_features(self):
        cf = features.ContourFeatures(
            self.times, self.freqs_hz, self.salience
        )
        cf.get_features()

class TestGetSampleRate(unittest.TestCase):

    def test_sample_rate1(self):
        times = np.array([1.0, 2.0, 3.0, 4.0])
        expected = 1.0
        actual = features.get_sample_rate(times)
        self.assertEqual(expected, actual)

    def test_sample_rate2(self):
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        expected = 10.0
        actual = features.get_sample_rate(times)
        self.assertEqual(expected, actual)

    def test_uneven_spacing(self):
        times = np.array([0.1, 1.0, 4.0, 6.0])
        with self.assertRaises(NotImplementedError):
            features.get_sample_rate(times)


class TestHzToCents(unittest.TestCase):

    def test_freq_series(self):
        freqs_hz = np.array([32.0, 64.0, 128.0])
        expected = np.array([0.0, 1200.0, 2400.0])
        actual = features.hz_to_cents(freqs_hz)
        self.assertTrue(array_equal(expected, actual))


class TestFitPoly(unittest.TestCase):

    def test_line(self):
        signal = np.array([0.0, 1.0])
        expected_coeffs = np.array([0.0, 1.0])
        expected_diff = 0.0
        actual_coeffs, actual_diff = features.fit_poly(
            signal, n_poly_degrees=1
        )
        self.assertTrue(array_equal(expected_coeffs, actual_coeffs))
        self.assertAlmostEqual(expected_diff, actual_diff)

    def test_cubic(self):
        signal = np.array([0, 1.0/27.0, 8.0/27.0, 1.0])
        expected_coeffs = np.array([0.0, 0.0, 0.0, 1.0])
        expected_diff = 0.0
        actual_coeffs, actual_diff = features.fit_poly(
            signal, n_poly_degrees=3
        )
        print expected_coeffs
        print actual_coeffs
        self.assertTrue(array_equal(expected_coeffs, actual_coeffs))
        self.assertAlmostEqual(expected_diff, actual_diff)


class TestTimeSeriesFeatures(unittest.TestCase):

    def test_even_grid(self):
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        freqs = np.array([440.0, 440.0, 440.0, 440.0, 440.0, 440.0, 440.0])
        sal = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        features.time_series_features(times, freqs, sal)

    def test_uneven_grid(self):
        times = np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        freqs = np.array([440.0, 440.0, 440.0, 440.0, 440.0, 440.0, 440.0])
        sal = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        features.time_series_features(times, freqs, sal)


class TestTotalVariation(unittest.TestCase):

    def test_flat(self):
        signal = np.array([0.0, 0.0, 0.0, 0.0])
        expected = 0.0
        actual = features.total_variation(signal)
        self.assertEqual(expected, actual)

    def test_unit_step(self):
        signal = np.array([0.0, 0.0, 1.0, 1.0])
        expected = 1.0
        actual = features.total_variation(signal)
        self.assertEqual(expected, actual)

    def test_unit_step_reverse(self):
        signal = np.array([1.0, 1.0, 0.0, 0.0])
        expected = 1.0
        actual = features.total_variation(signal)
        self.assertEqual(expected, actual)


class TestVibratoFeatures(unittest.TestCase):

    def test_flat(self):
        freqs_cents = np.array([440.0, 440.0, 440.0, 440.0])
        expected_rate = 0.0
        expected_extent = 0.0
        expected_coverate = 0.0
        actual = features.vibrato_features(freqs_cents, 44100)
        actual_rate = actual[0]
        actual_extent = actual[1]
        actual_coverage = actual[2]
        self.assertEqual(expected_rate, actual_rate)
        self.assertEqual(expected_extent, actual_extent)
        self.assertEqual(expected_coverate, actual_coverage)

    def test_pure_sine(self):
        sample_rate = 2000
        grid = np.linspace(0, 1, sample_rate)
        freqs_cents = 50.0*np.sin(2.0*np.pi*12.0*grid) + 100.0
        expected_rate = 10.21208791208791 #12.0
        expected_extent = 99.999587722367053 #100.0
        expected_coverage = 1.0
        actual = features.vibrato_features(freqs_cents, sample_rate)
        actual_rate = actual[0]
        actual_extent = actual[1]
        actual_coverage = actual[2]
        self.assertEqual(expected_rate, actual_rate)
        self.assertEqual(expected_extent, actual_extent)
        self.assertEqual(expected_coverage, actual_coverage)

    def test_half_sine(self):
        sample_rate = 2000
        grid = np.linspace(0, 1, sample_rate)
        freqs_cents = 50.0*np.sin(2.0*np.pi*12.0*grid) + 100.0
        freqs_cents[0:sample_rate / 2] = 100.0
        expected_rate = 9.3293730317778376 #12.0
        expected_extent = 94.408366757784748 #100.0
        expected_coverage = 0.76769230769230767 # 0.5
        actual = features.vibrato_features(freqs_cents, sample_rate)
        actual_rate = actual[0]
        actual_extent = actual[1]
        actual_coverage = actual[2]
        self.assertEqual(expected_rate, actual_rate)
        self.assertEqual(expected_extent, actual_extent)
        self.assertEqual(expected_coverage, actual_coverage)



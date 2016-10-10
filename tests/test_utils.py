"""Tests for motif/utils.py
"""
import numpy as np
import os
import unittest

from motif import utils


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


ANNOTATION_FILE = relpath('data/test_annotation.csv')


def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2))


class TestValidateContours(unittest.TestCase):

    def test_valid(self):
        expected = None
        actual = utils.validate_contours([0], [0], [0], [0])
        self.assertEqual(expected, actual)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            acutal = utils.validate_contours([0], [0], [0], [0, 1])


class TestFormatContourData(unittest.TestCase):

    def test_format_contour_data(self):
        frequencies = np.array([440.0, 0.0, -440.0, 0.0])
        actual_cents, actual_voicing = utils.format_contour_data(frequencies)
        expected_cents = np.array([6551.31794236, 0.0, 6551.31794236, 0.0])
        expected_voicing = np.array([True, False, False, False])
        self.assertTrue(array_equal(expected_cents, actual_cents))
        self.assertTrue(array_equal(expected_voicing, actual_voicing))


class TestFormatAnnotation(unittest.TestCase):

    def test_format_annotation(self):
        new_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        times = np.array([0.0, 1.0, 2.0])
        freqs = np.array([50.0, 60.0, 70.0])
        actual_cent, actual_voicing = utils.format_annotation(
            new_times, times, freqs
        )
        expected_cent = np.array([
            2786.31371386, 2944.13435737, 3101.95500087,
            3235.39045367, 3368.82590647
        ])
        expected_voicing = np.array([True, True, True, True, True])

        self.assertTrue(array_equal(expected_cent, actual_cent))
        self.assertTrue(array_equal(expected_voicing, actual_voicing))

    def test_format_annotation_same_times(self):
        new_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        freqs = np.array([50.0, 60.0, 70.0, 0.0, 0.0, 0.0])
        actual_cent, actual_voicing = utils.format_annotation(
            new_times, times, freqs
        )
        expected_cent = np.array([
            2786.31371386, 3101.95500087,
            3368.82590647, 0.0, 0.0, 0.0
        ])
        expected_voicing = np.array([True, True, True, False, False, False])

        self.assertTrue(array_equal(expected_cent, actual_cent))
        self.assertTrue(array_equal(expected_voicing, actual_voicing))


class TestGetSnippetIdx(unittest.TestCase):

    def test_get_snippet_idx(self):
        snippet = np.array([2.0, 3.2, 4.4])
        full_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([False, False, True, True, True, False])
        actual = utils.get_snippet_idx(snippet, full_array)
        self.assertTrue(array_equal(expected, actual))


class TestLoadAnnotation(unittest.TestCase):

    def test_load_annotation(self):
        expected_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        expected_freqs = np.array([440.0, 441.0, 55.0, 56.0, 57.0, 200.0])
        actual_times, actual_freqs = utils.load_annotation(ANNOTATION_FILE)
        self.assertTrue(array_equal(expected_times, actual_times))
        self.assertTrue(array_equal(expected_freqs, actual_freqs))

    def test_load_annotation_list(self):
        expected_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        expected_freqs = [[440.0], [441.0], [55.0], [56.0], [57.0], [200.0]]

        actual_times, actual_freqs = utils.load_annotation(
            ANNOTATION_FILE, to_array=False
        )
        self.assertTrue(array_equal(expected_times, actual_times))
        self.assertEqual(expected_freqs, actual_freqs)

    def test_file_not_exists(self):
        with self.assertRaises(IOError):
            utils.load_annotation('does/not/exist.csv')

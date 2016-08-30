"""Tests for contour_decoders/viterbi.py
"""
import unittest
import numpy as np

from motif.contour_decoders import viterbi


def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2))


class TestMvGaussian(unittest.TestCase):

    def setUp(self):
        self.dcd = viterbi.ViterbiDecoder()

    def test_decode(self):
        with self.assertRaises(NotImplementedError):
            self.dcd.decode(None, None)

    def test_get_id(self):
        expected = 'viterbi'
        actual = self.dcd.get_id()
        self.assertEqual(expected, actual)

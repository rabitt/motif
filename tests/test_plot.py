"""Tests for motif/plot.py
"""
from mock import patch
import numpy as np
import os
import unittest

from motif import core
from motif import plot


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


AUDIO_FILE = relpath("data/short.wav")
ANNOTATION_FILE = relpath('data/test_annotation.csv')


class TestPlotContours(unittest.TestCase):

    def setUp(self):
        self.index = np.array([0, 0, 1, 1, 1, 2])
        self.times = np.array([0.0, 0.1, 0.0, 0.1, 0.2, 0.5])
        self.freqs = np.array([440.0, 441.0, 50.0, 52.0, 55.0, 325.2])
        self.salience = np.array([0.2, 0.4, 0.5, 0.2, 0.4, 0.0])
        self.sample_rate = 10.0
        self.audio_fpath = AUDIO_FILE
        self.ctr = core.Contours(
            self.index, self.times, self.freqs, self.salience,
            self.sample_rate, self.audio_fpath
        )

    @unittest.skip("Plotting is failing on remote.")
    @patch("matplotlib.pyplot.show")
    def test_plot_with_annotation_single(self, mock_show):
        mock_show.return_value = None
        plot.plot_with_annotation(self.ctr, ANNOTATION_FILE, single_f0=True)

    @unittest.skip("Plotting is failing on remote.")
    @patch("matplotlib.pyplot.show")
    def test_plot_with_annotation_multi(self, mock_show):
        mock_show.return_value = None
        plot.plot_with_annotation(self.ctr, ANNOTATION_FILE, single_f0=False)


class TestPlotContours(unittest.TestCase):

    def setUp(self):
        self.index = np.array([0, 0, 1, 1, 1, 2])
        self.times = np.array([0.0, 0.1, 0.0, 0.1, 0.2, 0.5])
        self.freqs = np.array([440.0, 441.0, 50.0, 52.0, 55.0, 325.2])
        self.salience = np.array([0.2, 0.4, 0.5, 0.2, 0.4, 0.0])
        self.sample_rate = 10.0
        self.audio_fpath = AUDIO_FILE
        self.ctr = core.Contours(
            self.index, self.times, self.freqs, self.salience,
            self.sample_rate, self.audio_fpath
        )

    @unittest.skip("Plotting is failing on remote.")
    @patch("matplotlib.pyplot.show")
    def test_plot_contours(self, mock_show):
        mock_show.return_value = None
        plot.plot_contours(self.ctr, style='contour')

    @unittest.skip("Plotting is failing on remote.")
    @patch("matplotlib.pyplot.show")
    def test_plot_salience(self, mock_show):
        mock_show.return_value = None
        plot.plot_contours(self.ctr, style='salience')

"""Tests for motif/run.py
"""
import unittest
import os
import numpy as np

from motif import run
from motif import core
from motif.contour_extractors.hll import BINARY_AVAILABLE


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


SHORT_AUDIO_FILE = relpath("data/short.wav")
TRAIN_AUDIO_FILE = relpath("data/input.wav")
ANNOTATION_FILE = relpath('data/input_annotation.csv')


def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2))


@unittest.skipIf(not BINARY_AVAILABLE, "hll binary not available")
class TestProcess(unittest.TestCase):

    def test_process(self):
        audio_files = [SHORT_AUDIO_FILE]
        training_pairs = [(TRAIN_AUDIO_FILE, ANNOTATION_FILE)]
        testing_pairs = [(SHORT_AUDIO_FILE, ANNOTATION_FILE)]
        run.process(
            audio_files=audio_files, training_pairs=training_pairs,
            testing_pairs=testing_pairs, extract_id='hll',
            feature_id='bitteli', classifier_id='random_forest'
        )


class TestGetExtractModule(unittest.TestCase):

    def test_hll(self):
        etr = run.get_extract_module('hll')
        self.assertEqual('hll', etr.get_id())

    def test_salamon(self):
        etr = run.get_extract_module('salamon')
        self.assertEqual('salamon', etr.get_id())


class TestGetFeaturesModule(unittest.TestCase):

    def test_bitteli(self):
        ftr = run.get_features_module('bitteli')
        self.assertEqual('bitteli', ftr.get_id())

    def test_cesium(self):
        ftr = run.get_features_module('cesium')
        self.assertEqual('cesium', ftr.get_id())

    def test_melodia(self):
        ftr = run.get_features_module('melodia')
        self.assertEqual('melodia', ftr.get_id())


class TestGetClassifyModule(unittest.TestCase):

    def test_mv_gaussian(self):
        clf = run.get_classify_module('mv_gaussian')
        self.assertEqual('mv_gaussian', clf.get_id())

    def test_random_forest(self):
        clf = run.get_classify_module('random_forest')
        self.assertEqual('random_forest', clf.get_id())

"""Tests for motif/run.py
"""
import unittest
import os
import numpy as np

from motif import run


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


SHORT_AUDIO_FILE = relpath("data/short.wav")
TRAIN_AUDIO_FILE = relpath("data/input.wav")
ANNOTATION_FILE = relpath('data/input_annotation.csv')


def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2))


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

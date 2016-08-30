"""Tests for motif/core.py
"""
import unittest
import csv
import os
import numpy as np

from motif import core
# from motif import features
# from motif import MvGaussian


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


AUDIO_FILE = relpath("data/short.wav")
ANNOTATION_FILE = relpath('data/test_annotation.csv')
CONTOURS_FILE = relpath('data/contours.csv')


def array_equal(array1, array2):
    return np.all(np.isclose(array1, array2))


class TestContours(unittest.TestCase):

    def setUp(self):
        self.index = np.array([0, 0, 1, 1, 1, 2])
        self.times = np.array([0.0, 0.1, 0.0, 0.1, 0.2, 0.5])
        self.freqs = np.array([440.0, 441.0, 50.0, 52.0, 55.0, 325.2])
        self.salience = np.array([0.2, 0.4, 0.5, 0.2, 0.4, 0.0])
        self.sample_rate = 10.0
        self.audio_fpath = AUDIO_FILE
        self.ctr = core.Contours(
            self.index, self.times, self.freqs, self.salience, self.sample_rate,
            self.audio_fpath
        )

    def test_invalid_contours(self):
        with self.assertRaises(ValueError):
            core.Contours(
                self.index, self.times, self.freqs[:-1], self.salience,
                self.sample_rate, self.audio_fpath
            )

    def test_invalid_filepath(self):
        with self.assertRaises(IOError):
            core.Contours(
                self.index, self.times, self.freqs, self.salience,
                self.sample_rate, 'not/a/file.wav'
            )

    def test_index(self):
        expected = self.index
        actual = self.ctr.index
        self.assertTrue(array_equal(expected, actual))

    def test_times(self):
        expected = self.times
        actual = self.ctr.times
        self.assertTrue(array_equal(expected, actual))

    def test_freqs(self):
        expected = self.freqs
        actual = self.ctr.freqs
        self.assertTrue(array_equal(expected, actual))

    def test_salience(self):
        expected = np.array([0.4, 0.8, 1.0, 0.4, 0.8, 0.0])
        actual = self.ctr.salience
        self.assertTrue(array_equal(expected, actual))

    def test_sample_rate(self):
        expected = self.sample_rate
        actual = self.ctr.sample_rate
        self.assertEqual(expected, actual)

    def test_filepath(self):
        expected = self.audio_fpath
        actual = self.ctr.audio_filepath
        self.assertEqual(expected, actual)

    def test_nums(self):
        expected = [0, 1, 2]
        actual = self.ctr.nums
        self.assertEqual(expected, actual)

    def test_index_mapping(self):
        expected = {0: range(0, 2), 1: range(2, 5), 2: range(5, 6)}
        actual = self.ctr.index_mapping
        self.assertEqual(expected, actual)

    def test_duration(self):
        expected = 3.0
        actual = self.ctr.duration
        self.assertEqual(expected, actual)

    def test_uniform_times(self):
        expected = np.arange(0, 3.1, 0.1)
        actual = self.ctr.uniform_times
        self.assertTrue(array_equal(expected, actual))

    def test_features(self):
        expected = None
        actual = self.ctr._features
        self.assertEqual(expected, actual)

    def test_labels(self):
        expected = None
        actual = self.ctr._labels
        self.assertEqual(expected, actual)

    def test_overlaps(self):
        expected = None
        actual = self.ctr._overlaps
        self.assertEqual(expected, actual)

    def test_scores(self):
        expected = None
        actual = self.ctr._scores
        self.assertEqual(expected, actual)

    def test_contour_times(self):
        expected = np.array([0.0, 0.1, 0.2])
        actual = self.ctr.contour_times(1)
        self.assertTrue(array_equal(expected, actual))

    def test_contour_freqs(self):
        expected = np.array([440.0, 441.0])
        actual = self.ctr.contour_freqs(0)
        self.assertTrue(array_equal(expected, actual))

    def test_contour_salience(self):
        expected = np.array([0.0])
        actual = self.ctr.contour_salience(2)
        self.assertTrue(array_equal(expected, actual))

    def test_compute_labels_default(self):
        self.ctr.compute_labels(ANNOTATION_FILE)
        expected_overlaps = {0: 1.0, 1: 1.0/3.0, 2: 0.0}
        expected_labels = {0: 1, 1: 0, 2: 0}
        actual_overlaps = self.ctr._overlaps
        actual_labels = self.ctr._labels
        self.assertEqual(expected_overlaps, actual_overlaps)
        self.assertEqual(expected_labels, actual_labels)

    def test_compute_labels_default(self):
        self.ctr.compute_labels(
            ANNOTATION_FILE, overlap_threshold=0.2
        )
        expected_overlaps = {0: 1.0, 1: 1.0/3.0, 2: 0.0}
        expected_labels = {0: 1, 1: 1, 2: 0}
        actual_overlaps = self.ctr._overlaps
        actual_labels = self.ctr._labels
        self.assertEqual(expected_overlaps, actual_overlaps)
        self.assertEqual(expected_labels, actual_labels)

    def test_compute_features(self):
        pass

    def test_compute_scores(self):
        pass
        # clf = MvGaussian()
        # X = np.array([[1.0, 2.0], [0.0, 0.0], [0.5, 0.7]])
        # Y = np.array([1, 1, 0])
        # clf.fit(X, Y)

    def test_stack_features_none(self):
        with self.assertRaises(ReferenceError):
            self.ctr.stack_features()

    def test_stack_features(self):
        self.ctr._features = {0: [1, 1, 1], 2: [0, 0, 0], 1: [1, 1, 1]}
        expected = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
        actual = self.ctr.stack_features()
        self.assertTrue(array_equal(expected, actual))

    def test_stack_labels_none(self):
        with self.assertRaises(ReferenceError):
            self.ctr.stack_labels()

    def test_stack_labels(self):
        self.ctr._labels = {0: 1, 2: 0, 1: 0}
        expected = np.array([1, 0, 0])
        actual = self.ctr.stack_labels()
        self.assertTrue(array_equal(expected, actual))

    def test_to_multif0_format(self):
        expected_times = np.arange(0, 3.1, 0.1)
        expected_freqs = [[] for _ in expected_times]
        expected_freqs[0].extend([440.0, 50.0])
        expected_freqs[1].extend([441.0, 52.0])
        expected_freqs[2].append(55.0)
        expected_freqs[5].append(325.2)
        expected_freqs = [np.array(f) for f in expected_freqs]
        actual_times, actual_freqs = self.ctr.to_multif0_format()
        self.assertTrue(array_equal(expected_times, actual_times))
        self.assertEqual(len(expected_freqs), len(actual_freqs))

        for f_expected, f_actual in zip(expected_freqs, actual_freqs):
            self.assertTrue(array_equal(f_expected, f_actual))
        

    def test_score_coverage(self):
        actual = self.ctr.score_coverage(ANNOTATION_FILE)
        expected = {
            'Precision': 0.5,
            'Recall': 0.5,
            'Accuracy': 0.33333333333333331,
            'Substitution Error': 0.16666666666666666,
            'Miss Error': 0.33333333333333331,
            'False Alarm Error': 0.33333333333333331,
            'Total Error': 0.83333333333333337,
            'Chroma Precision': 0.5,
            'Chroma Recall': 0.5,
            'Chroma Accuracy': 0.33333333333333331,
            'Chroma Substitution Error': 0.16666666666666666,
            'Chroma Miss Error': 0.33333333333333331,
            'Chroma False Alarm Error': 0.33333333333333331,
            'Chroma Total Error': 0.83333333333333337
        }
        for k in expected.keys():
            self.assertEqual(expected[k], actual[k])

    def test_plot(self):
        pass

    def test_save_target_contours_none(self):
        with self.assertRaises(ReferenceError):
            self.ctr.save_target_contours(CONTOURS_FILE)

    def test_save_target_contours(self):
        self.ctr._scores = {0: 0.6, 1: 0.2, 2: 0.9}
        fpath = CONTOURS_FILE
        self.ctr.save_target_contours(fpath)
        expected = [
            [0, 0.0, 440.0, 0.4],
            [0, 0.1, 441.0, 0.8],
            [2, 0.5, 325.2, 0.0]
        ]
        with open(fpath, 'r') as fhandle:
            reader = csv.reader(fhandle, delimiter=',')
            actual = [[float(a) for a in line] for line in reader]

        os.remove(fpath)
        self.assertTrue(array_equal(expected, actual))

    def test_save(self):
        self.ctr._scores = {0: 0.6, 1: 0.2, 2: 0.9}
        fpath = CONTOURS_FILE
        self.ctr.save(fpath)
        expected = [
            [0, 0.0, 440.0, 0.4],
            [0, 0.1, 441.0, 0.8],
            [1, 0.0, 50.0, 1.0],
            [1, 0.1, 52.0, 0.4],
            [1, 0.2, 55.0, 0.8],
            [2, 0.5, 325.2, 0.0]
        ]

        with open(fpath, 'r') as fhandle:
            reader = csv.reader(fhandle, delimiter=',')
            actual = [[float(a) for a in line] for line in reader]

        os.remove(fpath)
        self.assertTrue(array_equal(expected, actual))

class TestValidateContours(unittest.TestCase):

    def test_valid(self):
        expected = None
        actual = core._validate_contours([0], [0], [0], [0])
        self.assertEqual(expected, actual)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            acutal = core._validate_contours([0], [0], [0], [0, 1])


class TestFormatContourData(unittest.TestCase):

    def test_format_contour_data(self):
        frequencies = np.array([440.0, 0.0, -440.0, 0.0])
        actual_cents, actual_voicing = core._format_contour_data(frequencies)
        expected_cents = np.array([6551.31794236, 0.0, 6551.31794236, 0.0])
        expected_voicing = np.array([True, False, False, False])
        self.assertTrue(array_equal(expected_cents, actual_cents))
        self.assertTrue(array_equal(expected_voicing, actual_voicing))


class TestFormatAnnotation(unittest.TestCase):

    def test_format_annotation(self):
        times = np.array([0.0, 1.0, 2.0])
        freqs = np.array([50.0, 60.0, 70.0])
        duration = 2.0
        sample_rate = 2.0
        actual_times, actual_cent, actual_voicing = core._format_annotation(
            times, freqs, duration, sample_rate
        )
        expected_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        expected_cent = np.array([
            2786.31371386, 2944.13435737, 3101.95500087,
            3235.39045367, 3368.82590647
        ])

        expected_voicing = np.array([True, True, True, True, True])

        self.assertTrue(array_equal(expected_times, actual_times))
        self.assertTrue(array_equal(expected_cent, actual_cent))
        self.assertTrue(array_equal(expected_voicing, actual_voicing))

    def test_format_annotation_same_times(self):
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        freqs = np.array([50.0, 60.0, 70.0, 0.0, 0.0, 0.0])
        duration = 5.0
        sample_rate = 1.0
        actual_times, actual_cent, actual_voicing = core._format_annotation(
            times, freqs, duration, sample_rate
        )
        expected_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        expected_cent = np.array([
            2786.31371386, 3101.95500087,
            3368.82590647, 0.0, 0.0, 0.0
        ])

        expected_voicing = np.array([True, True, True, False, False, False])

        self.assertTrue(array_equal(expected_times, actual_times))
        self.assertTrue(array_equal(expected_cent, actual_cent))
        self.assertTrue(array_equal(expected_voicing, actual_voicing))


class TestGetSnippetIdx(unittest.TestCase):

    def test_get_snippet_idx(self):
        snippet = np.array([2.0, 3.2, 4.4])
        full_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([False, False, True, True, True, False])
        actual = core._get_snippet_idx(snippet, full_array)
        self.assertTrue(array_equal(expected, actual))


class TestLoadAnnotation(unittest.TestCase):

    def test_load_annotation(self):
        expected_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        expected_freqs = np.array([440.0, 441.0, 55.0, 56.0, 57.0, 200.0])
        actual_times, actual_freqs = core._load_annotation(ANNOTATION_FILE)
        self.assertTrue(array_equal(expected_times, actual_times))
        self.assertTrue(array_equal(expected_freqs, actual_freqs))

    def test_load_annotation_list(self):
        expected_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        expected_freqs = [[440.0], [441.0], [55.0], [56.0], [57.0], [200.0]]

        actual_times, actual_freqs = core._load_annotation(
            ANNOTATION_FILE, to_array=False
        )
        self.assertTrue(array_equal(expected_times, actual_times))
        self.assertEqual(expected_freqs, actual_freqs)

    def test_file_not_exists(self):
        with self.assertRaises(IOError):
            core._load_annotation('does/not/exist.csv')


class TestExtractorRegistry(unittest.TestCase):

    def test_keys(self):
        actual = sorted(core.EXTRACTOR_REGISTRY.keys())
        expected = sorted(['hll', 'salamon'])
        self.assertEqual(expected, actual)

    def test_types(self):
        for val in core.EXTRACTOR_REGISTRY.values():
            self.assertTrue(issubclass(val, core.ContourExtractor))


class TestContourExtractor(unittest.TestCase):

    def setUp(self):
        self.cex = core.ContourExtractor()

    def test_inits(self):
        self.assertEqual(self.cex.audio_samplerate, 44100)
        self.assertEqual(self.cex.audio_channels, 1)
        self.assertEqual(self.cex.audio_bitdepth, 32)
        self.assertEqual(self.cex.audio_db_level, -3.0)

    def test_sample_rate(self):
        with self.assertRaises(NotImplementedError):
            self.cex.sample_rate

    def test_get_id(self):
        with self.assertRaises(NotImplementedError):
            self.cex.get_id()

    def test_compute_contours(self):
        with self.assertRaises(NotImplementedError):
            self.cex.compute_contours(AUDIO_FILE)

    def test_preprocess_audio(self):
        tmp_audio = self.cex._preprocess_audio(AUDIO_FILE)
        self.assertTrue(os.path.exists(tmp_audio))

    def test_preprocess_audio_passthrough(self):
        tmp_audio = self.cex._preprocess_audio(
            AUDIO_FILE, normalize_format=False, normalize_volume=False
        )
        self.assertTrue(os.path.exists(tmp_audio))

    def test_preprocess_audio_hpss(self):
        with self.assertRaises(NotImplementedError):
            self.cex._preprocess_audio(AUDIO_FILE, hpss=True)

    def test_preprocess_audio_equal_loudness_filter(self):
        with self.assertRaises(NotImplementedError):
            self.cex._preprocess_audio(AUDIO_FILE, equal_loudness_filter=True)

    def test_post_process_contours(self):
        with self.assertRaises(NotImplementedError):
            self.cex._postprocess_contours()


class TestFeaturesRegistry(unittest.TestCase):

    def test_keys(self):
        actual = sorted(core.FEATURES_REGISTRY.keys())
        expected = sorted(['cesium', 'melodia', 'bitteli'])
        self.assertEqual(expected, actual)

    def test_types(self):
        for val in core.FEATURES_REGISTRY.values():
            self.assertTrue(issubclass(val, core.ContourFeatures))


class TestContourFeatures(unittest.TestCase):

    def setUp(self):
        self.ftr = core.ContourFeatures()

    def test_get_feature_vector(self):
        times = np.array([0])
        freqs = np.array([0])
        salience = np.array([0])
        sample_rate = 1
        with self.assertRaises(NotImplementedError):
            self.ftr.get_feature_vector(times, freqs, salience, sample_rate)

    def test_set_feature_names(self):
        with self.assertRaises(NotImplementedError):
            self.ftr.feature_names

    def test_get_id(self):
        with self.assertRaises(NotImplementedError):
            self.ftr.get_id()

    def test_compute_all_feautres(self):
        pass


class TestClassifierRegistry(unittest.TestCase):

    def test_keys(self):
        actual = sorted(core.CLASSIFIER_REGISTRY.keys())
        expected = sorted(['mv_gaussian', 'random_forest'])
        self.assertEqual(expected, actual)

    def test_types(self):
        for val in core.CLASSIFIER_REGISTRY.values():
            self.assertTrue(issubclass(val, core.Classifier))


class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.clf = core.Classifier()

    def test_threshold(self):
        with self.assertRaises(NotImplementedError):
            self.clf.threshold

    def test_predict(self):
        with self.assertRaises(NotImplementedError):
            self.clf.predict(np.array([0, 1]))

    def test_fit(self):
        with self.assertRaises(NotImplementedError):
            self.clf.fit(np.array([0, 1]), np.array([0]))

    def test_get_id(self):
        with self.assertRaises(NotImplementedError):
            self.clf.get_id()



# -*- coding: utf-8 -*-
""" Core methods and base class definitions
"""
import csv
from mir_eval import melody, multipitch
import numpy as np
import os
import six
from sklearn import metrics
import sox
import tempfile as tmp

from .utils import validate_contours, format_contour_data, format_annotation
from .utils import get_snippet_idx, load_annotation


###############################################################################
class Contours(object):
    '''Class containing information about all contours in a single audio
    file.

    Attributes
    ----------
    nums : list
        Ordered list of contour numbers
    index_mapping : dict
        Mapping from contour number to the indices into times/freqs/salience
        where the contour is active
    index : array
        array of contour numbers
    times : array
        array of contour times
    freqs : array
        array of contour frequencies
    salience : array
        array of contour salience values
    _features : dict
        Mapping from contour number to computed features.
        Will not be set until the compute_features method is run
    _labels : dict
        Mapping from contour number to computed ground truth labels.
    _overlaps : dict
        Mapping from contour number to computed overlap with ground truth
    _scores : dict
        Mapping from contour number to computed classifier score

    '''
    def __init__(self, index, times, freqs, salience, sample_rate,
                 audio_filepath=None, audio_duration=None):
        '''
        Parameters
        ----------
        index : np.array
            Array of contour numbers
        times : np.array
            Array of contour times
        freqs : np.array
            Array of contour frequencies
        salience : np.array
            Array of contour saliences
        sample_rate : float
            Contour sample rate.
        audio_filepath : str or None
            Path to audio file contours were extracted from

        '''
        validate_contours(index, times, freqs, salience)
        if audio_filepath is not None and not os.path.exists(audio_filepath):
            raise IOError("audio_filepath does not exist.")
        elif audio_filepath is None and audio_duration is None:
            raise ValueError(
                "one of audio_filepath or audio_duration must be set.")

        # contour attributes
        self.index = index
        self.times = times
        self.freqs = freqs
        self.salience = self._set_salience(salience)
        self.sample_rate = sample_rate
        self.audio_filepath = audio_filepath
        self.audio_duration = audio_duration

        self.nums = self._compute_nums()
        self.index_mapping = self._compute_index_mapping()
        self.duration = self._compute_duration()
        self.uniform_times = self._compute_uniform_times()

    def _set_salience(self, salience):
        '''Set the salience attribute

        Returns
        -------
        salience : np.array
            Normalized salience.

        '''
        if len(salience) == 0 or np.max(salience) == 0:
            return salience
        else:
            return salience / np.max(salience)

    def _compute_nums(self):
        '''Compute the list of contour index numbers

        Returns
        -------
        nums : list
            Sorted list of contour index numbers

        '''
        return sorted(list(set(self.index)))

    def _compute_index_mapping(self):
        '''Computes the mapping from contour numbers to indices.

        Returns
        -------
        index_mapping : dict
            Mapping from contour numbers to indices.

        '''
        index_mapping = dict.fromkeys(self.nums)
        for num in self.nums:
            idxs = np.where(self.index == num)[0]
            index_mapping[num] = range(idxs[0], idxs[-1] + 1)
        return index_mapping

    def _compute_duration(self):
        '''Compute the duration of the audio file.

        Returns
        -------
        duration : float
            Audio file duration
        '''
        if self.audio_duration is not None:
            return self.audio_duration
        else:
            return sox.file_info.duration(self.audio_filepath)

    def _compute_uniform_times(self):
        '''Compute array of uniform time stamps at the sample rate

        Returns
        -------
        uniform_times : np.array
            Array of uniform time stamps at the sample rate
        '''
        n_stamps = int(np.ceil(self.duration * self.sample_rate)) + 1
        uniform_times = np.arange(0, n_stamps) / float(self.sample_rate)
        return uniform_times

    def contour_times(self, index):
        '''Get the time stamps for a particular contour number.

        Parameters
        ----------
        index : int
            contour number

        Returns
        -------
        contour_times : array
            array of contour times
        '''
        return self.times[self.index_mapping[index]]

    def contour_freqs(self, index):
        '''Get the frequency values for a particular contour number.

        Parameters
        ----------
        index : int
            contour number

        Returns
        -------
        contour_frequencies : array
            array of contour frequency values
        '''
        return self.freqs[self.index_mapping[index]]

    def contour_salience(self, index):
        '''Get the salience values for a particular contour number.

        Parameters
        ----------
        index : int
            contour number

        Returns
        -------
        contour_salience : array
            array of contour salience values
        '''
        return self.salience[self.index_mapping[index]]

    def compute_labels(self, annotation_fpath, overlap_threshold=0.5, single_f0=True):
        '''Compute overlaps with an annotation and labels for each contour.

        Parameters
        ----------
        annotation_fpath : str
            Path to annotation file.
        overlap_threshold : float, default=0.5
            The minimum amount of overlap with the annotation for a contour to
            be labeled as a positive example; between 0 and 1.

        '''
        if single_f0:
            annot_times, annot_freqs = load_annotation(annotation_fpath)
        else:
            raise NotImplementedError

        ref_cent, ref_voicing = format_annotation(
            self.uniform_times, annot_times, annot_freqs
        )
        est_cents, est_voicing = format_contour_data(self.freqs)

        labels = dict.fromkeys(self.nums)
        overlaps = dict.fromkeys(self.nums)

        for i in self.nums:
            gt_idx = get_snippet_idx(self.contour_times(i), self.uniform_times)

            this_est_cent, this_est_voicing = melody.resample_melody_series(
                self.contour_times(i), est_cents[self.index_mapping[i]],
                est_voicing[self.index_mapping[i]], self.uniform_times[gt_idx]
            )

            overlaps[i] = melody.overall_accuracy(
                ref_voicing[gt_idx], ref_cent[gt_idx],
                this_est_voicing,
                this_est_cent
            )
            labels[i] = 1 * (overlaps[i] > overlap_threshold)

        labels = np.array([labels[n] for n in self.nums])
        overlaps = np.array([overlaps[n] for n in self.nums])
        return labels, overlaps

    def to_multif0_format(self):
        '''Convert contours to multi-f0 format.

        Returns
        -------
        times : np.array
            uniform time stamps
        freqs : list of lists
            Each row has the form [time, freq1, freq2, ...]
            Each row may have any number of frequencies.
        '''
        n_uniform_times = len(self.uniform_times)
        freqs = [[] for i in range(n_uniform_times)]

        time_idx = np.round(self.times * self.sample_rate).astype(int)
        time_idx[time_idx >= n_uniform_times] = n_uniform_times - 1
        for i, freq in zip(time_idx, self.freqs):
            freqs[i].append(freq)
        freqs = [np.array(f).astype(float) for f in freqs]

        return self.uniform_times, freqs

    def coverage(self, annotation_fpath, single_f0=True):
        """ Compute how much the set of contours covers the annotation

        Parameters
        ----------
        annotation_fpath : str
            Path to annotation file.
        single_f0 : bool
            True for a file containing a single pitch per time stamp
            False for a file containing possibly multiple pitches / time stamp

        Returns
        -------
        scores : dict
            Dictionary of mutlipitch scores.

        """
        est_times, est_freqs = self.to_multif0_format()
        if single_f0:
            ref_times, ref_freqs = load_annotation(
                annotation_fpath, n_freqs=1, to_array=False, rm_zeros=True
            )

        else:
            ref_times, ref_freqs = load_annotation(
                annotation_fpath, n_freqs=None, to_array=False, rm_zeros=True
            )

        scores = multipitch.evaluate(
            ref_times, ref_freqs, est_times, est_freqs
        )
        return scores

    def save_contours_subset(self, output_fpath, output_nums):
        '''Save extracted contours where `score >= threshold` to a csv file.

        Parameters
        ----------
        output_fpath : str
            Path to save output csv file.
        output_nums : list
            List of contour numbers to save

        '''
        target_indices = []
        for num in output_nums:
            target_indices.extend(self.index_mapping[num])

        with open(output_fpath, 'w') as fhandle:
            writer = csv.writer(fhandle, delimiter=',')
            writer.writerows(zip(
                self.index[target_indices],
                self.times[target_indices],
                self.freqs[target_indices],
                self.salience[target_indices]
            ))

    def save(self, output_fpath):
        '''Save extracted contours to a csv file.

        Parameters
        ----------
        output_fpath : str
            Path to save output csv file.

        '''
        with open(output_fpath, 'w') as fhandle:
            writer = csv.writer(fhandle, delimiter=',')
            writer.writerows(zip(
                self.index,
                self.times,
                self.freqs,
                self.salience
            ))


###############################################################################
CONTOUR_EXTRACTOR_REGISTRY = {}  # All available extractors


class MetaContourExtractor(type):
    """Meta-class to register the available extractors."""
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class ContourExtractors
        if "ContourExtractor" in [base.__name__ for base in bases]:
            CONTOUR_EXTRACTOR_REGISTRY[cls.get_id()] = cls
        return cls


class ContourExtractor(six.with_metaclass(MetaContourExtractor)):
    """This class is an interface for all the contour extraction algorithms
    included in motif. Each extractor must inherit from it and implement the
    following method:
        - ``compute_contours``
    Additionally, two private helper functions are provided:
        - ``preprocess``
        - ``postprocess``
    These are meant to do common tasks for all the extractors and they should
    be called inside the process method if needed.

    Some methods may call a binary in the background, which creates a csv file.
    The csv file is loaded into memory and the file is deleted, unless
    ``clean=False``. When ``recompute=False``, this will first look for an
    existing precomputed contour file and if successful will load it directly.
    """
    def __init__(self):
        self.audio_channels = 1
        self.audio_bitdepth = 32
        self.audio_db_level = -3.0

    @property
    def audio_samplerate(self):
        """Property to get the sample rate of the output contours"""
        raise NotImplementedError("This property must return the sample rate "
                                  "of the output contours.")

    @property
    def sample_rate(self):
        """Property to get the sample rate of the output contours"""
        raise NotImplementedError("This property must return the sample rate "
                                  "of the output contours.")

    @property
    def min_contour_len(self):
        """Property to get the minimum length of a contour in seconds"""
        raise NotImplementedError("This property must return the minimum "
                                  "contour length in seconds.")

    @classmethod
    def get_id(cls):
        """Method to get the id of the extractor type"""
        raise NotImplementedError("This method must return a string identifier"
                                  " of the contour extraction type")

    def compute_contours(self, input_filepath):
        """Method for computing features for given file"""
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the contour extraction")

    def _preprocess_audio(self, audio_filepath, normalize_format=True,
                          normalize_volume=True, hpss=False,
                          equal_loudness_filter=False):
        '''Preprocess audio before computing contours

        Parameters
        ----------
        normalize : bool
            If True, normalize the audio
        hpss : bool
            If True, applies HPSS & computes contours on the harmonic compoment
        equal_loudness_filter : bool
            If True, applies an equal loudness filter to the audio

        '''
        tfm = sox.Transformer()
        if normalize_format:
            tfm.convert(
                samplerate=self.audio_samplerate,
                n_channels=self.audio_channels,
                bitdepth=self.audio_bitdepth
            )

        if normalize_volume:
            tfm.norm(db_level=self.audio_db_level)

        output_path = tmp.mktemp('.wav')
        tfm.build(audio_filepath, output_path)

        if hpss:
            raise NotImplementedError

        if equal_loudness_filter:
            raise NotImplementedError

        return output_path

    def _postprocess_contours(self, index, times, freqs, salience):
        """Remove contours that are too short.

        Parameters
        ----------
        index : np.array
            array of contour numbers
        times : np.array
            array of contour times
        freqs : np.array
            array of contour frequencies
        salience : np.array
            array of contour salience values

        Returns
        -------
        index_pruned : np.array
            Pruned array of contour numbers
        times_pruned : np.array
            Pruned array of contour times
        freqs_pruned : np.array
            Pruned array of contour frequencies
        salience_pruned : np.array
            Pruned array of contour salience values

        """
        keep_index = np.ones(times.shape).astype(bool)
        for i in set(index):
            this_idx = (index == i)
            if np.ptp(times[this_idx]) <= self.min_contour_len:
                keep_index[this_idx] = False

        return (index[keep_index], times[keep_index],
                freqs[keep_index], salience[keep_index])

    def _sort_contours(self, index, times, freqs, salience):
        """Sort contours by index and time.

        Parameters
        ----------
        index : np.array
            array of contour numbers
        times : np.array
            array of contour times
        freqs : np.array
            array of contour frequencies
        salience : np.array
            array of contour salience values

        Returns
        -------
        index_sorted : np.array
            Pruned array of contour numbers
        times_sorted : np.array
            Pruned array of contour times
        freqs_sorted : np.array
            Pruned array of contour frequencies
        salience_sorted : np.array
            Pruned array of contour salience values

        """
        sort_idx = np.lexsort((times, index))

        return (
            index[sort_idx], times[sort_idx], freqs[sort_idx],
            salience[sort_idx]
        )


###############################################################################
FEATURE_EXTRACTOR_REGISTRY = {}  # All available classifiers


class MetaFeatureExtractor(type):
    """Meta-class to register the available contour features."""
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class FeatureExtractor
        if "FeatureExtractor" in [base.__name__ for base in bases]:
            FEATURE_EXTRACTOR_REGISTRY[cls.get_id()] = cls
        return cls


class FeatureExtractor(six.with_metaclass(MetaFeatureExtractor)):
    """This class is an interface for all the feature extraction combinations
    included in motif. Each feature set must inherit from it and implement the
    following methods:
        - ``get_feature_vector``
            This should return a flat numpy array
        - ``feature_names``
            This should return a list of the same length as the above
            numpy array of what each dimension is. Can be as simple as an
            index, can be identfiers such as ['vibrato rate', 'vibrato extent']
    """
    def __init__(self):
        pass

    def get_feature_vector(self, times, freqs, salience, sample_rate):
        """Method for computing features for a given contour"""
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the contour feautres")

    @property
    def feature_names(self):
        """Set the array of features names."""
        raise NotImplementedError("This method must create and return a list "
                                  "of feature names, the same length as the"
                                  "feature vector.")

    @classmethod
    def get_id(cls):
        """Method to get the id of the feature type"""
        raise NotImplementedError("This method must return a string identifier"
                                  "of the feature type")

    def compute_all(self, ctr):
        """ Compute features for all contours.

        Parameters
        ----------
        ctr : Contour
            Instance of Contour object

        Returns
        -------
        features : np.array [n_contours, n_features]
            Feature matrix, ordered by contour number

        """
        features = []
        for i in ctr.nums:
            if len(ctr.index_mapping[i]) > 0:
                feature_vector = self.get_feature_vector(
                    ctr.contour_times(i),
                    ctr.contour_freqs(i),
                    ctr.contour_salience(i),
                    ctr.sample_rate
                )
                features.append(feature_vector)

        return np.array(features)


###############################################################################
CONTOUR_CLASSIFIER_REGISTRY = {}  # All available classifiers


class MetaContourClassifier(type):
    """Meta-class to register the available classifiers."""
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class ContourClassifier
        if "ContourClassifier" in [base.__name__ for base in bases]:
            CONTOUR_CLASSIFIER_REGISTRY[cls.get_id()] = cls
        return cls


class ContourClassifier(six.with_metaclass(MetaContourClassifier)):
    """This class is an interface for all the contour classifier algorithms
    included in motif. Each classifer must inherit from it and implement the
    following methods:
        - ``predict``
        - ``fit``
        - ``threshold``
    ``threshold`` should return a float whose determines the positive class
    threshold (e.g. ``score >= threshold`` : positive class,
    ``score < threshold`` : negative class)
    """
    def __init__(self):
        pass

    @property
    def threshold(self):
        """Property for setting threshold between classes"""
        raise NotImplementedError("This method most return a float that "
                                  "indicates the score cutoff between the "
                                  "positive and negative class.")

    def predict(self, X):
        """Method for predicting labels from input"""
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the prediction")

    def fit(self, X, Y):
        """Method for fitting the model"""
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the model fitting")

    @classmethod
    def get_id(cls):
        """Method to get the id of the extractor type"""
        raise NotImplementedError("This method must return a string identifier"
                                  " of the contour extraction type")

    def score(self, y_predicted, y_target, y_prob=None):
        """ Compute metrics on classifier predictions

        Parameters
        ----------
        y_predicted : np.array [n_samples]
            Predicted class labels
        y_target : np.array [n_samples]
            Target class labels
        y_prob : np.array [n_samples] or None, default=None
            predicted probabilties. If None, auc is not computed

        Returns
        -------
        scores : dict
            dictionary of scores for the following metrics:
            accuracy, matthews correlation coefficient, precision, recall, f1,
            support, confusion matrix, auc score
        """
        labels = set(y_target)
        labels.update(y_predicted)
        is_binary = len(labels) <= 2

        scores = {}
        scores['accuracy'] = metrics.accuracy_score(y_target, y_predicted)

        if is_binary:
            scores['mcc'] = metrics.matthews_corrcoef(y_target, y_predicted)
        else:
            scores['mcc'] = None

        (scores['precision'],
         scores['recall'],
         scores['f1'],
         scores['support']) = metrics.precision_recall_fscore_support(
             y_target, y_predicted
         )

        scores['confusion matrix'] = metrics.confusion_matrix(
            y_target, y_predicted, labels=list(labels)
        )

        if y_prob is not None:
            scores['auc score'] = metrics.roc_auc_score(
                y_target, y_prob + 1, average='weighted'
            )
        else:
            scores['auc score'] = None

        return scores


###############################################################################
CONTOUR_DECODER_REGISTRY = {}  # All available decoders


class MetaContourDecoder(type):
    """Meta-class to register the available decoders."""
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class ContourDecoder
        if "ContourDecoder" in [base.__name__ for base in bases]:
            CONTOUR_DECODER_REGISTRY[cls.get_id()] = cls
        return cls


class ContourDecoder(six.with_metaclass(MetaContourDecoder)):
    """This class is an interface for all the contour decoder algorithms
    included in motif. Each decoder must inherit from it and implement the
    following methods:
        - ``decode``
        - ``get_id``

    """
    def __init__(self):
        pass

    def decode(self, ctr, Y):
        """ Decode the output of the contour classifier.

        Parameters
        ----------
        ctr : Contours
            An instance of a Contours object
        Y : np.array [n_contours]
            Predicted contour scores.

        Returns
        -------
        times : np.ndarray
            Array of time stamps
        freqs : np.ndarray
            Array of f0 values in Hz

        """
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the decoder.")

    @classmethod
    def get_id(cls):
        """Method to get the id of the decoder type"""
        raise NotImplementedError("This method must return a string identifier"
                                  " of the contour decoder type")

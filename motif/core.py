# -*- coding: utf-8 -*-
""" Core methods and base class definitions
"""
import csv
import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import os
import seaborn as sns
import six
from sklearn import metrics
import sox
import tempfile as tmp

sns.set()


###############################################################################
class Contours(object):
    '''Class containing information about all contours in a single audio
    file.

    Attributes
    ----------
    nums : set
        Set of contour numbers
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
                 audio_filepath):
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
        audio_filepath : str
            Path to audio file contours were extracted from

        '''
        _validate_contours(index, times, freqs, salience)
        if not os.path.exists(audio_filepath):
            raise IOError("audio_filepath does not exist.")

        # contour attributes
        self.index = index
        self.times = times
        self.freqs = freqs
        self.salience = salience / np.max(salience)
        self.sample_rate = sample_rate
        self.audio_filepath = audio_filepath

        self.nums = self._compute_nums()
        self.index_mapping = self._compute_index_mapping()
        self.duration = self._compute_duration()
        self.uniform_times = self._compute_uniform_times()

        # computed attributes
        # self._features = None
        # self._labels = None
        # self._overlaps = None
        # self._scores = None

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
        return sox.file_info.duration(self.audio_filepath)

    def _compute_uniform_times(self):
        '''Compute array of uniform time stamps at the sample rate

        Returns
        -------
        uniform_times : np.array
            Array of uniform time stamps at the sample rate
        '''
        n_stamps = int(np.ceil(self.duration * self.sample_rate)) + 1
        uniform_times = np.arange(0, n_stamps) / self.sample_rate
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

    def compute_labels(self, annotation_fpath, overlap_threshold=0.5):
        '''Compute overlaps with an annotation and labels for each contour.

        Parameters
        ----------
        annotation_fpath : str
            Path to annotation file.
        overlap_threshold : float, default=0.5
            The minimum amount of overlap with the annotation for a contour to
            be labeled as a positive example; between 0 and 1.

        '''
        annot_times, annot_freqs = _load_annotation(annotation_fpath)
        ref_times, ref_cent, ref_voicing = _format_annotation(
            annot_times, annot_freqs, self.duration, self.sample_rate
        )
        est_cents, est_voicing = _format_contour_data(self.freqs)

        labels = dict.fromkeys(self.nums)
        overlaps = dict.fromkeys(self.nums)

        for i in self.nums:
            gt_idx = _get_snippet_idx(self.contour_times(i), ref_times)
            overlaps[i] = mir_eval.melody.overall_accuracy(
                ref_voicing[gt_idx], ref_cent[gt_idx],
                est_voicing[self.index_mapping[i]],
                est_cents[self.index_mapping[i]]
            )
            labels[i] = 1 * (overlaps[i] > overlap_threshold)

        labels = np.array([labels[n] for n in self.nums])
        overlaps = np.array([overlaps[n] for n in self.nums])
        return labels, overlaps

    # def compute_features(self, ctr_ftr):
    #     '''Compute features for each contour.

    #     Parameters
    #     ----------
    #     ctr_ftr : FeatureExtractor
    #         A FeatureExtractor object.

    #     '''
    #     self._features = ctr_ftr.compute_all_feautres(self)

    # def compute_scores(self, contour_classifier):
    #     '''Compute scores using a given classifier.

    #     Parameters
    #     ----------
    #     contour_classifier : ContourClassifier
    #         A trained ContourClassifier object.

    #     '''
    #     features = self.stack_features()
    #     labels = contour_classifier.predict(features)
    #     scores = {n: labels[n] for n in self.nums}
    #     self._scores = scores

    # def stack_features(self):
    #     '''Stack features into numpy array.

    #     Returns
    #     -------
    #     X : np.array [n_contours, n_features]
    #         Array of stacked features.

    #     '''
    #     if self._features is None:
    #         raise ReferenceError("Features have not yet been computed.")
    #     return np.array([self._features[n] for n in self.nums])

    # def stack_labels(self, labels):
    #     '''Stack labels into numpy array.

    #     Returns
    #     -------
    #     Y : np.array [n_contours, 1]
    #         Array of stacked labels.

    #     '''
    #     return np.array([self.labels[n] for n in self.nums])

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
        freqs = [[] for i in range(len(self.uniform_times))]
        time_idx = np.round(self.times * self.sample_rate).astype(int)
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
            False for a file containing possibly multiple pitches per time stamp

        Returns
        -------
        scores : dict
            Dictionary of mutlipitch scores.

        """
        est_times, est_freqs = self.to_multif0_format()
        if single_f0:
            ref_times, ref_freqs = _load_annotation(
                annotation_fpath, n_freqs=1, to_array=False
            )
        else:
            ref_times, ref_freqs = _load_annotation(
                annotation_fpath, n_freqs=None, to_array=False
            )
        scores = mir_eval.multipitch.evaluate(
            ref_times, ref_freqs, est_times, est_freqs
        )
        return scores

    def plot(self, style='contour'):
        '''Plot the contours.

        Parameters
        ----------
        style : str
            One of:
                - 'contour': plot each extracted contour, where each contour
                    gets its own color.
                - 'salience': plot the contours where the colors denote the
                    salience.

        '''
        if style == 'contour':
            for i in self.nums:
                plt.plot(self.contour_times(i), self.contour_freqs(i))
        elif style == 'salience':
            plt.scatter(
                self.times, self.freqs,
                c=(self.salience / np.max(self.salience)), cmap='BuGn',
                edgecolors='face', marker='.'
            )
            plt.colorbar()

        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.axis('tight')

    def save_contours_subset(self, output_fpath, output_nums):
        '''Save extracted contours where score >= threshold to a csv file.

        Parameters
        ----------
        output_fpath : str
            Path to save output csv file.
        output_nums : list
            List of contour numbers to save
        threshold : float
            Minimum score to be considered part of the target class.

        '''
        # nums_target = [n for n in self.nums if self._scores[n] >= threshold]
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


def _validate_contours(index, times, freqs, salience):
    '''Check that contour input is well formed.

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
    audio_filepath : str
        Path to audio file contours were extracted from

    '''
    N = len(index)
    if any([len(times) != N, len(freqs) != N, len(salience) != N]):
        raise ValueError(
            "the arrays index, times, freqs, and salience "
            "must be the same length."
        )


def _format_contour_data(frequencies):
    """ Convert contour frequencies to cents + voicing.

    Parameters
    ----------
    frequencies : np.array
        Contour frequency values

    Returns
    -------
    est_cent : np.array
        Contour frequencies in cents
    est_voicing : np.array
        Contour voicings

    """
    est_freqs, est_voicing = mir_eval.melody.freq_to_voicing(frequencies)
    est_cents = mir_eval.melody.hz2cents(est_freqs, 10.)
    return est_cents, est_voicing


def _format_annotation(annot_times, annot_freqs, duration, sample_rate):
    """ Format an annotation file and resample to a uniform timebase.

    Parameters
    ----------
    annot_times : np.array
        Annotation time stamps
    annot_freqs : np.array
        Annotation frequency values
    duration : float
        Length of the full audio file in seconds.
    sample_rate : float
        The target sample rate.

    Returns
    -------
    annot_times_new : np.array
        New annotation time stamps
    ref_cent : np.array
        Annotation frequencies in cents at the new timescale
    ref_voicing : np.array
        Annotation voicings at the new timescale

    """
    annot_times_new = np.arange(
        0, duration + 0.5 / sample_rate, 1.0 / sample_rate
    )

    ref_freq, ref_voicing = mir_eval.melody.freq_to_voicing(annot_freqs)
    ref_cent = mir_eval.melody.hz2cents(ref_freq, 10.)

    ref_cent, ref_voicing = mir_eval.melody.resample_melody_series(
        annot_times, ref_cent, ref_voicing, annot_times_new,
        kind='linear'
    )
    return annot_times_new, ref_cent, ref_voicing


def _get_snippet_idx(snippet, full_array):
    """ Find the indices of ``full_array`` where ``snippet`` is present.
    Assumes both ``snippet`` and ``full_array`` are ordered.

    Parameters
    ----------
    snippet : np.array
        Array of ordered time stamps
    full_array : np.array
        Array of ordered time stamps

    Returns
    -------
    idx : np.array
        Array of booleans indicating where in ``full_array`` ``snippet``
        is present.

    """
    idx = np.logical_and(
        full_array >= snippet[0], full_array <= snippet[-1]
    )
    return idx


def _load_annotation(annotation_fpath, n_freqs=1, to_array=True):
    """ Load an annotation from a csv file.

    Parameters
    ----------
    annotation_fpath : str
        Path to annotation file.
    n_freqs : int or None
        Number of frequencies to read, or None to use max
    to_array : bool
        If True, returns annot_freqs as a numpy array
        If False, returns annot_freqs as a list of lists.

    Returns
    -------
    annot_times : array
        Annotation time stamps
    annot_freqs : array
        Annotation frequency values

    """
    end_idx = None if n_freqs is None else n_freqs + 1
    if not os.path.exists(annotation_fpath):
        raise IOError("The annotation path {} does not exist.")

    annot_times = []
    annot_freqs = []
    with open(annotation_fpath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for row in reader:
            annot_times.append(row[0])
            annot_freqs.append([r for r in row[1:end_idx]])

    annot_times = np.array(annot_times, dtype=float)
    annot_freqs = [np.array(f).astype(float) for f in annot_freqs]
    if to_array:
        annot_freqs = np.array(annot_freqs, dtype=float).flatten()
    return annot_times, annot_freqs


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
            compute_contours()
    Additionally, two private helper functions are provided:
        - preprocess
        - postprocess
    These are meant to do common tasks for all the extractors and they should
    be called inside the process method if needed.

    Some methods may call a binary in the background, which creates a csv file.
    The csv file is loaded into memory and the file is deleted, unless
    clean=False. When recompute=False, this will first look for an existing
    precomputed contour file and if successful will load it directly.
    """
    def __init__(self):
        self.audio_samplerate = 44100
        self.audio_channels = 1
        self.audio_bitdepth = 32
        self.audio_db_level = -3.0

    @property
    def sample_rate(self):
        """Property to get the sample rate of the output contours"""
        raise NotImplementedError("This property must return the sample rate "
                                  "of the output contours.")

    @classmethod
    def get_id(cls):
        """Method to get the id of the extractor type"""
        raise NotImplementedError("This method must return a string identifier"
                                  " of the contour extraction type")

    def compute_contours(self, audio_filepath):
        """Method for computing features for given audio file"""
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the contour extraction")

    def _preprocess_audio(self, audio_filepath, normalize_format=True,
                          normalize_volume=True, hpss=False,
                          equal_loudness_filter=False):
        '''Preprocess the audio before computing contours

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


    def _postprocess_contours(self):
        """Remove contours that are too short.
        """
        raise NotImplementedError


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
    following method:
        get_feature_vector(self, times, freqs, salience, duration,
                           sample_rate)
            --> This should return a flat numpy array
        feature_names(self)
            --> This should return a list of the same length as the above
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
        predict(X)
        fit(X, y)
        threshold()
            Should return a float whose determines the positive class threshold
            (e.g. score >= threshold --> positive class,
             score < threshold --> negative class)
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

    def score(self, predicted_scores, y_target):
        """ Compute metrics on classifier predictions

        Parameters
        ----------
        predicted_scores : np.array [n_samples]
            predicted scores
        y_target : np.array [n_samples]
            Target class labels

        Returns
        -------
        scores : dict
            dictionary of scores for the following metrics:
            accuracy, matthews correlation coefficient, precision, recall, f1,
            support, confusion matrix, auc score
        """
        y_predicted = 1 * (predicted_scores >= self.threshold)
        scores = {}
        scores['accuracy'] = metrics.accuracy_score(y_target, y_predicted)
        scores['mcc'] = metrics.matthews_corrcoef(y_target, y_predicted)
        (scores['precision'],
         scores['recall'],
         scores['f1'],
         scores['support']) = metrics.precision_recall_fscore_support(
            y_target, y_predicted
        )
        scores['confusion matrix'] = metrics.confusion_matrix(
            y_target, y_predicted, labels=[0, 1]
        )
        scores['auc score'] = metrics.roc_auc_score(
            y_target, predicted_scores + 1, average='weighted'
        )
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
        """
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the decoder.")

    @classmethod
    def get_id(cls):
        """Method to get the id of the decoder type"""
        raise NotImplementedError("This method must return a string identifier"
                                  " of the contour decoder type")

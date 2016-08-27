# -*- coding: utf-8 -*-
""" Core methods and base class definitions
"""
import numpy as np
import csv
import mir_eval
import matplotlib.pyplot as plt
import seaborn as sns
import six
import sox

from . import features as F

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
    features : dict
        Mapping from contour number to computed features.
        Will not be set until the compute_features method is run
    labels : dict
        Mapping from contour number to computed ground truth labels.
    overlaps : dict
        Mapping from contour number to computed overlap with ground truth
    scores : dict
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

        # computed attributes
        self._features = None
        self._labels = None
        self._overlaps = None
        self._scores = None

    def _compute_nums(self):
        '''Compute the set of contour index numbers
        '''
        return list(set(self.index))

    def _compute_index_mapping(self):
        '''Computes the mapping from contour numbers to indices.
        '''
        index_mapping = dict.fromkeys(self.nums)
        for num in self.nums:
            idxs = np.where(self.index == num)[0]
            index_mapping[num] = range(idxs[0], idxs[-1] + 1)
        return index_mapping

    def _compute_duration(self):
        '''Compute the duration of the audio file.
        '''
        return sox.file_info.duration(self.audio_fpath)

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

    def get_labels(self, annotation_fpath, overlap_threshold=0.5):
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
        labels = dict.fromkeys(self.nums)
        overlaps = dict.fromkeys(self.nums)

        for i in self.nums:
            this_contour_times = self.contour_times(i)
            this_contour_freqs = self.contour_freqs(i)
            start_t = this_contour_times[0]
            end_t = this_contour_times[-1]
            gt_idx = np.logical_and(
                annot_times >= start_t, annot_times <= end_t
            )
            this_annot_times = annot_times[gt_idx]
            this_annot_freqs = annot_freqs[gt_idx]
            res = mir_eval.melody.evaluate(
                this_annot_times, this_annot_freqs,
                this_contour_times, this_contour_freqs
            )
            overlaps[i] = res['Overall Accuracy']
            labels[i] = 1 * (overlaps[i] > overlap_threshold)

        self._labels = labels
        self._overlaps = overlaps

    def compute_features(self):
        '''Compute features for each contour.
        '''
        features = dict.fromkeys(self.nums)

        for i in self.nums:
            cft = F.ContourFeatures(
                self.contour_times(i),
                self.contour_freqs(i),
                self.contour_salience(i)
            )
            features[i] = cft.get_features()

        self._features = features

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
                - 'score': plot the contours where the colors denote the
                    classifier score.
                - 'overlap': plot the contours where the color denotes the
                    amount of overlap with the annotation.

        '''
        if style == 'contour':
            for i in self.nums:
                plt.plot(self.contour_times(i), self.contour_freqs(i))
        elif style == 'salience':
            plt.scatter(
                self.times, self.freqs, c=(self.salience/np.max(self.salience)),
                cmap='BuGn', edgecolors='face', marker='.'
            )
            plt.colorbar()
        elif style == 'score':
            for i in self.nums:
                plt.plot(
                    self.contour_times(i),
                    self.contour_freqs(i),
                    color=self._scores[i]
                )
            plt.colorbar()
        elif style == 'overlap':
            for i in self.nums:
                plt.plot(
                    self.contour_times(i),
                    self.contour_freqs(i),
                    color=self._overlaps[i]
                )
            plt.colorbar()

        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.axis('tight')

    def save(self, output_fpath):
        with open(output_fpath, 'w') as fhandle:
            writer = csv.writer(fhandle, delimiter=',')
            writer.writerows(zip(
                self.index,
                self.times,
                self.freqs,
                self.salience
            ))

def _load_annotation(annotation_fpath):
    """ Load an annotation file into a pandas Series.
    Add column with frequency values also converted to cents.

    Parameters
    ----------
    annotation_fpath : str
        Path to annotation file.

    Returns
    -------
    annot_times : array
        Annotation time stamps
    annot_freqs : array
        Annotation frequency values
    """
    if annotation_fpath is not None:
        annot_times = []
        annot_freqs = []
        with open(annotation_fpath, 'r') as fhandle:
            reader = csv.reader(fhandle, delimiter=',')
            for row in reader:
                annot_times.append(row[0])
                annot_freqs.append(row[1])

        annot_times = np.array(annot_times, dtype=float)
        annot_freqs = np.array(annot_freqs, dtype=float)

    return annot_times, annot_freqs


###############################################################################
EXTRACTOR_REGISTRY = {}  # All available extractors


class MetaContourExtractor(type):
    """Meta-class to register the available extractors."""
    def __new__(mcs, meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class ContourExtractors
        if "ContourExtractor" in [base.__name__ for base in bases]:
            EXTRACTOR_REGISTRY[cls.get_id()] = cls
        return cls


class ContourExtractor(six.with_metaclass(MetaContourExtractor)):
    """This class is an interface for all the contour extraction algorithms
    included in motif. Each extractor must inherit from it and implement the
    following method:
            compute_contours()
    Additionally, two private helper functions are provided:
        - preprocess
        - postprocess
    These are meant to do common tasks for all the extractors and they should be
    called inside the process method if needed.

    Some methods may call a binary in the background, which creates a csv file.
    The csv file is loaded into memory and the file is deleted, unless
    clean=False. When recompute=False, this will first look for an existing
    precomputed contour file and if successful will load it directly.
    """
    def __init__(self, extractor="salamon"):
        '''
        Parameters
        ----------
        extractor : ?
            ?
        '''
        self.extractor = extractor
        self.recompute = True
        self.clean = True

    def compute_contours(self):
        """Method for computing features for given audio file"""
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the contour extraction")

    @classmethod
    def get_id(cls):
        """Method to get the id of the extractor type"""
        raise NotImplementedError("This method must return a string identifier"
                                  " of the contour extraction type")

    def _preprocess_audio(self, normalize=True, equal_loudness_filter=False,
                          hpss=False):
        '''Preprocess the audio before computing contours

        Parameters
        ----------
        normalize : bool
            If True, normalize the audio
        equal_loudness_fileter : bool
            If True, applies an equal loudness filter to the audio
        hpss : bool
            If True, applies HPSS & computes contours on the harmonic compoment

        '''
        raise NotImplementedError

    def _postprocess_contours(self):
        """Remove contours that are too short.
        """
        raise NotImplementedError


###############################################################################
FEATURES_REGISTRY = {}  # All available classifiers


class MetaContourFeatures(type):
    """Meta-class to register the available contour features."""
    def __new__(mcs, meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class ContourFeatures
        if "ContourFeatures" in [base.__name__ for base in bases]:
            FEATURES_REGISTRY[cls.get_id()] = cls
        return cls


class ContourFeatures(six.with_metaclass(MetaContourFeatures)):
    """This class is an interface for all the feature extraction combinations
    included in motif. Each feature set must inherit from it and implement the
    following method:
        get_feature_vector(self, times, freqs, salience, duration,
                           sample_rate)
            --> This should return a flat numpy array
        set_feature_names(self)
            --> This should return a list of the same length as the above
            numpy array of what each dimension is. Can be as simple as an
            index, can be idenfiers such as ['vibrato rate', 'vibrato extent']
    """
    def __init__(self, identifier="bittner2015"):
        '''
        Parameters
        ----------
        identifier : str
            ?
        '''
        self.identifier = identifier
        self.feature_names = self.set_feature_names()

    def get_feature_vector(self, times, freqs, salience, sample_rate):
        """Method for computing features for a given contour"""
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the contour feautres")

    def set_feature_names(self):
        """Set the array of features names."""
        raise NotImplementedError("This method must create and return a list "
                                  "of feature names, the same length as the"
                                  "feature vector.")

    @classmethod
    def get_id(cls):
        """Method to get the id of the feature type"""
        raise NotImplementedError("This method must return a string identifier"
                                  "of the feature type")

    def compute_all_feautres(self, ctr):
        features = dict.fromkeys(ctr.nums)

        for i in ctr.nums:
            feature_vector = self.compute_features(
                ctr.contour_times(i),
                ctr.contour_freqs(i),
                ctr.contour_salience(i),
                ctr.duration,
                ctr.sample_rate
            )
            features[i] = feature_vector

        return features

###############################################################################
CLASSIFIER_REGISTRY = {}  # All available classifiers


class MetaClassifier(type):
    """Meta-class to register the available classifiers."""
    def __new__(mcs, meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class Classifier
        if "Classifier" in [base.__name__ for base in bases]:
            CLASSIFIER_REGISTRY[cls.get_id()] = cls
        return cls


class Classifier(six.with_metaclass(MetaClassifier)):
    """This class is an interface for all the contour extraction algorithms
    included in motif. Each extractor must inherit from it and implement the
    following methods:
            predict(X)
            fit(X, y)
    """
    def __init__(self, **kwargs):
        pass

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

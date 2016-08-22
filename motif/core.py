import numpy as np
import csv
import os
import mir_eval
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from . import extract as C
from . import features as F


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
    def __init__(self, audio_fpath, method, recompute=True, clean=True):
        '''
        Parameters
        ----------
        audio_fpath : str
            Path to audio file
        method : str
            Which contour tracking method to use, one of:
                * 'hll': Harmonic Locked Loop tracking
                * 'salamon': Contour tracking from Melodia
                * 'bosch': NotImplemented
        recompute : bool, default=True
            If True, computes contours directly from the audio file.
            If False, first looks for an existing saved contour file.
        clean : bool, default=True
            If True, removes any temporary files created. If False, keeps files.

        '''
        # attributes from constructor
        self.audio_fpath = audio_fpath
        self.method = method
        self.recompute = recompute
        self.clean = clean

        # contour attributes
        self.nums = None
        self.index_mapping = None
        self.index = None
        self.times = None
        self.freqs = None
        self.salience = None

        # compute contours
        self._compute_contours()
        # create the index mapping
        self._set_index_mapping()

        # computed attributes
        self.features = None
        self.labels = None
        self.overlaps = None
        self.scores = None

    def _compute_contours(self):
        '''Compute contours for a track based on selected method.
        '''
        if self.method == 'hll':
            c_numbers, c_times, c_freqs, c_sal = C.hll(
                self.audio_fpath, recompute=self.recompute, clean=self.clean
            )
        elif self.method == 'salamon':
            c_numbers, c_times, c_freqs, c_sal = C.salamon(
                self.audio_fpath, recompute=self.recompute, clean=self.clean
            )

        self.index = c_numbers
        self.times = c_times
        self.freqs = c_freqs
        self.salience = c_sal/np.max(c_sal)

    def _set_index_mapping(self):
        '''Computes the mapping from contour numbers to indices.
        '''
        self.nums = set(self.index)
        self.index_mapping = dict.fromkeys(self.nums)
        for num in self.nums:
            self.index_mapping[num] = np.where(
                self.index == num)[0]

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
            labels[i] = 1*(overlaps[i] > overlap_threshold)

        self.labels = labels
        self.overlaps = overlaps

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

        self.features = features

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
                    color=self.scores[i]
                )
            plt.colorbar()
        elif style == 'overlap':
            for i in self.nums:
                plt.plot(
                    self.contour_times(i),
                    self.contour_freqs(i),
                    color=self.overlaps[i]
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


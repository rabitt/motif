# -*- coding: utf-8 -*-
"""Salamon's method for extracting contours
"""
from __future__ import print_function

import librosa
from mir_eval.melody import hz2cents
import numpy as np
import os
import scipy.signal
import subprocess
from subprocess import CalledProcessError

from motif.core import ContourExtractor
from motif.core import Contours
from motif.contour_extractors import utils



class DeepSal(ContourExtractor):
    '''Peak streaming based contour extraction as in [1]_
    on a deep learned salience representation as in [2]_

    .. [1] Salamon, Justin and Gómez, Emilia, and Bonada, Jordi.
        "Sinusoid extraction and salience function design for predominant
        melody estimation." 14th International Conference on Digital Audio
        Effects (DAFX11), Paris, France, 2011.
    .. [2]

    Parameters
    ----------
    hop_length : int, default=128
        Number of samples between frames.
    win_length : int, default=2048
        The window size in samples.
    n_fft : int, default=8192
        The fft size in samples.
    h_range : list, default=[1, 2, 3, 4, 5]
        The list of harmonics to use in salience function.
    h_weights : list, default=[1, 0.5, 0.25, 0.25, 0.25]
        The list of weights to apply to each harmonic in salience function.
    pitch_cont : float, default=80
        Pitch continuity threshold in cents.
    max_gap : float, default=0.01
        Threshold (in seconds) for how many values can be taken from S-.
    amp_thresh : float, default=0.9
        Threshold on how big a peak must be relative to the maximum in its
        frame.
    dev_thresh : float, default=0.9
        The maximum number of standard deviations below the mean a peak can
        be to survive.
    preprocess : bool, default=True
        If true, normalizes the volume and format of the audio before
        processing. Otherwise computes contours from original audio.

    Attributes
    ----------
    max_freq : float
        The maximum frequency allowed in a contour in Hz.
    hop_length : int
        Number of samples between frames.
    win_length : int
        The window size in samples.
    n_fft : int
        The fft size in samples.
    h_range : list
        The list of harmonics to use in salience function.
    h_weights : list
        The list of weights to apply to each harmonic in salience function.
    interpolation_type : str
        Frequency interpolation type. See scipy.signal.interp1d for details.
    pitch_cont : float
        Pitch continuity threshold in cents.
    max_gap : float
        Threshold (in seconds) for how many values can be taken from S-.
    amp_thresh : float
        Threshold on how big a peak must be relative to the maximum in its
        frame.
    dev_thresh : float
        The maximum number of standard deviations below the mean a peak can
        be to survive.
    preprocess : bool
        If true, normalizes the volume and format of the audio before
        processing. Otherwise computes contours from original audio.
    use_salamon_salience : bool
        If true, uses salamon vamp plugin to compute salience.

    '''
    def __init__(self, max_freq=3000.0, hop_length=128, win_length=2048,
                 n_fft=8192, h_range=[1, 2, 3, 4, 5],
                 h_weights=[1, 0.5, 0.25, 0.25, 0.25],
                 interpolation_type='linear', pitch_cont=80, max_gap=0.01,
                 amp_thresh=0.9, dev_thresh=0.9, preprocess=True):
        '''Init method.
        '''

        # self.max_freq = max_freq
        self.salience_sr = 22050
        self.salience_hop = 256
        self.salience_bins_per_octave = 60
        self.salience_n_octaves = 6
        self.salience_fmin = 32.7

        # # salience function parameters
        # self.hop_length = hop_length
        # self.win_length = win_length
        # self.n_fft = n_fft
        # self.h_range = h_range
        # self.h_weights = h_weights
        # self.interpolation_type = interpolation_type

        # peak streaming parameters
        self.pitch_cont = pitch_cont
        self.max_gap = max_gap
        self.amp_thresh = amp_thresh
        self.dev_thresh = dev_thresh

        self.preprocess = preprocess
        self.use_salamon_salience = use_salamon_salience

        ContourExtractor.__init__(self)

    @property
    def n_gap(self):
        """The number of time frames within the maximum gap

        Returns
        -------
        n_gap : float
            Number of time frames within the maximum gap.

        """
        return self.max_gap * self.sample_rate

    @property
    def audio_samplerate(self):
        """Sample rate of preprocessed audio.

        Returns
        -------
        audio_samplerate : float
            Number of samples per second.

        """
        return 44100.0

    @property
    def sample_rate(self):
        """Sample rate of output contours

        Returns
        -------
        sample_rate : float
            Number of samples per second.

        """
        return self.salience_sr / self.salience_hop

    @property
    def min_contour_len(self):
        """Minimum allowed contour length.

        Returns
        -------
        min_contour_len : float
            Minimum allowed contour length in seconds.

        """
        return 0.1

    @classmethod
    def get_id(cls):
        """Identifier of this extractor.

        Returns
        -------
        id : str
            Identifier of this extractor.

        """
        return "deepsal"

    def get_times(n_frames):
        time_grid = librosa.core.frames_to_time(
            range(n_frames), sr=self.salience_sr, hop_length=self.salience_hop
        )
        return time_grid

    def get_freqs():
        freq_grid = librosa.cqt_frequencies(
            self.salience_bins_per_octave*self.salience_n_octaves,
            self.salience_fmin, bins_per_octave=self.salience_bins_per_octave
        )
        return freq_grid

    def compute_contours(self, salience_npy_file):
        """Compute contours by peak streaming a salience output.

        Parameters
        ----------
        salience_npy_file : str
            Path to precomputed salience numpy file.

        Returns
        -------
        Instance of Contours object

        """
        if not os.path.exists(salience_npy_file):
            raise IOError(
                "The numpy file {} does not exist".format(salience_npy_file)
            )

        S = np.load(salience_npy_file)
        times = self.get_times(S.shape[1])
        freqs = self.get_freqs()

        psh = utils.PeakStreamHelper(
            S, times, freqs, self.amp_thresh, self.dev_thresh, self.n_gap,
            self.pitch_cont
        )

        c_numbers, c_times, c_freqs, c_sal = psh.peak_streaming()
        if len(c_numbers) > 0:
            c_numbers, c_times, c_freqs, c_sal = self._sort_contours(
                np.array(c_numbers), np.array(c_times), np.array(c_freqs),
                np.array(c_sal)
            )
            (c_numbers, c_times, c_freqs, c_sal) = self._postprocess_contours(
                c_numbers, c_times, c_freqs, c_sal
            )

        return Contours(
            c_numbers, c_times, c_freqs, c_sal, self.sample_rate,
            audio_filepath
        )


# -*- coding: utf-8 -*-
"""Salamon's method for extracting contours
"""
from __future__ import print_function

import librosa
from mir_eval.melody import hz2cents
import numpy as np
import os
import scipy.signal

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
    pitch_cont : float, default=80
        Pitch continuity threshold in cents.
    max_gap : float, default=0.01
        Threshold (in seconds) for how many values can be taken from S-.
    min_contour_freq : float, default=32.7
        Minimum contour frequency value
    max_contour_freq : float, default=3000.0
        Max contour frequency value
    peak_thresh : float, default=0.3
        Threshold between 0 and 1 to separate good peaks from bad peaks
    low_amp_thresh : float, default=0.005
        Threshold below which salience values are discarded from consideration

    Attributes
    ----------
    pitch_cont : float
        Pitch continuity threshold in cents.
    max_gap : float
        Threshold (in seconds) for how many values can be taken from S-.
    min_contour_freq : float
        Minimum contour frequency value
    max_contour_freq : float
        Max contour frequency value
    peak_thresh : float
        Threshold between 0 and 1 to separate good peaks from bad peaks
    low_amp_thresh : float
        Threshold below which salience values are discarded from consideration

    '''
    def __init__(self, pitch_cont=80, max_gap=0.2, min_contour_freq=32.7,
                 max_contour_freq=3000.0, peak_thresh=0.3,
                 low_amp_thresh=0.005):
        '''Init method.
        '''
        self.salience_sr = 22050
        self.salience_hop = 256
        self.salience_bins_per_octave = 60
        self.salience_n_octaves = 6
        self.salience_fmin = 32.7

        self.min_contour_freq = min_contour_freq
        self.max_contour_freq = max_contour_freq

        # peak streaming parameters
        self.pitch_cont = pitch_cont
        self.max_gap = max_gap
        self.peak_thresh = peak_thresh
        self.low_amp_thresh = low_amp_thresh

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
        return 0.05

    @classmethod
    def get_id(cls):
        """Identifier of this extractor.

        Returns
        -------
        id : str
            Identifier of this extractor.

        """
        return "deepsal"

    def get_times(self, n_frames):
        time_grid = librosa.core.frames_to_time(
            range(n_frames), sr=self.salience_sr, hop_length=self.salience_hop
        )
        return time_grid

    def get_freqs(self):
        freq_grid = librosa.cqt_frequencies(
            self.salience_bins_per_octave * self.salience_n_octaves,
            self.salience_fmin, bins_per_octave=self.salience_bins_per_octave
        )
        return freq_grid

    def compute_contours(self, salience_npy_file, audio_duration):
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
        S[S < self.low_amp_thresh] = 0.0
        times = self.get_times(S.shape[1])
        freqs = self.get_freqs()
        S[freqs < self.min_contour_freq, :] = 0
        S[freqs > self.max_contour_freq, :] = 0

        psh = utils.PeakStreamHelper(
            S, times, freqs, 0, 0, self.n_gap,
            self.pitch_cont, peak_thresh=self.peak_thresh
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
            audio_duration=audio_duration
        )

# -*- coding: utf-8 -*-
"""Salamon's method for extracting contours
"""
from __future__ import print_function

import librosa
import numpy as np
import os
# import scipy.signal
import subprocess
from subprocess import CalledProcessError

from motif.core import ContourExtractor
from motif.core import Contours
from motif.contour_extractors import utils


SALAMON_FPATH = "vamp_melodia-salience_melodia-salience_saliencefunction.csv"
VAMP_PLUGIN = b"vamp:melodia-salience:melodia-salience:saliencefunction"


def _check_binary():
    '''Check if the vamp plugin is available and can be called.

    Returns
    -------
    True if callable, False otherwise

    '''
    sonic_annotator_exists = True
    try:
        subprocess.check_output(['which', 'sonic-annotator'])
    except CalledProcessError:
        sonic_annotator_exists = False

    if sonic_annotator_exists:
        avail_plugins = subprocess.check_output(["sonic-annotator", "-l"])
        if VAMP_PLUGIN in avail_plugins:
            return True
        else:
            return False
    else:
        return False


BINARY_AVAILABLE = _check_binary()


class PeakStream(ContourExtractor):
    '''Peak streaming based contour extraction as in [1]_

    .. [1] Salamon, Justin and Gómez, Emilia, and Bonada, Jordi.
        "Sinusoid extraction and salience function design for predominant
        melody estimation." 14th International Conference on Digital Audio
        Effects (DAFX11), Paris, France, 2011.

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
                 amp_thresh=0.9, dev_thresh=0.9, preprocess=True,
                 use_salamon_salience=False):
        '''Init method.
        '''

        self.max_freq = max_freq

        # salience function parameters
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.h_range = h_range
        self.h_weights = h_weights
        self.interpolation_type = interpolation_type

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
        return self.audio_samplerate / self.hop_length

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
        return "peak_stream"

    def compute_contours(self, audio_filepath):
        """Compute contours as in Justin Salamon's melodia.
        This calls a vamp plugin in the background, which creates a csv file.
        The csv file is loaded into memory and the file is deleted.

        Parameters
        ----------
        audio_filepath : str
            Path to audio file.

        Returns
        -------
        Instance of Contours object

        """
        if not os.path.exists(audio_filepath):
            raise IOError(
                "The audio file {} does not exist".format(audio_filepath)
            )

        if self.preprocess:
            fpath = self._preprocess_audio(
                audio_filepath, normalize_format=True,
                normalize_volume=True
            )
        else:
            fpath = audio_filepath

        print("Computing salience...")
        if self.use_salamon_salience:
            times, freqs, S = self._compute_salience_salamon(fpath)
        else:
            y, sr = librosa.load(fpath, sr=self.audio_samplerate)
            times, freqs, S = self._compute_salience(y, sr)

        psh = utils.PeakStreamHelper(
            S, times, freqs, self.amp_thresh, self.dev_thresh, self.n_gap,
            self.pitch_cont, peak_thresh=None
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

    def _compute_salience(self, y, sr):
        """Computes salience function from audio signal using librosa's
        salience function.

        Parameters
        ----------
        y : np.array
            Audio signal
        sr : float
            Audio sample rate

        Returns
        -------
        times : np.array
            Array of times in seconds
        freqs : np.array
            Array of frequencies in Hz
        salience : np.array
            Salience matrix of shape (len(freqs), len(times))

        """
        # compute stft
        S = librosa.core.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        freqs = librosa.core.fft_frequencies(sr=sr, n_fft=self.n_fft)
        times = librosa.core.frames_to_time(
            np.arange(0, S.shape[1]), sr, hop_length=self.hop_length,
            n_fft=self.n_fft
        )

        # discard unneeded frequencies
        max_sal_freq = np.max(self.h_range) * self.max_freq
        max_sal_freq_index = np.argmin(np.abs(freqs - max_sal_freq))
        freqs_reduced = freqs[:max_sal_freq_index]

        S_sal = librosa.harmonic.salience(
            np.abs(S[:max_sal_freq_index, :]), freqs_reduced,
            self.h_range, weights=self.h_weights, kind=self.interpolation_type,
            filter_peaks=True, fill_value=0.0
        )

        max_freq_index = np.argmin(np.abs(freqs_reduced - self.max_freq))
        return times, freqs_reduced[:max_freq_index], S_sal[:max_freq_index, :]

    def _compute_salience_salamon(self, fpath):
        """Computes salience function from audio signal using melodia's
        salience function.

        Parameters
        ----------
        fpath : str
            Path to audio file.

        Returns
        -------
        times : np.array
            Array of times in seconds
        freqs : np.array
            Array of frequencies in Hz
        salience : np.array
            Salience matrix of shape (len(freqs), len(times))

        """
        if not BINARY_AVAILABLE:
            raise EnvironmentError(
                "Either the vamp plugin {} needed to compute these contours or "
                "sonic-annotator is not available.".format(VAMP_PLUGIN)
            )

        f_dir = os.path.dirname(fpath)
        f_name = os.path.basename(fpath)
        fpath_out = os.path.join(
            f_dir,
            "{}_{}".format(f_name.split('.')[0], SALAMON_FPATH)
        )
        if os.path.exists(fpath_out):
            os.remove(fpath_out)

        binary_call = [
            "sonic-annotator", "-d",
            "vamp:melodia-salience:melodia-salience:saliencefunction",
            fpath, "-w", "csv", "--csv-force"
        ]
        os.system(" ".join(binary_call))
        if not os.path.exists(fpath_out):
            raise IOError("output file does not exist")
        else:
            S_sal = np.loadtxt(fpath_out, dtype=float, delimiter=',')
            S_sal = (S_sal / np.max(S_sal, axis=0)).T
            times = librosa.core.frames_to_time(
                np.arange(0, S_sal.shape[1]), 44100, hop_length=128
            )
            freqs = 55.0 * np.power(2.0, (np.arange(0, 601)) / 120.0)
            os.remove(fpath_out)
        return times, freqs, S_sal



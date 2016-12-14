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

        psh = PeakStreamHelper(
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


class PeakStreamHelper(object):

    def __init__(self, S, times, freqs, amp_thresh, dev_thresh, n_gap,
                 pitch_cont):
        '''Init method.

        Parameters
        ----------
        S : np.array
            Salience matrix
        times : np.array
            Array of times in seconds
        freqs : np.array
            Array of frequencies in Hz
        amp_thresh : float
            Threshold on how big a peak must be relative to the maximum in its
            frame.
        dev_thresh : float
            The maximum number of standard deviations below the mean a peak can
            be to survive.
        n_gap : float
            Number of frames that can be taken from bad_peaks.
        pitch_cont : float
            Pitch continuity threshold in cents.

        '''
        self.S = S
        self.S_norm = self._get_normalized_S()
        self.times = times
        self.freqs_hz = freqs
        self.freqs = hz2cents(freqs)

        self.amp_thresh = amp_thresh
        self.dev_thresh = dev_thresh
        self.n_gap = n_gap
        self.pitch_cont = pitch_cont

        peaks = scipy.signal.argrelmax(S, axis=0)
        self.n_peaks = len(peaks[0])

        if self.n_peaks > 0:
            self.peak_index = np.arange(self.n_peaks)
            self.peak_time_idx = peaks[1]
            self.first_peak_time_idx = np.min(self.peak_time_idx)
            self.last_peak_time_idx = np.max(self.peak_time_idx)
            self.frame_dict = self._get_frame_dict()
            self.peak_freqs = self.freqs[peaks[0]]
            self.peak_freqs_hz = self.freqs_hz[peaks[0]]
            self.peak_amps = self.S[peaks[0], peaks[1]]
            self.peak_amps_norm = self.S_norm[peaks[0], peaks[1]]

            self.good_peaks, self.bad_peaks = self._partition_peaks()
            (self.good_peaks_sorted,
             self.good_peaks_sorted_index,
             self.good_peaks_sorted_avail,
             self.n_good_peaks) = self._create_good_peak_index()
            self.smallest_good_peak_idx = 0
        else:
            self.peak_index = np.array([])
            self.peak_time_idx = np.array([])
            self.first_peak_time_idx = None
            self.last_peak_time_idx = None
            self.frame_dict = {}
            self.peak_freqs = np.array([])
            self.peak_freqs_hz = np.array([])
            self.peak_amps = np.array([])
            self.peak_amps_norm = np.array([])
            self.good_peaks = set()
            self.bad_peaks = set()
            self.good_peaks_sorted = []
            self.good_peaks_sorted_index = {}
            self.good_peaks_sorted_avail = np.array([])
            self.n_good_peaks = 0
            self.smallest_good_peak_idx = 0

        self.gap = 0
        self.n_remaining = len(self.good_peaks)

        self.contour_idx = []
        self.c_len = []

    def _get_normalized_S(self):
        """Compute normalized salience matrix

        Returns
        -------
        S_norm : np.array
            Normalized salience matrix.

        """
        S_min = np.min(self.S, axis=0)
        S_norm = self.S - S_min
        S_max = np.max(S_norm, axis=0)
        S_max[S_max == 0] = 1.0
        S_norm = S_norm / S_max
        return S_norm

    def _get_frame_dict(self):
        """Get dictionary of frame index to peak index.

        Returns
        -------
        frame_dict : dict
            Dictionary mapping frame index to lists of peak indices

        """
        frame_dict = {k: [] for k in range(len(self.times))}
        for i, k in enumerate(self.peak_time_idx):
            frame_dict[k].append(i)

        for k, v in frame_dict.items():
            frame_dict[k] = np.array(v)

        return frame_dict

    def _partition_peaks(self):
        """Split peaks into good peaks and bad peaks.

        Returns
        -------
        good_peaks : set
            Set of good peak indices
        bad_peaks : set
            Set of bad peak indices

        """
        good_peaks = set(self.peak_index)
        bad_peaks = set()

        # peaks with amplitude below a threshold --> bad peaks
        bad_peak_idx = np.where(self.peak_amps_norm < self.amp_thresh)[0]
        bad_peaks.update(bad_peak_idx)

        # find indices of surviving peaks
        good_peaks.difference_update(bad_peaks)

        # compute mean and standard deviation of amplitudes of survivors
        mean_peak = np.mean(self.peak_amps[bad_peak_idx])
        std_peak = np.std(self.peak_amps[bad_peak_idx])

        # peaks with amplitude too far below the mean --> bad peaks
        bad_peaks.update(np.where(
            self.peak_amps < (mean_peak - (self.dev_thresh * std_peak)))[0])
        good_peaks.difference_update(bad_peaks)

        return good_peaks, bad_peaks

    def _create_good_peak_index(self):
        """Create a sorted index of peaks by amplitude.

        Returns
        -------
        good_peaks_sorted : np.ndarray
            Array of peak indices ordered by peak amplitude
        good_peaks_sorted_index : dict
            Dictionary mapping peak index to its position in good_peaks_sorted
        good_peaks_sorted_avail : np.ndarray
            Array of booleans indicating if a good peak has been used
        n_good_peaks : int
            Number of initial good peaks

        """
        good_peak_list = list(self.good_peaks)
        sort_idx = list(self.peak_amps[good_peak_list].argsort()[::-1])

        good_peaks_sorted = np.array(good_peak_list)[sort_idx]
        good_peaks_sorted_index = {
            j: i for i, j in enumerate(good_peaks_sorted)
        }

        n_good_peaks = len(good_peak_list)
        good_peaks_sorted_avail = np.ones((n_good_peaks, )).astype(bool)
        return (good_peaks_sorted, good_peaks_sorted_index,
                good_peaks_sorted_avail, n_good_peaks)

    def get_largest_peak(self):
        """Get the largest remaining good peak.

        Returns
        -------
        max_peak_idx : int
            Index of the largest remaining good peak

        """
        return self.good_peaks_sorted[self.smallest_good_peak_idx]

    def update_largest_peak_list(self, peak_index):
        """Update the list of largest peaks

        Parameters
        ----------
        peak_index : int
            Index of the largest remaining good peak

        """
        this_sorted_idx = self.good_peaks_sorted_index[peak_index]
        self.good_peaks_sorted_avail[this_sorted_idx] = False

        if this_sorted_idx <= self.smallest_good_peak_idx:
            i = this_sorted_idx
            while i < self.n_good_peaks:
                if self.good_peaks_sorted_avail[i]:
                    self.smallest_good_peak_idx = i
                    break
                else:
                    i += 1

    def get_closest_peak(self, current_f0, candidates):
        """Find the peak in `candidates` closest in frequency to `current_f0`.

        Parameters
        ----------
        current_f0 : float
            Current frequency value
        candidates : list
            List of peak candidates

        Returns
        -------
        closest_peak_idx : int
            Index of the closest peak to `current_f0`

        """
        min_dist = np.argmin(np.abs(self.peak_freqs[candidates] - current_f0))
        return candidates[min_dist]

    def get_peak_candidates(self, frame_idx, current_f0):
        """Get candidates in frame_idx at current_f0

        Parameters
        ----------
        frame_idx : int
            Frame index
        current_f0 : float
            Current frequency value

        Returns
        -------
        candidates : list or None
            List of peak candidates. None if no available peaks.
        from_good : bool or None
            True if candidates are "good", False if they are "bad",
            None if no available peaks.

        """

        # find candidates in time frame
        all_cands = self.frame_dict[frame_idx]

        # restrict to frames that satisfy pitch continuity
        all_cands = set(all_cands[
            np.abs(self.peak_freqs[all_cands] - current_f0) < self.pitch_cont
        ])

        if len(all_cands) == 0:
            return None, None

        cands = list(all_cands & self.good_peaks)

        if len(cands) > 0:
            self.gap = 0
            return cands, True

        bad_cands = list(all_cands & self.bad_peaks)

        if len(bad_cands) > 0:
            self.gap += 1
            return bad_cands, False

        return None, None

    def get_contour(self):
        """Get the next contour.

        Appends to `self.contour_idx` and `self.c_len`
        Removes peaks from `self.good_peaks` and `self.bad_peaks`
        as they are selected.
        """
        largest_peak = self.get_largest_peak()

        # time frame and freqency index of largest peak
        frame_idx = self.peak_time_idx[largest_peak]
        f0_val = self.peak_freqs[largest_peak]
        self.good_peaks.remove(largest_peak)
        self.update_largest_peak_list(largest_peak)
        self.n_remaining -= 1
        self.contour_idx.append(largest_peak)
        self.gap = 0
        c_len = 1

        # choose forward peaks for this contour
        while self.gap < self.n_gap:
            # go to next time frame
            frame_idx = frame_idx + 1
            if frame_idx > self.last_peak_time_idx:
                break

            cands, from_good = self.get_peak_candidates(frame_idx, f0_val)
            if cands is None:
                break

            closest_peak = self.get_closest_peak(f0_val, cands)

            # add this peak to the contour, remove it from candidates
            self.contour_idx.append(closest_peak)
            c_len += 1

            if from_good:
                self.good_peaks.remove(closest_peak)
                self.update_largest_peak_list(closest_peak)
                self.n_remaining -= 1
            else:
                self.bad_peaks.remove(closest_peak)

            # update target frequency
            f0_val = self.peak_freqs[closest_peak]

        # choose backward peaks for this contour
        frame_idx = self.peak_time_idx[largest_peak]
        f0_val = self.peak_freqs[largest_peak]
        self.gap = 0
        while self.gap < self.n_gap:
            # go to previous time frame
            frame_idx = frame_idx - 1
            if frame_idx < self.first_peak_time_idx:
                break

            cands, from_good = self.get_peak_candidates(frame_idx, f0_val)
            if cands is None:
                break

            closest_peak = self.get_closest_peak(f0_val, cands)

            # add this peak to the contour, change its label to 0
            self.contour_idx.append(closest_peak)
            c_len += 1

            if from_good:
                self.good_peaks.remove(closest_peak)
                self.update_largest_peak_list(closest_peak)
                self.n_remaining -= 1
            else:
                self.bad_peaks.remove(closest_peak)

            # update target frequency
            f0_val = self.peak_freqs[closest_peak]

        self.c_len.append(c_len)

    def peak_streaming(self):
        """Run peak streaming over salience function

        Returns
        -------
        c_numbers : np.array
            Contour numbers
        c_times : np.array
            Contour times in seconds
        c_freqs : np.array
            Contour frequencies
        c_sal : np.array
            Contour salience

        """
        # loop until there are no remaining peaks labeled with 1
        while self.n_remaining > 0:
            # print(self.n_remaining)
            self.get_contour()

        if len(self.c_len) > 0:
            c_numbers = np.repeat(range(len(self.c_len)), repeats=self.c_len)
            c_times = self.times[self.peak_time_idx[self.contour_idx]]
            c_freqs = self.peak_freqs_hz[self.contour_idx]
            c_sal = self.peak_amps[self.contour_idx]
        else:
            c_numbers = np.array([])
            c_times = np.array([])
            c_freqs = np.array([])
            c_sal = np.array([])

        return c_numbers, c_times, c_freqs, c_sal

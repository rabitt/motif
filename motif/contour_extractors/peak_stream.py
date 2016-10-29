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


class PeakStream(ContourExtractor):
    '''Peak streaming based contour extraction as in [1]_

    .. [1] Salamon, Justin and Gómez, Emilia, and Bonada, Jordi.
        "Sinusoid extraction and salience function design for predominant
        melody estimation." 14th International Conference on Digital Audio
        Effects (DAFX11), Paris, France, 2011.

    '''
    def __init__(self, hop_length=128, win_length=2048, n_fft=8192,
                 pitch_cont=80, max_gap=0.01, amp_thresh=0.9, dev_thresh=0.9,
                 preprocess=True):
        '''Init method.

        Parameters
        ----------
        hop_length : int
            Number of samples between frames.
        win_length : int
            The window size in samples.
        n_fft : int
            The fft size in samples.
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
        '''

        # salience function parameters
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft

        # peak streaming parameters
        self.pitch_cont = pitch_cont
        self.max_gap = max_gap
        self.amp_thresh = amp_thresh
        self.dev_thresh = dev_thresh

        self.preprocess = preprocess

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
                audio_filepath, normalize_format=True, normalize_volume=True
            )
        else:
            fpath = audio_filepath

        y, sr = librosa.load(fpath, sr=self.audio_samplerate)
        times, freqs, S = self._compute_salience(y, sr)

        psh = PeakStreamHelper(
            S, times, freqs, self.amp_thresh, self.dev_thresh, self.n_gap,
            self.pitch_cont
        )

        c_numbers, c_times, c_freqs, c_sal = psh.peak_streaming()
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
        """Computes salience function from audio signal.

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
        S = librosa.core.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        freqs = librosa.core.fft_frequencies(sr=sr, n_fft=self.n_fft)
        times = librosa.core.frames_to_time(
            np.arange(0, S.shape[1]), sr, hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        return times, freqs, np.abs(S)


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
        self.freqs = hz2cents(freqs)

        self.amp_thresh = amp_thresh
        self.dev_thresh = dev_thresh
        self.n_gap = n_gap
        self.pitch_cont = pitch_cont

        peaks = scipy.signal.argrelmax(S, axis=0)
        self.n_peaks = len(peaks[0])

        self.peak_index = np.arange(self.n_peaks)
        self.peak_time_idx = peaks[1]
        self.first_peak_time_idx = np.min(self.peak_time_idx)
        self.last_peak_time_idx = np.max(self.peak_time_idx)
        self.frame_dict = self._get_frame_dict()
        self.peak_freqs = self.freqs[peaks[0]]
        self.peak_amps = self.S[peaks[0], peaks[1]]
        self.peak_amps_norm = self.S_norm[peaks[0], peaks[1]]

        self.good_peaks, self.bad_peaks = self._partition_peaks()

        self.gap = 0

        self.contour_idx = []
        self.c_len = []

    @property
    def n_remaining(self):
        """Number of remaining good peaks

        Returns
        -------
        n_remaining : int
            Number of remaining good peaks.

        """
        return len(self.good_peaks)

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

        # mark peaks with amplitude below a threshold
        # of the maximum peak amplitude in the frame with label -1
        bad_peak_idx = np.where(self.peak_amps_norm < self.amp_thresh)[0]
        bad_peaks.update(bad_peak_idx)

        # find indices of surviving peaks
        good_peaks.difference_update(bad_peaks)

        # compute mean and standard deviation of amplitudes of survivors
        mean_peak = np.mean(self.peak_amps[bad_peak_idx])
        std_peak = np.std(self.peak_amps[bad_peak_idx])

        # mark peaks with amplitude too far below the mean with label -2
        bad_peaks.update(np.where(
            self.peak_amps < (mean_peak - (self.dev_thresh * std_peak)))[0])
        good_peaks.difference_update(bad_peaks)

        return good_peaks, bad_peaks

    def get_largest_peak(self):
        """Get the largest remaining good peak.

        Returns
        -------
        max_peak_idx : int
            Index of the largest remaining good peak

        """
        good_list = list(self.good_peaks)
        max_peak = np.argmax(self.peak_amps[good_list])
        max_peak_idx = self.peak_index[good_list][max_peak]
        return max_peak_idx

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
        all_cands = self.peak_index[self.frame_dict[frame_idx]]

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

        # initialize list of indices to be placed in this contour
        this_contour_idx = []
        largest_peak = self.get_largest_peak()

        # time frame and freqency index of largest peak
        frame_idx = self.peak_time_idx[largest_peak]
        f0_val = self.peak_freqs[largest_peak]
        self.good_peaks.remove(largest_peak)
        this_contour_idx.append(largest_peak)
        self.gap = 0
        c_len = 0

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
            self.get_contour()

        c_numbers = np.repeat(range(len(self.c_len)), repeats=self.c_len)
        c_times = self.times[self.peak_time_idx[self.contour_idx]]
        c_freqs = self.peak_freqs[self.contour_idx]
        c_sal = self.peak_amps[self.contour_idx]

        return c_numbers, c_times, c_freqs, c_sal

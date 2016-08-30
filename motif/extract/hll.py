# -*- coding: utf-8 -*-
"""HLL method for extracting contours.
"""
import csv
import librosa
import numpy as np
import os
from scipy import signal
import subprocess
from subprocess import CalledProcessError
import tempfile as tmp

from motif.core import ContourExtractor
from motif.core import Contours


def _check_binary():
    '''Check if the binary is available.

    Returns
    -------
    True if available, False otherwise
    '''
    hll_exists = True
    try:
        subprocess.check_output(['which', 'hll'])
    except CalledProcessError:
        hll_exists = False

    return hll_exists


BINARY_AVAILABLE = _check_binary()


class HLL(ContourExtractor):

    def __init__(self):
        # seed detection parameters
        self.hop_length = 8192
        self.n_cqt_bins = 12*6
        self.bins_per_octave = 12
        self.min_note = 'E1'
        self.med_filt_len = 5
        self.peak_thresh = 0.4
        ## librosa peak pick params for seed detection
        self.pre_max = 3
        self.post_max = 3
        self.pre_avg = 5
        self.post_avg = 7
        self.delta = 0.02
        self.wait = 10

        # HLL paramters
        self.n_harmonics = 5
        self.f_cutoff = 30  # Hz
        self.tracking_gain = 0.0005
        self.min_contour_len = 11025
        self.amplitude_threshold = 0.001
        self.tracking_update_threshold = 70.0

        ContourExtractor.__init__(self)

    @property
    def sample_rate(self):
        """Sample rate of output contours"""
        return 256.0/44100.0

    @classmethod
    def get_id(cls):
        """Identifier of this extractor."""
        return "hll"

    def compute_contours(self, audio_filepath):
        """Compute contours using Harmonic Locked Loops.
        This calls a binary in the background, which creates a csv file.
        The csv file is loaded into memory and the file is deleted, unless
        clean=False. When recompute=False, this will first look for an existing
        precomputed contour file and if successful will load it directly.

        Parameters
        ----------
        audio_filepath : str
            Path to audio file.

        Returns
        -------
        Instance of Contours object

        """
        if not BINARY_AVAILABLE:
            raise EnvironmentError(
                "The binary {} needed to compute these contours is "
                "not available. To fix this, copy the hll binary from "
                "``motif/extract/resources/`` into ``/usr/local/bin``."
            )

        if not os.path.exists(audio_filepath):
            raise IOError(
                "The audio file {} does not exist".format(audio_filepath)
            )

        tmp_audio = self._preprocess_audio(
            audio_filepath, normalize_format=True, normalize_volume=True
        )

        seed_fpath = self.get_seeds(tmp_audio)
        contours_fpath = tmp.mktemp('.csv')

        args = [
            "hll",
            "{}".format(tmp_audio),
            "{}".format(seed_fpath),
            "{}".format(contours_fpath),
            "{}".format(self.n_harmonics),
            "{}".format(self.f_cutoff),
            "{}".format(self.tracking_gain),
            "{}".format(self.min_contour_len),
            "{}".format(self.amplitude_threshold),
            "{}".format(self.tracking_update_threshold)
        ]
        os.system(' '.join(args))

        if not os.path.exists(contours_fpath):
            raise IOError(
                "Unable to find HLL output file {}".format(contours_fpath)
            )

        c_numbers, c_times, c_freqs, c_sal = self._load_contours(contours_fpath)

        os.remove(contours_fpath)
        os.remove(tmp_audio)
        os.remove(seed_fpath)

        return Contours(
            c_numbers, c_times, c_freqs, c_sal, self.sample_rate, audio_filepath
        )

    def get_seeds(self, audio_filepath):
        y, sr = librosa.load(audio_filepath, sr=None)
        cqt, samples, freqs = self._compute_cqt(y, sr)
        seeds = self._pick_seeds_cqt(cqt, freqs, samples)

        seeds_fpath = tmp.mktemp('.csv')
        with open(seeds_fpath, 'w') as fhandle:
            writer = csv.writer(fhandle, delimiter=',')
            writer.writerows(seeds)
        return seeds_fpath

    def _compute_cqt(self, y, sr):
        fmin = librosa.note_to_hz(self.min_note)
        cqt = np.abs(librosa.cqt(
            y, sr=sr, hop_length=self.hop_length, fmin=fmin, filter_scale=4,
            bins_per_octave=self.bins_per_octave, n_bins=self.n_cqt_bins,
            real=False
        ))
        n_time_frames = cqt.shape[1]
        freqs = librosa.cqt_frequencies(
            fmin=fmin, bins_per_octave=self.bins_per_octave,
            n_bins=self.n_cqt_bins
        )
        samples = librosa.frames_to_samples(
            range(n_time_frames), hop_length=self.hop_length
        )

        # compute log amplitude
        cqt_log = librosa.logamplitude(cqt**2, ref_power=np.max)
        cqt_log = cqt_log - np.min(np.min(cqt_log))
        cqt_log = cqt_log/(np.max(np.max(cqt_log)))

        return cqt_log, samples, freqs

    def _pick_seeds_cqt(self, cqt, cqt_freqs, samples):
        seeds = []
        for i, freq in enumerate(cqt_freqs):
            freq_band = cqt[i, :]
            freq_band = freq_band/np.max(freq_band)
            freq_band_smooth = signal.medfilt(freq_band, self.med_filt_len)
            peak_locs = librosa.util.peak_pick(
                freq_band_smooth, self.pre_max, self.post_max, self.pre_avg,
                self.post_avg, self.delta, self.wait
            )
            if len(peak_locs) > 0:
                peak_locs = peak_locs[
                    (freq_band_smooth[peak_locs] > self.peak_thresh)
                ]
                for peak_loc in peak_locs:
                    sample = samples[peak_loc]
                    seeds.append([sample, freq])
        seeds = np.array(seeds)
        return seeds

    def _load_contours(self, fpath):
        """ Load contour data from an HLL csv file.

        Parameters
        ----------
        fpath : str
            Path to output csv file.

        Returns
        -------
        index : np.array
            Array of contour numbers
        times : np.array
            Array of contour times
        freqs : np.array
            Array of contour frequencies
        contour_sal : np.array
            Array of contour saliences

        """
        index = []
        times = []
        freqs = []
        contour_sal = []
        with open(fpath, 'r') as fhandle:
            reader = csv.reader(fhandle, delimiter=',')
            for row in reader:
                index.append(row[0])
                times.append(row[1])
                freqs.append(row[2])
                contour_sal.append(row[3:])

        # Add column with annotation values in cents
        index = np.array(index, dtype=int)
        times = np.array(times, dtype=float) / self.sample_rate
        freqs = np.array(freqs, dtype=float)
        contour_sal = np.array(contour_sal, dtype=float)

        sort_idx = np.lexsort((times, index))

        return (
            index[sort_idx], times[sort_idx], freqs[sort_idx],
            contour_sal[sort_idx]
        )

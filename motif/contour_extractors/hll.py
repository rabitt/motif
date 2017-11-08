"""HLL method for extracting contours.
"""
import csv
import librosa
import numpy as np
import os
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
    '''HLL method for extracting contours.

    Parameters
    ----------
    hop_size : int, default=8192
        Seed detection CQT hop size.
    bins_per_octave : int, default=12
        Number of seed detection CQT bins per octave.
    min_note : str, default='E1'
        Minimum seed detection CQT note.
    peak_thresh : float, default=0.4
        Seed detection peak picking threshold.
    filter_scale : float, defualt=2.0
        CQT filter scale.
    wait : int >= 0, default=2
        Peak-picking number of samples to wait after picking a peak.
    avg_filt_len : int >= 0, default=12
        Peak-picking average filter length.
    pre_max : int >= 0, default=3
        Peak-picking num samples before `n` over which max is computed
    post_max : int >= 1, default=3
        Peak-picking num samples after `n` over which max is computed
    pre_avg : int >= 0, default=3
        Peak-picking num samples before `n` over which mean is computed
    post_avg : int >= 1, default=3
        Peak-picking num samples after `n` over which mean is computed
    delta : float >= 0, default=0.02
        Peak-picking threshold offset for mean
    n_harmonics : int, default=5
        Number of HLL harmonics.
    f_cutoff : float, default=30
        HLL cutoff frequency in Hz.
    tracking_gain : float, default=0.0005
        HLL tracking gain.
    min_contour_len_samples : int, default=11025
        HLL minimum number of samples in a single contour.
    amplitude_threshold : float, default=0.001
        HLL minimum amplitude threshold.
    tracking_update_threshold : float, default=70.0
        HLL tracking update threshold.

    Attributes
    ----------
    hop_size : int
        Seed detection CQT hop size.
    n_cqt_bins : int
        Number of seed detection CQT bins.
    bins_per_octave : int
        Number of seed detection CQT bins per octave.
    min_note : str
        Minimum seed detection CQT note.
    med_filt_len : int
        Seed detection frequency band median filter length.
    peak_thresh : float
        Seed detection peak picking threshold.
    pre_max : int >= 0
        Peak-picking number of samples before `n` over which max is computed
    post_max : int >= 1
        Peak-picking number of samples after `n` over which max is computed
    pre_avg : int >= 0
        Peak-picking number of samples before `n` over which mean is computed
    post_avg : int >= 1
        Peak-picking number of samples after `n` over which mean is computed
    delta : float >= 0
        Peak-picking threshold offset for mean
    wait : int >= 0
        Peak-picking number of samples to wait after picking a peak
    n_harmonics : int
        Number of HLL harmonics.
    f_cutoff : float
        HLL cutoff frequency in Hz.
    tracking_gain : float
        HLL tracking gain.
    min_contour_len_samples : int
        HLL minimum number of samples in a single contour.
    amplitude_threshold : float
        HLL minimum amplitude threshold.
    tracking_update_threshold : float
        HLL tracking update threshold.

    '''
    def __init__(self, hop_size=8192, n_octaves=6, bins_per_octave=12,
                 min_note='E1', peak_thresh=0.4, filter_scale=2.0, wait=2,
                 avg_filt_len=12, pre_max=3, post_max=3, pre_avg=3, post_avg=3,
                 delta=0.02, n_harmonics=5, f_cutoff=30, tracking_gain=0.0005,
                 min_contour_len_samples=11025, amplitude_threshold=0.001,
                 tracking_update_threshold=70.0):
        ''' Init method.
        '''
        # seed detection parameters
        self.hop_size = hop_size
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.min_note = min_note
        self.peak_thresh = peak_thresh
        self.filter_scale = filter_scale
        self.avg_filt_len = avg_filt_len

        # librosa peak pick params for seed detection
        self.pre_max = pre_max
        self.post_max = post_max
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.delta = delta
        self.wait = wait

        # HLL paramters
        self.n_harmonics = n_harmonics
        self.f_cutoff = f_cutoff  # Hz
        self.tracking_gain = tracking_gain
        self.min_contour_len_samples = min_contour_len_samples
        self.amplitude_threshold = amplitude_threshold
        self.tracking_update_threshold = tracking_update_threshold

        ContourExtractor.__init__(self)

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
        return self.audio_samplerate / 256.0

    @property
    def min_contour_len(self):
        """Minimum allowed contour length.

        Returns
        -------
        min_contour_len : float
            Minimum allowed contour length in seconds.

        """
        return self.min_contour_len_samples / self.audio_samplerate

    @classmethod
    def get_id(cls):
        """Identifier of this extractor.

        Returns
        -------
        id : str
            Identifier of this extractor.

        """
        return "hll"

    def compute_contours(self, audio_filepath):
        """Compute contours using Harmonic Locked Loops.
        This calls a binary in the background, which creates a csv file.
        The csv file is loaded into memory and the file is deleted.

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
            "{}".format(self.min_contour_len_samples),
            "{}".format(self.amplitude_threshold),
            "{}".format(self.tracking_update_threshold)
        ]
        os.system(' '.join(args))

        if not os.path.exists(contours_fpath):
            raise IOError(
                "Unable to find HLL output file {}".format(contours_fpath)
            )

        c_numbers, c_times, c_freqs, c_sal = self._load_contours(
            contours_fpath
        )

        c_numbers, c_times, c_freqs, c_sal = self._sort_contours(
            c_numbers, c_times, c_freqs, c_sal
        )

        os.remove(contours_fpath)
        os.remove(tmp_audio)
        os.remove(seed_fpath)

        (c_numbers, c_times, c_freqs, c_sal) = self._postprocess_contours(
            c_numbers, c_times, c_freqs, c_sal
        )

        return Contours(
            c_numbers, c_times, c_freqs, c_sal, self.sample_rate,
            audio_filepath
        )

    def get_seeds(self, audio_filepath):
        """Get the seeds file to pass to the HLL tracker.

        Parameters
        ----------
        audio_filepath : str
            Path to audio file.

        Returns
        -------
        seeds_fpath : str
            Path to the seeds output file.

        """
        y, sr = librosa.load(audio_filepath, sr=44100)
        y_harmonic = librosa.effects.harmonic(y)
        cqt, samples, freqs = self._compute_cqt(y_harmonic, sr)
        seeds = self._pick_seeds_cqt(cqt, freqs, samples)

        seeds_fpath = tmp.mktemp('.csv')
        with open(seeds_fpath, 'w') as fhandle:
            writer = csv.writer(fhandle, delimiter=',')
            writer.writerows(seeds)
        return seeds_fpath

    def _moving_average(self, a):
        """Compute the moving average of a signal.

        Parameters
        ----------
        a : np.array
            Signal

        Returns
        -------
        a_avg : np.array
            Moving average of signal.

        """
        n = self.avg_filt_len
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def _norm_matrix(self, mat, overall=True, time=True, freq=True):
        """Normalize a matrix overall, in time, or in frequency.

        Parameters
        ----------
        mat : np.array
            Matrix.
        overall : bool, default=True
            If True, normalizes first by the overall amplitudes.
        time : bool, default=True
            If True, normalizes in time.
        freq : bool, default=True
            If True, normalizes in frequency.

        Returns
        -------
        mat_norm : np.array
            Normalized matrix

        """
        if overall:
            mat = mat - np.min(mat)
            m = np.max(mat)
            if m == 0:
                m = 1
            mat = mat / m

        if time:
            mat = (mat.T - np.min(mat, axis=1)).T
            m = np.max(mat, axis=1)
            m[m == 0] = 1
            mat = (mat.T / m).T

        if freq:
            mat = mat - np.min(mat, axis=0)
            m = np.max(mat, axis=0)
            m[m == 0] = 1
            mat = mat / m

        return mat

    def _compute_cqt(self, y, sr):
        """Compute a CQT.

        Parameters
        ----------
        y : np.array
            Audio signal
        sr : float
            Audio singal sample rate

        Returns
        -------
        cqt_log : np.array [n_samples, n_freqs]
            Log amplitude CQT.
        samples : np.array [n_samples]
            CQT time stamps.
        freqs : np.array [n_freqs]
            CQT frequencies.

        """
        fmin = librosa.note_to_hz(self.min_note)
        bins_per_octave = 12
        n_cqt_bins = bins_per_octave * self.n_octaves
        cqt = np.abs(librosa.cqt(
            y, sr=sr, hop_length=self.hop_size, fmin=fmin,
            filter_scale=self.filter_scale,
            bins_per_octave=bins_per_octave, n_bins=n_cqt_bins
        ))

        cqt = self._norm_matrix(cqt)

        n_time_frames = cqt.shape[1]

        freqs = librosa.cqt_frequencies(
            fmin=fmin, bins_per_octave=bins_per_octave,
            n_bins=n_cqt_bins
        )
        samples = librosa.frames_to_samples(
            range(n_time_frames), hop_length=self.hop_size
        )

        return cqt, samples, freqs

    def _pick_seeds_cqt(self, cqt, cqt_freqs, samples):
        """Compute a CQT.

        Parameters
        ----------
        cqt : np.array [n_samples, n_freqs]
            Log amplitude CQT.
        freqs : np.array [n_freqs]
            CQT frequencies.
        samples : np.array [n_samples]
            CQT time stamps.

        Returns
        -------
        seeds : np.array [n_seeds, 2]
            Array of time, frequency seeds

        """
        seeds = []
        for i, freq in enumerate(cqt_freqs):
            freq_band = cqt[i, :]

            freq_band_smooth = self._moving_average(freq_band)
            peak_locs = librosa.util.peak_pick(
                freq_band_smooth, self.pre_max, self.post_max, self.pre_avg,
                self.post_avg, self.delta, self.wait
            )
            if len(peak_locs) > 0:
                peak_locs = peak_locs[
                    (freq_band[peak_locs] > self.peak_thresh)
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
                contour_sal.append(row[3])  # TODO: was 3: - generalize later!

        # Add column with annotation values in cents
        index = np.array(index, dtype=int)
        times = np.array(times, dtype=float) / self.audio_samplerate
        freqs = np.array(freqs, dtype=float)
        contour_sal = np.array(contour_sal, dtype=float)

        return (index, times, freqs, contour_sal)

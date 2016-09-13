# -*- coding: utf-8 -*-
"""Salamon's method for extracting contours
"""
import csv
import numpy as np
import os
import subprocess
from subprocess import CalledProcessError

from motif.core import ContourExtractor
from motif.core import Contours

OUTPUT_FILE_STRING = "vamp_melodia-contours_melodia-contours_contoursall"
VAMP_PLUGIN = "vamp:melodia-contours:melodia-contours:contoursall"


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


class Salamon(ContourExtractor):
    '''Salamon's method for extracting contours
    '''

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
        return 44100.0 / 128.0

    @property
    def min_contour_len(self):
        """Minimum allowed contour length.

        Returns
        -------
        min_contour_len : float
            Minimum allowed contour length in seconds.

        """
        return 0.0

    @classmethod
    def get_id(cls):
        """Identifier of this extractor.

        Returns
        -------
        id : str
            Identifier of this extractor.

        """
        return "salamon"

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
        if not BINARY_AVAILABLE:
            raise EnvironmentError(
                "Either the vamp plugin {} needed to compute these contours or "
                "sonic-annotator is not available.".format(VAMP_PLUGIN)
            )

        if not os.path.exists(audio_filepath):
            raise IOError(
                "The audio file {} does not exist".format(audio_filepath)
            )

        tmp_audio = self._preprocess_audio(
            audio_filepath, normalize_format=True, normalize_volume=True
        )

        input_file_name = os.path.basename(tmp_audio)
        output_file_name = "{}_{}.csv".format(
            input_file_name.split('.')[0], OUTPUT_FILE_STRING
        )
        output_dir = os.path.dirname(tmp_audio)
        output_path = os.path.join(output_dir, output_file_name)

        args = [
            "sonic-annotator", "-d", VAMP_PLUGIN,
            "{}".format(tmp_audio), "-w", "csv", "--csv-force"
        ]
        os.system(' '.join(args))

        if not os.path.exists(output_path):
            raise IOError(
                "Unable to find vamp output file {}".format(output_path)
            )

        c_numbers, c_times, c_freqs, c_sal = _load_contours(output_path)

        os.remove(output_path)
        os.remove(tmp_audio)

        return Contours(
            c_numbers, c_times, c_freqs, c_sal, self.sample_rate, audio_filepath
        )


def _load_contours(fpath):
    """ Load contour data from vamp output csv file.

    Parameters
    ----------
    fpath : str
        Path to vamp output csv file.

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
        contour_num = 0
        for row in reader:
            index.extend([contour_num]*len(row[14::3]))
            times.extend(row[14::3])
            freqs.extend(row[15::3])
            contour_sal.extend(row[16::3])
            contour_num += 1

    n_rows = np.min([
        len(index), len(times),
        len(freqs), len(contour_sal)
    ])

    index = np.array(index[:n_rows], dtype=int)
    times = np.array(times[:n_rows], dtype=float)
    freqs = np.array(freqs[:n_rows], dtype=float)
    contour_sal = np.array(contour_sal[:n_rows], dtype=float)

    non_nan_rows = ~(np.logical_or(
        np.logical_or(np.isnan(times), np.isnan(freqs)),
        np.isnan(contour_sal)))
    index = index[non_nan_rows]
    times = times[non_nan_rows]
    freqs = freqs[non_nan_rows]
    contour_sal = contour_sal[non_nan_rows]

    return index, times, freqs, contour_sal

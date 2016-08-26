# -*- coding: utf-8 -*-
"""Salamon's method for extracting contours
"""
import csv
import numpy as np
import os

from motif.core import ContourExtractor
from motif.core import Contours

SALAMON_CONTOUR_STRING = "vamp_melodia-contours_melodia-contours_contoursall"


class Salamon(ContourExtractor):

    @classmethod
    def get_id(cls):
        """Identifier of this extractor."""
        return "salamon"

    def compute_contours(self):
        """Compute contours as in Justin Salamon's melodia.
        This calls a vamp plugin in the background, which creates a csv file.
        The csv file is loaded into memory and the file is deleted, unless
        clean=False. When recompute=False, this will first look for an existing
        precomputed contour file and if successful will load it directly.

        Returns
        -------
        Instance of Contours object
        """
        input_file_name = os.path.basename(self.audio_filepath)
        output_file_name = "{}_{}.csv".format(
            input_file_name.split('.')[0], SALAMON_CONTOUR_STRING
        )
        output_dir = os.path.dirname(self.audio_filepath)
        output_path = os.path.join(output_dir, output_file_name)
        if self.recompute or not os.path.exists(output_path):
            args = [
                "sonic-annotator", "-d", 
                "vamp:melodia-contours:melodia-contours:contoursall",
                "{}".format(self.audio_filepath), "-w", "csv", "--csv-force"
            ]
            os.system(' '.join(args))

        if not os.path.exists(output_path):
            raise IOError(
                "Unable to find vamp output file {}".format(output_path)
            )

        c_numbers, c_times, c_freqs, c_sal = _load_contours(output_path)

        if self.clean:
            os.remove(output_path)

        return Contours(c_numbers, c_times, c_freqs, c_sal)


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

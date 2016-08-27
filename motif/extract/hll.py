# -*- coding: utf-8 -*-
"""HLL method for extracting contours.
"""
import numpy as np
import csv
import os
import tempfile as tmp
from motif.core import ContourExtractor
from motif.core import Contours


class HLL(ContourExtractor):

    @classmethod
    def get_id(cls):
        """Identifier of this extractor."""
        return "hll"

    def compute_contours(self):
        """Compute contours using Harmonic Locked Loops.
        This calls a binary in the background, which creates a csv file.
        The csv file is loaded into memory and the file is deleted, unless
        clean=False. When recompute=False, this will first look for an existing
        precomputed contour file and if successful will load it directly.

        Returns
        -------
        Instance of Contours object
        """
        if self.recompute:
            output_file_object = tmp.NamedTemporaryFile('csv')
            output_path = output_file_object.name
        else:
            input_name = os.path.basename(self.audio_filepath)
            input_dir = os.path.dirname(self.audio_filepath)
            output_name = "{}_HLL_contours.csv".format(input_name.split('.'))
            output_path = os.path.join(input_dir, output_name)

        if not os.path.exists(output_path):
            args = [
                "run_hll",
                "{}".format(self.audio_filepath), "{}".format(output_path.name)
            ]
            os.system(' '.join(args))

        if not os.path.exists(output_path):
            raise IOError(
                "Unable to find HLL output file {}".format(output_path)
            )

        c_numbers, c_times, c_freqs, c_sal = _load_contours(output_path)

        if self.clean:
            os.remove(output_path)

        return Contours(c_numbers, c_times, c_freqs, c_sal)


def _load_contours(fpath):
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
    times = np.array(times, dtype=float)
    freqs = np.array(freqs, dtype=float)
    contour_sal = np.array(contour_sal, dtype=float)

    return index, times, freqs, contour_sal

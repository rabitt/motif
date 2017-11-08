# -*- coding: utf-8 -*-
""" Utility methods for motif
"""
import csv
from mir_eval import melody
import numpy as np
import os


def validate_contours(index, times, freqs, salience):
    '''Check that contour input is well formed.

    Parameters
    ----------
    index : np.array
        Array of contour numbers
    times : np.array
        Array of contour times
    freqs : np.array
        Array of contour frequencies
    salience : np.array
        Array of contour saliences
    sample_rate : float
        Contour sample rate.
    audio_filepath : str
        Path to audio file contours were extracted from

    '''
    N = len(index)
    if any([len(times) != N, len(freqs) != N, len(salience) != N]):
        raise ValueError(
            "the arrays index, times, freqs, and salience "
            "must be the same length."
        )


def format_contour_data(frequencies):
    """ Convert contour frequencies to cents + voicing.

    Parameters
    ----------
    frequencies : np.array
        Contour frequency values

    Returns
    -------
    est_cent : np.array
        Contour frequencies in cents
    est_voicing : np.array
        Contour voicings

    """
    est_freqs, est_voicing = melody.freq_to_voicing(frequencies)
    est_cents = melody.hz2cents(est_freqs)
    return est_cents, est_voicing


def format_annotation(new_times, annot_times, annot_freqs):
    """ Format an annotation file and resample to a uniform timebase.

    Parameters
    ----------
    new_times : np.array
        Times to resample to
    annot_times : np.array
        Annotation time stamps
    annot_freqs : np.array
        Annotation frequency values

    Returns
    -------

    ref_cent : np.array
        Annotation frequencies in cents at the new timescale
    ref_voicing : np.array
        Annotation voicings at the new timescale

    """
    ref_freq, ref_voicing = melody.freq_to_voicing(annot_freqs)
    ref_cent = melody.hz2cents(ref_freq)

    ref_cent, ref_voicing = melody.resample_melody_series(
        annot_times, ref_cent, ref_voicing, new_times,
        kind='linear'
    )
    return ref_cent, ref_voicing


def get_snippet_idx(snippet, full_array):
    """ Find the indices of ``full_array`` where ``snippet`` is present.
    Assumes both ``snippet`` and ``full_array`` are ordered.

    Parameters
    ----------
    snippet : np.array
        Array of ordered time stamps
    full_array : np.array
        Array of ordered time stamps

    Returns
    -------
    idx : np.array
        Array of booleans indicating where in ``full_array`` ``snippet``
        is present.

    """
    idx = np.logical_and(
        full_array >= snippet[0], full_array <= snippet[-1]
    )
    return idx


def load_annotation(annotation_fpath, n_freqs=1, to_array=True, rm_zeros=False):
    """ Load an annotation from a csv file.

    Parameters
    ----------
    annotation_fpath : str
        Path to annotation file.
    n_freqs : int or None
        Number of frequencies to read, or None to use max
    to_array : bool
        If True, returns annot_freqs as a numpy array
        If False, returns annot_freqs as a list of lists.

    Returns
    -------
    annot_times : array
        Annotation time stamps
    annot_freqs : array
        Annotation frequency values

    """
    end_idx = None if n_freqs is None else n_freqs + 1
    if not os.path.exists(annotation_fpath):
        raise IOError("The annotation path {} does not exist.")

    annot_times = []
    annot_freqs = []
    with open(annotation_fpath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for row in reader:
            annot_times.append(row[0])
            if rm_zeros:
                temp_freqs = [r for r in row[1:end_idx] if float(r) > 0]
            else:
                temp_freqs = [r for r in row[1:end_idx]]
            annot_freqs.append(temp_freqs)

    annot_times = np.array(annot_times, dtype=float)
    annot_freqs = [np.array(f).astype(float) for f in annot_freqs]
    if to_array:
        annot_freqs = np.array(annot_freqs, dtype=float).flatten()
    return annot_times, annot_freqs

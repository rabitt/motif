# -*- coding: utf-8 -*-
""" Plotting helper functions
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

from .utils import load_annotation


def plot_with_annotation(ctr, annotation_fpath, single_f0=True):
    '''Plot all contours in a single color against an annotation.
    Useful for viewing contour coverage/accuracy.

    Parameters
    ----------
    ctr : Contours
        An instance of a Contours object
    annotation_fpath : str
        Path to an annotation file.
    single_f0 : bool
        If True, assumes the annotation contains one pitch at a time.
        If False, assumes there may be multiple ground truth pitches.

    '''
    if single_f0:
        ref_times, ref_freqs = load_annotation(
            annotation_fpath, n_freqs=1, to_array=False
        )
        ref_freqs = [f if f[0] != 0 else np.array([]) for f in ref_freqs]

    else:
        ref_times, ref_freqs = load_annotation(
            annotation_fpath, n_freqs=None, to_array=False
        )

    r_times = []
    r_freqs = []
    for t, freq in zip(ref_times, ref_freqs):
        r_times.extend([t for f in freq])
        r_freqs.extend([f for f in freq])

    # plot annotation
    plt.semilogy(
        np.array(r_times), np.array(r_freqs), 'ok', basey=2, markersize=5
    )

    # plot contours
    c1 = sns.color_palette('deep', 1)[0]
    for i in ctr.nums:
        plt.semilogy(ctr.contour_times(i), ctr.contour_freqs(i), c1,
                     basey=2, markersize=2)

    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.axis('tight')



def plot_contours(ctr, style='contour'):
    '''Plot contours.

    Parameters
    ----------
    ctr : Contours
        An instance of a Contours object
    style : str
        One of:
            - 'contour': plot each extracted contour, where each contour
                gets its own color.
            - 'salience': plot the contours where the colors denote the
                salience.

    '''
    if style == 'contour':
        for i in ctr.nums:
            plt.plot(ctr.contour_times(i), ctr.contour_freqs(i))
    elif style == 'salience':
        plt.scatter(
            ctr.times, ctr.freqs,
            c=(ctr.salience / np.max(ctr.salience)), cmap='BuGn',
            edgecolors='face', marker='.'
        )
        plt.colorbar()

    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.axis('tight')

"""Melodia feature extractor.
"""
import numpy as np

from motif.core import FeatureExtractor
from motif.feature_extractors import utils


class MelodiaFeatures(FeatureExtractor):
    '''Melodia feature extractor

    Attributes
    ----------
    ref_hz : float
        Reference frequency (Hz) for converting to cents.

    '''
    def __init__(self):
        self.ref_hz = 55.0
        FeatureExtractor.__init__(self)

    def get_feature_vector(self, times, freqs_hz, salience, sample_rate):
        """Get feature vector for a contour.

        Parameters
        ----------
        times : np.array
            Contour times
        freqs_hz : np.array
            Contour frequencies (Hz)
        salience : np.array
            Contour salience
        sample_rate : float
            Contour sample rate.

        Returns
        -------
        feature_vector : np.array
            Feature vector.

        """
        freqs_cents = utils.hz_to_cents(freqs_hz, ref_hz=self.ref_hz)

        features = [
            utils.get_contour_onset(times),
            utils.get_contour_offset(times),
            utils.get_contour_duration(times),
            utils.get_mean(freqs_cents),
            utils.get_std(freqs_cents),
            utils.get_mean(salience),
            utils.get_std(salience),
            utils.get_sum(salience),
            utils.vibrato_essentia(freqs_cents, sample_rate)  # dimension 4
        ]

        return np.concatenate(features)

    @property
    def feature_names(self):
        """Get feature names.

        Returns
        -------
        feature_names : list
            List of feature names.

        """
        feature_names = [
            'onset',
            'offset',
            'duration',
            'pitch mean (cents)',
            'pitch stddev (cents)',
            'salience mean',
            'salience stdev',
            'salience total',
            'vibrato',
            'vibrato rate',
            'vibrato extent (cents)',
            'vibrato coverage'
        ]
        return feature_names

    @classmethod
    def get_id(cls):
        """ The FeatureExtractor identifier

        Returns
        -------
        id : string
            class identifier
        """
        return 'melodia'

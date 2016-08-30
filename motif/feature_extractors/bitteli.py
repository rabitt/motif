"""Features as in Melodia.
"""
from motif.core import FeatureExtractor
from motif.feature_extractors import utils
import numpy as np

class BitteliFeatures(FeatureExtractor):

    def __init__(self):
        self.ref_hz = 55.0
        self.poly_degree = 5
        self.min_freq = 3
        self.max_freq = 30
        self.freq_step = 0.1
        self.vibrato_threshold = 0.25
        FeatureExtractor.__init__(self)

    def get_feature_vector(self, times, freqs_hz, salience, sample_rate):
        """
        Parameters
        ----------


        Returns
        -------

        """
        freqs_cents = utils.hz_to_cents(freqs_hz, ref_hz=self.ref_hz)

        features = [
            utils.get_contour_shape_features(
                times, freqs_cents, sample_rate, poly_degree=self.poly_degree,
                min_freq=self.min_freq, max_freq=self.max_freq,
                freq_step=self.freq_step,
                vibrato_threshold=self.vibrato_threshold),
            utils.get_polynomial_fit_features(
                times, salience, n_deg=self.poly_degree, norm=False),
            utils.get_contour_onset(times),
            utils.get_contour_offset(times),
            utils.get_contour_duration(times),
            utils.get_mean(freqs_cents),
            utils.get_std(freqs_cents),
            utils.get_range(freqs_cents),
            utils.get_total_variation(freqs_cents),
            utils.get_mean(salience),
            utils.get_std(salience),
            utils.get_range(salience),
            utils.get_sum(salience),
            utils.get_total_variation(salience)
        ]

        return np.concatenate(features)

    @property
    def feature_names(self):
        feature_names = [
            'vibrato rate',
            'vibrato extent',
            'vibrato coverage',
            'vibrato coverage - beginning',
            'vibrato coverage - middle',
            'vibrato coverage - end',
            '0th polynomial coeff - freq',
            '1st polynomial coeff - freq',
            '2nd polynomial coeff - freq',
            '3rd polynomial coeff - freq',
            '4th polynomial coeff - freq',
            '5th polynomial coeff - freq',
            'polynomial fit residual - freq',
            'overall model fit residual - freq',
            '0th polynomial coeff - salience',
            '1st polynomial coeff - salience',
            '2nd polynomial coeff - salience',
            '3rd polynomial coeff - salience',
            '4th polynomial coeff - salience',
            '5th polynomial coeff - salience',
            'polynomial fit residual - salience',
            'onset',
            'offset',
            'duration',
            'pitch mean (cents)',
            'pitch stddev (cents)',
            'pitch range (cents)',
            'pitch total variation',
            'salience mean',
            'salience stdev',
            'salience range',
            'salience total',
            'salience total variation'
        ]
        return feature_names

    @classmethod
    def get_id(cls):
        """Method to get the id of the feature type"""
        return 'bitteli'

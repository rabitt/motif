"""Features as in Melodia.
"""
from motif.core import ContourFeatures
from motif.features import utils
import numpy as np

class MelodiaFeatures(ContourFeatures):

    def __init__(self):
        self.ref_hz = 55.0
        ContourFeatures.__init__(self)

    def get_feature_vector(self, times, freqs_hz, salience, sample_rate):
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

    def set_feature_names(self):
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
        """Method to get the id of the feature type"""
        return 'melodia'

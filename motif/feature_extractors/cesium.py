"""Features from cesium module.
"""
# from cesium import science_feature_tools
from motif.core import FeatureExtractor
import numpy as np


class CesiumFeatures(FeatureExtractor):

    def get_feature_vector(self, times, freqs_hz, salience, sample_rate):
        raise NotImplementedError
        # contour_norm = freqs_hz / np.max(freqs_hz)
        # error = 1 - salience  # error is opposite of salience
        # ts_feat_dict = science_feature_tools.generate_science_features(
        #     times, contour_norm, error
        # )
        # features = np.array(ts_feat_dict.values())
        # return features

    @property
    def feature_names(self):
        feature_names = range(80)
        return feature_names

    @classmethod
    def get_id(cls):
        """Method to get the id of the feature type"""
        return 'cesium'

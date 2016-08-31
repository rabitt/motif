"""Celsium feature extractor.
"""
# from cesium import science_feature_tools
from motif.core import FeatureExtractor
import numpy as np


class CesiumFeatures(FeatureExtractor):
    '''Cesium feature extractor

    Attributes
    ----------

    '''
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
        """Get feature names.

        Returns
        -------
        feature_names : list
            List of feature names.

        """
        feature_names = range(80)
        return feature_names

    @classmethod
    def get_id(cls):
        """ The FeatureExtractor identifier

        Returns
        -------
        id : string
            class identifier
        """
        return 'cesium'

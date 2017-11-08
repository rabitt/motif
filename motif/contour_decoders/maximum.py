"""Maximum contour decoder.
"""
from __future__ import print_function
import numpy as np
from motif.core import ContourDecoder


class MaxDecoder(ContourDecoder):
    ''' Maximum contour decoder.
    '''
    def __init__(self, thresh=0.5, use_salience=False):
        '''Init method.
        '''
        self.use_salience = use_salience
        self.thresh = thresh
        ContourDecoder.__init__(self)

    def decode(self, ctr, Y):

        n_uniform_times = len(ctr.uniform_times)
        freqs = [[] for i in range(n_uniform_times)]
        scores = [[] for i in range(n_uniform_times)]

        time_idx = np.round(ctr.times * ctr.sample_rate).astype(int)
        time_idx[time_idx >= n_uniform_times] = n_uniform_times - 1

        contour_num_map = {}
        for k, n in enumerate(ctr.nums):
            contour_num_map[n] = k

        for i, freq in zip(time_idx, ctr.freqs):
            n = ctr.index[i]
            k = contour_num_map[n]
            if Y[k] >= self.thresh:
                freqs[i].append(freq)
            else:
                freqs[i].append(-1.0 * freq)

            if self.use_salience:
                scores[i].append(Y[k] * ctr.salience[i])
            else:
                scores[i].append(Y[k])

        single_f0 = []
        for i, f in enumerate(freqs):
            if len(f) > 0:

                single_f0.append(f[np.argmax(scores[i])])
            else:
                single_f0.append(0.0)

        return np.array(ctr.uniform_times), np.array(single_f0)

    @classmethod
    def get_id(cls):
        return 'maximum'

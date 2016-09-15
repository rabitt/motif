"""Viterbi contour decoder.
"""
from motif.core import ContourDecoder


class ViterbiDecoder(ContourDecoder):
    ''' Viterbi contour decoder.
    '''
    def decode(self, ctr, Y):
        raise NotImplementedError

    @classmethod
    def get_id(cls):
        return 'viterbi'

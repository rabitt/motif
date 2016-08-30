"""Viterbi decoder.
"""
from motif.core import ContourDecoder

class ViterbiDecoder(ContourDecoder):

    def decode(self, ctr, Y):
        raise NotImplementedError

    @classmethod
    def get_id(cls):
        return 'viterbi'

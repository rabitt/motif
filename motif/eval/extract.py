# # -*- coding: utf-8 -*-
# """ Evaluate the output of contour extraction
# TODO: This is currently old and hacky...make it nice!
# """
# from mir_eval import multipitch

# import csv
# import numpy as np
# import argparse

# HOP = 256
# FS = 44100


# def get_time_stamps(total_duration, hop):
#     time_stamps = []
#     n_stamps = int(np.ceil((total_duration*FS)/hop)) + 1
#     time_stamps = np.arange(n_stamps) * (hop / FS)
#     return time_stamps


# def sec_to_idx(time_in_seconds, hop, fs=FS):
#     return int(np.round(time_in_seconds*float(fs)/float(hop)))


# def make_blank_multif0_sequence(total_duration, hop):
#     time_stamps = get_time_stamps(total_duration, hop)
#     multif0 = [[t] for t in time_stamps]
#     return multif0


# def timefreq_to_mirex(times, freqs, hop):
#     duration = np.max(times)
#     multif0 = make_blank_multif0_sequence(duration, hop)
#     for time, freq in zip(times, freqs):
#         time_idx = sec_to_idx(time, hop)
#         multif0[time_idx].append(freq)
#     return multif0


# def eval(ref_times, ref_freqs, est_contour_times, est_contour_freqs):
#     hop = 256 # TODO: fix this
#     multif0 = timefreq_to_mirex(est_contour_times, est_contour_freqs, hop)
#     multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs)

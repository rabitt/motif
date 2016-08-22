import numpy as np
import csv
import os
import tempfile as tmp


def hll(audio_fpath, recompute=True, clean=True):
    output_path = tmp.NamedTemporaryFile('csv')
    if recompute or not os.path.exists(output_path):
        args = [
            "run_hll", 
            "{}".format(audio_fpath), "{}".format(output_path.name)
        ]
        os.system(' '.join(args))

    if not os.path.exists(output_path):
        raise IOError("Unable to find HLL output file {}".format(output_path))

    c_numbers, c_times, c_freqs, c_sal = load_contours(output_path)

    if clean:
        os.remove(output_path)

    return c_numbers, c_times, c_freqs, c_sal


def load_contours(fpath):
    index = []
    times = []
    freqs = []
    contour_sal = []
    with open(fpath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for row in reader:
            index.append(row[0])
            times.append(row[1])
            freqs.append(row[2])
            contour_sal.append(row[3:])

    # Add column with annotation values in cents
    index = np.array(index, dtype=int)
    times = np.array(times, dtype=float)
    freqs = np.array(freqs, dtype=float)
    contour_sal = np.array(contour_sal, dtype=float)

    return index, times, freqs, contour_sal

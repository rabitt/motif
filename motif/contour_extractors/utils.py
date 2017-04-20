import numpy as np

class PeakStreamHelper(object):

    def __init__(self, S, times, freqs, amp_thresh, dev_thresh, n_gap,
                 pitch_cont):
        '''Init method.

        Parameters
        ----------
        S : np.array
            Salience matrix
        times : np.array
            Array of times in seconds
        freqs : np.array
            Array of frequencies in Hz
        amp_thresh : float
            Threshold on how big a peak must be relative to the maximum in its
            frame.
        dev_thresh : float
            The maximum number of standard deviations below the mean a peak can
            be to survive.
        n_gap : float
            Number of frames that can be taken from bad_peaks.
        pitch_cont : float
            Pitch continuity threshold in cents.

        '''
        self.S = S
        self.S_norm = self._get_normalized_S()
        self.times = times
        self.freqs_hz = freqs
        self.freqs = hz2cents(freqs)

        self.amp_thresh = amp_thresh
        self.dev_thresh = dev_thresh
        self.n_gap = n_gap
        self.pitch_cont = pitch_cont

        peaks = scipy.signal.argrelmax(S, axis=0)
        self.n_peaks = len(peaks[0])

        if self.n_peaks > 0:
            self.peak_index = np.arange(self.n_peaks)
            self.peak_time_idx = peaks[1]
            self.first_peak_time_idx = np.min(self.peak_time_idx)
            self.last_peak_time_idx = np.max(self.peak_time_idx)
            self.frame_dict = self._get_frame_dict()
            self.peak_freqs = self.freqs[peaks[0]]
            self.peak_freqs_hz = self.freqs_hz[peaks[0]]
            self.peak_amps = self.S[peaks[0], peaks[1]]
            self.peak_amps_norm = self.S_norm[peaks[0], peaks[1]]

            self.good_peaks, self.bad_peaks = self._partition_peaks()
            (self.good_peaks_sorted,
             self.good_peaks_sorted_index,
             self.good_peaks_sorted_avail,
             self.n_good_peaks) = self._create_good_peak_index()
            self.smallest_good_peak_idx = 0
        else:
            self.peak_index = np.array([])
            self.peak_time_idx = np.array([])
            self.first_peak_time_idx = None
            self.last_peak_time_idx = None
            self.frame_dict = {}
            self.peak_freqs = np.array([])
            self.peak_freqs_hz = np.array([])
            self.peak_amps = np.array([])
            self.peak_amps_norm = np.array([])
            self.good_peaks = set()
            self.bad_peaks = set()
            self.good_peaks_sorted = []
            self.good_peaks_sorted_index = {}
            self.good_peaks_sorted_avail = np.array([])
            self.n_good_peaks = 0
            self.smallest_good_peak_idx = 0

        self.gap = 0
        self.n_remaining = len(self.good_peaks)

        self.contour_idx = []
        self.c_len = []

    def _get_normalized_S(self):
        """Compute normalized salience matrix

        Returns
        -------
        S_norm : np.array
            Normalized salience matrix.

        """
        S_min = np.min(self.S, axis=0)
        S_norm = self.S - S_min
        S_max = np.max(S_norm, axis=0)
        S_max[S_max == 0] = 1.0
        S_norm = S_norm / S_max
        return S_norm

    def _get_frame_dict(self):
        """Get dictionary of frame index to peak index.

        Returns
        -------
        frame_dict : dict
            Dictionary mapping frame index to lists of peak indices

        """
        frame_dict = {k: [] for k in range(len(self.times))}
        for i, k in enumerate(self.peak_time_idx):
            frame_dict[k].append(i)

        for k, v in frame_dict.items():
            frame_dict[k] = np.array(v)

        return frame_dict

    def _partition_peaks(self):
        """Split peaks into good peaks and bad peaks.

        Returns
        -------
        good_peaks : set
            Set of good peak indices
        bad_peaks : set
            Set of bad peak indices

        """
        good_peaks = set(self.peak_index)
        bad_peaks = set()

        # peaks with amplitude below a threshold --> bad peaks
        bad_peak_idx = np.where(self.peak_amps_norm < self.amp_thresh)[0]
        bad_peaks.update(bad_peak_idx)

        # find indices of surviving peaks
        good_peaks.difference_update(bad_peaks)

        # compute mean and standard deviation of amplitudes of survivors
        mean_peak = np.mean(self.peak_amps[bad_peak_idx])
        std_peak = np.std(self.peak_amps[bad_peak_idx])

        # peaks with amplitude too far below the mean --> bad peaks
        bad_peaks.update(np.where(
            self.peak_amps < (mean_peak - (self.dev_thresh * std_peak)))[0])
        good_peaks.difference_update(bad_peaks)

        return good_peaks, bad_peaks

    def _create_good_peak_index(self):
        """Create a sorted index of peaks by amplitude.

        Returns
        -------
        good_peaks_sorted : np.ndarray
            Array of peak indices ordered by peak amplitude
        good_peaks_sorted_index : dict
            Dictionary mapping peak index to its position in good_peaks_sorted
        good_peaks_sorted_avail : np.ndarray
            Array of booleans indicating if a good peak has been used
        n_good_peaks : int
            Number of initial good peaks

        """
        good_peak_list = list(self.good_peaks)
        sort_idx = list(self.peak_amps[good_peak_list].argsort()[::-1])

        good_peaks_sorted = np.array(good_peak_list)[sort_idx]
        good_peaks_sorted_index = {
            j: i for i, j in enumerate(good_peaks_sorted)
        }

        n_good_peaks = len(good_peak_list)
        good_peaks_sorted_avail = np.ones((n_good_peaks, )).astype(bool)
        return (good_peaks_sorted, good_peaks_sorted_index,
                good_peaks_sorted_avail, n_good_peaks)

    def get_largest_peak(self):
        """Get the largest remaining good peak.

        Returns
        -------
        max_peak_idx : int
            Index of the largest remaining good peak

        """
        return self.good_peaks_sorted[self.smallest_good_peak_idx]

    def update_largest_peak_list(self, peak_index):
        """Update the list of largest peaks

        Parameters
        ----------
        peak_index : int
            Index of the largest remaining good peak

        """
        this_sorted_idx = self.good_peaks_sorted_index[peak_index]
        self.good_peaks_sorted_avail[this_sorted_idx] = False

        if this_sorted_idx <= self.smallest_good_peak_idx:
            i = this_sorted_idx
            while i < self.n_good_peaks:
                if self.good_peaks_sorted_avail[i]:
                    self.smallest_good_peak_idx = i
                    break
                else:
                    i += 1

    def get_closest_peak(self, current_f0, candidates):
        """Find the peak in `candidates` closest in frequency to `current_f0`.

        Parameters
        ----------
        current_f0 : float
            Current frequency value
        candidates : list
            List of peak candidates

        Returns
        -------
        closest_peak_idx : int
            Index of the closest peak to `current_f0`

        """
        min_dist = np.argmin(np.abs(self.peak_freqs[candidates] - current_f0))
        return candidates[min_dist]

    def get_peak_candidates(self, frame_idx, current_f0):
        """Get candidates in frame_idx at current_f0

        Parameters
        ----------
        frame_idx : int
            Frame index
        current_f0 : float
            Current frequency value

        Returns
        -------
        candidates : list or None
            List of peak candidates. None if no available peaks.
        from_good : bool or None
            True if candidates are "good", False if they are "bad",
            None if no available peaks.

        """

        # find candidates in time frame
        all_cands = self.frame_dict[frame_idx]

        # restrict to frames that satisfy pitch continuity
        all_cands = set(all_cands[
            np.abs(self.peak_freqs[all_cands] - current_f0) < self.pitch_cont
        ])

        if len(all_cands) == 0:
            return None, None

        cands = list(all_cands & self.good_peaks)

        if len(cands) > 0:
            self.gap = 0
            return cands, True

        bad_cands = list(all_cands & self.bad_peaks)

        if len(bad_cands) > 0:
            self.gap += 1
            return bad_cands, False

        return None, None

    def get_contour(self):
        """Get the next contour.

        Appends to `self.contour_idx` and `self.c_len`
        Removes peaks from `self.good_peaks` and `self.bad_peaks`
        as they are selected.
        """
        largest_peak = self.get_largest_peak()

        # time frame and freqency index of largest peak
        frame_idx = self.peak_time_idx[largest_peak]
        f0_val = self.peak_freqs[largest_peak]
        self.good_peaks.remove(largest_peak)
        self.update_largest_peak_list(largest_peak)
        self.n_remaining -= 1
        self.contour_idx.append(largest_peak)
        self.gap = 0
        c_len = 1

        # choose forward peaks for this contour
        while self.gap < self.n_gap:
            # go to next time frame
            frame_idx = frame_idx + 1
            if frame_idx > self.last_peak_time_idx:
                break

            cands, from_good = self.get_peak_candidates(frame_idx, f0_val)
            if cands is None:
                break

            closest_peak = self.get_closest_peak(f0_val, cands)

            # add this peak to the contour, remove it from candidates
            self.contour_idx.append(closest_peak)
            c_len += 1

            if from_good:
                self.good_peaks.remove(closest_peak)
                self.update_largest_peak_list(closest_peak)
                self.n_remaining -= 1
            else:
                self.bad_peaks.remove(closest_peak)

            # update target frequency
            f0_val = self.peak_freqs[closest_peak]

        # choose backward peaks for this contour
        frame_idx = self.peak_time_idx[largest_peak]
        f0_val = self.peak_freqs[largest_peak]
        self.gap = 0
        while self.gap < self.n_gap:
            # go to previous time frame
            frame_idx = frame_idx - 1
            if frame_idx < self.first_peak_time_idx:
                break

            cands, from_good = self.get_peak_candidates(frame_idx, f0_val)
            if cands is None:
                break

            closest_peak = self.get_closest_peak(f0_val, cands)

            # add this peak to the contour, change its label to 0
            self.contour_idx.append(closest_peak)
            c_len += 1

            if from_good:
                self.good_peaks.remove(closest_peak)
                self.update_largest_peak_list(closest_peak)
                self.n_remaining -= 1
            else:
                self.bad_peaks.remove(closest_peak)

            # update target frequency
            f0_val = self.peak_freqs[closest_peak]

        self.c_len.append(c_len)

    def peak_streaming(self):
        """Run peak streaming over salience function

        Returns
        -------
        c_numbers : np.array
            Contour numbers
        c_times : np.array
            Contour times in seconds
        c_freqs : np.array
            Contour frequencies
        c_sal : np.array
            Contour salience

        """
        # loop until there are no remaining peaks labeled with 1
        while self.n_remaining > 0:
            # print(self.n_remaining)
            self.get_contour()

        if len(self.c_len) > 0:
            c_numbers = np.repeat(range(len(self.c_len)), repeats=self.c_len)
            c_times = self.times[self.peak_time_idx[self.contour_idx]]
            c_freqs = self.peak_freqs_hz[self.contour_idx]
            c_sal = self.peak_amps[self.contour_idx]
        else:
            c_numbers = np.array([])
            c_times = np.array([])
            c_freqs = np.array([])
            c_sal = np.array([])

        return c_numbers, c_times, c_freqs, c_sal

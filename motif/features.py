# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:39:57 2016

@author: mariapanteli
"""

import numpy as np
import numpy.polynomial.polynomial as Poly
import librosa
from cesium import science_feature_tools


class ContourFeatures(object):
    def __init__(self, times, freqs_hz, salience):
        '''
        Parameters
        ----------
        times : np.ndarray
            Time series, with time stamps in seconds
        freqs_hz : np.ndarray
            Frequency estimates in Hz
        salience : np.ndarray
            Salience estimates
        '''
        self.times = times
        self.sample_rate = get_sample_rate(times)
        self.freqs_hz = freqs_hz
        self.freqs_cents = hz_to_cents(freqs_hz)
        self.salience = salience

    def get_freq_polynomial_coeffs(self, n_poly_degrees=5):
        '''Fit polynomial to frequency estimates
        
        Parameters
        ----------
        n_poly_degrees : int
            Number of polynomial degrees
        
        Returns:
        --------
        poly_coeff : np.ndarray
            Polynomial coefficients
        diff : float
            Reconstruction error
        '''
        poly_coeff, diff = fit_poly(
            self.freqs_hz, n_poly_degrees=n_poly_degrees
        )
        return np.concatenate((poly_coeff, [diff]))
    
    def get_salience_polynomial_coeffs(self, n_poly_degrees=5):
        '''Fit polynomial to salience estimates
        
        Parameters
        ----------
        n_poly_degrees : int
            Number of polynomial degrees
        
        Returns:
        --------
        poly_coeff : np.ndarray
            Polynomial coefficients
        diff : float
            Reconstruction error
        '''
        poly_coeff, diff = fit_poly(
            self.salience, n_poly_degrees=n_poly_degrees
        )
        return np.concatenate((poly_coeff, [diff]))

    def get_vibrato_features(self):
        '''Get vibrato features
        
        Returns:
        --------
        vibrato_features : np.ndarray
            Vibrato rate, extent, coverage
        '''
        return vibrato_features(
            self.freqs_cents, self.sample_rate
        )

    def get_time_series_features(self):
        '''Get time series features
        
        Returns:
        --------
        series_features : np.ndarray
            The time series features
        '''    
        return time_series_features(
            self.times, self.freqs_hz, self.salience
        )

    def get_duration(self):
        '''Get duration in seconds
        
        Returns:
        --------
        duration : float
            Duration in seconds
        '''
        return np.array([self.times[-1] - self.times[0]])

    def get_pitch_mean(self):
        '''Get the pitch mean in Hz
        
        Returns:
        --------
        pitch_mean : float
            Pitch mean in Hz
        '''
        return np.array([np.mean(self.freqs_hz)])

    def get_pitch_std(self):
        '''Get the pitch standard deviation in Hz
        
        Returns:
        --------
        pitch_std : float
            Pitch standard deviation in Hz
        '''
        return np.array([np.std(self.freqs_hz)])

    def get_pitch_range(self):
        '''Get the pitch range in Hz
        
        Returns:
        --------
        pitch_range : float
            Pitch range in Hz
        '''
        return np.array([np.max(self.freqs_hz) - np.min(self.freqs_hz)])

    def get_freq_total_variation(self):
        '''Get the total variation of the frequency estimates
        
        Returns:
        --------
        total_variation : float
            Total variation of frequency estimates
        '''
        return np.array([total_variation(self.freqs_cents)])

    def get_sal_total_variation(self):
        '''Get the total variation of the salience estimates
        
        Returns:
        --------
        total_variation : float
            Total variation of salience estimates
        '''
        return np.array([total_variation(self.salience)])

    def get_features(self):
        '''Get contour features
        
        Returns:
        --------
        features : np.ndarray
            All extracted contour features
        '''
        features = []
        features.append(self.get_freq_polynomial_coeffs(n_poly_degrees=1))
        features.append(self.get_salience_polynomial_coeffs(n_poly_degrees=1))
        features.append(self.get_vibrato_features())
        features.append(self.get_duration())
        features.append(self.get_pitch_mean())
        features.append(self.get_pitch_std())
        features.append(self.get_pitch_range())
        # features.append(self.get_time_series_features())
        return np.concatenate(features)


def get_sample_rate(times):
    '''Get the sample rate from a time series

    Parameters
    ----------
    times : np.ndarray
        Time series, with time stamps in seconds

    Returns:
    --------
    sample_rate : float
        The number of samples per second
    '''
    steps = np.diff(times)
    first_step = steps[0]
    steps = steps - first_step

    if np.all(np.isclose(steps, np.zeros(steps.shape), atol=1e-02, rtol=1e-12)):
        sample_rate = 1.0 / float(first_step)
    else:
        print times
        print steps
        raise NotImplementedError("times must be on an evenly spaced grid")

    return sample_rate


def hz_to_cents(freq_hz, ref_hz=32.0):
    """ convert frequency values from Hz to cents
    """
    freq_cents = 1200.0 * np.log2(freq_hz/ref_hz)
    return freq_cents


def fit_poly(signal, n_poly_degrees=5):
    # RACHEL: What should the behavior be when len(signal) 
    # < n_poly_degrees? 
    """ fit a polynomial and return the coefficients
    """
    n_points = len(signal)
    if n_points < n_poly_degrees + 1:
        raise ValueError('signal must be at least as long as n_poly_degrees')

    signal_norm = signal / np.max(signal)  # normalize
    grid = np.linspace(0, 1, num=n_points)
    poly_coeff = Poly.polyfit(grid, signal_norm, n_poly_degrees)
    poly_fitted = Poly.polyval(grid, poly_coeff)
    diff = np.linalg.norm(signal_norm - poly_fitted)

    return poly_coeff, diff


def time_series_features(times, freqs_hz, salience):
    # RACHEL: This breaks for salience values = 1 or bigger
    # maybe we should normalize?
    contour_norm = freqs_hz / np.max(freqs_hz)
    if salience is None:
        error = 0.01 * np.ones(len(freqs_hz))
    else:
        error = 1 - salience  # error is opposite of salience
    ts_feat_dict = science_feature_tools.generate_science_features(
        times, contour_norm, error
    )
    return np.array(ts_feat_dict.values())


def total_variation(signal):
    return np.sum(np.abs(signal[1:] - signal[:-1]))


def vibrato_features(freqs_cents, sr):
    # RACHEL: These features work but aren't very precise. (e.g a perfect
    # 12 Hz sine wav estimates a rate of 9.8) also the computation is a bit slow

    """ estimate vibrato and return vibrato rate, extent, coverage
    """
    #11. vibrato rate (in Hz, 0 if vibrato=0)
    #12. vibrato extent (in cents, 0 if vibrato=0)
    #13. vibrato coverage (fraction of contour where vibrato detected,
    #   range [0,1])
    # estimate vibrato as stated in Salamon & Gomez 2012:
    # we apply the FFT to the contour's pitch trajectory 
    # (after subtracting the mean) and check for a prominent peak 
    # in the expected frequency range for human vibrato (5-8Hz)."

    #contour = frequencies / np.max(frequencies) - np.mean(frequencies)
    #contour = (frequencies - np.mean(frequencies))/np.max(frequencies)
    #contour = frequencies / np.max(frequencies)

    contour = freqs_cents - np.mean(freqs_cents)
    frame_size = int(np.round(0.35 * sr))
    hop_size = 1        
    fft_size = 4*frame_size
    frames = np.arange(0, len(contour) - frame_size, hop_size)
    freqs = np.fft.fftfreq(fft_size, 1. / sr)[0:int(fft_size / 2.)]
    vib_inds = np.where((freqs >= 2) & (freqs <= 20))[0] # vibrato 2-20 Hz
    rate = []
    extent = []
    coverage = []
    
    for frame in frames:
        contour_segment = (
            contour[frame:frame + frame_size] -
            np.mean(contour[frame:frame + frame_size])
        )

        spec = np.abs(np.fft.fft(contour_segment, n=fft_size))
        #spec = 20*np.log10(spec[0:int(fft_size/2.)])  # db spectrum
        peak_inds = librosa.util.peak_pick(spec, 3, 3, 3, 5, 0.5, 10)
        # top 3 peaks
        if len(peak_inds) > 0:
            top_peak_idx = np.argsort(spec[peak_inds])[::-1][:3]
            peak_inds = peak_inds[top_peak_idx]

        vib_peaks = list(
            np.intersect1d(vib_inds, peak_inds, assume_unique=True)
        )

        if len(vib_peaks) > 0:
            rate.append(np.mean(freqs[vib_peaks]))
            extent.append(
                np.max(contour[frame:frame+frame_size]) - 
                np.min(contour[frame:frame+frame_size])
            )
            # append '1' if current frame has vibrato
            coverage.append(1)

    try:
        if len(rate) > 0:
            rate = np.mean(rate)
        else:
            rate = 0.0

        if len(extent) > 0:
            extent = np.mean(extent)
        else:
            extent = 0.0

        if len(coverage) > 0:
            coverage = sum(coverage)/float(len(frames))
        else:
            coverage = 0.0

        feats = np.array([rate, extent, coverage])

    except:
        feats = np.array([0, 0, 0])  # default 0's if no vibrato

    return feats

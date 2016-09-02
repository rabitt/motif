"""Utility functions for computing contour features.
Each of these functions computes single sets of contour features using
information such as the times, frequencies, salience, sample rate, etc.

Each function returns a flattened numpy array for easy concatenation.

@author: mariapanteli, rabitt
"""
from __future__ import print_function
import librosa
import numpy as np
import numpy.polynomial.polynomial as Poly
import scipy


def hz_to_cents(freq_hz, ref_hz=32.0):
    '''Convert frequency values from Hz to cents

    Parameters
    ----------
    freq_hz : np.array
        Array of contour frequencies in Hz
    ref_hz : float
        Reference frequency in Hz

    Returns
    -------
    freq_cents : np.array
        Array of contour frequencies in cents
    '''
    freq_cents = 1200.0 * np.log2(freq_hz / ref_hz)
    return freq_cents


def get_contour_onset(times):
    '''Get the first time stamp of a contour

    Parameters
    ----------
    times : np.array
        Array of contour times

    Returns
    -------
    onset : float
        The contour onset in seconds
    '''
    return np.array([times[0]])


def get_contour_offset(times):
    '''Get the last time stamp of a contour

    Parameters
    ----------
    times : np.array
        Array of contour times

    Returns
    -------
    offset : float
        The contour offset in seconds
    '''
    return np.array([times[-1]])


def get_contour_duration(times):
    '''Get contour duration in seconds

    Parameters
    ----------
    times : np.array
        Array of contour times

    Returns
    -------
    duration : float
        Duration in seconds
    '''
    return np.array([times[-1] - times[0]])


def get_mean(signal):
    '''Get the mean of a signal.

    Parameters
    ----------
    signal : np.array
        Array of values

    Returns
    -------
    mean : float
        The mean of the signal
    '''
    return np.array([np.mean(signal)])


def get_std(signal):
    '''Get the standard deviation of a signal.

    Parameters
    ----------
    signal : np.array
        Array of values

    Returns
    -------
    std : float
        The standard deviation of the signal
    '''
    return np.array([np.std(signal)])


def get_sum(signal):
    '''Get the sum of a signal.

    Parameters
    ----------
    signal : np.array
        Array of values

    Returns
    -------
    sum : sum of the signal
    '''
    return np.array([np.sum(signal)])


def get_range(signal):
    '''Get the range of a signal.

    Parameters
    ----------
    signal : np.array
        Array of values

    Returns
    -------
    range : float
        The range of the signal
    '''
    return np.array([np.max(signal) - np.min(signal)])


def get_total_variation(signal):
    '''Get the total variation of a signal.

    Parameters
    ----------
    signal : np.array
        Array of values

    Returns
    -------
    total_variation : float
        The total variation of the signal
    '''
    return np.array([np.sum(np.abs(signal[1:] - signal[:-1]))])


def get_polynomial_fit_features(times, signal, n_deg=5, norm=False):
    '''Fit a signal to a polynomial, return coefficients of polynomial and
    residual error.

    Parameters
    ----------
    times : np.array
        Array of contour times.
    signal : np.array
        Array of values to fit.
    n_deg : int, default=5
        Number of polynomial degrees to fit.
    norm : bool, default=False
        If True, scales the signal to be between 0 and 1
        If False, the signal is not altered.

    Returns
    -------
    poly_coeff : np.array
        The coefficients of the polynomial.
    poly_approx : np.array
        The polynomial approximation of the signal.
    residual : np.array
        The pointwise difference between the signal and the polynomial.

    '''
    poly_coeff, _, residual = _fit_poly(
        n_deg, signal, grid=times, norm=norm
    )
    return np.concatenate([poly_coeff, [np.linalg.norm(residual)]])


def _fit_poly(n_poly_degrees, signal, grid=None, norm=False):
    '''Fit a signal to a polynomial. If grid is not given, assumes a uniform
    grid between 0 and 1 of length len(signal).

    Parameters
    ----------
    n_poly_degrees : int
        Number of polynomial degrees to fit.
    signal : np.array
        Array of values to fit.
    grid : np.array or None, default=None
        Array of x-values, or None.
        If None, uses a uniform time grid between 0 and 1.
    norm : bool, default=False
        If True, scales the signal to be between 0 and 1
        If False, the signal is not altered.

    Returns
    -------
    poly_coeff : np.array
        The coefficients of the polynomial.
    poly_approx : np.array
        The polynomial approximation of the signal.
    residual : np.array
        The pointwise difference between the signal and the polynomial.

    '''
    n_points = len(signal)
    if n_points < n_poly_degrees + 1:
        raise ValueError('signal must be at least as long as n_poly_degrees')

    if norm:
        signal = signal - np.min(signal)
        max_val = np.max(signal)
        if max_val > 0:
            signal = signal / max_val

    if grid is None:
        grid = np.linspace(0, 1, num=n_points)

    poly_coeff = Poly.polyfit(grid, signal, n_poly_degrees)
    poly_approx = Poly.polyval(grid, poly_coeff)
    residual = signal - poly_approx

    return poly_coeff, poly_approx, residual


def _fit_normalized_cosine(x, y, min_freq, max_freq, step):
    '''Assuming the amplitude is 1, find the optimal frequency and phase of a
    to fit a cosine. Searches within the frequency range (min_freq, max_freq).

    Fits to a cosine of the form:
        y = cos((2pi * freq * x) - phase)

    Parameters
    ----------
    x : np.array
        Array of evenly spaced x-values.
    y : np.array
        Array of y-values.
    min_freq : float
        The minimum allowed vibrato frequency.
    max_freq : float
        The maximum allowed vibrato frequency.
    step : float
        The step size between vibrato search frequencies.

    Returns
    -------
    freq : float
        The estimated optimal frequency.
    phase : float
        The estimated optimal phase (in radians)
    '''
    freqs = np.arange(min_freq, max_freq, step)
    dot_prod = np.dot(
        np.exp(2.0 * np.pi * 1j * np.multiply.outer(freqs, x)),
        y
    )
    dot_prod_mag = np.abs(dot_prod)

    peak_idx = list(scipy.signal.argrelmax(dot_prod_mag)[0])
    if len(peak_idx) == 0:
        return 0, 0

    idx = peak_idx[np.argmax(dot_prod_mag[peak_idx])]
    freq = freqs[idx]
    phase = np.angle(dot_prod[idx])
    return freq, phase


def _compute_coverage_array(y_sinfit_diff, cycle_length, vibrato_threshold):
    '''Given an array of residual differences, compute the vibrato coverage
    over time by splitting the interval up chunks of size cycle_length.

    Parameters
    ----------
    y_sinfit_diff : np.array
        Array of residual differences between 0 and 1.
    cycle_length : float
        Optimal number of intervals for the estimated vibrato frequency
    vibrato_threshold : float
        Value between 0 and 1 to determine whether the fit is good enough.

    Returns
    -------
    coverage : np.array
        Array of booleans indicating if the current frame contains vibrato.
    '''
    n_points = len(y_sinfit_diff)
    half_period_idx = list(
        np.round(
            cycle_length * np.arange(
                0, int(np.ceil(float(n_points) / float(cycle_length)) + 1)
            )
        ).astype(int)
    )

    if half_period_idx[-1] > n_points:
        half_period_idx = half_period_idx[:-1]

    if half_period_idx[-1] < n_points:
        half_period_idx.append(n_points)

    # compute the goodness of fit for each half period
    diff_thresh = np.zeros(y_sinfit_diff.shape)
    diffs = np.zeros((len(half_period_idx) - 1, ))
    for k, (i, j) in enumerate(zip(half_period_idx[:-1], half_period_idx[1:])):
        diffs[k] = np.mean(y_sinfit_diff[i:j])
        diff_thresh[i:j] = diffs[k]

    # vibrato is active when the fit diff is below a threshold
    coverage = np.less_equal(diff_thresh, vibrato_threshold)
    diff_coverage = np.less_equal(diffs, vibrato_threshold)

    # If vibrato is active for less than 2 full periods, set coverage to None
    if sum(diff_coverage) <= 3:
        coverage[:] = False

    return coverage


def get_contour_shape_features(times, freqs, sample_rate, poly_degree=5,
                               min_freq=3, max_freq=30, freq_step=0.1,
                               vibrato_threshold=0.25):
    '''Fit contour to a low order polynomial plus sinusoidal vibrato.

    Parameters
    ----------
    times : np.array
        Sequence of contour times
    freqs : np.array
        Sequence of contour frequencies
    sample_rate : float
        Contour sample rate
    poly_degree : float, default=5
        Low order polynomial degree
    min_freq : float, default=3
        The minimum allowed vibrato frequency
    max_freq : float, default=30
        The maximum allowed vibrato frequency
    freq_step : float, default=0.1
        The step size between vibrato search frequencies
    vibrato_threshold : float, default=0.25
        The fitness threshold for a half period to be considered vibrato.
        Regions with normalized fitness differences below vibrato_threshold are
        considered to have vibrato.

    Returns
    -------
    features : np.array
        Array of feautres. Elements (in order) are:
            - vibrato rate (in Hz)
            - vibrato extent (in the same units as freqs)
            - vibrato coverage (between 0 and 1)
            - vibrato coverage beginning (between 0 and 1)
            - vibrato coverage middle (between 0 and 1)
            - vibrato coverage end (between 0 and 1)
            - 0th polynomial coefficient
            - 1st polynomial coefficient
            - ...
            - Kth polynomial coefficient (K = poly_degree)
            - polynomial fit residual
            - overall model fit residual
    '''
    n_points = len(freqs)

    # fit contour to a low order polynomial
    poly_coeffs, y_poly, y_diff = _fit_poly(
        poly_degree, freqs, grid=times
    )
    # remove amplitude envelope using hilbert transform
    y_hilbert = np.abs(scipy.signal.hilbert(y_diff))
    y_sin = y_diff / y_hilbert

    # get ideal vibrato parameters from resulting signal
    vib_freq, vib_phase = _fit_normalized_cosine(
        times, y_sin, min_freq=min_freq, max_freq=max_freq, step=freq_step
    )
    y_sinfit = np.cos(2. * np.pi * vib_freq * times - vib_phase)

    # get residual of sinusoidal fit
    y_sinfit_diff = np.abs(y_sin - y_sinfit)

    # compute vibrato coverage
    if vib_freq > 0:
        cycle_length = 0.5 * ((sample_rate) / vib_freq)
        coverage = _compute_coverage_array(
            y_sinfit_diff, cycle_length, vibrato_threshold
        )
    else:
        coverage = np.zeros((n_points, )).astype(bool)

    # compute percentage of coverage
    vib_coverage = coverage.mean()

    # if vibrato is present, set extent and rate. Otherwise they are zero.
    if vib_coverage > 0:
        vib_extent = np.mean(y_hilbert[coverage])
        vib_rate = vib_freq
    else:
        vib_extent = 0.0
        vib_rate = 0.0

    # compute the overall model fit
    y_vib = vib_extent * y_sinfit
    y_vib[~coverage] = 0
    y_modelfit = y_vib + y_poly

    # compute residuals
    polyfit_residual = np.linalg.norm(y_diff) / float(n_points)
    modelfit_residual = np.linalg.norm(freqs - y_modelfit) / float(n_points)

    # aggregate features
    thirds = int(np.round(n_points / 3.0))
    return np.concatenate([
        np.array([vib_rate, vib_extent, vib_coverage]),
        np.array([coverage[:thirds].mean(), coverage[thirds:2 * thirds].mean(),
                  coverage[2 * thirds:].mean()]),
        poly_coeffs,
        np.array([polyfit_residual, modelfit_residual])
    ])


def vibrato_essentia(freqs, sample_rate, hop_size=1):
    """Estimate vibrato parameters as in essentia.

    Warning: These features work but aren't very precise (e.g a perfect
    12 Hz sine wav estimates a rate of 9.8).

    Parameters
    ----------
    freqs : np.array
        Sequence of contour frequencies
    sample_rate : float
        Contour sample rate
    hop_size : int, default=1
        Number of samples to advance each frame

    Returns
    -------
    features : np.array
        Array of feautres. Elements (in order) are:
            - vibrato active (1 if active, 0 if not)
            - vibrato rate (in Hz, 0 if inactive)
            - vibrato extent (in the same units as freqs, 0 if inactive)
            - vibrato coverage (between 0 and 1, 0 if inactive)
    """
    contour = freqs - np.mean(freqs)
    frame_size = int(np.round(0.35 * sample_rate))
    fft_size = 4 * frame_size
    n_frames = len(contour) - frame_size
    freqs = np.fft.fftfreq(fft_size, 1. / sample_rate)[0:int(fft_size / 2.)]
    vib_inds = np.where((freqs >= 2) & (freqs <= 20))[0]  # vibrato 2-20 Hz
    rate = []
    extent = []
    coverage = []

    for frame in np.arange(0, n_frames, hop_size):
        contour_segment = (
            contour[frame:frame + frame_size] -
            np.mean(contour[frame:frame + frame_size])
        )

        spec = np.abs(np.fft.fft(contour_segment, n=fft_size))
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
                np.max(contour[frame:frame + frame_size]) -
                np.min(contour[frame:frame + frame_size])
            )
            # append '1' if current frame has vibrato
            coverage.append(1.0)

    rate = np.mean(rate) if len(rate) > 0 else 0.0
    extent = np.mean(extent) if len(extent) > 0 else 0.0
    coverage = sum(coverage) / n_frames if len(coverage) > 0 else 0.0
    vib_params = [rate, extent, coverage]

    if vib_params == [0.0, 0.0, 0.0]:
        feats = np.array([0] + vib_params)
    else:
        feats = np.array([1] + vib_params)

    return feats

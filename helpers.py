import numpy as np
from scipy import signal, optimize
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import scipy.stats as stats
import pandas as pd

from temp_data import temp_dict


def improved_circular_normalize(delays, led_period):
    """
    Improved circular normalization using known LED period

    Parameters:
    - delays: Time delays between firefly and LED flashes
    - led_period: Known period of LED flashes

    Returns:
    -normalized delays: normalized delays in the range [-0.5, 0.5]
    """
    # Normalize using the known LED period
    normalized_delays = (delays % led_period) / led_period

    # Shift to [-0.5, 0.5] range
    normalized_delays = np.where(
        normalized_delays > 0.5,
        normalized_delays - 1.0,
        normalized_delays
    )
    return [normalized_delays]


def find_time_delays(time_series_led, time_series_ff, window_length):
    """
    Find time delays between led and firefly timeseries over a sliding window

    Parameters:
    - time_series_led: Timestamps of LED-ON
    - time_series_ff: Timestamps of Firefly-ON
    - window_length: Number of seconds determining window within which delays are calculated

    Returns:
    - closest delays: normalized delays in the range [-0.5, 0.5]
    - period_change_data: the change in period between flash points
    """
    led_times_array = np.array(time_series_led)
    firefly_times_array = np.array(time_series_ff)

    closest_delays = []
    period_change_data = []

    led_period = round(np.median(np.diff(time_series_led)[np.diff(time_series_led) > 0.1]), 3)
    if not led_period:
        return [], []

    for i, firefly_time in enumerate(firefly_times_array):
        window_mask = (firefly_times_array >= firefly_time - window_length / 2) & \
                      (firefly_times_array <= firefly_time + window_length / 2)

        if np.sum(window_mask) >= 3:
            closest_led_index = np.argmin(np.abs(firefly_time - led_times_array))

            raw_delay = firefly_time - led_times_array[closest_led_index]

            normalized_delays = improved_circular_normalize(raw_delay, led_period)

            mean_delay = np.mean(normalized_delays)
            for delay in normalized_delays:
                closest_delays.append(float(delay))

            if i >= 2:
                period1 = firefly_times_array[i - 1] - firefly_times_array[i - 2]
                period2 = firefly_times_array[i] - firefly_times_array[i - 1]

                period_change = period2 - period1
                period_change_data.append((mean_delay, period_change))

    return closest_delays, period_change_data


def dfa(time_series, scale_range=None, polynomial_order=2):
    """
    Perform Detrended Fluctuation Analysis on a time series.

    Parameters:
    - time_series: the time series data to analyze.
    - scale_range: range of scales to consider (min_scale, max_scale). If None, defaults to (5, len(time_series)//4).
    - polynomial_order: order of polynomial for detrending (default is 2).

    Returns:
    - scales: scales at which fluctuations were calculated.
    - fluctuations: fluctuation values at each scale.
    - alpha: scaling exponent (slope of log-log plot).
    - confidence_interval: (lower_bound, upper_bound) of the 95% confidence interval for alpha.
    """
    x = np.asarray(time_series)

    # Integrate the time series (cumulative sum of deviations from mean)
    x_integrated = np.cumsum(x - np.mean(x))

    # Determine the scale range if not provided
    N = len(x)
    if scale_range is None:
        min_scale = 5
        max_scale = N // 4  # Don't go beyond N/4 for statistical reliability
        scale_range = (min_scale, max_scale)
    else:
        min_scale, max_scale = scale_range

    # Create a set of scales that are approximately logarithmically spaced
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=20, dtype=int)
    scales = np.unique(scales)  # Remove duplicates

    # Calculate fluctuations for each scale
    fluctuations = np.zeros(len(scales))

    for i, scale in enumerate(scales):
        # Number of non-overlapping segments
        n_segments = int(np.floor(N / scale))

        if n_segments == 0:
            continue

        # Calculate local trends and fluctuations
        local_fluctuations = np.zeros(n_segments)

        for j in range(n_segments):
            # Extract segment
            segment = x_integrated[j * scale:(j + 1) * scale]

            # Create time index for the segment
            time_index = np.arange(scale)

            # Fit polynomial and calculate residuals
            try:
                coeffs = np.polyfit(time_index, segment, polynomial_order)
                trend = np.polyval(coeffs, time_index)
                local_fluctuations[j] = np.sqrt(np.mean((segment - trend) ** 2))
            except np.linalg.LinAlgError:
                local_fluctuations[j] = np.nan

        # Root mean square of all segments
        fluctuations[i] = np.sqrt(np.mean(local_fluctuations ** 2))

    valid = ~np.isnan(fluctuations)
    fluctuations = fluctuations[valid]
    scales = scales[:len(fluctuations)]
    # Calculate scaling exponent (alpha) using log-log linear regression
    log_scales = np.log10(scales)
    log_fluctuations = np.log10(fluctuations)

    # Perform linear regression
    X = add_constant(log_scales)
    model = OLS(log_fluctuations, X).fit()
    alpha = model.params[1]  # Slope

    # Calculate 95% confidence interval
    ci = model.conf_int(alpha=0.05)
    confidence_interval = (ci[1][0], ci[1][1])

    return scales, fluctuations, alpha, confidence_interval


def whittle_mle(time_series):
    """
    Estimate the Hurst exponent using Whittle Maximum Likelihood Estimation.

    Parameters:
    - time_series: time series data to analyze.

    Returns:
    - H: estimated Hurst exponent.
    - ci: (lower_bound, upper_bound) of the 95% confidence interval.
    """
    # Ensure time series is a numpy array
    x = np.asarray(time_series)

    # Compute periodogram
    f, pxx = signal.periodogram(x)

    # Remove zero frequency
    f = f[1:]
    pxx = pxx[1:]

    # Define the spectral density function for fractional Gaussian noise
    def fgn_spectrum(f, H):
        # Spectral density for fractional Gaussian noise
        return (2 * np.sin(np.pi * f)) ** (-2 * H)

    # Negative log-likelihood function to minimize
    def neg_log_likelihood(H):
        spectrum = fgn_spectrum(f, H)
        return np.sum(np.log(spectrum) + pxx / spectrum)

    # Optimize to find the best H value
    result = optimize.minimize_scalar(neg_log_likelihood, bounds=(0, 1), method='bounded')
    H = result.x

    # Compute confidence interval (using Fisher information)
    # This is an approximation based on the curvature of the likelihood function
    step = 0.001
    H_plus = H + step
    H_minus = H - step

    # Compute second derivative of negative log-likelihood
    d2l = (neg_log_likelihood(H_plus) - 2 * neg_log_likelihood(H) + neg_log_likelihood(H_minus)) / (step ** 2)

    # Standard error based on Fisher information
    std_error = 1 / np.sqrt(d2l)

    # 95% confidence interval
    ci = (H - 1.96 * std_error, H + 1.96 * std_error)

    return H, ci


def sliding_time_window_derivative(times, phases, window_seconds=3.0):
    """
    Compute phase derivative using a sliding time window approach.

    Parameters:
    - times : array of time values.
    - phases: array of phase values.
    - window_seconds: Window size in seconds.

    Returns:
    - derivatives: List of derivative values, with None included where windowing is not possible.
    """
    half_window = window_seconds / 2
    derivatives = [None] * len(times)

    for i in range(len(times)):
        # Find indices of points within the time window [t - half_window, t + half_window]
        start_idx = next((j for j in range(i, -1, -1) if times[i] - times[j] > half_window), 0)
        end_idx = next((j for j in range(i, len(times)) if times[j] - times[i] > half_window),
                       len(times) - 1)

        dt = times[end_idx] - times[start_idx]  # Total time difference in window
        dphi = phases[end_idx] - phases[start_idx]  # Phase difference in window

        derivatives[i] = dphi / dt if dt != 0 else 0  # Avoid division by zero

    return derivatives


def dedupe(flash_list, eps):
    """
    Dedupes a timeseries such that flashes of length > 1frame are treated as the same flash

    Parameters:
    - flash_list: timeseries of flash points
    - eps: time delay treated as the same flash

    Returns:
    - deduped_flash_lst: Deduplicated flash times to be included in the analysis
    """
    deduped_flash_list = []
    if len(flash_list) > 0:

        sorted_l = np.sort(flash_list)
        current_flash_group = [sorted_l[0]]
        for i in range(1, len(sorted_l)):
            if sorted_l[i] - current_flash_group[-1] <= eps:
                current_flash_group.append(sorted_l[i])
            else:
                deduped_flash_list.append(current_flash_group[0])
                current_flash_group = [sorted_l[i]]
        if current_flash_group:
            deduped_flash_list.append(current_flash_group[0])
    return deduped_flash_list


def compute_phase_response_curve(time_series_led, time_series_ff, epsilon=0.04, do_responses_relative_to_ff=False):
    """
    Compute the phase response curve (PRC) of fireflies relative to LED flashes.

    Parameters:
    - time_series_led: array of LED flash timestamps.
    - time_series_ff: array of firefly flash timestamps.
    - epsilon: maximum time difference (in seconds) between frames to be considered the same flash
    - do_responses_relative_to_ff: boolean indicating whether to calculate FL-Response Time [FF_t - LED_(t-1)] or
                                   FF-Response Time [FF_t - FF_(t-1)]


    Returns:
    - phases: list of normalized phases (firefly phase relative to LED)
    - response_times: list of response times (time difference between firefly flash and previous LED)
    - phase_time_diff_pairs: list of (phase, response_time, firefly_time) tuples
    """
    led_times_array = np.array(time_series_led)
    firefly_times_array = np.array(time_series_ff)
    phases = []
    response_times = []
    phase_time_diff_pairs = []

    if len(led_times_array) < 2 or len(firefly_times_array) < 1:
        return [], [], []

    # deduplicate firefly flash times array
    # accounting to ensure two- or more frame flashes are seen as the same flash
    deduplicated_ff_times = dedupe(firefly_times_array, epsilon)
    deduplicated_led_times = dedupe(led_times_array, epsilon)
    deduplicated_ff_times = np.array(deduplicated_ff_times)
    deduplicated_led_times = np.array(deduplicated_led_times)

    if do_responses_relative_to_ff:
        for i in range(1, len(deduplicated_ff_times)):
            firefly_time = deduplicated_ff_times[i]
            previous_firefly_time = deduplicated_ff_times[i - 1]

            closest_led_idx = np.argmin(np.abs(deduplicated_led_times - firefly_time))
            closest_led_time = deduplicated_led_times[closest_led_idx]

            phase = firefly_time - closest_led_time
            phases.append(phase)

            response_time = firefly_time - previous_firefly_time
            response_times.append(response_time)

            if -0.5 <= phase <= 0.5:
                phase_time_diff_pairs.append((phase, response_time, firefly_time))

    else:
        # Now compute the phase response curve with deduplicated flashes
        for firefly_time in deduplicated_ff_times:
            closest_led_idx = np.argmin(np.abs(deduplicated_led_times - firefly_time))
            closest_led_time = deduplicated_led_times[closest_led_idx]

            phase = firefly_time - closest_led_time
            phases.append(phase)

            previous_led_candidates = deduplicated_led_times[(deduplicated_led_times < firefly_time) &
                                                             (deduplicated_led_times != closest_led_time)]

            if len(previous_led_candidates) > 0:
                previous_led_time = np.max(previous_led_candidates)

                response_time = firefly_time - previous_led_time
                response_times.append(response_time)

                phase_time_diff_pairs.append((phase, response_time, firefly_time))

    return phases, response_times, phase_time_diff_pairs


def get_starts_of_flashes(ff_xs, ff_ys):
    """
    Gets the beginning of flashes from firefly timeseries

    Parameters:
    - ff_xs: timeseries of firefly flash points
    - ff_ys: timeseries of firefly flash points

    Returns:
    - flash_times_to_include: flash times to be included in the analysis
    """
    flash_times = [x[0] for x in list(zip(ff_xs, ff_ys)) if x[1] == 1.0]
    good_idxs = np.where((np.diff(flash_times) * 1000) > 100)[0]
    flash_times_to_include = []
    for idx in range(len(good_idxs)):
        flash_times_to_include.append(flash_times[good_idxs[idx]])
    return flash_times_to_include


def get_start_index(flash_times_to_include, start_time):
    """
    Gets the start index of a flash timeseries

    Parameters:
    - flash_times_to_include: timeseries of flash points
    - start_time: time threshold

    Returns:
    - start_index: index indicating start of times to be included in the analysis occurring on/after the time threshold
    """
    start_index = 0
    for ix in range(len(flash_times_to_include)):
        if flash_times_to_include[ix] > start_time:
            start_index = ix
            break
    return start_index


def get_introduced_time(k, instance):
    """
    Gets the time LED is introduced into the experiment

    Parameters:
    - k: LED frequency
    - instance: Experiment instance

    Returns:
    - led_introduced: known experimental artifact of the time when the LED is introduced into the system
    """
    if k == '300':
        if instance == 0:
            led_introduced = 41.2
        elif instance == 1:
            led_introduced = 82.5
        elif instance == 2:
            led_introduced = 11.0
        elif instance == 3:
            led_introduced = 78.2
        else:
            led_introduced = 38.4
        return led_introduced

    if k == '400':
        if instance == 0:
            led_introduced = 86.33
        elif instance == 1:
            led_introduced = 81.98
        elif instance == 2:
            led_introduced = 49.95
        elif instance == 3:
            led_introduced = 43.3
        else:
            led_introduced = 30
        return led_introduced


def get_temp_from_experiment_date(date, index):
    """
    Gets the temperature of the experiment

    Parameters:
    - date: Experiment date code
    - index: Experiment instance

    Returns:
    - temp: known temperature of the system when the LED is introduced
    """
    key = date + '_' + index
    temp = temp_dict.temp_dict[key]

    return temp


def t_f_conversion(t):
    """
    Converts temperature into frequency

    Parameters:
    - t: temperature

    Returns:
    Predicted frequency at that temperature
    """
    return (0.186 * (t ** 2) - 11.14 * t + 194.1) / 60


def tighten(k):
    """
     Converts index into appropriate index

     Parameters:
     - k: experimental frequency

     Returns:
     Tightened frequency , matching code keys
     """
    if k == '780':
        return '770'
    if k == '710':
        return '700'
    if k == '860':
        return '850'
    if k == '70':
        return '700'
    if k == '610':
        return '600'
    if k == '640':
        return '600'
    if k == '60':
        return '600'
    if k == '50':
        return '500'
    if k == '630':
        return '600'
    if k == '470':
        return '500'
    if k == '560':
        return '600'
    if k == '10.0':
        return '1000'
    return k


def get_offset(p):
    """
     Get experimental offset, where necessary

     Parameters:
     - p: experimental index

     Returns:
     Necessary offset subtraction based on experimental index
     """
    return (0.56 if p else 0), (0.79 if p else 0), (0.08 if p else 0)


def double_std(arr):
    """
     Doubles the standard deviation of an arr

     Parameters:
     - arr: experimental data

     Returns:
     Doubled standard deviation
     """
    return np.std(arr) * 2


def adjust_for_offset(k1, l1, l2, tp):
    """
    Set experimental offset, where necessary

    Parameters:
     - k1, k2, l1, l2: experimental data
     - tp: trial parameters

     Returns:
     - l1, l2: modified lists
     """

    omb, omb_, offset = tp
    if k1:
        l1 = [x if (x < omb or x > omb_) else x - offset for x in l1]
        l2 = [x if (x < omb or x > omb_) else x - offset for x in l2]

    return l1, l2


def get_rolling_window_flash_times(flash_times_to_include, window_size_seconds):
    """
    Parameters:
     - flash_times_to_include: relevant flash information
     - window_size_seconds: size of the sliding window on which statistics are being measured

     Returns:
     - rolling_flash_avg_flash_times: average time periods over the window
     """

    rolling_flash_avg_flash_times = []
    start_time = min(flash_times_to_include) + window_size_seconds / 2

    start_index = get_start_index(flash_times_to_include, start_time)

    for i in range(start_index, len(flash_times_to_include)):
        ith_list = [flash_times_to_include[i]]
        for k in range(0, start_index):
            if flash_times_to_include[i] - flash_times_to_include[k] > window_size_seconds / 2:
                break
            else:
                ith_list.append(flash_times_to_include[k])
        for j in range(i + 1, len(flash_times_to_include)):
            if flash_times_to_include[j] - flash_times_to_include[i] > window_size_seconds / 2:
                break
            else:
                ith_list.append(flash_times_to_include[j])
        rolling_flash_avg_flash_times.append(ith_list)
    return rolling_flash_avg_flash_times


def r_means(flashes, led_introduced):
    """
    Gets rolling means for any set of data
    Parameters:
     - flashes: relevant flash information
     - led_introduced: start time of experiment

     Returns:
     - b,a,r,f: rolling before, rolling after, rolling mean, and flash timings
     """
    b = []
    a = []
    r = []
    f = []
    for rfl in flashes:
        x = np.diff(rfl)
        if not np.isnan(np.mean(x)):
            if rfl[-1] < led_introduced:
                b.append(np.mean([y for y in x]))
            else:
                a.append(np.mean([y for y in x]))
            r.append(np.mean([y for y in x]))
            f.append(rfl[0])
    return b, a, r, f

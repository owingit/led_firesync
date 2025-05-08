import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, optimize, stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

import plotting_helpers
from temp_data import temp_dict

MIN_CUTOFF_PERIOD = 0.25
MAX_CUTOFF_PERIOD = 2.5


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

    normalized_delays = np.where(
        normalized_delays < 0,  # Check for negative values
        1.0 + normalized_delays,  # Map [-0.5, 0) to [0.5, 1]
        normalized_delays  # Keep [0, 0.5] unchanged
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


def check_frame_rate(input_csv):
    df = pd.read_csv(input_csv)
    df['LED times'] = df['LED times'].apply(eval)
    df['FF times'] = df['FF times'].apply(eval)

    df['LED_timestamps'] = df['LED times'].apply(lambda x: x[0])
    df['LED_states'] = df['LED times'].apply(lambda x: x[1])
    df['FF_timestamps'] = df['FF times'].apply(lambda x: x[0])
    df['FF_states'] = df['FF times'].apply(lambda x: x[1])

    frame_rate = df['LED_timestamps'].iloc[1] - df['LED_timestamps'].iloc[0]
    frame_rate_ff = df['FF_timestamps'].iloc[1] - df['FF_timestamps'].iloc[0]
    try:
        assert frame_rate_ff == frame_rate
        return input_csv, frame_rate
    except AssertionError:
        print(f"File {input_csv} frame rate mismatch found, excluding")
        return None, None


def generate_simulated_phases(t, length=500, noise=0.05):
    """
    Generate synthetic phase differences between two time series in the range [-0.5, 0.5].

    Parameters:
    - t: Type of phase relationship ('drifting', 'synchronized', 'phase_locked', 'random')
    - length: Number of time points
    - noise: Noise level for phase perturbations

    Returns:
    - p: Array of phase differences in range [-0.5, 0.5]
    """
    import numpy as np

    if t == 'drifting':
        # Phase difference changes at a constant rate over time
        # Create a smooth linear progression from -0.4 to 0.4
        base_drift = np.linspace(-0.4, 0.4, length)
        p = base_drift + np.random.normal(0, noise, length)
        p = (p + 0.5) % 1 - 0.5  # Ensure values stay in [-0.5, 0.5] range

    elif t == 'synchronized':
        # Signals are in-phase, phase difference is very close to or equal to 0
        # Small noise around zero
        p = np.random.normal(0, noise, length)
        p = (p + 0.5) % 1 - 0.5  # Wrap properly

    elif t == 'phase_locked':
        # Constant non-zero phase difference (e.g., locked at 0.25)
        # Same principle as synchronized but with a non-zero mean
        fixed_phase = 0.25  # A clear phase offset, but constant
        p = np.random.normal(fixed_phase, noise, length)
        p = (p + 0.5) % 1 - 0.5  # Wrap properly

    else:  # t == 'random'
        # Completely random phase differences
        p = np.random.uniform(-0.5, 0.5, length)

    return p


def renorm_phases(p):
    """
    Map phase values from the interval [-0.5, 0.5] to [0.0, 1.0]

    Parameters:
    - p: list of phases

    Returns:
    - phase_list_mapped: phases mapped to appropriate range
    """

    phase_list_mapped = []
    for x in p:
        if x >= 0:
            phase_list_mapped.append(x)
        else:
            phase_list_mapped.append((x + 1))
    return phase_list_mapped


def compute_cross_correlation(led_times, ff_times, max_lag_ms=1000, bin_size_ms=1):
    """
    Compute cross-correlation between two sets of event times.

    Parameters:
    - led_times: Array of LED event times
    - ff_times: Array of FF (responding) event times
    - max_lag_ms: Maximum lag to compute correlation (in milliseconds)
    - bin_size_ms: Size of time bins for correlation (in milliseconds)

    Returns:
    - lags: Array of lag times
    - correlation: Cross-correlation values
    """
    min_time = min(np.min(led_times), np.min(ff_times))
    max_time = max(np.max(led_times), np.max(ff_times))

    bins = np.arange(min_time, max_time + bin_size_ms, bin_size_ms)
    led_hist, _ = np.histogram(led_times, bins=bins)
    ff_hist, _ = np.histogram(ff_times, bins=bins)

    # Compute cross-correlation
    correlation = signal.correlate(
        led_hist - np.mean(led_hist),
        ff_hist - np.mean(ff_hist),
        mode='full'
    )

    # Normalize correlation
    correlation = correlation / (np.std(led_hist) * np.std(ff_hist) * len(led_hist))

    # Generate lags
    lags = np.linspace(-max_lag_ms / 2, max_lag_ms / 2, len(correlation))

    return lags, correlation


def delay_embedding(x, m, tau):
    """
    Perform delay coordinate embedding using Takens' theorem

    Parameters:
    - x: Time series
    - m: Embedding dimension
    - tau: Time delay

    Returns:
    Embedded trajectory matrix
    """
    # Ensure enough points for embedding
    N = len(x)
    if N < (m - 1) * tau + 1:
        raise ValueError("Insufficient data points for given embedding parameters")

    # Create embedding matrix
    X = np.zeros((N - (m - 1) * tau, m))
    for i in range(m):
        X[:, i] = x[i * tau: N - (m - 1) * tau + i * tau]

    return X


def recurrence_with_embedding(x, m=2, tau=1, epsilon=None, norm='max'):
    """
    Compute recurrence plot with sophisticated embedding and distance metrics

    Parameters:
    - x: Time series
    - m: Embedding dimension
    - tau: Time delay
    - epsilon: Percent of max to be threshold (if None, 5%)
    - norm: Distance norm ('max', 'euclidean')

    Returns:
    Distance matrix and threshold
    """
    # Embed the time series
    X = delay_embedding(x, m, tau)

    # Compute pairwise distances
    if norm == 'max':
        # Chebyshev (max) norm - common in dynamical systems
        distances = np.max(np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]), axis=-1)
    elif norm == 'euclidean':
        # Euclidean norm
        distances = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=-1)
    else:
        raise ValueError("Unsupported norm type")

    if epsilon is None:
        # Typical method: 5% of maximum distance
        thresh = 0.05 * np.max(distances)
    else:
        thresh = epsilon * np.max(distances)

    return distances, thresh


def cao_embedding_dim(x, max_dim=10, tau=1):
    """
    Estimate optimal embedding dimension using Cao's method.

    Parameters:
    - x: Time series
    - max_dim: Maximum embedding dimension to test
    - tau: Time delay

    Returns:
    - E1: Array of E1(m) values
    - Recommended embedding dimension
    """
    N = len(x)

    def create_embedding(x, m, tau):
        N = len(x)
        if N < (m - 1) * tau + 1:
            raise ValueError("Insufficient data points for embedding")
        X = np.zeros((N - (m - 1) * tau, m))
        for i in range(m):
            X[:, i] = x[i * tau: N - (m - 1) * tau + i * tau]

        return X

    def compute_distance_stabilization(m):
        try:
            X_m = create_embedding(x, m, tau)
            X_m1 = create_embedding(x, m + 1, tau)
        except ValueError:
            return 2

        # Compute nearest-neighbor distances in m and m+1 dimensions
        d_m = np.linalg.norm(X_m[:, np.newaxis, :] - X_m[np.newaxis, :, :], axis=-1)
        d_m1 = np.linalg.norm(X_m1[:, np.newaxis, :] - X_m1[np.newaxis, :, :], axis=-1)

        # Avoid self-comparison by setting diagonals to infinity
        np.fill_diagonal(d_m, np.inf)
        np.fill_diagonal(d_m1, np.inf)

        # Find nearest neighbors in m-dimension
        min_indices = np.argmin(d_m, axis=1)
        safe_min_indices = np.clip(min_indices[:len(d_m1)], 0, d_m1.shape[1] - 1)
        valid_indices = np.arange(len(d_m1))  # Match d_m1 row count
        E1_m = np.mean(d_m1[valid_indices, safe_min_indices] / d_m[valid_indices, safe_min_indices])

        return E1_m

    E1 = np.array([compute_distance_stabilization(m) for m in range(1, max_dim + 1)])
    try:
        embedding_dim = np.argmax(np.abs(E1[1:] - E1[:-1]) < 0.01) + 1
        return embedding_dim
    except TypeError:
        return E1


def estimate_embedding_params(x, max_tau=10):
    """
    Estimate embedding dimension and delay using mutual information

    Parameters:
    - x: Time series
    - max_tau: Maximum time delay to consider

    Returns:
    Recommended tau and embedding dimension
    """

    def mutual_information(x, tau):
        x1 = x[:-tau]
        x2 = x[tau:]
        return np.abs(np.cov(x1, x2)[0, 1])

    mis = [mutual_information(x, tau) for tau in range(1, max_tau + 1)]
    tau = np.argmin(mis) + 1

    embed_dim = cao_embedding_dim(x, 10, tau)

    return tau, embed_dim


def sinusoid(x, A, B, C, D):
    return A * np.sin(B * x + C) + D


def recurrence_period_density_entropy(R, normalize=True):
    n = R.shape[0]
    period_counts = {}

    # Loop over all possible time delays (i.e., diagonals above the main one)
    for tau in range(1, n):
        diag = np.diag(R, k=tau)
        recurrence_count = np.sum(diag)
        if recurrence_count > 0:
            period_counts[tau] = recurrence_count

    if not period_counts:
        return 0.0  # no recurrence periods found

    # Convert counts to probability distribution
    total_counts = sum(period_counts.values())
    p = np.array([count / total_counts for count in period_counts.values()])

    # Shannon entropy
    entropy = -np.sum(p * np.log(p))

    if normalize:
        entropy /= np.log(len(period_counts))  # normalized to [0, 1]

    return entropy


def circular_autocorrelation(phases, max_lag=None):
    phases = np.array(phases)
    n = len(phases)
    if max_lag is None:
        max_lag = n - 1

    # Convert to complex form on the unit circle
    z = np.exp(1j * phases)

    ac = []
    for tau in range(1, max_lag + 1):
        r = np.mean(z[:-tau] * np.conj(z[tau:]))
        ac.append(np.abs(r))  # or r.real for cosine component
    return np.array(ac)


def fit_prc(phases, shifts, d, k, i):
    """
    Fit mean phase response curve

    Parameters:
    - phases: phase differences
    - shifts: phase shifts
    - d, k, i: date, key, index of experiment

    Returns:
    - fit_y: best fit (poly or sinusoidal) for the phase response curve
    """
    to_plot_phase_avgs = improved_circular_normalize(
        np.array(phases), float(k) / 1000)[0]

    sorted_data = sorted(zip(to_plot_phase_avgs, shifts), key=lambda x: x[0], reverse=True)
    tps, tss = zip(*sorted_data)
    tps, tss = np.array(tps), np.array(tss)
    initial_guess = [np.ptp(tss) / 2, 2 * np.pi / np.ptp(tps), 0, np.ptp(tss)]
    lower_bounds = [-np.inf, -np.inf, -np.inf, 0]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]
    bounds = (lower_bounds, upper_bounds)

    try:
        params, _ = optimize.curve_fit(sinusoid, tps, tss, p0=initial_guess, bounds=bounds)
    except RuntimeError:
        params = initial_guess
        print(f'using initial guess on {d, k, i}')

    fit_x = np.linspace(0.0, 1.0, 200)
    poly_coeffs = np.polyfit(tps, tss, deg=3)  # Get coefficients [a, b, c, d]
    poly_func = np.poly1d(poly_coeffs)  # Create a polynomial function
    poly_fit_y = poly_func(fit_x)
    sinusoid_residuals = tss - sinusoid(tps, *params)
    sinusoid_rmse = np.sqrt(np.mean(sinusoid_residuals ** 2))
    poly_residuals = tss - poly_func(tps)
    poly_rmse = np.sqrt(np.mean(poly_residuals ** 2))

    # Select the better model
    if sinusoid_rmse < poly_rmse:
        best_fit_y = sinusoid(fit_x, *params)
        fit_label = "Sinusoidal Fit"
        best_residuals = sinusoid_residuals
    else:
        best_fit_y = poly_fit_y
        fit_label = "Cubic Polynomial Fit"
        best_residuals = poly_residuals

    ss_total = np.sum((tss - np.mean(tss)) ** 2)
    ss_residuals = np.sum(best_residuals ** 2)
    r_squared = 1 - (ss_residuals / ss_total)
    print(f"R² Score: {r_squared}")
    baseline_rmse = np.sqrt(np.mean((tss - np.mean(tss)) ** 2))  # RMSE of using mean as predictor
    rmse = np.sqrt(np.mean(best_residuals ** 2))  # RMSE of sinusoidal fit
    if baseline_rmse > 0:
        improvement = (1 - rmse / baseline_rmse) * 100
    else:
        improvement = 0
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Model RMSE with {fit_label}: {rmse:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    fit_y = best_fit_y

    return fit_y


def recurrence_metrics(recurrence_matrix):
    """
    Calculate metrics on the recurrence matrix

    Parameters:
    - recurrence matrix: np array of recurrence plot

    Returns:
    - recurrence rate, determinance, laminarity
    """
    n = recurrence_matrix.shape[0]

    # Recurrence Rate (RR)
    rr = np.sum(recurrence_matrix) / (n * n)

    # Find diagonal lines of length >= 2
    diag_lengths = []
    for offset in range(-n + 1, n):
        diag = np.diagonal(recurrence_matrix, offset=offset)
        lengths = np.diff(np.where(np.concatenate(([0], diag, [0])) == 0)[0]) - 1
        diag_lengths.extend(lengths[lengths >= 2])

    # Determinism (DET)
    if np.sum(recurrence_matrix) == 0:
        det = 0  # Avoid division by zero
    else:
        det = np.sum(diag_lengths) / np.sum(recurrence_matrix)

    # Find vertical lines of length >= 2
    vert_lengths = []
    for j in range(n):
        col = recurrence_matrix[:, j]
        lengths = np.diff(np.where(np.concatenate(([0], col, [0])) == 0)[0]) - 1
        vert_lengths.extend(lengths[lengths >= 2])

    # Laminarity (LAM)
    if np.sum(recurrence_matrix) == 0:
        lam = 0
    else:
        lam = np.sum(vert_lengths) / np.sum(recurrence_matrix)
    rpde = recurrence_period_density_entropy(recurrence_matrix)

    return rr, det, lam, rpde


def recurrence(ps, method='median', do_stats=False):
    """
    Calculate recurrence plot on list of phases

    Parameters:
    - ps: list of phase differences
    - method: epsilon distance method for recurrence closeness

    Returns:
    - recurrence matrix, as well as the tau and embedding dimension if method == 'embedding',
      and statistics about the matrix or None if unspecified
    """
    phases = np.array(ps)
    distances = np.abs(phases[:, np.newaxis] - phases[np.newaxis, :])

    tau = None
    embed_dim = None
    if method == 'median':
        threshold = np.median(distances)
    elif method == 'percentage':
        threshold = 0.05 * np.max(np.abs(phases[:, np.newaxis] - phases[np.newaxis, :]))
    elif method == 'point_density':
        threshold = np.percentile(distances, 1)
    else:
        tau, embed_dim = estimate_embedding_params(phases, 10)
        distances, threshold = recurrence_with_embedding(phases, embed_dim, tau)
    recurrence_matrix = distances <= threshold

    if do_stats:
        return recurrence_matrix, tau, embed_dim, recurrence_metrics(recurrence_matrix)

    else:
        return recurrence_matrix, tau, embed_dim, None


def bout_analysis(pargs):
    """
    Investigate bouts that appear near-synchronous

    Parameters:
    - pargs: command line arguments with some flags and such
    """
    fpaths = os.listdir(pargs.data_path)

    # date_key_index: [min_time, max_time]
    bouts_to_inspect = {
        '20210527_500_47': [195, 205],
        '20230523_300_106': [238, 246],
        '20220529_400_101': [347, 352],
        '20210526_600_40': [225, 237],
        '20210522_700_10': [94, 131],
        '20210525_600_34': [305, 322],
        '20210528_850_53': [210, 242],
        '20210529_1000_51': [122, 142],
        '20221521_770_5': [150, 165],
        '20220524_500_83': [308, 322],
    }
    for fp in fpaths:
        path, framerate = check_frame_rate(pargs.data_path + '/' + fp)
        if path is None:
            continue

        date = path.split('_')[1].split('/')[1]
        key = path.split('_')[2]
        index = path.split('_')[3].split('.')[0]

        if pargs.log:
            print(f'Bout Analysis on timeseries from {date} with led freq {key}')
        with open(path, 'r') as data_file:
            ts = {'ff': [], 'led': []}
            data = list(csv.reader(data_file))
            if date[0] == '0':
                date = date[-4:] + date[:-4]
            for line in data[1:]:
                try:
                    ts['ff'].append(line[0])
                    ts['led'].append(line[1])
                except IndexError:
                    print(f'Error loading data with {fp}')

            # Prepare time series data
            expname = '{}_{}_{}'.format(date, key, index)
            try:
                min_x, max_x = bouts_to_inspect[expname]
                boutname = '{}_[{}_{}]'.format(expname, min_x, max_x)

                ff_xs = []
                ff_ys = []
                for x in ts['ff']:
                    x_val, y_val = eval(x)
                    x_val = float(x_val)
                    if min_x <= x_val <= max_x:
                        ff_xs.append(x_val)
                        ff_ys.append(0.0 if float(y_val) == 0.0 else 0.5)

                led_xs = []
                led_ys = []
                for x in ts['led']:
                    x_val, y_val = eval(x)
                    x_val = float(x_val)
                    if min_x <= x_val <= max_x:
                        led_xs.append(x_val)
                        led_ys.append(0.5 if float(y_val) == 1.0 else 0.502)

                led_xs_flashes = [x for x, y in zip(led_xs, led_ys) if y == 0.502]
                ff_xs_flashes = [x for x, y in zip(ff_xs, ff_ys) if y == 0.5]
                _, _, pairs, period = compute_phase_response_curve(time_series_led=led_xs_flashes,
                                                                   time_series_ff=ff_xs_flashes,
                                                                   epsilon=0.08,
                                                                   m=float(key)/1000,
                                                                   do_responses_relative_to_ff=pargs.do_ffrt,
                                                                   only_lookback=pargs.re_norm)

                times, phases, responses, shifts = zip(*[(t, p, r, s) for p, r, s, t in pairs])
                # PRC
                from collections import defaultdict
                shift_sums = defaultdict(lambda: [0, 0])  # [sum of shifts, count]

                # Aggregate shifts per phase
                for p, s in zip(phases, shifts):
                    p = round(p, 6)
                    if MIN_CUTOFF_PERIOD <= s <= MAX_CUTOFF_PERIOD:
                        shift_sums[p][0] += s
                        shift_sums[p][1] += 1

                # Compute averages
                to_plot_phase_avgs = []
                to_plot_shift_avgs = []

                for p, (shift_sum, count) in shift_sums.items():
                    to_plot_phase_avgs.append(p)
                    to_plot_shift_avgs.append(shift_sum / count)

                to_plot_phases = []
                to_plot_shifts = []
                for p, s in zip(phases, shifts):
                    if MIN_CUTOFF_PERIOD <= s <= MAX_CUTOFF_PERIOD:
                        to_plot_phases.append(p)
                        to_plot_shifts.append(s)
                fig, ax = plt.subplots()
                ax.scatter(to_plot_phases, to_plot_shifts, color='orange')
                ax.axhline(1.0, color='black', linestyle='-')
                ax.set_xlabel('Phase difference')
                ax.set_xlabel('Phase shift')
                ax.set_xlim([0.0, float(key)/1000])
                ax.set_ylim([0.0, 2.75])
                plt.savefig('figs/prc_{}'.format(boutname))
                plt.close()
            except KeyError:
                continue


def get_bout_indices(flash_times, max_gap=2.0):
    """
    Returns a list of (start_idx, end_idx) index pairs for each bout,
    where a bout is a sequence of flashes with inter-flash gaps ≤ max_gap.
    """
    bout_indices = []
    if not flash_times:
        return bout_indices

    start = 0
    for i in range(1, len(flash_times)):
        if flash_times[i] - flash_times[i - 1] > max_gap:
            if i - 1 >= start:
                bout_indices.append((start, i - 1))
            start = i
    if start <= len(flash_times) - 1:
        bout_indices.append((start, len(flash_times) - 1))  # Final bout
    return bout_indices


def compute_det_and_lam_and_rr(R, l_min=2, v_min=2):
    """
    Computes DET and LAM and Recurrence Rate for a given recurrence submatrix R.
    l_min: min diagonal length
    v_min: min vertical length
    """
    if np.sum(R) == 0:
        return 0.0, 0.0, 0.0

    # --- Determinism ---
    diag_lengths = []
    n = R.shape[0]
    for offset in range(-n + 1, n):
        diag = np.diagonal(R, offset=offset)
        lengths = np.diff(np.where(np.concatenate(([0], diag, [0])) == 0)[0]) - 1
        diag_lengths.extend(lengths[lengths >= l_min])
    det = np.sum(diag_lengths) / np.sum(R)

    # --- Laminarity ---
    vert_lengths = []
    for col in range(n):
        column = R[:, col]
        lengths = np.diff(np.where(np.concatenate(([0], column, [0])) == 0)[0]) - 1
        vert_lengths.extend(lengths[lengths >= v_min])
    lam = np.sum(vert_lengths) / np.sum(R)

    rr = np.sum(R) / (n * n)

    return det, lam, rr


def windowed_rqa(recurrence_matrix, flash_times, led_start_time, name, times, phases, shifts, responses, max_gap=2.0):
    ft = [x for x in flash_times if x >= led_start_time]
    prior_flashes = [x for x in flash_times if x < led_start_time]

    if len(prior_flashes) >= 2:
        prior_periods = np.diff(prior_flashes)
        prior_period = np.median(prior_periods)
        prior_period_var = np.var(prior_periods)
    else:
        prior_period = np.nan
        prior_period_var = np.nan

    flash_times = dedupe(ft, eps=0.08)
    bout_ranges = get_bout_indices(flash_times, max_gap=max_gap)
    results = []

    for start, end in bout_ranges:
        sub_R = recurrence_matrix[start:end, start:end]
        det, lam, rr = compute_det_and_lam_and_rr(sub_R)

        bout_start_time = flash_times[start]
        bout_end_time = flash_times[end]

        # Subset the PRC-related values
        bout_phases = []
        bout_shifts = []
        bout_responses = []
        for t, p, s, r in zip(times, phases, shifts, responses):
            if bout_start_time <= t <= bout_end_time:
                bout_phases.append(p)
                bout_shifts.append(s)
                bout_responses.append(r)

        results.append({
            "start_idx": bout_start_time,
            "end_idx": bout_end_time,
            "duration": bout_end_time - bout_start_time,
            "num_flashes": end - start,
            "det": det,
            "lam": lam,
            "rr": rr,
            "name": name,
            "phases": bout_phases,
            "shifts": bout_shifts,
            "responses": bout_responses,
            "prior_period": prior_period,
            "prior_period_var": prior_period_var,
        })

    return results


def extract_top_bottom_bouts(metrics_dict, bout_gap_length, pargs, top_n=10, num_flashes=10, plot_prc=False):
    """
    Extracts, saves, and plots the top 10 and bottom 10 bouts by DET + LAM.

    Parameters:
    - metrics_dict: dict containing bout start and end indices and statistics
    - bout_gap_length: gap length to use (2-5)
    - pargs: program arguments
    - top_n: how many to show (default 10)
    - num_flashes: how many flashes need to be in the top
    - plot_prc: whether to plot the phase response curve for the bouts
    """
    bouts = []
    for key, bouts_list in metrics_dict[bout_gap_length].items():
        for bout in bouts_list:
            if bout['det'] > 0.1 and bout['lam'] > 0.1:
                score = bout['det'] + bout['lam']
                bouts.append((key, bout, score, bout['num_flashes']))

    # Sort by score and only select top if they have enough flashes
    sorted_bouts = sorted(bouts, key=lambda x: x[2], reverse=True)
    top_bouts = [b for b in sorted_bouts if b[3] > num_flashes][:top_n]
    bottom_bouts = sorted_bouts[-top_n:]
    selected_bouts = top_bouts + bottom_bouts

    for i, (key, bout, score, nf) in enumerate(selected_bouts):
        date, led_freq, index = key.split('_')

        # Construct file path
        fname = f"{date}_{led_freq}_{index}.csv"
        fpath = os.path.join(pargs.data_path, fname)

        if not os.path.exists(fpath):
            date = date[-4:] + date[:-4]
            fname = f"{date}_{led_freq}_{index}.csv"
            fpath = os.path.join(pargs.data_path, fname)
            if not os.path.exists(fpath):
                continue

        # Load data
        with open(fpath, 'r') as data_file:
            ts = {'ff': [], 'led': []}
            data = list(csv.reader(data_file))
            for line in data[1:]:
                try:
                    ts['ff'].append(line[0])
                    ts['led'].append(line[1])
                except IndexError:
                    print(f'Error loading data with {fpath}')

            # Prepare time series data
            if date[0] == '0':
                date = date[-4:] + date[:-4]
            expname = '{}_{}_{}'.format(date, key, index)
            min_x, max_x = bout['start_idx'], bout['end_idx']
            boutname = '{}_[{}_{}]'.format(expname, min_x, max_x)

            ff_xs = []
            ff_ys = []
            for x in ts['ff']:
                x_val, y_val = eval(x)
                x_val = float(x_val)
                if min_x <= x_val <= max_x:
                    ff_xs.append(x_val)
                    ff_ys.append(0.0 if float(y_val) == 0.0 else 0.5)
            led_xs = []
            led_ys = []
            for x in ts['led']:
                x_val, y_val = eval(x)
                x_val = float(x_val)
                if min_x <= x_val <= max_x:
                    led_xs.append(x_val)
                    led_ys.append(0.5 if float(y_val) == 1.0 else 0.502)
            led_xs_flashes = [x for x, y in zip(led_xs, led_ys) if y == 0.502]
            led_xs_flashes = dedupe(led_xs_flashes, 0.08)
            ff_xs_flashes = [x for x, y in zip(ff_xs, ff_ys) if y == 0.5]
            ff_xs_flashes = dedupe(ff_xs_flashes, 0.08)
            # Save to CSV
            outname = f'bout_{i:02d}_score_{score:.2f}_{key}_[{min_x}-{max_x}].csv'
            outpath = os.path.join(f'bouts_{bout_gap_length}', outname)
            os.makedirs(f'bouts_{bout_gap_length}', exist_ok=True)

            with open(outpath, 'w', newline='') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow(['FF', 'LED'])
                writer.writerows(zip(ff_xs_flashes, led_xs_flashes))
            print(f"Saved {outpath}")
            plt.figure(figsize=(6, 2))

            # Highlight flash points
            plt.scatter(ff_xs_flashes, [0.5] * len(ff_xs_flashes), color='green', s=10, marker='o', label='FF flashes')
            plt.scatter(led_xs_flashes, [0.51] * len(led_xs_flashes), color='purple', s=10, marker='x',
                        label='LED flashes')

            plt.ylim(0.49, 0.52)
            plt.title(f'{boutname}\nScore: {score:.2f}')
            plt.xlabel('Time')
            plt.ylabel('Signal')
            plt.legend(loc='upper right', fontsize='x-small', frameon=False)
            plt.tight_layout()

            # Save plot
            plotname = f'bout_{i:02d}_score_{score:.2f}_{key}[{min_x}-{max_x}].png'
            plotpath = os.path.join(f'bouts_{bout_gap_length}', plotname)
            plt.savefig(plotpath, dpi=150)
            plt.close()

            if plot_prc:
                _, _, pairs, period = compute_phase_response_curve(
                    time_series_led=led_xs_flashes,
                    time_series_ff=ff_xs_flashes,
                    epsilon=0.08,
                    m=float(key)/1000,
                    do_responses_relative_to_ff=pargs.do_ffrt,
                    only_lookback=pargs.re_norm
                )
                times, phases, responses, shifts = zip(*[(t, p, r, s) for p, r, s, t in pairs])
                from collections import defaultdict
                shift_sums = defaultdict(lambda: [0, 0])  # [sum of shifts, count]

                # Aggregate shifts per phase
                for p, s in zip(phases, shifts):
                    p = round(p, 6)
                    if MIN_CUTOFF_PERIOD <= s <= MAX_CUTOFF_PERIOD:
                        shift_sums[p][0] += s
                        shift_sums[p][1] += 1

                # Compute averages
                to_plot_phase_avgs = []
                to_plot_shift_avgs = []

                for p, (shift_sum, count) in shift_sums.items():
                    to_plot_phase_avgs.append(p)
                    to_plot_shift_avgs.append(shift_sum / count)

                to_plot_phases = []
                to_plot_shifts = []
                for p, s in zip(phases, shifts):
                    if MIN_CUTOFF_PERIOD <= s <= MAX_CUTOFF_PERIOD:
                        to_plot_phases.append(p)
                        to_plot_shifts.append(s)
                fig, ax = plt.subplots()
                ax.scatter(to_plot_phases, to_plot_shifts, color='grey')
                ax.scatter(to_plot_phase_avgs, to_plot_shift_avgs, color='orange')
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.5, 2.0)
                ax.axhline(1.0, linestyle='-', color='black', alpha=0.7)
                plotname = f'bout_{i:02d}_score_{score:.2f}_{key}[{min_x}-{max_x}].png'

                plotpath = os.path.join(f'bouts_{bout_gap_length}', 'prc_{}'.format(plotname))
                plt.savefig(plotpath, dpi=150)
                plt.close()


def do_nonlinear_analysis(pargs):
    """
    Perform various non-linear analyses on the dataset
    Right now:
        - gets each file in the data path
        - splits the LED and FF timeseries
        - calculates phase response curve from the two timseries, as well as phases, firefly periods, etc
        - recurrence plots
        - detrended fluctuation analysis
        - cross- and auto-correlation
        - poincare plots

    Parameters:
    - pargs: command line arguments with some flags and such
    """
    fpaths = os.listdir(pargs.data_path)
    metrics_dict = {bl: {} for bl in [2.0, 3.0, 4.0, 5.0]}
    phases_dict = {bl: {} for bl in [2.0, 3.0, 4.0, 5.0]}

    do_bouts = True
    for fp in fpaths:
        path, framerate = check_frame_rate(pargs.data_path + '/' + fp)
        if path is None:
            continue

        date = path.split('_')[1].split('/')[1]
        key = path.split('_')[2]
        index = path.split('_')[3].split('.')[0]

        if pargs.log:
            print(f'DFA Analysis on timeseries from {date} with led freq {key}')
        with open(path, 'r') as data_file:
            ts = {'ff': [], 'led': []}
            data = list(csv.reader(data_file))
            if date[0] == '0':
                date = date[-4:] + date[:-4]
            for line in data[1:]:
                try:
                    ts['ff'].append(line[0])
                    ts['led'].append(line[1])
                except IndexError:
                    print(f'Error loading data with {fp}')

            # Prepare time series data
            ff_xs = np.array([float(eval(x)[0]) for x in ts['ff']])
            ff_ys = np.array([0.0 if float(eval(x)[1]) == 0.0 else 0.5 for x in ts['ff']])
            led_xs = np.array([float(eval(x)[0]) for x in ts['led']])
            led_ys = np.array([0.5 if float(eval(x)[1]) == 1.0 else 0.502 for x in ts['led']])
            led_xs_flashes = [x for x, y in zip(led_xs, led_ys) if y == 0.502]
            ff_xs_flashes = [x for x, y in zip(ff_xs, ff_ys) if y == 0.5]

            ### NLAnalysis
            _, _, pairs, period = compute_phase_response_curve(time_series_led=led_xs_flashes,
                                                               time_series_ff=ff_xs_flashes,
                                                               epsilon=0.08,  # one frame
                                                               m=float(key) / 1000,
                                                               do_responses_relative_to_ff=pargs.do_ffrt,
                                                               only_lookback=pargs.re_norm)

            times, phases, responses, shifts = zip(*[(t, p, r, s) for p, r, s, t in pairs])
            figtitle = 'figs/analysis/DFA_Analysis_and_Cross-Correlation_of_phases_{}_{}_{}.png'.format(date, key, index)

            if pargs.re_norm:
                # phases = renorm_phases(phases)
                figtitle = 'figs/analysis/DFA_Analysis_and_Cross-Correlation_of_phases_0-1_{}_{}_{}.png'.format(date,
                                                                                                                key,
                                                                                                                index)
            scales, fluctuations, alpha, ci = dfa(phases)
            lags, correlation = compute_cross_correlation(led_xs_flashes, ff_xs_flashes)
            if pargs.log:
                if alpha is not None:
                    phase_hurst, _ = whittle_mle(phases)
                    peak_index = np.argmax(correlation)
                    peak_lag = lags[peak_index]
                    peak_correlation = correlation[peak_index]
                    print(f"DFA, Cross-correlation complete for {date} {key} {index}")
                    print(f"Alpha: {alpha:.4f}")
                    print(f"Hurst exp: {phase_hurst:.4f} ")
                    print(f"Peak Correlation: {peak_correlation:.4f}")
                    print(f"Peak Lag: {peak_lag:.4f} ms")

            pfigtitle = 'figs/analysis/autocorrelation_of_phases_{}_{}_{}.png'.format(date, key, index)
            if pargs.re_norm:
                pfigtitle = 'figs/analysis/autocorrelation_of_phases_0-1_{}_{}_{}.png'.format(date, key, index)
            plotting_helpers.plot_autocorrelation(phases, pfigtitle)

            if pargs.do_poincare:
                pcare_figtitle = 'figs/analysis/Poincare_of_phases_{}_{}_{}.png'.format(date, key, index)
                if pargs.re_norm:
                    pcare_figtitle = 'figs/analysis/Poincare_of_phases_0-1_{}_{}_{}.png'.format(date, key, index)
                plotting_helpers.plot_poincare(phases, pcare_figtitle)

            fig, axes = plt.subplots(2)
            if alpha is not None:
                phase_hurst, _ = whittle_mle(phases)
                plotting_helpers.plot_dfa_results(scales, fluctuations, alpha, ci, phase_hurst, axes[0])
            else:
                print(f"DFA, Cross-correlation failed for {date} {key} {index}")
                plt.close(fig)
                continue
            plotting_helpers.plot_cross_correlation(lags, correlation, axes[1])
            fig.suptitle(f"DFA_Analysis_and_Cross-Correlation_{date}_{key}_{index}")
            plt.tight_layout()
            plt.savefig(figtitle)
            plt.close(fig)

            for data_type in ['real']:
                for bout_gap_length in [2.0, 3.0, 4.0, 5.0]:
                    if data_type == 'real':
                        for method in ['percentage']:
                        #for method in ['median', 'percentage', 'point_density', 'embedding']:
                            r_currences, t, e, metrics = recurrence(phases, method, do_stats=True)
                            if not do_bouts:
                                if method == 'percentage':
                                    metrics_dict[bout_gap_length]['{}_{}_{}'.format(date, key, index)] = metrics
                                    phases_dict[bout_gap_length]['{}_{}_{}'.format(date, key, index)] = phases
                            else:
                                if method == 'percentage':
                                    led_start_time = led_xs_flashes[0]
                                    metrics = windowed_rqa(r_currences, ff_xs_flashes, led_start_time,
                                                           '{}_{}_{}'.format(date, key, index),
                                                           times, phases, shifts, responses, max_gap=bout_gap_length)
                                    metrics_dict[bout_gap_length]['{}_{}_{}'.format(date, key, index)] = metrics
                                    phases_dict[bout_gap_length]['{}_{}_{}'.format(date, key, index)] = phases

                                    r_currences, t, e, metrics = recurrence(phases, method, do_stats=True)

                            fig, ax = plt.subplots()
                            plotting_helpers.plot_recurrence(r_currences, ax, method, t, e, metrics)
                            figtitle = 'figs/analysis/Recurrence_by_{}thresh_of_phases_{}_{}_{}.png'.format(
                                method, date, key, index
                            )
                            if pargs.re_norm:
                                figtitle = 'figs/analysis/Recurrence_by_{}thresh_of_phases_0-1_{}_{}_{}.png'.format(
                                    method, date, key, index
                                )
                            plt.tight_layout()
                            plt.savefig(figtitle)
                            plt.close(fig)

    for bout_gap_length in [2.0, 3.0, 4.0, 5.0]:
        plotting_helpers.plot_recurrence_bouts(metrics_dict[bout_gap_length], bout_gap_length)
        plotting_helpers.plot_prc_by_key(metrics_dict[bout_gap_length], bout_gap_length)
        extract_top_bottom_bouts(metrics_dict, bout_gap_length, pargs,
                                 top_n=10, num_flashes=10, plot_prc=True)
    print('here')


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
        min_scale = 4
        max_scale = N // 3  # Don't go beyond N/4 for statistical reliability
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
    try:
        alpha = model.params[1]  # Slope
    except IndexError:
        return None, None, None, None

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


def compute_phase_response_curve(time_series_led, time_series_ff, epsilon=0.08, m=0.0,
                                 do_responses_relative_to_ff=False,
                                 only_lookback=False):
    """
    Compute the phase response curve (PRC) of fireflies relative to LED flashes.

    Parameters:
    - time_series_led: array of LED flash timestamps.
    - time_series_ff: array of firefly flash timestamps.
    - epsilon: maximum time difference (in seconds) between frames to be considered the same flash
    - do_responses_relative_to_ff: boolean indicating whether to calculate FL-Response Time [FF_t - LED_(t-1)] or
                                   FF-Response Time [FF_t - FF_(t-1)]
    - only_lookback: only calculate phases relative to prior LED time, not subsequent


    Returns:
    - phases: list of normalized phases (firefly phase relative to LED)
    - response_times: list of response times (time difference between firefly flash and previous LED)
    - phase_time_diff_pairs: list of (phase, response_time, firefly_time) tuples
    - period: endogenous period of the firefly, if it flashed before the LED was started
    """
    led_times_array = np.array(time_series_led)
    firefly_times_array = np.array(time_series_ff)
    phases = []
    response_times = []
    phase_time_diff_pairs = []
    phase_shifts = []

    if len(led_times_array) < 2 or len(firefly_times_array) < 1:
        return [], [], [], [], None

    # deduplicate firefly flash times array
    # accounting to ensure two- or more frame flashes are seen as the same flash
    deduplicated_ff_times = dedupe(firefly_times_array, epsilon)
    deduplicated_led_times = dedupe(led_times_array, epsilon)
    deduplicated_ff_times = np.array(deduplicated_ff_times)
    deduplicated_led_times = np.array(deduplicated_led_times)
    period = detect_period_before_first_led_flash(deduplicated_ff_times, deduplicated_led_times)
    if period is None:
        # Fallback if we can't determine the period from pre-LED data
        print("Warning: Could not determine period from pre-LED data")
        if len(np.diff(deduplicated_ff_times)) > 0:
            period = {'period_estimate': stats.mode(np.diff(deduplicated_ff_times))[0][0]}
        else:
            return [], [], [], [], None
    if do_responses_relative_to_ff:
        previous_response_time = period['period_estimate']
        for i in range(1, len(deduplicated_ff_times)):
            firefly_time = deduplicated_ff_times[i]
            previous_firefly_time = deduplicated_ff_times[i - 1]

            closest_led_idx = np.argmin(np.abs(deduplicated_led_times - firefly_time))
            closest_led_time = deduplicated_led_times[closest_led_idx]
            if only_lookback:
                previous_led_candidates = deduplicated_led_times[deduplicated_led_times <= firefly_time]
                if len(previous_led_candidates) == 0:
                    continue
                else:
                    previous_led_time = np.max(previous_led_candidates)
                    closest_led_time = previous_led_time
            phase = firefly_time - closest_led_time
            phases.append(phase)

            response_time = firefly_time - previous_firefly_time
            response_times.append(response_time)

            phase_shift = response_time / previous_response_time
            phase_shifts.append(phase_shift)
            previous_response_time = response_time
            if only_lookback:
                if 0.0 <= phase <= m:
                    phase_time_diff_pairs.append((phase, response_time, phase_shift, firefly_time))
            else:
                if -0.5 <= phase <= 0.5:
                    phase_time_diff_pairs.append((phase, response_time, phase_shift, firefly_time))

    else:
        previous_response_time = period['period_estimate']
        # Now compute the phase response curve with deduplicated flashes
        for firefly_time in deduplicated_ff_times:
            closest_led_idx = np.argmin(np.abs(deduplicated_led_times - firefly_time))
            closest_led_time = deduplicated_led_times[closest_led_idx]
            if only_lookback:
                previous_led_candidates = deduplicated_led_times[deduplicated_led_times <= firefly_time]
                if len(previous_led_candidates) == 0:
                    continue
                else:
                    previous_led_time = np.max(previous_led_candidates)
                    closest_led_time = previous_led_time
            phase = firefly_time - closest_led_time
            phases.append(phase)

            previous_led_candidates = deduplicated_led_times[(deduplicated_led_times < firefly_time) &
                                                             (deduplicated_led_times != closest_led_time)]

            if len(previous_led_candidates) > 0:
                previous_led_time = np.max(previous_led_candidates)

                response_time = firefly_time - previous_led_time
                response_times.append(response_time)

                phase_shift = response_time / previous_response_time
                phase_shifts.append(phase_shift)
                previous_response_time = response_time

                if only_lookback:
                    if 0.0 <= phase <= m:
                        phase_time_diff_pairs.append((phase, response_time, phase_shift, firefly_time))
                else:
                    if -0.5 <= phase <= 0.5:
                        phase_time_diff_pairs.append((phase, response_time, phase_shift, firefly_time))

    return phases, response_times, phase_time_diff_pairs, period


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


def detect_period_before_first_led_flash(ff_xs_flashes, led_xs_flashes):
    """
    Detect the period of the ff_xs timeseries before the first LED flash.

    Parameters:
    -----------
    ff_xs : array-like
        Timestamps of the ff timeseries
    led_xs_flashes : array-like
        Timestamps of LED flashes

    Returns:
    --------
    dict with period estimation details and list of periods used
    """
    # Find the first LED flash time
    first_led_flash = min(led_xs_flashes) if len(led_xs_flashes) > 0 else np.inf

    # Filter ff_xs and ff_ys to only include data before first LED flash
    pre_flash_xs = [x for x in ff_xs_flashes if x < first_led_flash]
    pre_flash_xs = np.array(pre_flash_xs)

    all_periods = np.diff(pre_flash_xs)
    all_periods_valid = [x for x in all_periods if MAX_CUTOFF_PERIOD > x > MIN_CUTOFF_PERIOD]
    all_periods_valid = np.array(all_periods_valid)

    # If no data before LED flash, return None
    if len(all_periods_valid) < 3:
        return None

    pstats = {
        'mean': np.mean(all_periods_valid),
        'median': np.median(all_periods_valid),
        'std': np.std(all_periods_valid),
        'min': np.min(all_periods_valid),
        'max': np.max(all_periods_valid)
    }

    pstats['coefficient_of_variation'] = pstats['std'] / pstats['mean'] if pstats['mean'] != 0 else np.inf

    # Direct period estimation using median
    period_estimate = pstats['median']

    # Confidence interval for the period
    confidence_interval = stats.t.interval(
        alpha=0.95,  # 95% confidence interval
        df=len(all_periods_valid) - 1,  # degrees of freedom
        loc=period_estimate,  # center of interval
        scale=pstats['std'] / np.sqrt(len(all_periods_valid))  # standard error
    )

    return {
        'statistics': pstats,
        'period_estimate': period_estimate,
        'period_confidence_interval': confidence_interval,
        'periods': (pre_flash_xs, all_periods_valid)
    }


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
    return b, f, r, a

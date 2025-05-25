import helpers
import numpy as np
from scipy import stats


def calculate_statistics(d, key, instance, pargs):
    """
    Calculate a wide range of flash timing statistics for a single trial instance under a specific stimulus frequency

    Processes pairs of firefly flash data and LED timing data to compute:
    - Inter-flash periods before and after LED introduction
    - Windowed and rolling average flash statistics
    - Synchrony with LED flashes
    - Phase response curve information
    - Connected component analysis for assessing bout structure
    - Root mean squared error (RMSE) and variability of flash timing
    - Differences in timing between firefly flashes and LED flashes
    and more

    Parameters:
    -----------
    d: data dictionary containing LED and firefly flash timing information.
       Should have structure like
            d[freq_key][instance]['led'] and d[freq_key][instance]['ff']
    key: frequency string indicating the LED interflash interval in milliseconds (e.g., '300', '600').
    instance: instance identifier for a particular recording or observation.
    pargs: program args

    Returns:
    --------
    dict
        Dictionary of statistical results including:
            - Flash period statistics (mean, variance, median, RMSE)
            - Flash timing distributions before/after LED onset
            - Rolling window flash analysis
            - Phase response metrics
            - Flash-LED synchronization measures
            - Connected component statistics (optional)
            - Error metrics relative to stimulus frequency
            - Flash density and recurrence characteristics
        and more
    """
    for arg_name, arg_value in vars(pargs).items():
        if arg_name not in ['save_folder', 'p', 'investigate', 'window_size_seconds', 'data_path', 'save_data']:
            if arg_value:  # Check if the argument is True
                print('Calculating statistics for {} instance {} of freq {}'.format(arg_name, instance, key))
    # flash timing limits
    min_cutoff = 0.166
    max_cutoff = 3.0

    # window size (events)
    min_flashes_in_window = 3
    k1 = (key == '300')
    k2 = (key == '600')

    # LED information
    led_interflash = (float(key) / 1000)
    ts_led = d[key][instance]['led']
    led_xs = [float(eval(x)[0]) for x in ts_led]
    led_ys = [float(eval(x)[1]) for x in ts_led]
    led_introduced = [l[0] for l in zip(led_xs, led_ys) if int(l[1]) == 2][0]

    # Firefly information
    ts_ff = d[key][instance]['ff']
    ff_xs = [float(eval(x)[0]) for x in ts_ff]
    ff_ys = [float(eval(x)[1]) for x in ts_ff]

    # Combined information
    flash_times_to_include = helpers.get_starts_of_flashes(ff_xs, ff_ys)
    trial_params = helpers.get_offset(pargs.p)
    flashes_before = [x for x in flash_times_to_include if x < led_introduced]
    flashes_after = [x for x in flash_times_to_include if x >= led_introduced]
    _all_periods_before = np.diff(flashes_before)
    _all_periods_after = np.diff(flashes_after)

    if key == '300' or key == '400':
        all_periods_before = [x for x in _all_periods_before if x > min_cutoff]
        all_periods_after = [x for x in _all_periods_after if x > min_cutoff]
    else:
        all_periods_before = [x for x in _all_periods_before]
        all_periods_after = [x for x in _all_periods_after]

    all_periods_after_wo_outliers = [x for x in all_periods_after if min_cutoff <= x <= max_cutoff]
    all_periods_before_wo_outliers = [x for x in all_periods_before if min_cutoff <= x <= max_cutoff]

    # Accumulate the windowed periods and calculate rolling averages
    rolling_flash_avg_flash_times = helpers.get_rolling_window_flash_times(flash_times_to_include,
                                                                           args.window_size_seconds)
    rolling_flash_avg_flash_times_before = [f for f in rolling_flash_avg_flash_times if f[-1] < led_introduced]
    rolling_flash_avg_flash_times_after = [f for f in rolling_flash_avg_flash_times if f[-1] >= led_introduced]
    rolling_flash_periods_before = [y for x in rolling_flash_avg_flash_times_before if len(x) >= min_flashes_in_window
                                    for y in np.diff(x)]
    rolling_flash_periods_after = [y for x in rolling_flash_avg_flash_times_after if len(x) >= min_flashes_in_window
                                   for y in np.diff(x)]
    rolling_flash_periods_after, all_periods_after = helpers.adjust_for_offset(k1, rolling_flash_periods_after,
                                                                               all_periods_after, trial_params)

    rolling_avg_interflash, flash_times_to_plot, mean_interflashes_before, mean_interflashes_after = helpers.r_means(
        rolling_flash_avg_flash_times, led_introduced)

    rolling_flash_periods_after = [x for x in rolling_flash_periods_after if x > min_cutoff]

    # Look for first flash synchrony
    first_after = flashes_after[0]
    immediate_inhibition = first_after - led_introduced
    if immediate_inhibition < (led_interflash) / 2:
        immediate_inhibition = flashes_after[1] - led_introduced
        first_after = flashes_after[1]
    absolute_difference_function = lambda list_value: abs(list_value - first_after)
    led_xs_flashes = [x for x,y in zip(led_xs, led_ys) if int(y) == 2]
    first_led_flash_after_inhibition = min(led_xs_flashes, key=absolute_difference_function)
    led_f = (float(key) / 1000)
    first_flash_synchrony = (first_after - first_led_flash_after_inhibition) / led_f
    actual = [led_interflash] * len(rolling_avg_interflash)
    actual_before = [led_interflash] * len(mean_interflashes_before)
    actual_after = [led_interflash] * len(mean_interflashes_after)

    # Find time delays
    time_delays = helpers.find_time_delays(led_xs_flashes, flashes_after, pargs.window_size_seconds)
    if pargs.do_ffrt:
        phases, phase_shifts, phase_time_difs, period = helpers.compute_phase_response_curve(
            led_xs_flashes, flashes_after, 0.033, float(key)/1000, True
        )
    else:
        phases, phase_shifts, phase_time_difs, period = helpers.compute_phase_response_curve(
            led_xs_flashes, flashes_after, 0.033, float(key)/1000, False
        )
    led_ff_diffs = []
    for x in led_xs_flashes:
        x_min_diff = 2 * led_f
        for y in flashes_after:
            diff = x - y
            if abs(diff) < x_min_diff:
                x_min_diff = diff
        if -led_f < x_min_diff <= led_f:
            led_ff_diffs.append(x_min_diff / led_f)

    ff_led_diffs = []
    for set_of_xs in rolling_flash_avg_flash_times_after:
        if len(set_of_xs) > 2:
            for x in set_of_xs:
                _set = False
                x_min_diff = 2 * led_f
                for y in led_xs_flashes:
                    diff = x - y
                    if abs(diff) < x_min_diff:
                        _set = True
                        x_min_diff = diff
                    else:
                        if _set is True:
                            break
                if 'key' == '300' or key == '400':
                    if x_min_diff > led_f:
                        x_min_diff = x_min_diff - led_f
                    elif x_min_diff < -led_f:
                        x_min_diff = x_min_diff + led_f
                    ff_led_diffs.append(x_min_diff / led_f)
                else:
                    ff_led_diffs.append(x_min_diff / led_f)

        _nl = []
        _ln = []
    bouts_before = helpers.get_bout_count_variance(flashes_before)
    bouts_after = helpers.get_bout_count_variance(flashes_after)
    # aggregate level statistics
    rmse = np.sqrt(((np.array(rolling_avg_interflash) - np.array(actual)) ** 2).mean())
    rmse_before = np.sqrt(((np.array(mean_interflashes_before) - np.array(actual_before)) ** 2).mean())
    rmse_after = np.sqrt(((np.array(mean_interflashes_after) - np.array(actual_after)) ** 2).mean())
    not_squared_error_before = np.array(mean_interflashes_before) - np.array(actual_before)
    not_squared_error_after = np.array(mean_interflashes_after) - np.array(actual_after)
    mean_before = np.mean(mean_interflashes_before)
    mean_after = np.mean(mean_interflashes_after)
    var_before = np.var(mean_interflashes_before)
    var_after = np.var(mean_interflashes_after)
    rmses_over_time = np.sqrt(((np.array(mean_interflashes_after) - np.array(actual_after)) ** 2))

    # Sanity checks
    try:
        flashes_per_window_before = len(mean_interflashes_before) / (len(flashes_before) - 1) # ratio of interflahes to flashes
    except ZeroDivisionError:
        flashes_per_window_before = 0
    try:
        flashes_per_window_after = len(mean_interflashes_after) / (len(flashes_after) - 1) # ratio of interflahes to flashes
    except ZeroDivisionError:
        flashes_per_window_after = 0
    try:
        mode_b = stats.mode(np.diff(flashes_before))[0][0]
    except IndexError:
        mode_b = np.nan
    try:
        mode_a = stats.mode(np.diff(flashes_after))[0][0]
    except IndexError:
        mode_a = np.nan

    # Connected components
    if pargs.do_cc:
        try:
            cc_before = 0
            all_cc_lens_b = []
            l_cc_b = 0
            cc_b = []
            max_cc_b = 0
            last_list = rolling_flash_avg_flash_times_before[0]
            for l in rolling_flash_avg_flash_times_before[1:]:
                found = False
                for x in l:
                    if x in last_list:
                        found = True
                        cc_b.append(x)
                        l_cc_b += 1
                        break
                if not found:
                    if l_cc_b > max_cc_b:
                        max_cc_b = l_cc_b
                    if len(cc_b) > 1:
                        all_cc_lens_b.append(cc_b[-1] - cc_b[0])
                    cc_b = []
                    l_cc_b = 0
                    cc_before += 1
                last_list = l
            time_before = led_introduced - flashes_before[0]
            area_before = sum(all_cc_lens_b) / time_before
            cc_before = cc_before / len(rolling_flash_avg_flash_times_before)
        except IndexError:
            cc_before = 0
            max_cc_b = 0
            all_cc_lens_b = []
            area_before = 0

        try:
            l_cc_a = 0
            cc_a = []
            all_cc_lens_a = []
            max_cc_a = 0
            cc_after = 0
            last_list = rolling_flash_avg_flash_times_after[0]
            for l in rolling_flash_avg_flash_times_after[1:]:
                found = False
                for x in l:
                    if x in last_list:
                        found = True
                        cc_a.append(x)
                        l_cc_a += 1
                        break
                if not found:
                    if l_cc_a > max_cc_a:
                        max_cc_a = l_cc_a
                    if len(cc_a) > 1:
                        all_cc_lens_a.append(cc_a[-1] - cc_a[0])
                    cc_a = []
                    l_cc_a = 0
                    cc_after += 1
                last_list = l
            cc_after = cc_after / len(rolling_flash_avg_flash_times_after)
            time_after = flashes_after[-1] - led_introduced
            area_after = sum([a for a in all_cc_lens_a if a > 1]) / time_after
        except IndexError:
            cc_after = 0
            max_cc_a = 0
            all_cc_lens_a = []
            area_after = 0

    else:
        area_before = None
        area_after = None
        max_cc_b = None
        max_cc_a = None
        all_cc_lens_a = None
        all_cc_lens_b = None

    return {'rmse': rmse,
            'rmse_after': rmse_after,
            'rmse_before': rmse_before,
            'bouts_before': bouts_before,
            'bouts_after': bouts_after,
            'phases': phases,
            'phase_shifts': phase_shifts,
            'phase_time_diffs': phase_time_difs,
            'phase_time_diffs_instanced': {instance: phase_time_difs},
            'flashes_before': flashes_before if len(flashes_before) > 0 else np.nan,
            'flashes_after': flashes_after if len(flashes_after) > 0 else np.nan,
            'windowed_period_before': rolling_flash_periods_before if len(rolling_flash_periods_before) > 0 else np.nan,
            'windowed_period_after': rolling_flash_periods_after if len(rolling_flash_periods_after) > 0 else np.nan,
            'not_squared_before': not_squared_error_before,
            'not_squared_after': not_squared_error_after,
            'mean_before': mean_before,
            'mean_after': mean_after,
            'var_before': var_before,
            'var_after': var_after,
            'individual_before': mean_interflashes_before,
            'individual_after': mean_interflashes_after,
            'before': mean_interflashes_before,
            'after': mean_interflashes_after,
            'rmses_over_time': rmses_over_time,
            'flashes_per_window_before': flashes_per_window_before,
            'flashes_per_window_after': flashes_per_window_after,
            'connected_components_before': area_before,
            'connected_components_after': area_after,
            'longest_cc_before': max_cc_b,
            'longest_cc_after': max_cc_a,
            'all_cc_lens_before': all_cc_lens_b,
            'all_cc_lens_after': all_cc_lens_a,
            'mode_before': mode_b,
            'mode_after': mode_a,
            'median_before': np.median(mean_interflashes_before),
            'median_after': np.median(mean_interflashes_after),
            'immediate_inhibition': immediate_inhibition,
            'first_flash_synchrony': first_flash_synchrony,
            'led_ff_diffs': [x for x in time_delays[0]],
            'phase_response': [x for x in time_delays[1]],
            'ff_led_diffs': ff_led_diffs,
            'all_periods_before': all_periods_before if len(all_periods_before) > 0 else np.nan,
            'all_periods_after': all_periods_after if len(all_periods_after) > 0 else np.nan,
            'all_periods_before_wo_outliers': all_periods_before_wo_outliers if len(all_periods_before_wo_outliers) > 0 else np.nan,
            'all_periods_after_wo_outliers': all_periods_after_wo_outliers if len(all_periods_after_wo_outliers) > 0 else np.nan
            }


def sliding_window_stats(d, pargs):
    """
    This function loops over the input data and computes:
    - Aggregated inter-flash periods across windows
    - Time delays between firefly and LED flashes
    - Flash synchrony events and their timing
    - Statistical summaries such as median, mode, and RMSE over time
    - Recurrence and stability of flash patterns across time
    - Phase-based differences when LED is active

    This serves as a higher-level aggregation and summarization tool which calls --calculate_statistics--
    on different conditions

    Parameters:
    -----------
    d: dataset containing all LED and firefly flash instances.
    pargs: program args

    Returns:
    --------
    dict
        Aggregated statistics and temporal patterns from all data instances.
    """
    ks = ['300', '400', '500', '600', '700', '770', '850', '1000']
    rmses = {'rmse': dict.fromkeys(ks, None),
             'rmse_after': dict.fromkeys(ks, None),
             'rmse_before': dict.fromkeys(ks, None),
             'bouts_before': dict.fromkeys(ks, None),
             'bouts_after': dict.fromkeys(ks, None),
             'phases': dict.fromkeys(ks, None),
             'phase_shifts': dict.fromkeys(ks, None),
             'phase_time_diffs': dict.fromkeys(ks, None),
             'phase_time_diffs_instanced': dict.fromkeys(ks, None),
             'flashes_before': dict.fromkeys(ks, None),
             'flashes_after': dict.fromkeys(ks, None),
             'not_squared_before': dict.fromkeys(ks, None),
             'not_squared_after': dict.fromkeys(ks, None),
             'windowed_period_before': dict.fromkeys(ks, None),
             'windowed_period_after': dict.fromkeys(ks, None),
             'mean_before': dict.fromkeys(ks, None),
             'mean_after': dict.fromkeys(ks, None),
             'var_before': dict.fromkeys(ks, None),
             'var_after': dict.fromkeys(ks, None),
             'individual_before': dict.fromkeys(ks, None),
             'individual_after': dict.fromkeys(ks, None),
             'ff_led_diffs': dict.fromkeys(ks, None),
             'led_ff_diffs': dict.fromkeys(ks, None),
             'phase_response': dict.fromkeys(ks, None),
             'before': dict.fromkeys(ks, None),
             'after': dict.fromkeys(ks, None),
             'rmses_over_time': dict.fromkeys(ks, None),
             'flashes_per_window_before': dict.fromkeys(ks, None),
             'flashes_per_window_after': dict.fromkeys(ks, None),
             'connected_components_before': dict.fromkeys(ks, None),
             'connected_components_after': dict.fromkeys(ks, None),
             'mode_before': dict.fromkeys(ks, None),
             'mode_after': dict.fromkeys(ks, None),
             'median_before': dict.fromkeys(ks, None),
             'median_after': dict.fromkeys(ks, None),
             'longest_cc_before': dict.fromkeys(ks, None),
             'longest_cc_after': dict.fromkeys(ks, None),
             'all_cc_lens_before': dict.fromkeys(ks, None),
             'all_cc_lens_after': dict.fromkeys(ks, None),
             'immediate_inhibition': dict.fromkeys(ks, None),
             'first_flash_synchrony': dict.fromkeys(ks, None),
             'all_periods_before': dict.fromkeys(ks, None),
             'all_periods_after': dict.fromkeys(ks, None),
             'all_periods_before_wo_outliers': dict.fromkeys(ks, None),
             'all_periods_after_wo_outliers': dict.fromkeys(ks, None),
             }
    if pargs.do_means:
        rmse_means = {'rmse': dict.fromkeys(ks, 0),
                      'rmse_after': dict.fromkeys(ks, 0),
                      'rmse_before': dict.fromkeys(ks, 0),
                      'mean_before': dict.fromkeys(ks, 0),
                      'mean_after': dict.fromkeys(ks, 0),
                      'flashes_per_window_before': dict.fromkeys(ks, 0),
                      'flashes_per_window_after': dict.fromkeys(ks, 0),

                      'connected_components_before': dict.fromkeys(ks, 0),
                      'connected_components_after': dict.fromkeys(ks, 0),
                      'longest_cc_before': dict.fromkeys(ks, 0),
                      'longest_cc_after': dict.fromkeys(ks, 0),
                      'mode_before': dict.fromkeys(ks, 0),
                      'mode_after': dict.fromkeys(ks, 0),
                      'median_before': dict.fromkeys(ks, 0),
                      'median_after': dict.fromkeys(ks, 0),
                      'all_cc_lens_before': dict.fromkeys(ks, 0),
                      'all_cc_lens_after': dict.fromkeys(ks, 0),
                      'immediate_inhibition': dict.fromkeys(ks, 0),
                      'first_flash_synchrony': dict.fromkeys(ks,0),
                      }
    else:
        rmse_means = None

    for k in ks:
        for j in range(len(list(d[k]))):
            rmse = calculate_statistics(d, k, j, pargs)

            # Aggregate each trial by key
            for stat_key in rmses.keys():
                if stat_key == 'before' or stat_key == 'after':
                    if not rmses[stat_key].get(k):
                        rmses[stat_key][k] = [s for s in rmse[stat_key] if not np.isnan(s)]
                    else:
                        rmses[stat_key][k].extend([s for s in rmse[stat_key] if not np.isnan(s)])
                elif stat_key == 'individual_before' \
                        or stat_key == 'individual_after' \
                        or stat_key == 'ff_led_diffs' or stat_key == 'led_ff_diffs':
                    if not rmses[stat_key].get(k):
                        rmses[stat_key][k] = [rmse[stat_key]]
                    else:
                        rmses[stat_key][k].append(rmse[stat_key])
                elif stat_key == 'phase_time_diffs_instanced':
                    if not rmses[stat_key].get(k):
                        rmses[stat_key][k] = [rmse[stat_key][j]]
                    else:
                        rmses[stat_key][k].append(rmse[stat_key][j])
                elif stat_key == 'rmses_over_time':
                    if not rmses[stat_key].get(k):
                        rmses[stat_key][k] = [rmse[stat_key]]
                    else:
                        rmses[stat_key][k].append(rmse[stat_key])

                elif stat_key == 'all_cc_lens_before' or stat_key == 'all_cc_lens_after':
                    try:
                        if not rmses[stat_key].get(k):
                            rmses[stat_key][k] = [s for s in rmse[stat_key]]
                        else:
                            rmses[stat_key][k].extend([s for s in rmse[stat_key]])
                    except ValueError:
                        if not rmses[stat_key].get(k):
                            rmses[stat_key][k] = [rmse[stat_key]]
                        else:
                            rmses[stat_key][k].append(rmse[stat_key])
                    except TypeError:
                        if not rmses[stat_key].get(k):
                            rmses[stat_key][k] = [rmse[stat_key]]
                        else:
                            rmses[stat_key][k].append(rmse[stat_key])
                else:
                    try:
                        if not rmses[stat_key].get(k):
                            rmses[stat_key][k] = [s for s in rmse[stat_key]]
                        else:
                            rmses[stat_key][k].extend([s for s in rmse[stat_key]])
                    except ValueError:
                        if not rmses[stat_key].get(k):
                            rmses[stat_key][k] = [rmse[stat_key]]
                        else:
                            rmses[stat_key][k].append(rmse[stat_key])
                    except TypeError:
                        if not rmses[stat_key].get(k):
                            rmses[stat_key][k] = [rmse[stat_key]]
                        else:
                            rmses[stat_key][k].append(rmse[stat_key])

        if pargs.do_means and rmse_means:
            for mean_key in rmse_means.keys():
                xs = [x for x in rmses[mean_key][k] if not np.isnan(x)]
                rmse_means[mean_key][k] = np.mean(xs)

    return rmses, rmse_means

import numpy as np
import pandas as pd


def find_time_delays(time_series_led, time_series_ff, window_length):
    # Function to find the time delay
    delays = []
    prc = []
    time_series_led = np.array(time_series_led)
    time_series_ff = np.array(time_series_ff)

    for i, time_led in enumerate(time_series_led):
        # Define the window for the current time point in A
        window_start = time_led - window_length / 2
        window_end = time_led + window_length / 2

        # Find points in time series B within the window
        points_in_window_indices = np.where((time_series_ff >= window_start) & (time_series_ff <= window_end))[0]
        points_in_window = time_series_ff[points_in_window_indices]

        # Check if the window contains at least 3 points from time series B
        if len(points_in_window) >= 3:
            time_delays = points_in_window - time_led
            min_index = np.argmin(np.abs(time_delays))
            nearest_delay = time_delays[min_index]
            try:
                period_change = np.diff(points_in_window)[1] - np.diff(points_in_window)[0]
            except IndexError:
                continue  # will pick up next window

            # Normalize the delay
            if 0 < i < len(time_series_led) - 2:
                prev_led = time_series_led[i - 1]
                next_led = time_series_led[i + 2]
                norm_delay = (nearest_delay / (next_led - prev_led)) * 2  # Scale to -0.5 to 0.5
            elif i == 0:
                next_led = time_series_led[i + 1]
                norm_delay = nearest_delay / (next_led - time_led) - 0.5
            else:
                prev_led = time_series_led[i - 1]
                norm_delay = (nearest_delay / (time_led - prev_led)) + 0.5

            if abs(norm_delay) <= 0.5:
                delays.append(norm_delay)
                prc.append((norm_delay, period_change))

    return delays, prc


def get_starts_of_flashes(ff_xs, ff_ys):
    flash_times = [x[0] for x in list(zip(ff_xs, ff_ys)) if x[1] == 1.0]
    good_idxs = np.where((np.diff(flash_times) * 1000) > 100)[0]
    flash_times_to_include = []
    for idx in range(len(good_idxs)):
        flash_times_to_include.append(flash_times[good_idxs[idx]])
    return flash_times_to_include


def get_start_index(flash_times_to_include, start_time):
    start_index = 0
    for ix in range(len(flash_times_to_include)):
        if flash_times_to_include[ix] > start_time:
            start_index = ix
            break
    return start_index


def get_introduced_time(k, instance):
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


def get_temperature_2022(led_freq, j, cutoff_j):
    t = pd.read_csv('temp_data/export_Peleglabkestrel3_2022_6_4_18_47_39.csv')
    t['FORMATTED DATE-TIME'] = pd.to_datetime(t['FORMATTED DATE-TIME'])
    start_date=None
    end_date=None
    if led_freq == '770':
        if j - cutoff_j < 1:
            start_date = '05-18-2022 21:15:00'
            end_date = '05-18-2022 21:25:00'
        elif 1 <= j - cutoff_j < 3:
            start_date = '05-18-2022 22:00:00'
            end_date = '05-18-2022 22:30:00'
        else:
            start_date = '05-20-2022 21:35:00'
            end_date = '05-20-2022 21:45:00'

    elif led_freq == '600':
        if j - cutoff_j < 2:
            start_date = '05-18-2022 21:20:00'
            end_date = '05-18-2022 21:40:00'
        elif 2 <= j - cutoff_j < 3:
            start_date = '05-19-2022 21:45:00'
            end_date = '05-19-2022 21:55:00'
        elif 3 <= j - cutoff_j < 4:
            start_date = '05-25-2022 21:15:00'
            end_date = '05-25-2022 21:25:00'
        else:
            start_date = '05-26-2022 21:10:00'
            end_date = '05-26-2022 21:20:00'

    elif led_freq == '700':
        if 0 <= j - cutoff_j < 3:
            start_date = '05-18-2022 22:00:00'
            end_date = '05-18-2022 22:30:00'
        elif 3 <= j - cutoff_j < 5:
            start_date = '05-29-2022 21:15:00'
            end_date = '05-29-2022 21:35:00'
        else:
            start_date = '05-25-2022 21:35:00'
            end_date = '05-25-2022 21:45:00'

    elif led_freq == '850':
        if 0 <= j - cutoff_j < 1:
            start_date = '05-19-2022 21:35:00'
            end_date = '05-19-2022 21:45:00'
        else:
            start_date = '05-20-2022 22:10:00'
            end_date = '05-20-2022 22:25:00'

    elif led_freq == '1000':
        start_date = '05-26-2022 21:40:00'
        end_date = '05-26-2022 21:50:00'

    elif led_freq == '500':
        if 0 <= j - cutoff_j < 1:
            start_date = '05-19-2022 21:00:00'
            end_date = '05-19-2022 21:10:00'
        elif 1 <= j - cutoff_j < 2:
            start_date = '05-21-2022 22:25:00'
            end_date = '05-21-2022 22:35:00'
        elif 2 <= j - cutoff_j < 4:
            start_date = '05-24-2022 21:05:00'
            end_date = '05-24-2022 21:25:00'
        elif 4 <= j - cutoff_j < 5:
            start_date = '05-26-2022 21:00:00'
            end_date = '05-26-2022 21:10:00'
        else:
            start_date = '05-26-2022 22:00:00'
            end_date = '05-26-2022 22:20:00'

    elif led_freq == '400':
        if 0 <= j - cutoff_j < 2:
            start_date = '05-29-2022 21:05:00'
            end_date = '05-29-2022 21:20:00'
        elif 2 <= j - cutoff_j < 4:
            start_date = '05-29-2022 21:30:00'
            end_date = '05-29-2022 21:45:00'
        elif 4 <= j - cutoff_j < 5:
            start_date = '05-29-2022 22:00:00'
            end_date = '05-29-2022 22:10:00'
        else:
            start_date = '05-29-2022 22:15:00'
            end_date = '05-29-2022 22:35:00'

    elif led_freq == '300':
        if 0 <= j - cutoff_j < 2:
            start_date = '05-29-2022 21:20:00'
            end_date = '05-29-2022 21:30:00'
        elif 2 <= j - cutoff_j < 4:
            start_date = '05-29-2022 21:45:00'
            end_date = '05-29-2022 22:00:00'
        else:
            start_date = '05-29-2022 22:05:00'
            end_date = '05-29-2022 22:15:00'

    mask = (t['FORMATTED DATE-TIME'] > start_date) & (t['FORMATTED DATE-TIME'] <= end_date)
    t_masked = t.loc[mask]
    return t_masked['Temperature'].values


def get_temperature_2021(led_freq, j):
    may_28 = pd.read_csv('temp_data/export_Peleglabkestrel2_2021_5_28_23_46_30.csv')
    may_28['FORMATTED DATE-TIME'] = pd.to_datetime(may_28['FORMATTED DATE-TIME'])
    start_date=None
    end_date=None
    if led_freq == '770':
        if j < 6:
            start_date = '05-21-2021 20:30:00'
            end_date = '05-21-2021 21:20:00'
        elif 6 <= j < 8:
            start_date = '05-24-2021 20:50:00'
            end_date = '05-24-2021 21:30:00'
        else:
            start_date = '05-28-2021 21:20:00'
            end_date = '05-28-2021 22:00:00'
    elif led_freq == '600':
        if j < 4:
            start_date = '05-24-2021 21:00:00'
            end_date = '05-24-2021 21:40:00'
        elif 4 <= j < 10:
            start_date = '05-25-2021 21:45:00'
            end_date = '05-25-2021 22:30:00'
        else:
            start_date = '05-26-2021 21:50:00'
            end_date = '05-26-2021 22:00:00'
    elif led_freq == '700':
        if 0 <= j < 6:
            start_date = '05-23-2021 20:45:00'
            end_date = '05-23-2021 21:45:00'
        elif j == 6:
            start_date = '05-23-2021 22:40:00'
            end_date = '05-23-2021 22:50:00'
        elif 7 <= j < 10:
            start_date = '05-24-2021 22:35:00'
            end_date = '05-24-2021 22:50:00'
        else:
            start_date = '05-25-2021 21:25:00'
            end_date = '05-25-2021 21:45:00'
    elif led_freq == '850':
        if j < 4:
            start_date = '05-23-2021 22:00:00'
            end_date = '05-23-2021 22:40:00'
        elif 4 <= j < 6:
            start_date = '05-24-2021 20:35:00'
            end_date = '05-24-2021 21:10:00'
        elif 6 == j:
            start_date = '05-27-2021 22:00:00'
            end_date = '05-27-2021 22:10:00'
        else:
            start_date = '05-28-2021 21:00:00'
            end_date = '05-28-2021 21:40:00'
    elif led_freq == '1000':
        start_date = '05-28-2021 21:00:00'
        end_date = '05-28-2021 22:00:00'
    mask = (may_28['FORMATTED DATE-TIME'] > start_date) & (may_28['FORMATTED DATE-TIME'] <= end_date)
    may_28_masked = may_28.loc[mask]
    return may_28_masked['Temperature'].values


def extend_temperatures_2022(k, j):
    may_27 = pd.read_csv('temp_data/export_Peleglabkestrel3_2022_5_27_0_14_12.csv')
    may_27['FORMATTED DATE-TIME'] = pd.to_datetime(may_27['FORMATTED DATE-TIME'])
    start_date = None
    end_date = None
    if k == '600':
        if j == 0:
            start_date = '05-18-2022 21:00:00'
            end_date = '05-18-2022 21:10:00'
        elif j == 1:
            start_date = '05-18-2022 21:15:00'
            end_date = '05-18-2022 21:20:00'
        elif j == 2:
            start_date = '05-18-2022 21:20:00'
            end_date = '05-18-2022 21:30:00'
        elif j == 3:
            start_date = '05-20-2022 21:15:00'
            end_date = '05-20-2022 21:25:00'
        elif j == 4:
            start_date = '05-20-2022 21:25:00'
            end_date = '05-20-2022 21:35:00'
        elif j == 5:
            start_date = '05-20-2022 21:55:00'
            end_date = '05-20-2022 22:10:00'
        elif j == 6:
            start_date = '05-21-2022 21:45:00'
            end_date = '05-21-2022 21:55:00'
        elif j == 7:
            start_date = '05-21-2022 21:55:00'
            end_date = '05-21-2022 22:05:00'
        elif j == 8:
            start_date = '05-22-2022 20:50:00'
            end_date = '05-22-2022 21:00:00'
        elif j == 9:
            start_date = '05-24-2022 21:05:00'
            end_date = '05-24-2022 21:15:00'
        elif j == 10:
            start_date = '05-24-2022 21:25:00'
            end_date = '05-24-2022 21:35:00'
        elif j == 11:
            start_date = '05-24-2022 21:35:00'
            end_date = '05-24-2022 21:40:00'
        elif j == 12:
            start_date = '05-25-2022 21:15:00'
            end_date = '05-25-2022 21:25:00'
        elif j == 13:
            start_date = '05-25-2022 21:50:00'
            end_date = '05-25-2022 22:00:00'
        elif j == 14:
            start_date = '05-26-2022 21:10:00'
            end_date = '05-26-2022 21:20:00'
    elif k == '700':
        if j == 0:
            start_date = '05-18-2022 21:45:00'
            end_date = '05-18-2022 21:55:00'
        elif j == 1:
            start_date = '05-18-2022 21:55:00'
            end_date = '05-18-2022 22:05:00'
        elif j == 2:
            start_date = '05-19-2022 21:05:00'
            end_date = '05-19-2022 21:15:00'
        elif j == 3:
            start_date = '05-19-2022 21:15:00'
            end_date = '05-19-2022 21:25:00'
        elif j == 4:
            start_date = '05-19-2022 21:25:00'
            end_date = '05-19-2022 21:35:00'
        elif j == 5:
            start_date = '05-19-2022 22:15:00'
            end_date = '05-19-2022 22:30:00'
        elif j == 6:
            start_date = '05-20-2022 21:45:00'
            end_date = '05-20-2022 22:00:00'
        elif j == 7:
            start_date = '05-21-2022 22:05:00'
            end_date = '05-21-2022 22:15:00'
        elif j == 8:
            start_date = '05-22-2022 21:00:00'
            end_date = '05-22-2022 21:10:00'
        elif j == 9:
            start_date = '05-24-2022 21:50:00'
            end_date = '05-24-2022 22:00:00'
        elif j == 10:
            start_date = '05-25-2022 21:30:00'
            end_date = '05-25-2022 21:40:00'
        elif j == 11:
            start_date = '05-26-2022 21:20:00'
            end_date = '05-26-2022 21:30:00'
        elif j == 12:
            start_date = '05-26-2022 22:10:00'
            end_date = '05-26-2022 22:20:00'
    elif k == '770':
        if j == 0:
            start_date = '05-20-2022 21:35:00'
            end_date = '05-20-2022 21:45:00'
        elif j == 1:
            start_date = '05-21-2022 22:25:00'
            end_date = '05-21-2022 22:35:00'
        elif j == 2:
            start_date = '05-25-2022 22:05:00'
            end_date = '05-25-2022 22:20:00'
        elif j == 3:
            start_date = '05-25-2022 21:40:00'
            end_date = '05-25-2022 21:50:00'
        elif j == 4:
            start_date = '05-26-2022 21:50:00'
            end_date = '05-26-2022 22:00:00'
    elif k == '850':
        if 0 <= j < 3:
            start_date = '05-19-2022 21:30:00'
            end_date = '05-19-2022 22:15:00'
        elif j == 3:
            start_date = '05-20-2022 22:05:00'
            end_date = '05-20-2022 22:20:00'
        elif 4 <= j < 6:
            start_date = '05-21-2022 21:20:00'
            end_date = '05-21-2022 21:40:00'
        elif j == 6:
            start_date = '05-21-2022 22:15:00'
            end_date = '05-21-2022 22:25:00'
        elif j == 7:
            start_date = '05-24-2022 21:40:00'
            end_date = '05-24-2022 21:50:00'
        elif j == 8:
            start_date = '05-25-2022 21:05:00'
            end_date = '05-25-2022 21:15:00'
        elif j == 9:
            start_date = '05-25-2022 22:00:00'
            end_date = '05-25-2022 22:10:00'
        elif j == 10:
            start_date = '05-26-2022 21:30:00'
            end_date = '05-26-2022 21:40:00'
        elif j == 11:
            start_date = '05-26-2022 22:20:00'
            end_date = '05-26-2022 22:30:00'
    elif k == '1000':
        if j == 0:
            start_date = '05-21-2022 21:40:00'
            end_date = '05-21-2022 21:45:00'
        elif 1 <= j < 4:
            start_date = '05-24-2022 20:55:00'
            end_date = '05-24-2022 21:20:00'
        elif j == 4:
            start_date = '05-24-2022 22:00:00'
            end_date = '05-24-2022 22:10:00'
        elif j == 5:
            start_date = '05-25-2022 21:00:00'
            end_date = '05-25-2022 21:10:00'
        elif j == 6:
            start_date = '05-26-2022 21:40:00'
            end_date = '05-26-2022 21:50:00'
    mask = (may_27['FORMATTED DATE-TIME'] > start_date) & (may_27['FORMATTED DATE-TIME'] <= end_date)
    may_27_masked = may_27.loc[mask]
    return may_27_masked['Temperature'].values


def t_f_conversion(t):
    return (0.186 * (t ** 2) - 11.14 * t + 194.1) / 60


def tighten(k):
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
    return (0.76 if p else 0), (0.77 if p else 0), (0.18 if p else 0)


def epsilon_closeness(lists, k, epsilon):
    retlists = []
    retlistkeys=[]
    for j, i in enumerate(lists):
        if len(i) < 0:
            continue
        else:
            if k < epsilon:
                if 1.0 > abs(np.median(i) - k) < (1 - epsilon):
                    retlists.append(i)
                    retlistkeys.append(j)
            else:
                if 1.0 > abs(np.median(i) - k) < (3 * (1 - epsilon)) / 2:
                    retlists.append(i)
                    retlistkeys.append(j)
    return retlists, retlistkeys


def double_std(arr):
    return np.std(arr) * 2


def adjust_for_offset(k1, k2, l1, l2, tp):
    # Adjust for camera timing offset from background stack (where necessary)

    omb, omb_, offset = tp
    if k1:
        l1 = [x if (x < omb or x > omb_) else x - offset for x in l1]
        l2 = [x if (x < omb or x > omb_) else x - offset for x in l2]

    elif k2:
        l1 = [x - (offset/3) for x in l1]
        l2 = [x - (offset/3) for x in l2]
    return l1, l2


def get_rolling_window_flash_times(flash_times_to_include, window_size_seconds):
    # Get a list of flash times that are valid inclusions
    # (belong to a window of size window_size_seconds with sufficiently many flashes)

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
    # If the rolling flash length is before than the LED introduction timing, it becomes part of _flash before_ list
    # Otherwise it is an after-LED introduction flash, and its period can be included
    # returns: list of before and after flash periods, their means, and their first timestamp for organization
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

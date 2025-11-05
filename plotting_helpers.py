import csv
import itertools
import os
import pickle

import matplotlib
import matplotlib.colors as mcolors
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import scipy
import scipy.interpolate
import seaborn as sns
from matplotlib import cm
from plotly.subplots import make_subplots
from scipy import stats, optimize
from scipy.stats import trim_mean, sem, t
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

import helpers

from dataclasses import dataclass
from scipy.optimize import least_squares
from pathlib import Path



DET_FILTER_LEVEL = 0.7
LAM_FILTER_LEVEL = 0.7


def boxplots(all_befores, all_afters, plot_params):
    data = {k: {'before': all_befores[k], 'induced': all_afters[k]} for k in all_befores.keys()}
    df = pd.DataFrame.from_dict({(i, j): data[i][j]
                                 for i in data.keys()
                                 for j in data[i].keys()},
                                orient='index')
    user_ids = []
    frames = []

    for user_id, d in data.items():
        user_ids.append(int(user_id))
        frames.append(pd.DataFrame.from_dict(d, orient='index'))

    df = pd.concat(frames, keys=user_ids)
    df = df.T
    df_melted = df.stack(level=[0, 1]).reset_index()
    df_melted.columns = ['Index', 'Key', 'State', 'Values']

    flierprops = dict(markerfacecolor='0.75', markersize=5,
                      linestyle='none')
    colors = plt.cm.viridis_r(np.linspace(0, 1, 24))  # viridis colormap with 8 colors

    alphas = [0.6, 0.9]
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Key', y='Values', hue='State', data=df_melted, showfliers=False, hue_order=['before', 'induced'])
    plt.title('Firefly periods prior to and following LED introduction')

    for i, key in enumerate(sorted(data.keys(), key=lambda x: int(x))):
        key = int(key)
        state1_values = df_melted[(df_melted['Key'] == key) & (df_melted['State'] == 'before')]['Values']
        state2_values = df_melted[(df_melted['Key'] == key) & (df_melted['State'] == 'induced')]['Values']

        paired_ttest = stats.mannwhitneyu(state1_values, state2_values)
        var1 = np.var(state1_values, ddof=1)
        var2 = np.var(state2_values, ddof=1)

        # F-test statistic
        F = var1 / var2 if var1 > var2 else var2 / var1
        dof1 = len(state1_values) - 1
        dof2 = len(state2_values) - 1
        paired_ftest_pvalue = 2 * (1 - stats.f.cdf(F, dof1, dof2))
        print(paired_ttest, 'F-test: (', F, ',', paired_ftest_pvalue, ')')
        if paired_ttest.pvalue < 0.05:
            if paired_ftest_pvalue < 0.05:
                q1_state1, q3_state1 = np.percentile(state1_values, [25, 75])
                q1_state2, q3_state2 = np.percentile(state2_values, [25, 75])
                iqr_state1, iqr_state2 = q3_state1 - q1_state1, q3_state2 - q1_state2

                upper_whisker_state1 = q3_state1 + 1.5 * iqr_state1
                upper_whisker_state2 = q3_state2 + 1.5 * iqr_state2
                plt.text(i, max(upper_whisker_state1, upper_whisker_state2), '*', ha='center', fontsize=12)
        colors_extended = [color for color in colors[::3] for _ in range(2)]
        for patch, color in zip(plt.gca().artists, colors_extended):
            patch.set_color(color)
        alpha_cycle = itertools.cycle(alphas)
        for patch in plt.gca().artists:
            patch.set_alpha(next(alpha_cycle))

    plt.gca().set_xticklabels([f'{int(key) / 1000:.3f}' for key in data.keys()])

    plt.ylabel('Firefly period (s)')
    plt.xlabel('LED period (s)')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.legend([], [], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(plot_params.save_folder + '/boxplots.png')


def plot_recurrence_bouts(d, bgl):
    bests = [
        "20210521_770_4<br>297.549-307.62",
        "20210522_700_8<br>185.626-197.699",
        "20220524_500_83<br>99.6-104.567",
        "20210523_850_12<br>255.311-266.1",
        "20210526_600_40<br>415.008-417.859",
        "20210521_500_47<br>170.702-177.289",
        "20210522_700_10<br>133.984-155.078",
        "20210525_500_38<br>279.406-292.096",
        "20220529_400_100<br>158.467-162.8",
        "20210525_600_32<br>325.68-334.234",
    ]
    worsts = [
        "20210526_600_43<br>148.241-165.366",
        "20220518_770_68<br>376.233-392.7",
        "20220529_300_102<br>44.067-61.9",
        "20220521_500_81<br>53.233-101.033",
        "20230523_300_114<br>205.133-275.2",
        "20210529_1000_61<br>67.734-106.136",
        "20220526_1000_60<br>177.405-201.134",
        "20230523_400_107<br>425.633-474.9",
        "20220518_770_73<br>83.267-100.2",
        "20210527_850_44<br>308.721-324.579",
    ]
    metrics_dict = d
    records = []
    for key, bouts in metrics_dict.items():
        k = key.split('_')[1]
        for i, m in enumerate(bouts):
            det = m['det']
            lam = m['lam']
            rr = m['rr']
            if np.isnan(det) or np.isnan(lam) or det == 0 or lam == 0:
                continue

            if m['num_flashes'] > 3:
                records.append({
                    "key": k,
                    "bout": f"{round(m['start_idx'],3)}-{round(m['end_idx'],3)}",
                    "det": det,
                    "lam": lam,
                    "rr": rr,
                    "count": f"{m['num_flashes']}",
                    "hover_label": f"{key}<br>{round(m['start_idx'],3)}-{round(m['end_idx'],3)}"
                })

    df = pd.DataFrame(records)

    # Create scatter plot with hover labels
    colors = plt.cm.viridis_r(np.linspace(0, 1, 24))  # returns RGBA
    colors = [mcolors.to_hex(c) for c in colors]  # convert to hex strings
    colormap = {
        300: colors[2],
        400: colors[5],
        500: colors[8],
        600: colors[11],
        700: 'yellow',
        770: colors[17],
        850: colors[20],
        1000: colors[23],
        'best_10': 'black',
        'worst_10': 'grey'
    }
    fig = px.scatter(
        df,
        x="det",
        y="lam",
        color="key",
        hover_name="hover_label",
        labels={"det": "Determinism (DET)", "lam": "Laminarity (LAM)"},
        title="Determinism vs. Laminarity per Bout",
        color_discrete_map={str(k): v for k, v in colormap.items()},
        width=800,
        height=600
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(showlegend=False)
    fig.write_html('figs/analysis/bout_det_vs_lam_{}bgl.html'.format(bgl))
    print('bout vs lam written')


def compute_confidence_interval(data, confidence=0.68):
    if len(data) < 2:
        return np.nan, np.nan
    mean = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean - h, mean + h


def trimmed_data(data, proportiontocut=0.10):
    data_sorted = np.sort(data)
    n = len(data)
    trim = int(n * proportiontocut)
    return data_sorted[trim: n - trim] if n > 2 * trim else np.array([])


def compute_bin_summaries(phases, shifts, num_bins=10, trim_prop=0.025):
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    trimmed_means, ci_lower, ci_upper = [], [], []

    for i in range(num_bins):
        in_bin = (phases >= bin_edges[i]) & (phases < bin_edges[i + 1])
        if np.any(in_bin):
            subset = trimmed_data(shifts[in_bin], proportiontocut=trim_prop)
            if len(subset) > 1:
                mean = np.mean(subset)
                lo, hi = compute_confidence_interval(subset)
            else:
                mean, lo, hi = np.nan, np.nan, np.nan
        else:
            mean, lo, hi = np.nan, np.nan, np.nan
        trimmed_means.append(mean)
        ci_lower.append(lo)
        ci_upper.append(hi)

    return bin_centers, trimmed_means, ci_lower, ci_upper


def bootstrap_ci(data, num_bootstrap=1000, confidence=0.68):
    if len(data) < 2:
        return np.nan, np.nan
    bootstraps = np.random.choice(data, size=(num_bootstrap, len(data)), replace=True)
    medians = np.median(bootstraps, axis=1)
    lower = np.percentile(medians, (1 - confidence) / 2 * 100)
    upper = np.percentile(medians, (1 + confidence) / 2 * 100)
    return lower, upper


def compute_bin_summaries_median(phases, shifts, num_bins=10, trim_prop=0.025):
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    medians, ci_lower, ci_upper = [], [], []

    for i in range(num_bins):
        in_bin = (phases >= bin_edges[i]) & (phases < bin_edges[i + 1])
        if np.any(in_bin):
            subset = trimmed_data(shifts[in_bin], proportiontocut=trim_prop)
            if len(subset) > 1:
                median = np.median(subset)
                lo, hi = bootstrap_ci(subset)
            else:
                median, lo, hi = np.nan, np.nan, np.nan
        else:
            median, lo, hi = np.nan, np.nan, np.nan
        medians.append(median)
        ci_lower.append(lo)
        ci_upper.append(hi)

    return bin_centers, medians, ci_lower, ci_upper


def simulate_null_model(median_response, led_period, num_bins=10, trim_prop=0.025, trials=50):
    null_curves = []
    for _ in range(trials):
        firefly_times = np.cumsum(np.random.exponential(scale=1 / median_response, size=1000))
        led_times = np.arange(0, firefly_times[-1], led_period)
        phases, shifts = [], []

        for t_led in led_times:
            prev = firefly_times[firefly_times < t_led]
            next_ = firefly_times[firefly_times > t_led]
            if len(prev) == 0 or len(next_) == 0:
                continue
            t_prev, t_next = prev[-1], next_[0]
            cycle = t_next - t_prev
            if cycle == 0: continue
            phase = (t_led - t_prev) / cycle
            shift = (next_[0] - (t_prev + cycle)) / cycle + 1
            if 0.5 < shift < 2.0:
                phases.append(phase)
                shifts.append(shift)

        phases = np.array(phases)
        shifts = np.array(shifts)
        bin_centers, means, *_ = compute_bin_summaries(phases, shifts, num_bins=num_bins, trim_prop=trim_prop)
        null_curves.append(means)

    return bin_centers, np.nanmean(null_curves, axis=0)


def plot_prc(bin_centers, means, ci_lo, ci_hi, raw_phases, raw_shifts, null_curve=None,
             title="", color="purple", show=True, filename=None, scatter=False):
    plt.figure(figsize=(6, 4))
    err_low = np.array(means) - np.array(ci_lo)
    err_high = np.array(ci_hi) - np.array(means)

    plt.errorbar(bin_centers, means, yerr=[err_low, err_high],
                 fmt='o-', capsize=3, color=color, label='Mean ± CI')

    for i, center in enumerate(bin_centers):
        in_bin = (raw_phases >= center - 0.05) & (raw_phases < center + 0.05)
        if np.any(in_bin):
            jittered_x = center + np.random.normal(0, 0.005, np.sum(in_bin))
            if scatter:
                plt.scatter(jittered_x, raw_shifts[in_bin], color='black', alpha=0.2, s=6)

    if null_curve is not None:
        plt.plot(bin_centers, null_curve, linestyle='--', color='gray', label='Null model')

    plt.title(title)
    plt.xlabel("Normalized Phase (0–1)")
    plt.ylabel("Phase Shift")
    plt.grid(True)
    ymin = np.nanmin([*ci_lo] + ([] if null_curve is None else list(null_curve))) - 0.05
    ymax = np.nanmax([*ci_hi] + ([] if null_curve is None else list(null_curve))) + 0.05
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    elif show:
        plt.show()


def process_treatment_data(results, flter):
    grouped = defaultdict(lambda: {"phases": [], "shifts": [], "responses": []})

    threshold = None
    if flter == 'DET+LAM*BOUT_LENGTH':
        all_scores = [
            res['det'] + res['lam'] * res['num_flashes']
            for k in results
            for res in results[k]
        ]
        threshold = np.percentile(all_scores, 90)

    for k, reslist in results.items():
        for res in reslist:
            key = k.split('_')[1]
            max_phase = float(key) / 1000
            if res['num_flashes'] < 5:
                continue
            if flter == 'DET_LAM' and not (res["lam"] > LAM_FILTER_LEVEL or res["det"] > DET_FILTER_LEVEL):
                continue
            if flter == 'DET' and not (res["det"] > DET_FILTER_LEVEL):
                continue
            if flter == 'LAM' and not (res["lam"] > LAM_FILTER_LEVEL):
                continue
            if flter == 'PERIOD_DIFF' and abs(res['prior_period'] - max_phase) > res['prior_period'] / 2:
                continue
            if flter == 'DET+LAM*BOUT_LENGTH':
                score = res['det'] + res['lam'] * res['num_flashes']
                if score < threshold:
                    continue

            grouped[key]["phases"].extend(res["phases"])
            grouped[key]["shifts"].extend(res["shifts"])
            grouped[key]["responses"].extend(res["responses"])

    return grouped


def plot_individual_treatment_prcs(results, flter=False, bout_gap_length=2.0):
    grouped = process_treatment_data(results, flter)
    for key, data in grouped.items():
        try:
            phase_max = float(key) / 1000
        except ValueError:
            continue
        shifts = np.array(data["shifts"])
        phases = np.array(data["phases"]) / phase_max
        responses = np.array(data["responses"])
        valid = (~np.isnan(shifts)) & (shifts > 0.5) & (shifts < 2.0)
        if np.sum(valid) < 10:
            continue

        shifts, phases = shifts[valid], phases[valid]
        bin_centers, means, ci_lo, ci_hi = compute_bin_summaries(phases, shifts)
        _, null_curve = simulate_null_model(np.median(responses), 1.0 / (1000 / float(key)))

        suffix = "_filtered_by_{}".format(flter) if flter else ""
        plot_prc(
            bin_centers, means, ci_lo, ci_hi,
            raw_phases=phases, raw_shifts=shifts, null_curve=null_curve,
            title=f"Normalized PRC — Key {key}",
            filename=f"figs/analysis/{bout_gap_length}bgl_prc_{key}{suffix}.png",
            scatter=True
        )


def plot_individual_treatment_responses(results, flter=False, bout_gap_length=2.0):
    grouped = process_treatment_data(results, flter)
    for key, data in grouped.items():
        try:
            phase_max = float(key) / 1000
        except ValueError:
            continue
        shifts = np.array(data["shifts"])
        phases = np.array(data["phases"]) / phase_max
        responses = np.array(data["responses"])
        valid = (~np.isnan(shifts)) & (shifts > 0.5) & (shifts < 2.0)
        if np.sum(valid) < 10: continue

        responses, phases = responses[valid], phases[valid]
        bin_centers, means, ci_lo, ci_hi = compute_bin_summaries_median(phases, responses)

        suffix = "_filtered_by_{}".format(flter) if flter else ""
        plot_prc(
            bin_centers, means, ci_lo, ci_hi,
            raw_phases=phases, raw_shifts=responses, null_curve=None,
            title=f"Normalized PRC — Key {key}",
            filename=f"figs/analysis/{bout_gap_length}bgl_prc_response_{key}{suffix}.png",
            scatter=True
        )


def plot_grand_prc(results, flter=False, bout_gap_length=2.0):
    all_phases, all_shifts = [], []
    threshold = None
    if flter == 'DET+LAM*BOUT_LENGTH':
        all_scores = [
            res['det'] + res['lam'] * res['num_flashes']
            for reslist in results.values()
            for res in reslist
            if 'det' in res and 'lam' in res and 'num_flashes' in res
        ]
        threshold = np.percentile(all_scores, 90)

    for k, reslist in results.items():
        for res in reslist:
            key = k.split('_')[1]
            max_phase = float(key) / 1000

            if res['num_flashes'] < 5:
                continue
            if flter == 'DET_LAM' and not (res["lam"] > LAM_FILTER_LEVEL or res["det"] > DET_FILTER_LEVEL):
                continue
            if flter == 'DET' and not (res["det"] > DET_FILTER_LEVEL):
                continue
            if flter == 'LAM' and not (res["lam"] > LAM_FILTER_LEVEL):
                continue
            if flter == 'PERIOD_DIFF' and abs(res['prior_period'] - max_phase) > res['prior_period'] / 2:
                continue
            if flter == 'DET+LAM*BOUT_LENGTH':
                score = res['det'] + res['lam'] * res['num_flashes']
                if score < threshold:
                    continue

            all_phases.extend([p / max_phase for p in res["phases"]])
            all_shifts.extend(res["shifts"])

    shifts = np.array(all_shifts)
    phases = np.array(all_phases)
    valid = (~np.isnan(shifts)) & (shifts > 0.5) & (shifts < 2.0)
    if np.sum(valid) < 10: return

    shifts, phases = shifts[valid], phases[valid]
    bin_centers, means, ci_lo, ci_hi = compute_bin_summaries(phases, shifts)
    plot_prc(
        bin_centers, means, ci_lo, ci_hi,
        raw_phases=phases, raw_shifts=shifts, null_curve=None,
        title="Normalized Grand PRC" + (" (Filtered)" if flter else ""),
        filename=f"figs/analysis/{bout_gap_length}bgl_prc_grand{'_filtered' if flter else ''}.png",
        scatter=False
    )


def plot_combined_prc(results, flter=False, bout_gap_length=2.0):
    grouped = process_treatment_data(results, flter)
    plt.figure(figsize=(10, 8))
    treatment_keys = sorted(grouped.keys(), key=lambda x: int(x))
    colors = plt.get_cmap('viridis_r')(np.linspace(0, 1, len(treatment_keys)))
    all_lines = []

    for i, (color, key) in enumerate(zip(colors, treatment_keys)):
        data = grouped[key]
        try:
            phase_max = float(key) / 1000
        except ValueError:
            continue
        shifts = np.array(data["shifts"])
        phases = np.array(data["phases"]) / phase_max
        valid = (~np.isnan(shifts)) & (shifts > 0.5) & (shifts < 2.0)
        if np.sum(valid) < 10: continue
        shifts, phases = shifts[valid], phases[valid]
        bin_centers, means, *_ = compute_bin_summaries(phases, shifts)
        plt.plot(bin_centers, means, label=f"Key {key}", color=color, linewidth=2)
        all_lines.append((bin_centers, means))

    plot_grand = True
    if plot_grand:
        all_phases, all_shifts = [], []
        threshold = None
        if flter == 'DET+LAM*BOUT_LENGTH':
            all_scores = [
                res['det'] + res['lam'] * res['num_flashes']
                for reslist in results.values()
                for res in reslist
                if 'det' in res and 'lam' in res and 'num_flashes' in res
            ]
            threshold = np.percentile(all_scores, 90)

        for k, reslist in results.items():
            for res in reslist:
                key = k.split('_')[1]
                max_phase = float(key) / 1000

                if res['num_flashes'] < 5:
                    continue
                if flter == 'DET_LAM' and not (res["lam"] > LAM_FILTER_LEVEL or res["det"] > DET_FILTER_LEVEL):
                    continue
                if flter == 'DET' and not (res["det"] > DET_FILTER_LEVEL):
                    continue
                if flter == 'LAM' and not (res["lam"] > LAM_FILTER_LEVEL):
                    continue
                if flter == 'PERIOD_DIFF' and abs(res['prior_period'] - max_phase) > res['prior_period'] / 2:
                    continue
                if flter == 'DET+LAM*BOUT_LENGTH':
                    score = res['det'] + res['lam'] * res['num_flashes']
                    if score < threshold:
                        continue

                all_phases.extend([p / max_phase for p in res["phases"]])
                all_shifts.extend(res["shifts"])
        shifts = np.array(all_shifts)
        phases = np.array(all_phases)
        valid = (~np.isnan(shifts)) & (shifts > 0.5) & (shifts < 2.0)
        shifts, phases = shifts[valid], phases[valid]
        bin_centers, means, *_ = compute_bin_summaries(phases, shifts)
        plt.plot(bin_centers, means, color='black', linewidth=4, label="Grand Mean")

    plt.title("Combined PRC: Grand + All Treatments")
    plt.xlabel("Normalized Phase (0–1)")
    plt.ylabel("Phase Shift")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    suffix = "_filtered_by_{}".format(flter) if flter else ""
    plt.savefig(f"figs/analysis/{bout_gap_length}_bgl_prc_combined{suffix}.png")
    plt.close()


def plot_prc_by_key(results, bgl):
    for flter in [False, 'DET_LAM', 'PERIOD_DIFF', 'DET', 'LAM', 'DET+LAM*BOUT_LENGTH']:
        plot_individual_treatment_prcs(results, flter, bgl)
        plot_individual_treatment_responses(results, flter, bgl)
        plot_grand_prc(results, flter, bgl)
        plot_combined_prc(results, flter, bgl)


def plot_recurrence_stats(d):
    metrics_dict_s = d
    fig, ax = plt.subplots()
    colors = plt.cm.viridis_r(np.linspace(0, 1, 24))

    colormap = {
        300: colors[2],
        400: colors[5],
        500: colors[8],
        600: colors[11],
        700: 'yellow',
        770: colors[17],
        850: colors[20],
        1000: colors[23],
    }
    bests = ['20210527_500_47', '20210525_600_34',
             '20210528_850_53', '20210521_770_4', '20210522_700_10',
             '20210526_600_40', '20210521_770_5',
             '20220524_500_83', '20210522_700_10', '20210522_700_8']
    worsts = ['20220526_600_86', '20210526_600_43',
             '20210525_700_37', '20220518_770_73', '20210527_850_44',
             '20220526_1000_88', '20220529_400_99', '20220529_300_102',
             '20230523_400_109', '20220518_600_69']
    seen = {
        300: False,
        400: False,
        500: False,
        600: False,
        700: False,
        770: False,
        850: False,
        1000: False,
        'best_10': False,
        'worst_10': False
    }
    for k in metrics_dict_s.keys():
        key = int(k.split('_')[1])
        if k in bests:
            color = 'black'
            size = 9
            key = 'best_10'
        elif k in worsts:
            color = 'grey'
            size = 3
            key = 'worst_10'
        else:
            color = colormap[key]
            size = 6
        if seen[key]:
            ax.scatter(metrics_dict_s[k][1], metrics_dict_s[k][2], color=color, s=size)
        else:
            ax.scatter(metrics_dict_s[k][1], metrics_dict_s[k][2],
                       color=color, s=size, label=key)
            seen[key] = True
    ax.set_xlabel('Determinism of recurrence matrix')
    ax.set_ylabel('Laminarity of recurrence matrix')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('figs/analysis/recurrence_stats.png')
    plt.close()


def plot_recurrence(r, ax, method, t, e, metrics):
    """
    Visualize the recurrence plot.

    Parameters:
    -----------
    - r: Binary matrix representing recurrence points
    - ax: axes on which to plot
    - method: method used
    - t: tau for the embedding
    - e: embedding dimension
    - metrics: recurrence rate, determinism, and laminarity of the matrix
    """
    if metrics is not None:
        rr = round(metrics[0], 3)
        det = round(metrics[1], 3)
        lam = round(metrics[2], 3)
        rpde = round(metrics[3], 3)
    else:
        rr = 'n/a'
        det = 'n/a'
        lam = 'n/a'
        rpde = 'n/a'
    if t is not None:
        title_string = 'Recurrence Plot from {}: Tau = {}, embedding dim = {}\n{}rr, {}det, {}lam, {}rpde'.format(
            method, t, e, rr, det, lam, rpde)
    else:
        title_string = 'Recurrence Plot from {}, {}rr, {}det, {}lam, {}rpde'.format(method, rr, det, lam, rpde)
    cax = ax.imshow(r, cmap='binary', origin='lower')
    ax.set_title(title_string)
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Time Index')
    plt.colorbar(cax, label='Recurrence', ax=ax)


def plot_cross_correlation(lags, correlation, ax):
    """
    Plot the cross-correlation results.

    Parameters:
    - lags: Array of lag times
    - correlation: Cross-correlation values
    - ax: axes on which to plot
    """
    ax.plot(lags, correlation)
    ax.set_title('Cross-Correlation between LED and FF Event Times')
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Normalized Cross-Correlation')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.grid(True)


def plot_dfa_results(scales, fluctuations, alpha, confidence_interval, hurst_exp, ax):
    """
    Plot the results of DFA analysis.

    Parameters:
    - scales: the scales at which fluctuations were calculated.
    - fluctuations: the fluctuation values at each scale.
    - alpha: scaling exponent (slope of log-log plot).
    - confidence_interval: (lower_bound, upper_bound) of the 95% confidence interval for alpha.
    - title: Title for the plot.

    Returns:
    - fig, ax: figure and axis objects for the plot.
    """
    # Plot fluctuations versus scale
    ax.loglog(scales, fluctuations, 'o', markersize=8, label='Fluctuations')

    # Plot the fitted line
    log_scales = np.log10(scales)
    log_fluct = np.log10(fluctuations)

    # Perform linear regression for the plot
    X = add_constant(log_scales)
    model = OLS(log_fluct, X).fit()

    # Generate predicted values
    x_line = np.linspace(min(log_scales), max(log_scales), 100)
    y_line = model.params[0] + model.params[1] * x_line
    # Plot the regression line
    ax.loglog(10 ** x_line, 10 ** y_line, 'r-',
              label=f'Fit: α = {alpha:.3f} (95% CI: {confidence_interval[0]:.3f}-{confidence_interval[1]:.3f})')

    ax.set_xlabel('Scale (s)', fontsize=12)
    ax.set_ylabel('Fluctuation F(s)', fontsize=12)
    ax.set_title(f"DFA analysis: Alpha={alpha:.4f},Hurst={hurst_exp:.4f}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.2)


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


def before_vs_mode_freq(rmses, plot_params):
    fig, ax = plt.subplots()
    colormap = cm.get_cmap('viridis_r', 24)
    allxs = []
    allys = []
    for i, k in enumerate(rmses['mode_before'].keys()):
        diffs = []
        befores = []
        for j, x in enumerate(rmses['mode_before'][k]):
            y = rmses['mode_after'][k][j]
            if not np.isnan(x):
                if not np.isnan(y):
                    befores.append(x)
                    diffs.append(y - x)
                    allxs.extend(befores)
                    allys.extend(diffs)
        print(befores, diffs)
        ax.scatter(befores, diffs, color=colormap.__call__(i * 3), label=k)

    ax.set_xlabel('Mode freq before[s]')
    ax.set_ylabel('Diff in mode freq (after-before)[s]')
    m, b = np.polyfit(allxs, allys, 1)
    ax.plot(allxs, m * np.array(allxs) + b, linestyle='dotted', color='black',
            label=r'$y={}x+{}$'.format(round(m, 3), round(b, 3)))
    plt.legend()
    plt.savefig(plot_params.save_folder + '/before_vs_mode_20240607.png')
    plt.close()


def barbell_modes(rmses, plot_params):
    fig, ax = plt.subplots()
    colormap = cm.get_cmap('viridis_r', 24)
    for i, k in enumerate(rmses['mode_before'].keys()):
        befores = []
        afters = []
        for j, x in enumerate(rmses['mode_before'][k]):
            y = rmses['mode_after'][k][j]
            if not np.isnan(x):
                if not np.isnan(y):
                    befores.append(x)
                    afters.append(y)
        if len(befores) == len(afters):
            for q, individual in enumerate(zip(befores, afters)):
                xs = [q, q + 0.5]
                ys = [individual[0], individual[1]]
                if q == 0:
                    ax.scatter(xs, ys, color=colormap.__call__(i * 3), label=k)
                    ax.plot(xs, ys, color=colormap.__call__(i * 3))
                else:
                    ax.scatter(xs, ys, color=colormap.__call__(i * 3))
                    ax.plot(xs, ys, color=colormap.__call__(i * 3))
    ax.set_xlabel('Individual')
    ax.set_ylabel('Most frequent ff frequency (before->after)[s]')
    plt.legend()
    plt.savefig(plot_params.save_folder + '/barbell_modes.png')
    plt.close()


def plot_poincare(phases, title):
    """
    Create a Poincaré plot of phase differences.

    Parameters:
    - phases  list of consecutive phase differences
    - title: figtitle
    """
    # Convert to numpy array if not already
    phase_diff = np.array(phases)

    # Create Poincaré plot
    plt.figure(figsize=(6, 6))
    plt.scatter(phase_diff[:-1], phase_diff[1:], alpha=0.5, s=10, color="black")
    plt.xlabel(r'$\Delta \phi_i$')
    plt.ylabel(r'$\Delta \phi_{i+1}$')
    plt.title("Poincaré Plot of Phase Differences")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(title)
    plt.close()


def plot_autocorrelation(phases, title):
    """
    Create an autocorrelation plot of phase differences.

    Parameters:
    - phases  list of consecutive phase differences
    - title: figtitle
    """
    # Convert to numpy array if not already
    phase_diff = np.array(phases)
    autocorrelation = helpers.circular_autocorrelation(phase_diff)
    lags = np.arange(1, len(autocorrelation) + 1)

    plt.figure(figsize=(6, 6))
    plt.plot(lags, autocorrelation, marker='o')
    plt.xlabel("Lag")
    plt.ylabel("Circular Autocorrelation Magnitude")
    plt.title("Circular Autocorrelation of Phase Differences")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(title)
    plt.close()


def plot_alpha_vs_dist_period(dist_ps, alphas, keys):
    fig, ax = plt.subplots()
    xs = dist_ps
    colormap = cm.get_cmap('viridis_r', len(keys))
    colors = [colormap(i) for i in range(len(keys))]

    ax.scatter(xs, alphas, color=colors, s=5, )
    ax.set_xlabel('T_{ff} - T_{LED}')
    ax.set_ylabel('DFA Alpha')
    ax.set_xlim([-1, 1])
    ax.set_ylim([0.1, 1.5])
    plt.savefig('figs/analysis/Alpha_vs_period_diff.png')
    plt.close()


def format_data(pargs):
    fpaths = os.listdir(pargs.data_path)

    for fp in fpaths:
        if fp == '.DS_Store':
            continue
        elif not os.path.isdir(pargs.data_path + '/' + fp):
            path, framerate = check_frame_rate(pargs.data_path + '/' + fp)
            if path is None:
                continue

            date = path.split('_')[1].split('/')[1]
            key = path.split('_')[2]
            index = path.split('_')[3].split('.')[0]

            if pargs.log:
                print(f'Writing timeseries from {date} with led freq {key}')
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

                ff_all_times = {round(float(x), 6) for x in ff_xs}
                led_all_times = {round(float(x), 6) for x in led_xs}
                ff_flash_times = {round(float(x), 6) for x in ff_xs_flashes}
                led_flash_times = {round(float(x), 6) for x in led_xs_flashes}

                all_times = sorted(ff_all_times.union(led_all_times))

                save_path = f"data_paths/formatted/f{date}_{key}_{index}.csv"
                with open(save_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['timestamp', 'FF', 'LED'])
                    for t in all_times:
                        ff = 1 if t in ff_flash_times else 0
                        led = 1 if t in led_flash_times else 0
                        writer.writerow([t, ff, led])


def write_timeseries_figs(pargs):
    #####
    # Write the timeseries figures path objects
    # Interactive timeseries plots for any given day - the raw data
    #####
    keys = ['300', '400', '500', '600', '700', '770', '850', '1000']
    p_hases = {k: [] for k in keys}
    s_hifts = {k: [] for k in keys}
    fpaths = os.listdir(pargs.data_path)
    #format_data(pargs)
    all_fits = {key: [] for key in keys}
    all_derivs = {key: [] for key in keys}
    all_phases = {key: [] for key in keys}

    for fp in fpaths:
        if fp == '.DS_Store':
            continue
        path, framerate = check_frame_rate(pargs.data_path + '/' + fp)
        if path is None:
            continue

        date = path.split('_')[1].split('/')[1]
        key = path.split('_')[2]
        index = path.split('_')[3].split('.')[0]

        if pargs.log:
            print(f'Writing timeseries from {date} with led freq {key}')
        with open(path, 'r') as data_file:
            ts = {'ff': [], 'led': []}
            data = list(csv.reader(data_file))
            temp = helpers.get_temp_from_experiment_date(date, index)
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
            masked_ff = ff_ys > 0.01
            masked_led = led_ys > 0.50

            if pargs.with_stats:
                _, _, pairs, period = helpers.compute_phase_response_curve(
                    time_series_led=led_xs_flashes,
                    time_series_ff=ff_xs_flashes,
                    epsilon=0.08,
                    m=float(key)/1000,
                    do_responses_relative_to_ff=pargs.do_ffrt,
                    only_lookback=pargs.re_norm
                )
                if period is None:
                    ff_period = 'N/A'
                    print('ff period is N/A prior to LED in {}-{}-{} exp'.format(date, key, index))
                else:
                    ff_period = period['period_estimate']
                    ff_period = f"{round(ff_period * 1000)}"

                times, phases, responses, shifts = zip(*[(t, p, r, s) for p, r, s, t in pairs])
                if pargs.re_norm:
                    phases = helpers.renorm_phases(phases)

                phase_derivative = helpers.sliding_time_window_derivative(times, phases,
                                                                          float(key)/1000,
                                                                          window_seconds=3.0)

                valid_indices = [i for i, val in enumerate(phase_derivative) if val is not None]
                valid_times = [times[i] for i in valid_indices]
                valid_derivatives = [phase_derivative[i] for i in valid_indices]
                all_derivs[key].append(valid_derivatives)

                phase_acceleration = helpers.sliding_time_window_derivative(valid_times, valid_derivatives,
                                                                            float(key)/1000,
                                                                            window_seconds=3.0)
                valid_indices_2nd = [i for i, val in enumerate(phase_acceleration) if val is not None]
                valid_times_2nd = [valid_times[i] for i in valid_indices_2nd]
                valid_accelerations = [phase_acceleration[i] for i in valid_indices_2nd]

                date_str = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
                title_text = f"<br>LED+FF Experiment<br>Date: {date_str}<br>LED Period: {key}ms<br>Firefly Period prior to introduction: {ff_period}ms<br>Temp: {temp}°C<br>Frame Rate: {round(1 / framerate, 3)} fps<br><br><br><br>"
                x_min = min(min(led_xs), min(ff_xs))
                x_max = max(max(led_xs), max(ff_xs))

                led_y_value = 0.505
                ff_y_value = 0.495

                trace1_led = go.Scatter(
                    x=led_xs[masked_led],
                    y=[led_y_value] * len(led_xs[masked_led]),
                    mode='markers',
                    name='LED',
                    marker=dict(color='black', size=5)
                )

                trace1_ff = go.Scatter(
                    x=ff_xs[masked_ff],
                    y=[ff_y_value] * len(ff_xs[masked_ff]),
                    mode='markers',
                    name='Firefly',
                    marker=dict(color='orange', size=5)
                )

                trace2 = go.Scatter(
                    x=times,
                    y=phases,
                    mode='markers',
                    name='Phase Differences',
                    line=dict(color='black', width=1, dash='solid')
                )
                all_phases[key].append(phases)

                trace2_baseline = go.Scatter(x=[x_min, x_max], y=[0, 0], mode='lines',
                                             line=dict(color='blue', dash='dash', width=1), showlegend=False)

                if period.get('periods'):
                    p_xs = period['periods'][0]
                    p_ys = period['periods'][1]
                    xtimes = [*p_xs, *times]
                    yresponses = [*p_ys, *responses]
                else:
                    xtimes = times
                    yresponses = responses
                trace3 = go.Scatter(
                    x=xtimes,
                    y=yresponses,
                    mode='markers',
                    name='Firefly Period',
                    marker=dict(color='red', size=5),
                )
                trace3_baseline = go.Scatter(x=[min(xtimes), x_max], y=[float(key) / 1000, float(key) / 1000], mode='lines',
                                             line=dict(color='blue', dash='dash', width=1), showlegend=False)
                trace_derivative = go.Scatter(
                    x=valid_times,
                    y=valid_derivatives,
                    mode='markers',
                    name='Phase Derivative over 3s window',
                    marker=dict(color='purple', size=5)
                )
                trace_acceleration = go.Scatter(
                    x=valid_times_2nd,
                    y=valid_accelerations,
                    mode='lines',
                    name='Phase Second Derivative ([1/s^2])',
                    line=dict(color='blue', width=2, dash='dash')  # Dashed line for distinction
                )

                fig = make_subplots(rows=4, cols=1,
                                    shared_xaxes=True,
                                    vertical_spacing=0.08,  # Reduced spacing between subplots
                                    subplot_titles=("", "Phase Differences", "Firefly Period", "Phase Derivative"))

                fig.add_trace(trace1_led, row=1, col=1)
                fig.add_trace(trace1_ff, row=1, col=1)
                fig.add_trace(trace2, row=2, col=1)
                fig.add_trace(trace2_baseline, row=2, col=1)
                fig.add_trace(trace3, row=3, col=1)
                fig.add_trace(trace3_baseline, row=3, col=1)
                fig.add_trace(trace_derivative, row=4, col=1)
                fig.add_trace(trace_acceleration, row=4, col=1)

                fig.update_layout(
                    height=1200,
                    plot_bgcolor='white',
                    width=800,
                    showlegend=False,
                    xaxis=dict(range=[x_min, x_max]),
                    xaxis2=dict(range=[x_min, x_max]),
                    xaxis3=dict(range=[min(xtimes), x_max]),
                    xaxis4=dict(range=[x_min, x_max]),
                    yaxis=dict(
                        title="Signal presence",
                        tickmode='array',
                        tickvals=[ff_y_value, led_y_value],
                        ticktext=['Firefly', 'LED'],
                        range=[0.425, 0.575]
                    ),
                    yaxis2=dict(title="Phase difference [s]"),
                    yaxis3=dict(title="Firefly period [s]"),
                    yaxis4=dict(title="Phase derivative [1/s]"),
                    margin=dict(t=180, l=80, r=80, b=80),
                    title={
                        'text': title_text,
                        'y': 0.98,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    }
                )

                fig.update_xaxes(title_text="Time (s)", row=1, col=1, showticklabels=True,
                                 gridcolor='lightgrey'
                                 )
                fig.update_xaxes(title_text="Time (s)", row=2, col=1, showticklabels=True,
                                 gridcolor='lightgrey')
                fig.update_xaxes(title_text="Time (s)", row=3, col=1, showticklabels=True,
                                 gridcolor='lightgrey'
                                 )
                fig.update_xaxes(title_text="Time (s)", row=4, col=1, showticklabels=True,
                                 gridcolor='lightgrey')

                config = {
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'editable': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{date}_{key}_{index}',
                        'height': 1200,
                        'width': 800,
                        'scale': 2
                    }
                }

                fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)

                # PRC
                from collections import defaultdict
                shift_sums = defaultdict(lambda: [0, 0])  # [sum of shifts, count]

                # Aggregate shifts per phase
                for p, s in zip(phases, shifts):
                    p = round(p, 6)
                    if helpers.MIN_CUTOFF_PERIOD <= s <= helpers.MAX_CUTOFF_PERIOD:
                        shift_sums[p][0] += s
                        shift_sums[p][1] += 1

                # Compute averages
                to_plot_phase_avgs = []
                to_plot_shift_avgs = []

                for p, (shift_sum, count) in shift_sums.items():
                    to_plot_phase_avgs.append(p)
                    to_plot_shift_avgs.append(shift_sum / count)

                p_hases[key].extend(to_plot_phase_avgs)
                s_hifts[key].extend(to_plot_shift_avgs)

                to_plot_phases = []
                to_plot_shifts = []
                for p, s in zip(phases, shifts):
                    if helpers.MIN_CUTOFF_PERIOD <= s <= helpers.MAX_CUTOFF_PERIOD:
                        to_plot_phases.append(p)
                        to_plot_shifts.append(s)
                if len(to_plot_shifts) > 3:
                    phase_response_trace = go.Scatter(
                        x=to_plot_phases,
                        y=to_plot_shifts,
                        mode='markers',
                        name='Phase response curve',
                        marker=dict(color='black', size=5)

                    )
                    phase_response_trace_2 = go.Scatter(
                        x=to_plot_phase_avgs,
                        y=to_plot_shift_avgs,
                        mode='markers',
                        name='Avg Phase response curve',
                        marker=dict(color='gray', size=7)

                    )
                    phase_response_baseline = go.Scatter(x=[0.0, float(key)/1000],
                                                         y=[1.0, 1.0], mode='lines',
                                                         line=dict(color='blue', dash='dash', width=1),
                                                         showlegend=False)
                    fig2 = make_subplots(rows=1, cols=1,
                                         vertical_spacing=0.08,
                                         subplot_titles=["Phase response curve"])
                    fig2.add_trace(phase_response_trace, row=1, col=1)
                    fig2.add_trace(phase_response_baseline, row=1, col=1)
                    fig2.add_trace(phase_response_trace_2, row=1, col=1)

                    fig2.update_layout(
                        height=400,
                        width=800,
                        showlegend=True,
                        xaxis=dict(range=[0.0, float(key)/1000]),

                        yaxis=dict(
                            title="Percent of period response",
                            range=[0.0, 2.75]
                        ),
                        margin=dict(t=180, l=80, r=80, b=80),
                    )
                    fig2.update_xaxes(title_text="Phase difference (s)", row=1, col=1, showticklabels=True)

                    config2 = {
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'editable': False,
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'{date}_{key}_{index}',
                            'height': 400,
                            'width': 800,
                            'scale': 2
                        }
                    }
                    fig2_html = pio.to_html(fig2, full_html=False, include_plotlyjs=False, config=config2)

                    html_content = f"""
                                    <html>
                                    <head>
                                        <title>Timeseries Analysis</title>
                                        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                                        <style>
                                            body {{ 
                                                font-family: Arial, sans-serif; 
                                                text-align: center;
                                                display: flex;
                                                justify-content: center;
                                                align-items: center;
                                                flex-direction: column;
                                                margin: 0;
                                                padding: 0;
                                                min-height: 100vh;
                                            }}
                                            h1 {{ color: #333; }}
                                            .figure-container {{ 
                                                margin: 20px auto;
                                                width: 800px;
                                                height: 1200px;
                                                display: flex;
                                                justify-content: center;
                                            }}
                                            .figure-container-2 {{ 
                                                margin: 20px auto;
                                                width: 800px;
                                                height: 400px;
                                                display: flex;
                                                justify-content: center;
                                            }}
                                        </style>
                                    </head>
                                    <body>
                                        <div class="figure-container">
                                            {fig_html}
                                        </div>
                                        <div class="figure-container-2">
                                            {fig2_html}
                                        </div>
                                    </body>
                                    </html>
                                    """

                    # aggregate PRC
                    fit_y = helpers.fit_prc(to_plot_phase_avgs, to_plot_shift_avgs, date, key, index)
                    is_always_positive = True
                    for fy in fit_y:
                        if fy < 0:
                            is_always_positive = False
                        elif fy > 3.0:
                            is_always_positive = False

                    if is_always_positive:
                        all_fits[key].append(fit_y)

                else:
                    html_content = f"""
                                                        <html>
                                                        <head>
                                                            <title>Timeseries Analysis</title>
                                                            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                                                            <style>
                                                                body {{ 
                                                                    font-family: Arial, sans-serif; 
                                                                    text-align: center;
                                                                    display: flex;
                                                                    justify-content: center;
                                                                    align-items: center;
                                                                    flex-direction: column;
                                                                    margin: 0;
                                                                    padding: 0;
                                                                    min-height: 100vh;
                                                                }}
                                                                h1 {{ color: #333; }}
                                                                .figure-container {{ 
                                                                    margin: 20px auto;
                                                                    width: 800px;
                                                                    height: 1200px;
                                                                    display: flex;
                                                                    justify-content: center;
                                                                }}
                                                                .figure-container-2 {{ 
                                                                    margin: 20px auto;
                                                                    width: 800px;
                                                                    height: 400px;
                                                                    display: flex;
                                                                    justify-content: center;
                                                                }}
                                                            </style>
                                                        </head>
                                                        <body>
                                                            <div class="figure-container">
                                                                {fig_html}
                                                            </div>
                                                        </body>
                                                        </html>
                                                        """

                if pargs.do_ffrt:
                    if not pargs.re_norm:
                        output_path = f"{pargs.save_folder}/timeseries/LED_Period={key}ms/{date}_{key}_{index}_phase_ffrt.html"
                    else:
                        output_path = f"{pargs.save_folder}/timeseries/LED_Period={key}ms/{date}_{key}_{index}_phase_ffrt_0-1.html"
                else:
                    if not pargs.re_norm:
                        output_path = f"{pargs.save_folder}/timeseries/LED_Period={key}ms/{date}_{key}_{index}_phase_rt.html"
                    else:
                        output_path = f"{pargs.save_folder}/timeseries/LED_Period={key}ms/{date}_{key}_{index}_phase_rt_0-1.html"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(output_path, 'w') as f:
                    f.write(html_content)
                print(f'Written {date}_{key}_{index}')

            else:
                fig = make_subplots(
                    rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0
                )

                # Plot led
                fig.add_trace(
                    go.Scatter(x=led_xs[masked_led], y=led_ys[masked_led], mode='markers', name='LED',
                               marker=dict(color='black')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(x=led_xs[masked_led], y=led_ys[masked_led] - 0.5, name='LED',
                           marker=dict(color='black'), base=0.5),
                    row=1, col=1,
                )

                # Plot firefly
                fig.add_trace(
                    go.Scatter(x=ff_xs[masked_ff], y=ff_ys[masked_ff], name='FF', mode='markers',
                               marker=dict(color='orange')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(x=ff_xs[masked_ff], y=ff_ys[masked_ff] - 0.498, name='FF',
                           marker=dict(color='orange'), base=0.498),
                    row=1, col=1
                )

                fig.update_layout(
                    height=600,
                    showlegend=True,
                    xaxis_title="T [s]", yaxis_visible=False,
                    title={
                        'y': 0.95,
                        'x': 0.5,
                        'text': "LED+FF<br>Date: {}-{}-{}<br>LED Period: {}<br>Temp: {}°C<br>Frame Rate: {} fps".format(
                            date[0:4], date[4:6], date[6:8],
                            key, temp, round((1 / framerate), 3)),
                        'xanchor': 'center',
                        'yanchor': 'top',
                    },
                    margin={'t': 100},
                )

                fig.update_yaxes(range=[0.494, 0.506], row=1, col=1)
                fig.update_xaxes(matches='x')
                fig.update_yaxes(matches='y1')
                fig.write_html(pargs.save_folder + '/timeseries/LED_Period={}ms/{}_{}_{}__.html'.format(
                    key, date, key, index)
                )
    if pargs.latex:
        latex_rows = []
        for fp in fpaths:
            if fp == '.DS_Store':
                continue
            path, framerate = check_frame_rate(pargs.data_path + '/' + fp)
            if path is None:
                continue

            date = path.split('_')[1].split('/')[1]
            key = path.split('_')[2]
            index = path.split('_')[3].split('.')[0]
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

                if not led_xs_flashes or not ff_xs_flashes:
                    pre_led_duration = "N/A"
                    post_led_duration = "N/A"
                else:
                    pre_led_duration = round(led_xs_flashes[0] - ff_xs_flashes[0], 2)
                    post_led_duration = round(led_xs_flashes[-1] - led_xs_flashes[0], 2)

                try:
                    temp = helpers.get_temp_from_experiment_date(date, index)
                except Exception:
                    temp = "N/A"

                year = date[:4]
                month = date[4:6]
                day = date[6:8]

                row = [year, month, day, key, f"{temp}", f"{pre_led_duration}", f"{post_led_duration}"]
                latex_rows.append((int(key), row))

        # Sort by LED frequency
        latex_rows.sort()

        # Generate LaTeX
        print("\\begin{table}[H]")
        print("\\centering")
        print("\\scriptsize")
        print("\\caption{Experimental Metadata. All \\textbf{$T_{\\mathrm{LED}}$} values in milliseconds (ms).}")
        print("\\label{table:table1}")
        print("\\setlength{\\extrarowheight}{-0.8pt}")
        print("\\begin{tabular}{")
        print("|" + "|".join([">{\\centering\\arraybackslash}m{0.7cm}"] * 14) + "|}")
        print("\\hline")
        print("\\textbf{yyyy} & \\textbf{mm} & \\textbf{dd} & \\textbf{$T_{\\mathrm{LED}}$} & \\textbf{°C} & \\textbf{$t_{\\mathrm{pre}}$} & \\textbf{$t_{\\mathrm{post}}$} " * 2 + "\\\\")
        print("\\hline")

        # Output two rows per line
        for i in range(0, len(latex_rows), 2):
            left = latex_rows[i][1]
            right = latex_rows[i + 1][1] if i + 1 < len(latex_rows) else [""] * 7
            row = " & ".join(left + right) + " \\\\ \\hline"
            print(row)

        # Example frequency counts (edit or compute dynamically if needed)
        print("\\multicolumn{14}{|c|}{")
        print("\\textbf{N}: \\quad 300 (\\textbf{19}), 400 (\\textbf{13}), 500 (\\textbf{18}), 600 (\\textbf{17}),")
        print(
            "700 (\\textbf{18}), 770 (\\textbf{13}), 850 (\\textbf{15}), 1000 (\\textbf{14}), All (\\textbf{127})")
        print("} \\\\ \\hline")

        print("\\end{tabular}")
        print("\\end{table}")
    plot_aggregate_fitted_prc(all_fits)
    plot_aggregate_phase_derivatives(all_derivs, all_phases)


def plot_aggregate_phase_derivatives(allderivs, all_phases):
    ks = sorted(allderivs.keys(), reverse=True)
    num_bins = 71
    CHANGE_TOLERANCE = 1e-6
    all_flat_filtered_derivs = []
    filtered_derivs_by_group = {}
    for key in ks:
        derivs_of_sublists = allderivs[key]
        current_group_filtered_derivs = []
        for sublist in derivs_of_sublists:
            # Apply the filtering logic to the current sublist
            # Now iterate up to len(sublist) - 2 to allow checking i, i+1, and i+2
            for i in range(len(sublist) - 2):
                # Check if sublist[i] is stable with sublist[i+1]
                # AND if sublist[i+1] is stable with sublist[i+2]
                if (abs(sublist[i] - sublist[i + 1]) <= CHANGE_TOLERANCE and
                        abs(sublist[i + 1] - sublist[i + 2]) <= CHANGE_TOLERANCE):
                    current_group_filtered_derivs.append(sublist[i])

        filtered_derivs_by_group[key] = current_group_filtered_derivs
        all_flat_filtered_derivs.extend(current_group_filtered_derivs)

    bin_min, bin_max = -0.5, 0.5
    bins = np.linspace(bin_min, bin_max, num_bins + 1)
    ps_fig, ps_ax = plt.subplots(len(ks), figsize=(10, 8), sharex=True)
    colormap = cm.get_cmap('viridis_r', len(ks) * 3)

    max_height = 0
    # First pass: get max height with uniform bins (using filtered data)
    for key in ks:
        filtered_derivs = filtered_derivs_by_group[key]
        if filtered_derivs:
            counts, _ = np.histogram(filtered_derivs, bins=bins, density=True)
            max_height = max(max_height, counts.max())

    # Second pass: plot with uniform bins and y limit (using filtered data)
    for i, key in enumerate(ks):
        filtered_derivs = filtered_derivs_by_group[key]
        current_ax = ps_ax[i] if len(ks) > 1 else ps_ax
        color = colormap(i * 3)
        if filtered_derivs:
            current_ax.hist(filtered_derivs, bins=bins, alpha=0.6, color=color, edgecolor='black', linewidth=0.5,
                            density=True)

        current_ax.set_ylim(0, max_height * 1.1)
        current_ax.set_xlim(bin_min, bin_max)

        if i != len(ks) - 1:
            current_ax.xaxis.set_visible(False)
        current_ax.axvline(x=0, color='black', linestyle=':', linewidth=1)
        current_ax.grid(False)
        current_ax.tick_params(axis='both', which='both', length=0)
        current_ax.spines['top'].set_visible(False)
        current_ax.spines['right'].set_visible(False)

    last_ax = ps_ax[-1] if len(ks) > 1 else ps_ax
    last_ax.set_xlabel('Stable phase derivative', fontsize=12, labelpad=10)  # Changed label

    ps_fig.text(0.01, 0.5, 'pdf', va='center', rotation='vertical')

    plt.savefig('figs/analysis/filtered_stable_derivatives.png', dpi=300, bbox_inches='tight')
    plt.close()
    df_data = []
    for group_name in ks:
        derivs = filtered_derivs_by_group.get(group_name, [])
        for deriv_value in derivs:
            df_data.append({'Group': group_name, 'Derivative': deriv_value})

    df = pd.DataFrame(df_data)
    y_min, y_max = -0.5, 0.5
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = [colormap(i * 3) for i in range(len(ks))]
    sns.violinplot(
        x='Group',
        y='Derivative',
        data=df,
        ax=ax,
        inner='quartile',
        palette=palette,
        edgecolor='black',
        linewidth=1
    )
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel('Stable Phase Derivative', fontsize=14)
    ax.set_xlabel('Group', fontsize=14)
    ax.set_title('Distribution of Stable Phase Derivatives by Group', fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('figs/analysis/stable_derivs_violin_plot_seaborn.png', dpi=300)
    plt.close()


def plot_aggregate_fitted_prc(allfits):
    ps_fig, ps_ax = plt.subplots()
    colormap = {
        '300': 'darkred',
        '400': 'red',
        '500': 'orange',
        '600': 'yellow',
        '700': 'mediumseagreen',
        '770': 'royalblue',
        '850': 'turquoise',
        '1000': 'blueviolet'
    }
    mean_fits_collection = []
    fit_x = np.linspace(0.0, 1.0, 200)

    for key in allfits.keys():
        af = np.array(allfits[key])

        mean_fit = np.mean(af, axis=0)
        std_fit = np.std(af, axis=0)
        mean_fits_collection.append(mean_fit)

        ps_ax.plot(fit_x, mean_fit, linewidth=2.5, color=colormap[key], alpha=1.0,
                   label='Mean PRC: LED_Period={}ms'.format(key), zorder=3)

    grand_mean_fit = np.mean(mean_fits_collection, axis=0)
    ps_ax.plot(fit_x, grand_mean_fit, color='black', linewidth=2, linestyle='--',
               label='Grand Mean PRC', zorder=4)
    ps_ax.set_xlabel('Phase difference [s]', fontsize=14, labelpad=10)
    ps_ax.set_ylabel('Phase shift', fontsize=14, labelpad=10)
    ps_ax.set_title('Phase response curve for Photuris frontalis', fontsize=16, pad=20)
    ps_ax.set_xlim([0.0, 1.0])
    ps_ax.axhline(1.0, linestyle='dotted', linewidth=1.5, color='black', zorder=1)
    ps_ax.legend(fontsize=6, loc='upper left', frameon=False, title='PRC by LED Period',
                 bbox_to_anchor=(1.02, 1), borderaxespad=0, ncol=1)
    ps_ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.savefig('figs/analysis/all_prc.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_statistics(rmses, ks, plot_params):
    all_befores = {}
    all_afters = {}

    colormap = cm.get_cmap('viridis_r', len(ks) * 3)

    if plot_params.do_delay_plot:
        fig, axes = plt.subplots(len(ks))
        x = np.linspace(0.0, 1.0, 1000)

        for i, k in enumerate(ks):
            all_delays = []
            for individual in rmses['led_ff_diffs'][k]:
                delays = [i for i in individual]
                try:
                    all_delays.extend(delays)
                except TypeError:
                    all_delays.append(delays)

            mu = np.mean(all_delays)
            sigma = np.std(all_delays)
            gaussian = stats.norm.pdf(x, mu, sigma / 2)

            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            color = colormap.__call__(i * 3)

            axes[i].plot(x, gaussian, color=color)
            axes[i].hist(all_delays, density=True, bins=np.arange(0.0, 1.0, 0.033), color=color, alpha=0.75)

            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)

            axes[i].set_xlim(0.0, 1.0)
            axes[i].set_ylim(0.0, 4.0)

        axes[int(len(ks) / 2)].set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/delay_distributions_hist_normalized')
        plt.close(fig)

        colormap = cm.get_cmap('viridis_r', len(ks) * 3)
        fig, ax = plt.subplots(8)

        for i, k in enumerate(ks):
            color = colormap.__call__(i * 3)
            all_phases = []
            all_phase_shifts = []
            for ps_individual in rmses['phases'][k]:
                all_phase_shifts.append(helpers.improved_circular_normalize(ps_individual, (float(k) / 1000))[0].item())
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            if i != len(ks) - 1:
                ax[i].xaxis.set_visible(False)

            ax[i].hist(all_phase_shifts, color=color, density=True, bins=np.arange(-0.5, 0.5, 0.03))
            ax[i].set_xlim([-0.5, 0.5])
            ax[i].set_ylim([0, 3])
        ax[len(ks) - 1].set_xlabel(r"${\Delta \phi}$")
        plt.savefig(plot_params.save_folder + '/phase_response_histograms')
        plt.close(fig)
        from matplotlib.colors import LinearSegmentedColormap, to_rgb
        from scipy.stats import gaussian_kde

        colormap = cm.get_cmap('viridis_r', len(ks) * 3)
        fig, ax = plt.subplots(8)

        for i, k in enumerate(ks):
            color = colormap.__call__(i * 3)
            all_phases_ = []
            all_freqs_ = []
            for phase, freq,_ in rmses['phase_time_diffs'][k]:
                if 0 < freq <= 0.99:
                    all_phases_.append(helpers.improved_circular_normalize(phase, float(k)/1000)[0].item())
                    all_freqs_.append(freq)

            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            if i != len(ks) - 1:
                ax[i].xaxis.set_visible(False)
            else:
                ax[i].tick_params(axis='x', colors='white')

            start = np.array([22, 22, 22]) / 255
            target = np.array(to_rgb(color))
            colors = [start, target]
            cmap = LinearSegmentedColormap.from_list('custom_mono', colors)

            # Create histogram
            ax[i].hist2d(all_phases_, all_freqs_,
                         bins=[np.arange(-0.5, 0.5, 0.1), np.arange(0.0, 1.1, 0.1)],
                         cmap=cmap)
            ax[i].set_ylim(0.0, 1.0)
            ax[i].tick_params(axis='y', colors='white')

        ax[len(ks) - 1].set_xlabel('Time delay between LED and forward firefly [s]', color='white')
        ax[int(len(ks) / 2) - 1].set_ylabel('Response period [s]', color='white')
        fig.set_facecolor(np.array([22, 22, 22]) / 255)

        plt.savefig(plot_params.save_folder + '/response_period_vs_delay')
        plt.close(fig)
        colormap = plt.cm.get_cmap('viridis_r', len(ks)*3)

        # delay
        for i, k in enumerate(ks):
            fig, ax = plt.subplots(len(rmses['phase_time_diffs_instanced'][k]), figsize=(14, 16), sharex=True,
                                   gridspec_kw={'hspace': 0.4})
            color = colormap.__call__(i * 3)
            for kk, individual in enumerate(rmses['phase_time_diffs_instanced'][k]):
                times = []
                phases = []

                for ii, (phase, freq, time) in enumerate(individual):
                    times.append(time)
                    phases.append(phase)

                if len(phases) > 2:
                    times = np.array(times)
                    phases = np.array(phases)
                    alphas = [1.0 - abs(phase) for phase in phases]

                    ax[kk].scatter(times, phases, color=color, alpha=alphas, s=10)
                ax[kk].axhline(0.0, color='white', linestyle='--', linewidth=1)
                ax[kk].spines['top'].set_visible(False)
                ax[kk].spines['right'].set_visible(False)
                ax[kk].spines['bottom'].set_visible(True)
                ax[kk].spines['left'].set_visible(True)
                ax[kk].spines['bottom'].set_edgecolor('gray')
                ax[kk].spines['left'].set_edgecolor('gray')
                ax[kk].set_facecolor(np.array([22, 22, 22]) / 255)
                ax[kk].tick_params(axis='y', colors='gray')
                ax[kk].grid(False)

            ax[0].set_ylabel('Phase\nDiff [s]', color='gray')
            ax[-1].set_xlabel('Experiment time [s]', color='gray')
            ax[-1].tick_params(axis='x', colors='gray')

            fig.set_facecolor(np.array([22, 22, 22]) / 255)
            plt.savefig('figs/phase_traj/phase_trajectories_{}'.format(k))
            plt.close(fig)

        # RT
        for i, k in enumerate(ks):
            fig, ax = plt.subplots(len(rmses['phase_time_diffs_instanced'][k]), figsize=(14, 16), sharex=True,
                                   gridspec_kw={'hspace': 0.4})
            color = colormap.__call__(i * 3)
            if i == 4:
                color = 'yellow'
            for kk, individual in enumerate(rmses['phase_time_diffs_instanced'][k]):
                times = []
                freqs = []

                for ii, (phase, freq, time) in enumerate(individual):
                    times.append(time)
                    freqs.append(freq)

                if len(freqs) > 2:
                    times = np.array(times)
                    freqs = np.array(freqs)
                    ref_freq = float(k) / 1000

                    diffs = np.abs(freqs - ref_freq)
                    max_diff = np.max(diffs) if np.max(diffs) > 0 else 1
                    normalized_diffs = diffs / max_diff

                    alphas = 1.0 - 0.5 * normalized_diffs
                    ax[kk].scatter(times, freqs, color=color, alpha=alphas, s=10)
                ax[kk].axhline(float(k)/1000, color='white', linestyle='--', linewidth=1)
                ax[kk].spines['top'].set_visible(False)
                ax[kk].spines['right'].set_visible(False)
                ax[kk].spines['bottom'].set_visible(True)
                ax[kk].spines['left'].set_visible(True)
                ax[kk].spines['bottom'].set_edgecolor('gray')
                ax[kk].spines['left'].set_edgecolor('gray')
                ax[kk].set_facecolor(np.array([22, 22, 22]) / 255)
                ax[kk].tick_params(axis='y', colors='gray')
                ax[kk].grid(False)
            ax[0].set_ylabel('Response\nTime [s]', color='gray')
            ax[-1].set_xlabel('Experiment time [s]', color='gray')
            ax[-1].tick_params(axis='x', colors='gray')

            fig.set_facecolor(np.array([22, 22, 22]) / 255)
            plt.savefig('figs/rt_traj/response_time_trajectories_{}'.format(k))
            plt.close(fig)

    if plot_params.do_prc:
        fig, axes = plt.subplots(len(ks))
        for i, k in enumerate(ks):
            xx_ = np.linspace(-0.5, 0.5, 1000)

            all_delay_responses = [a for a in rmses['phase_response'][k] if np.abs(a[1]) < (float(k) / 1000)]
            xs = np.array([a[0] for a in all_delay_responses])
            ys = np.array([a[1] for a in all_delay_responses])

            x_dict = {}
            for q, x in enumerate(xs):
                rounded_x = round(x, 2)
                if rounded_x not in x_dict:
                    x_dict[rounded_x] = []
                x_dict[rounded_x].append(ys[q])

            to_plot_xs = np.array(sorted(x_dict.keys()))
            to_plot_ys = np.array([np.mean(x_dict[j]) for j in to_plot_xs])
            to_plot_stds = np.array([np.std(x_dict[j]) for j in to_plot_xs])

            if len(to_plot_xs) > 3:  # Ensure enough points for interpolation
                spline = scipy.interpolate.make_interp_spline(to_plot_xs, to_plot_ys, k=3)
                smooth_ys = spline(xx_)

                spline_upper = scipy.interpolate.make_interp_spline(to_plot_xs, to_plot_ys + to_plot_stds, k=3)
                spline_lower = scipy.interpolate.make_interp_spline(to_plot_xs, to_plot_ys - to_plot_stds, k=3)

                smooth_upper = spline_upper(xx_)
                smooth_lower = spline_lower(xx_)
            else:
                smooth_ys = np.interp(xx_, to_plot_xs, to_plot_ys)
                smooth_upper = np.interp(xx_, to_plot_xs, to_plot_ys + to_plot_stds)
                smooth_lower = np.interp(xx_, to_plot_xs, to_plot_ys - to_plot_stds)

            color = 'yellow' if i == 4 else colormap.__call__(i * 3)

            axes[i].plot(xx_, smooth_ys, color=color, linewidth=2)
            axes[i].fill_between(xx_, smooth_lower, smooth_upper, color='gray', alpha=0.25)
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(-0.5, 0.5)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        axes[int(len(ks) / 2)].set_ylabel('Period shift [s]')
        plt.savefig(plot_params.save_folder + '/prc')
        plt.close(fig)
        fig, axes = plt.subplots(len(ks), figsize=(8, 6))

        bin_size = 0.033  # Bin width for histogram
        all_all_delay_responses = []

        for i, k in enumerate(ks):
            all_delay_responses = [a for a in rmses['phase_response'][k] if np.abs(a[1]) < (float(k) / 1000)]
            xs = np.array([a[0] for a in all_delay_responses])
            ys = np.array([a[1] for a in all_delay_responses])
            all_all_delay_responses.extend(all_delay_responses)

            bin_edges = np.arange(0.0, 1.0 + bin_size, bin_size)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            binned_means = []
            binned_stds = []
            for j in range(len(bin_edges) - 1):
                mask = (xs >= bin_edges[j]) & (xs < bin_edges[j + 1])
                if np.sum(mask) > 0:
                    binned_means.append(np.mean(ys[mask]))
                    binned_stds.append(np.std(ys[mask]))
                else:
                    binned_means.append(np.nan)
                    binned_stds.append(np.nan)

            binned_means = np.array(binned_means)
            binned_stds = np.array(binned_stds)

            color = 'yellow' if i == 4 else colormap.__call__(i * 3)
            for j in range(len(bin_centers)):
                if np.isnan(binned_means[j]):
                    continue

                alpha = 1.0 if binned_means[j] > 0 else 0.6
                linewidth = 2.0 if binned_means[j] > 0 else 1.0

                axes[i].bar(
                    bin_centers[j], binned_means[j], width=bin_size * 0.9,
                    color=color, edgecolor='black', alpha=alpha, linewidth=linewidth
                )
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)

            axes[i].set_xlim(0.0, 1.0)
            axes[i].set_ylim(-0.2, 0.2)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

        axes[int(len(ks) - 1)].set_xlabel('Normalized Time Delay')
        axes[int(len(ks) / 2)].set_ylabel('Average Period Shift [s]')
        plt.savefig(plot_params.save_folder + '/binned_prc')
        plt.close(fig)

        # aggregate PRC
        fig, axes = plt.subplots()
        xs = np.array([a[0] for a in all_all_delay_responses])
        ys = np.array([a[1] for a in all_all_delay_responses])

        bin_edges = np.arange(0.0, 1.0 + bin_size, bin_size)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        binned_means = []
        binned_stds = []
        for j in range(len(bin_edges) - 1):
            mask = (xs >= bin_edges[j]) & (xs < bin_edges[j + 1])
            if np.sum(mask) > 0:
                binned_means.append(np.mean(ys[mask]))
                binned_stds.append(np.std(ys[mask]))
            else:
                binned_means.append(np.nan)
                binned_stds.append(np.nan)

        binned_means = np.array(binned_means)
        binned_stds = np.array(binned_stds)

        color = 'purple'
        for j in range(len(bin_centers)):
            if np.isnan(binned_means[j]):
                continue

            alpha = 1.0 if binned_means[j] > 0 else 0.6
            linewidth = 2.0 if binned_means[j] > 0 else 1.0

            axes.bar(
                bin_centers[j], binned_means[j], width=bin_size * 0.9, color=color, edgecolor='black', alpha=alpha,
                linewidth=linewidth
            )

        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(-0.2, 0.2)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.set_xlabel('Normalized time delay')
        axes.set_ylabel('Period shift [s]')
        plt.savefig(plot_params.save_folder + '/aggregate_prc')
        plt.close(fig)

    if plot_params.do_initial_distribution:
        # figure 1 a
        fig, axes = plt.subplots(len(ks), figsize=(8, 6))
        all_before = []
        all_befores = {}
        for i, k in enumerate(ks):
            all_befores[k] = []
            for individual in rmses['windowed_period_before'][k]:
                try:
                    all_befores[k].extend(individual)
                    all_before.extend(individual)
                except TypeError:
                    all_before.append(individual)
                    all_befores[k].append(individual)
        for i, k in enumerate(ks):
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].hist(all_befores[k], density=True, bins=np.arange(0.0, 2.0, 0.03), color='black', alpha=0.25)

            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(0.0, 1.5)
            axes[i].set_ylim(0.0, 6.0)
        axes[len(ks) - 1].set_xlabel('T[s]')
        axes[int(len(ks) / 2)].set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_windowed_period_2025_before_nolines')
        plt.close(fig)

        fig, axes = plt.subplots(2, figsize=(8, 6))
        bout_lengths_before = []
        bout_lengths_after = []
        bout_vars_before = []
        bout_vars_after = []
        for i, k in enumerate(ks):
            for div in rmses['bouts_before'][k]:
                bout_lengths_before.append(div[0])
                bout_vars_before.append(div[1])
        for i, k in enumerate(ks):
            for ind in rmses['bouts_after'][k]:
                bout_lengths_after.append(ind[0])
                bout_vars_after.append(ind[1])

        axes[0].hist(bout_lengths_before, density=True, bins=np.arange(0.0, 100, 1.0),
                     color='black', alpha=0.25, label='Before')
        axes[0].hist(bout_lengths_after, density=True, bins=np.arange(0.0, 100, 1.0),
                     color='green', alpha=0.25, label='After')
        axes[1].hist(bout_vars_before, density=True, bins=np.arange(0.0, 1.00, 0.01), color='black', alpha=0.25)
        axes[1].hist(bout_vars_after, density=True, bins=np.arange(0.0, 1.00, 0.01), color='green', alpha=0.25)

        axes[0].set_xlabel('Bout length (flashes)')
        axes[1].set_xlabel('Bout interflash variance [s]')
        axes[0].legend()

        plt.savefig(plot_params.save_folder + '/bout_statistics')
        plt.close(fig)

        # fig 1b
        fig, axes = plt.subplots()
        all_before = []
        all_befores = {}
        for i, k in enumerate(ks):
            all_befores[k] = []
            for individual in rmses['windowed_period_before'][k]:
                try:
                    all_befores[k].extend(individual)
                    all_before.extend(individual)
                except TypeError:
                    all_before.append(individual)
                    all_befores[k].append(individual)
        axes.hist(all_before, density=True, bins=np.arange(0.0, 2.0, 0.03), color='black', alpha=0.25)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.set_xlim(0.0, 1.5)
        axes.set_ylim(0.0, 6.0)
        axes.set_xlabel('T[s]')
        axes.set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_windowed_period_before_aggregate_all2025')
        plt.close(fig)

        fig, axes = plt.subplots()
        all_before = []
        all_befores = {}
        for i, k in enumerate(ks):
            all_befores[k] = []
            for individual in rmses['windowed_period_after'][k]:
                try:
                    all_befores[k].extend(individual)
                    all_before.extend(individual)
                except TypeError:
                    all_before.append(individual)
                    all_befores[k].append(individual)
        axes.hist(all_before, density=True, bins=np.arange(0.0, 2.0, 0.03), color='black', alpha=0.25)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.set_xlim(0.0, 1.5)
        axes.set_ylim(0.0, 6.0)
        axes.set_xlabel('T[s]')
        axes.set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_windowed_period_after_aggregate_all')
        plt.close(fig)

        # variance before and after
        sns.set(style='whitegrid', context='talk')
        fig, ax = plt.subplots(figsize=(10, 8))

        x_labels = []
        x_positions = {}
        k_sorted = sorted(ks)

        for i, k in enumerate(k_sorted):
            x_labels.extend([f'{k}\nBefore', f'{k}\nAfter'])
            x_positions[(k, 'Before')] = i * 2
            x_positions[(k, 'After')] = i * 2 + 1

        color_map = sns.color_palette("husl", len(k_sorted))

        for i, k in enumerate(k_sorted):
            individual_idx = 0
            befores = rmses['individual_before'][k]
            afters = rmses['individual_after'][k]

            for before, after in zip(befores, afters):
                try:
                    var_before = np.var(before)
                except TypeError:
                    var_before = np.nan

                try:
                    var_after = np.var(after)
                except TypeError:
                    var_after = np.nan

                if np.isnan(var_before) or np.isnan(var_after):
                    continue

                x_b = x_positions[(k, 'Before')]
                x_a = x_positions[(k, 'After')]

                if var_before > var_after:
                    print(individual_idx, k)
                ax.plot([x_b, x_a], [var_before, var_after], color=color_map[i], alpha=0.6, linewidth=1)
                ax.scatter([x_b, x_a], [var_before, var_after], color=color_map[i], s=30,
                           label=f'k={k}' if individual_idx == 0 else None)
                individual_idx += 1

        ax.set_xticks([x_positions[key] for key in x_positions])
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_ylabel('Variance of T [s]')
        ax.set_title('Windowed period variance before and after per individual, grouped by k')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best')

        plt.tight_layout()
        plt.savefig(plot_params.save_folder + '/LED_variance_dumbbell_by_k_colored2024.png')
        plt.close()

    if plot_params.do_windowed_period_plot:
        fig, axes = plt.subplots(len(ks), figsize=(10, 8))
        for i, k in enumerate(ks):
            axes[i].grid(False)
            axes[i].tick_params(axis='both', which='both', length=0)
            all_befores[k] = []
            all_afters[k] = []
            for individual in rmses['windowed_period_after'][k]:
                try:
                    all_afters[k].extend(individual)
                except TypeError:
                    all_afters[k].append(individual)
            for individual in rmses['windowed_period_before'][k]:
                try:
                    all_befores[k].extend(individual)
                except TypeError:
                    all_befores[k].append(individual)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            all_befores[k] = [x for x in all_befores[k] if not np.isnan(x)]
            all_afters[k] = [x for x in all_afters[k] if not np.isnan(x)]
            color = colormap.__call__(i * 3)
            axes[i].hist(all_befores[k], density=True, bins=np.arange(0.0, 2.0, 0.03), color='black', alpha=0.25)
            axes[i].hist(all_afters[k], density=True, bins=np.arange(0.0, 2.0, 0.03), color=color, alpha=0.75)
            if i != 0:
                axes[i].axvline(float(k) / 1000, color='black')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--')
                axes[i].axvline(2.0 * (float(k) / 1000), color='black', linestyle=':')
            else:
                axes[i].axvline(float(k) / 1000, color='black', label='LED period')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--', label='0.5 * LED period')
                axes[i].axvline(2.0 * (float(k) / 1000), color='black', linestyle=':', label='2.0 * LED period')
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(0.0, 2.05)
            axes[i].set_ylim(0.0, 6.0)

        if plot_params.save_data:
            with open(plot_params.save_folder + '/all_befores_from_experiments.pickle', 'wb') as handle:
                pickle.dump(all_befores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            all_befores_list = []
            for k in all_befores.keys():
                li = [x for x in all_befores[k] if not np.isnan(x)]
                all_befores_list.extend(str(li))
            with open(plot_params.save_folder + '/single_firefly_intervals.csv', 'w') as csv_handle:
                csv_handle.write('\n'.join(all_befores_list))
            with open(plot_params.save_folder + '/all_afters_from_experiments.pickle', 'wb') as handle:
                pickle.dump(all_afters, handle, protocol=pickle.HIGHEST_PROTOCOL)

        axes[len(ks) - 1].set_xlabel('T[s]')
        fig.text(0.01, 0.5, 'pdf', va='center', rotation='vertical')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_period_2025_windowed_beforeafter_w_linesyes')
        plt.close(fig)

        if plot_params.do_boxplots:
            boxplots(all_befores, all_afters, plot_params)

        fig, axes = plt.subplots(len(ks), figsize=(10, 8))
        for i, k in enumerate(ks):
            axes[i].grid(False)
            axes[i].tick_params(axis='both', which='both', length=0)
            all_afters[k] = []
            for individual in rmses['windowed_period_after'][k]:
                try:
                    all_afters[k].extend(individual)
                except TypeError:
                    all_afters[k].append(individual)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            all_afters[k] = [x for x in all_afters[k] if not np.isnan(x)]
            color = colormap.__call__(i * 3)
            axes[i].hist(all_afters[k], density=True, bins=np.arange(0.0, 2.0, 0.03), color=color, alpha=0.75)
            if i != 0:
                axes[i].axvline(float(k) / 1000, color='black')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--')
                axes[i].axvline(2.0 * (float(k) / 1000), color='black', linestyle=':')
            else:
                axes[i].axvline(float(k) / 1000, color='black', label='LED period')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--', label='0.5 * LED period')
                axes[i].axvline(2.0 * (float(k) / 1000), color='black', linestyle=':', label='2.0 * LED period')
                axes[i].legend()
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(0.0, 2.05)
            axes[i].set_ylim(0.0, 6.0)
        axes[len(ks) - 1].set_xlabel('T[s]')
        fig.text(0.01, 0.5, 'pdf', va='center', rotation='vertical')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_period_2025_windowed_after_w_lines')
        plt.close(fig)

        fig, axes = plt.subplots(len(ks) + 1, figsize=(12, 8))
        all_befores = []

        for kk in rmses['windowed_period_before'].keys():
            for individual in rmses['windowed_period_before'][kk]:
                try:
                    all_befores.extend(individual)
                except TypeError:
                    all_befores.append(individual)

        axes[0].hist(all_befores, density=True, bins=np.arange(0.0, 2.0, 0.03), color='gray', alpha=0.75)

        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].grid(False)
        axes[0].tick_params(axis='both', which='both', length=0)
        axes[0].set_xlim(0.0, 2.05)
        axes[0].set_ylim(0.0, 6.0)
        for i, k in enumerate(ks):
            axes[i + 1].grid(False)
            axes[i + 1].tick_params(axis='both', which='both', length=0)
            all_afters[k] = []
            for individual in rmses['windowed_period_after'][k]:
                try:
                    all_afters[k].extend(individual)
                except TypeError:
                    all_afters[k].append(individual)

            axes[i + 1].spines['top'].set_visible(False)
            axes[i + 1].spines['right'].set_visible(False)
            all_afters[k] = [x for x in all_afters[k] if not np.isnan(x)]
            color = colormap.__call__(i * 3)
            axes[i + 1].hist(all_afters[k], density=True, bins=np.arange(0.0, 2.0, 0.03), color=color, alpha=0.75)
            if i != 0:
                axes[i + 1].axvline(float(k) / 1000, color='black')
                axes[i + 1].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--')
                axes[i + 1].axvline(2.0 * (float(k) / 1000), color='black', linestyle=':')
            else:
                axes[i + 1].axvline(float(k) / 1000, color='black', label='LED period')
                axes[i + 1].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--', label='0.5 * LED period')
                axes[i + 1].axvline(2.0 * (float(k) / 1000), color='black', linestyle=':', label='2.0 * LED period')
            if i != len(ks):
                axes[i].xaxis.set_visible(False)
            axes[i + 1].set_xlim(0.0, 2.05)
            axes[i + 1].set_ylim(0.0, 6.0)
        axes[len(ks)].set_xlabel('T[s]')
        fig.text(0.01, 0.5, 'pdf', va='center', rotation='vertical')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_period_2025_windowed_after_w_lines_and_top')
        plt.close(fig)

        fig, axes = plt.subplots(len(ks), figsize=(10, 8))
        for i, k in enumerate(ks):
            axes[i].grid(False)
            axes[i].tick_params(axis='both', which='both', length=0)
            all_befores[k] = []
            all_afters[k] = []
            for individual in rmses['windowed_period_after'][k]:
                try:
                    all_afters[k].extend(individual)
                except TypeError:
                    all_afters[k].append(individual)
            for individual in rmses['windowed_period_before'][k]:
                try:
                    all_befores[k].extend(individual)
                except TypeError:
                    all_befores[k].append(individual)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            all_befores[k] = [x for x in all_befores[k] if not np.isnan(x)]
            all_afters[k] = [x for x in all_afters[k] if not np.isnan(x)]
            color = colormap.__call__(i * 3)
            axes[i].hist(all_befores[k], density=True, bins=np.arange(0.0, 2.0, 0.03), color='black', alpha=0.25)
            axes[i].hist(all_afters[k], density=True, bins=np.arange(0.0, 2.0, 0.03), color=color, alpha=0.75)
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(0.0, 2.05)
            axes[i].set_ylim(0.0, 6.0)

        axes[len(ks) - 1].set_xlabel('T[s]')
        fig.text(0.01, 0.5, 'pdf', va='center', rotation='vertical')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_period_2025_windowed_beforeafter_without_lines')
        plt.close(fig)

        fig, axes = plt.subplots(len(ks), figsize=(10, 8))
        for i, k in enumerate(ks):
            axes[i].grid(False)
            axes[i].tick_params(axis='both', which='both', length=0)
            all_afters[k] = []
            for individual in rmses['windowed_period_after'][k]:
                try:
                    all_afters[k].extend(individual)
                except TypeError:
                    all_afters[k].append(individual)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            all_afters[k] = [x for x in all_afters[k] if not np.isnan(x)]
            color = colormap.__call__(i * 3)
            axes[i].hist(all_afters[k], density=True, bins=np.arange(0.0, 2.0, 0.03), color=color, alpha=0.75)
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(0.0, 2.05)
            axes[i].set_ylim(0.0, 6.0)
        axes[len(ks) - 1].set_xlabel('T[s]')
        fig.text(0.01, 0.5, 'pdf', va='center', rotation='vertical')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_period_2025_windowed_after')
        plt.close(fig)

        fig, axes = plt.subplots(len(ks), figsize=(10, 8))
        all_before = []
        all_afters = {}
        all_befores = {}
        for i, k in enumerate(ks):
            all_befores[k] = []
            all_afters[k] = []
            for individual in rmses['windowed_period_after'][k]:
                try:
                    all_afters[k].extend(individual)
                except TypeError:
                    all_afters[k].append(individual)
            for individual in rmses['windowed_period_before'][k]:
                try:
                    all_befores[k].extend(individual)
                    all_before.extend(individual)
                except TypeError:
                    all_before.append(individual)
                    all_befores[k].append(individual)
        for i, k in enumerate(ks):
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].fill_betweenx(
                np.arange(0.0, 7.0, 1.0), np.nanpercentile(all_before, 50) - (np.nanstd(all_before) / 2),
                                          np.nanpercentile(all_before, 50) + (np.nanstd(all_before) / 2),
                facecolor='black', alpha=0.2
            )
            color = colormap.__call__(i * 3)
            axes[i].hist(all_afters[k], density=True, bins=np.arange(0.0, 3.0, 0.03), color=color)
            if i != 0:
                axes[i].axvline(float(k) / 1000, color='black')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--')
                axes[i].axvline(2.0 * (float(k) / 1000), color='black', linestyle=':')
            else:
                axes[i].axvline(float(k) / 1000, color='black', label='LED period')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--', label='0.5 * LED period')
                axes[i].axvline(2.0 * (float(k) / 1000), color='black', linestyle=':', label='2.0 * LED period')
                axes[i].legend()
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(0.0, 1.5)
            axes[i].set_ylim(0.0, 6.0)
        axes[len(ks) - 1].set_xlabel('T[s]')
        axes[int(len(ks) / 2)].set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_windowed_period_2025_w_aggregatebefore_after')
        plt.close(fig)
        print('here')

    if plot_params.do_scatter_overall_stats:
        fig,ax = plt.subplots()
        # Mean and variance before and after LD
        ks_hz = [(float(k) / 1000) for k in ks]
        means_b = [np.mean([scipy.stats.mode(y)[0] for y in rmses['all_periods_before'][k]]) for k in ks]
        means_a = [np.mean([scipy.stats.mode(y)[0] for y in rmses['all_periods_after'][k]]) for k in ks]
        vars_b = [np.mean([np.var(y) for y in rmses['all_periods_before'][k]]) for k in ks]
        vars_a = [np.mean([np.var(y) for y in rmses['all_periods_after'][k]]) for k in ks]
        ax.scatter(ks_hz, means_b, color='deepskyblue', label='Mean period before LED introduced')
        ax.scatter(ks_hz, means_a, color='darkcyan', label='Mean period after LED introduced')
        ax.scatter(ks_hz, vars_b, color='orangered', label='Var of period before LED introduced')
        ax.scatter(ks_hz, vars_a, color='darkred', label='Var of period after LED introduced')
        ax.set_xlabel('LED period)')
        ax.set_ylabel('Mean and variance of firefly period (1/s, 1/(s^2))')
        ax.scatter(ks_hz, [(float(k) / 1000) for k in ks], color='black', marker='+', label='LED period')
        plt.legend()
        plt.savefig(plot_params.save_folder + '/summary')
        plt.close(fig)


def inflection_score(phases, cycle_length, stability_thresh=0.05, min_region_len=5):
    """
    Compute a score based on variance difference before and after a stability transition
    in phase values. Detects both stable→unstable and unstable→stable inflections.

    Parameters:
    - phases: list or array of phase values (can be circular, normalized [0,1] or radians).
    - cycle_length: normalization constant for circular difference (e.g., 1.0 or 2π).
    - stability_thresh: threshold for smoothed circular difference to define stability.
    - min_region_len: minimum number of points required on both sides of a transition.

    Returns:
    - best_score: float indicating max variance difference across a valid inflection.
    """
    if len(phases) < 2 * min_region_len + 1:
        return 0.0

    phases = np.array(phases)
    dphi = helpers.circular_diff(phases, cycle_length)
    smooth = np.convolve(np.abs(dphi), np.ones(3) / 3, mode='valid')
    is_stable = smooth < stability_thresh

    # Find both types of transitions: up (unstable → stable), down (stable → unstable)
    transitions = np.where(np.diff(is_stable.astype(int)) != 0)[0] + 1

    best_score = 0.0

    for idx in transitions:
        pre_start = idx - min_region_len
        post_end = idx + min_region_len
        if pre_start < 0 or post_end > len(phases):
            continue

        pre_var = helpers.circular_variance(phases[pre_start:idx], cycle_length)
        post_var = helpers.circular_variance(phases[idx:post_end], cycle_length)

        score = pre_var - post_var
        best_score = max(best_score, abs(score))

    return best_score


def inflection_points(metrics_dict, bout_gap_length, top_n=10, num_flashes=5):
    bouts = []
    for key, bouts_list in metrics_dict[bout_gap_length].items():
        for bout in bouts_list:
            if bout['det'] > 0.1 and bout['lam'] > 0.1 and bout['num_flashes'] > num_flashes:
                score = inflection_score(bout['phases'], float(key) / 1000)
                bouts.append((key, bout, score, bout['num_flashes']))
    colormap = cm.get_cmap('viridis_r', 24)
    grouped = defaultdict(list)
    for bout in bouts:
        name = bout[0]
        middle_number = name.split('_')[1]
        grouped[middle_number].append(bout)

    # Sort each group descending by the third element
    for key in grouped:
        grouped[key].sort(key=lambda x: x[2], reverse=True)
    fig, axes = plt.subplots(figsize=(10, 10))

    top_bouts_per_freq = {'300': [], '400': [], '500': [], '600': [], '700': [], '770': [], '850': [], '1000': []}
    for i, (group_key, group_bouts) in enumerate(sorted(grouped.items(), key=lambda item: int(item[0]))):
        top = group_bouts[:top_n]
        top_bouts_per_freq[group_key].extend(top)

    all_phases = []
    all_shifts = []
    for i, group_key in enumerate(top_bouts_per_freq.keys()):
        print(i, group_key)
        print('here!!')
        group_bouts = top_bouts_per_freq[group_key]
        phases = [y for x in group_bouts for y in x[1]['phases']]
        shifts = [y for x in group_bouts for y in x[1]['shifts']]
        p_hases = []
        s_hifts = []

        color = colormap(i * 3)

        for p, s in zip(phases, shifts):
            if s < 3.0:
                p_hases.append(p)
                s_hifts.append(s)

        p_hases = np.array(p_hases)
        p_hases = helpers.improved_circular_normalize(p_hases, float(group_key) / 1000)[0]
        if not isinstance(p_hases, np.ndarray):
            p_hases = np.array(p_hases)
        s_hifts = np.array([np.nan if s is None else s for s in s_hifts], dtype=float)
        valid_mask = ~np.isnan(s_hifts)
        p_hases = p_hases[valid_mask]
        s_hifts = s_hifts[valid_mask]
        all_phases.extend(p_hases)
        all_shifts.extend(s_hifts)
        sorted_indices = np.argsort(p_hases)
        p_hases_sorted = p_hases[sorted_indices]
        s_hifts_sorted = s_hifts[sorted_indices]
        bins = np.linspace(min(p_hases_sorted), max(p_hases_sorted), 20)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        medians = []
        for i in range(1, len(bins)):
            in_bin = (p_hases_sorted >= bins[i - 1]) & (p_hases_sorted < bins[i])
            if np.any(in_bin):
                medians.append(np.median(s_hifts_sorted[in_bin]))

            else:
                medians.append(np.nan)

        medians = np.array(medians)

        # Smooth with rolling average first (this will retain NaNs at gaps)
        medians_smooth = pd.Series(medians).rolling(window=5, center=True, min_periods=1).mean()

        # Plot
        axes.plot(bin_centers, medians_smooth, color=color)
        # ax.plot(unique_phases, medians, color=color)
        # ax.scatter(p_hases, s_hifts, color=color, s=7)
        axes.axhline(1.0, color='black', linestyle='--')
        axes.spines['top'].set_visible(False)
    all_phases = np.array(all_phases)
    all_shifts = np.array(all_shifts)

    # Sort
    sorted_indices = np.argsort(all_phases)
    p_hases_sorted = all_phases[sorted_indices]
    s_hifts_sorted = all_shifts[sorted_indices]

    bins = np.linspace(0, 1, 20)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    medians = []
    for i in range(1, len(bins)):
        in_bin = (p_hases_sorted >= bins[i - 1]) & (p_hases_sorted < bins[i])
        if np.any(in_bin):
            medians.append(np.median(s_hifts_sorted[in_bin]))
        else:
            medians.append(np.nan)

    medians = np.array(medians)

    # Smooth and fill nan
    medians_smooth = pd.Series(medians).rolling(window=5, center=True, min_periods=1).mean()
    # axes2 = axes.twinx()  # create a second y-axis that shares the same x-axis

    # Plot thick black average
    axes.plot(bin_centers, medians_smooth, color='black', linewidth=7, label='Overall average')
    # axes2.set_ylim(0.99, 1.027)
    # axes2.set_ylabel('Overall PRC')
    axes.set_ylabel(r'$\frac{T_{i+1}}{T_i}$', labelpad=30, rotation='vertical', fontsize=20)
    axes.set_xlabel('Phase difference between firefly and driving signal [s]')
    plt.tight_layout()
    plt.savefig('figs/prcs_{}_sameaxis_inflections.png'.format(bout_gap_length))
    plt.close()


def top_bottom_aggregate_statistics(metrics_dict, bout_gap_length, top_n=10, num_flashes=5):
    bouts = []
    for key, bouts_list in metrics_dict[bout_gap_length].items():
        for bout in bouts_list:
            if bout['det'] > 0.1 and bout['lam'] > 0.1 and bout['num_flashes'] > num_flashes:
                score = bout['det'] + bout['lam']
                bouts.append((key, bout, score, bout['num_flashes']))

    grouped = defaultdict(list)
    for bout in bouts:
        name = bout[0]
        middle_number = name.split('_')[1]
        grouped[middle_number].append(bout)

    # Sort each group descending by the third element
    for key in grouped:
        grouped[key].sort(key=lambda x: x[2], reverse=True)

    num_groups = len(grouped)
    colormap = cm.get_cmap('viridis_r', 24)
    fig, axes = plt.subplots(num_groups, 1, figsize=(10, 10), sharex=True)
    if num_groups == 1:
        axes = [axes]

    for i, (ax, (group_key, group_bouts)) in enumerate(
            zip(axes, sorted(grouped.items(), key=lambda item: int(item[0])))):
        top = group_bouts[:top_n]
        bottom = group_bouts[-top_n:]
        color = colormap(i * 3)

        phase_derivs = []
        for bout in top:
            phase_derivs.extend(helpers.circular_diff(bout[1]['phases'], float(group_key) / 1000))
        ax.hist(phase_derivs, density=True, bins=np.arange(-0.51, 0.51, 0.033), color=color, alpha=0.6)
        phase_derivs = []
        for bout in bottom:
            phase_derivs.extend(helpers.circular_diff(bout[1]['phases'], float(group_key) / 1000))
        ax.hist(phase_derivs, density=True, bins=np.arange(-0.51, 0.51, 0.033), color='black', alpha=0.6)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axes[-1].set_xlabel(r'Circular Phase Derivative $\frac{d\phi}{dt}$ (s$^{-1}$)', fontsize=12)
    fig.text(0.01, 0.5, 'pdf', va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig('figs/phase_derivs_topbottom_{}.png'.format(bout_gap_length))
    plt.close()
    fig, axes = plt.subplots(num_groups, 1, figsize=(10, 10), sharex=True)
    if num_groups == 1:
        axes = [axes]

    top_bouts_per_freq = {'300': [],'400': [], '500': [], '600': [], '700': [], '770': [],'850': [], '1000': []}
    for i, (ax, (group_key, group_bouts)) in enumerate(
            zip(axes, sorted(grouped.items(), key=lambda item: int(item[0])))):
        top = group_bouts[:top_n]
        top_bouts_per_freq[group_key].extend(top)
        bottom = group_bouts[-top_n:]
        color = colormap(i * 3)

        phase_derivs = []
        for bout in top:
            phase_derivs.extend(helpers.circular_diff_normalized(bout[1]['phases'], float(group_key) / 1000))
        ax.hist(phase_derivs, density=True, bins=np.arange(-0.51, 0.51, 0.05), color=color, alpha=0.6)
        phase_derivs = []
        for bout in bottom:
            phase_derivs.extend(helpers.circular_diff_normalized(bout[1]['phases'], float(group_key) / 1000))
        ax.hist(phase_derivs, density=True, bins=np.arange(-0.51, 0.51, 0.05), color='black', alpha=0.6)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axes[-1].set_xlabel(r'Normalized Circular Phase Derivative $\frac{d\phi}{dt}$ (s$^{-1}$)', fontsize=12)
    fig.text(0.01, 0.5, 'pdf', va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig('figs/phase_derivs_topbottom_{}_normalized.png'.format(bout_gap_length))
    plt.close()

    fig, axes = plt.subplots(len(grouped), 1, figsize=(10, 10), sharex=True)
    if len(grouped) == 1:
        axes = [axes]
    sorted_group_keys = sorted(grouped.keys(), key=lambda x: int(x))

    for i, (ax, group_key) in enumerate(zip(axes, sorted_group_keys)):
        group_bouts = grouped[group_key]
        top = group_bouts[:top_n]
        bottom = group_bouts[-top_n:]
        color = 'xkcd:sun yellow' if i == 4 else colormap(i * 3)
        top_responses = [r for bout in top for r in bout[1].get('responses', [])]
        bottom_responses = [r for bout in bottom for r in bout[1].get('responses', [])]

        ax.hist(top_responses, bins=np.arange(0.0, 2.0, 0.03), density=True,
                color=color, alpha=0.6, label='Top 10')
        ax.hist(bottom_responses, bins=np.arange(0.0, 2.0, 0.03), density=True,
                color='black', alpha=0.6, label='Bottom 10')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        y_value = float(group_key) / 1000
        ax.axvline(x=y_value, color='black', linestyle='-', linewidth=3)
        y_value = float(group_key) / 2000
        ax.axvline(x=y_value, color='black', linestyle='--', linewidth=1)
        y_value = float(group_key) / 500
        ax.axvline(x=y_value, color='black', linestyle='-', linewidth=1)

        ax.set_ylabel('pdf')

    axes[-1].set_xlabel('Firefly induced period [s]')
    plt.tight_layout()
    plt.savefig('figs/period_responses_topbottom_{}.png'.format(bout_gap_length))
    plt.close()
    fig, ax, results = analyze_firefly_led_synchronization(grouped)
    print_sync_statistics(results)


def analyze_firefly_led_synchronization(grouped, figsize=(12, 8)):
    """
    Analyze synchronization between firefly and LED signals
    Creates a figure similar to the target showing phase relationships
    """
    from pathlib import Path

    # ---------------- knobs ----------------
    SAVE_DIR = "figs"
    figsize = (9, 5)
    MIN_ISI = 0.30
    exclude = 5
    thresh = 2.0
    n_bins = 35
    phase_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    phase_centers_global = 0.5 * (phase_edges[:-1] + phase_edges[1:])
    tol = 1e-3  # 1 ms tolerance for boundary comparisons

    # LED selection within a firefly cycle:
    # "first": keep the first LED strictly inside (t_prev, t_next)
    # "midpoint": keep the LED inside (t_prev, t_next) nearest the mid-cycle time
    SELECTION_MODE = "midpoint"  # or "midpoint"

    def _wrap_half(x):
        return x - np.floor(x + 0.5)

    # ---------------- pass 1: per-frequency curves ----------------
    sorted_keys = sorted(grouped.keys(), key=lambda x: float(x))
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(sorted_keys)))

    all_results = {}

    for ci, freq in enumerate(sorted_keys):
        print(f"Processing {freq} ms frequency...")
        color = colors[ci]
        led_period = float(freq) / 1000.0

        phases = []
        ratios = []
        flash_counts = []

        for j, trial in enumerate(grouped[freq]):
            try:
                f_times = np.asarray(trial[1]['full_timeseries']['firefly'], dtype=float)
                l_times = np.asarray(trial[1]['full_timeseries']['led'], dtype=float)
                start = float(trial[1]['start_idx'])
                end = float(trial[1]['end_idx'])

                if int(trial[1]['num_flashes']) < exclude:
                    continue

                # we only consider LED pulses that occur in the window
                l_win = l_times[(l_times >= start) & (l_times <= end)]
                if f_times.size < 3 or l_win.size == 0:
                    continue
                flash_counts.append(trial[1]['num_flashes'])

                # -------- one LED per firefly cycle --------
                # iterate over *firefly cycles* that overlap the LED window
                # i goes from the 2nd flash to the penultimate flash (so we have prev & next)
                for idx_prev in range(1, f_times.size - 1):
                    t_prevprev = f_times[idx_prev - 1]
                    t_prev = f_times[idx_prev]
                    t_next = f_times[idx_prev + 1]

                    # skip cycles that lie completely outside the LED window
                    if (t_next <= start) or (t_prev >= end):
                        continue

                    T_prev = t_prev - t_prevprev
                    T_next = t_next - t_prev
                    if (T_prev < MIN_ISI) or (T_next < MIN_ISI):
                        continue
                    if T_next >= thresh * led_period:
                        continue

                    # LEDs strictly inside the cycle (with tolerance)
                    m = (l_win > t_prev + tol) & (l_win < t_next - tol)
                    leds_inside = l_win[m]
                    if leds_inside.size == 0:
                        continue

                    if SELECTION_MODE == "first":
                        L = leds_inside[0]
                    elif SELECTION_MODE == "midpoint":
                        mid = 0.5 * (t_prev + t_next)
                        L = leds_inside[np.argmin(np.abs(leds_inside - mid))]
                    else:
                        L = leds_inside[0]  # default

                    # phase relative to the *ongoing* cycle
                    phi = _wrap_half((t_prev - L) / T_prev)  # [-0.5, 0.5)
                    phases.append(phi)

                    # ordinate: next/prev ratio
                    ratios.append(T_next / T_prev)

            except Exception:
                continue

        if len(phases) == 0:
            continue

        # bin to global edges
        phases = np.asarray(phases, float)
        ratios = np.asarray(ratios, float)
        good = np.isfinite(phases) & np.isfinite(ratios) & (phases >= -0.5) & (phases < 0.5)
        phases = phases[good];
        ratios = ratios[good]

        bin_idx = np.digitize(phases, phase_edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        y_mean = np.full(n_bins, np.nan, float)
        y_sem = np.full(n_bins, np.nan, float)
        n_per = np.zeros(n_bins, int)

        for b in range(n_bins):
            m = (bin_idx == b)
            if np.any(m):
                vals = ratios[m]
                y_mean[b] = np.mean(vals)
                y_sem[b] = np.std(vals, ddof=1) / np.sqrt(vals.size) if vals.size >= 2 else 0.0
                n_per[b] = vals.size

        valid = np.isfinite(y_mean)
        if not np.any(valid):
            continue

        all_results[freq] = {
            "color": color,
            "x": phase_centers_global[valid],
            "y": y_mean[valid],
            "sem": y_sem[valid],
            "n": n_per[valid],
            "counts_full": n_per.copy(),
        }

    # ---------------- Figure A: combined ----------------
    fig_comb, ax_comb = plt.subplots(figsize=figsize)
    for freq in sorted(all_results.keys(), key=lambda x: float(x)):
        rec = all_results[freq]
        ax_comb.plot(
            rec["x"], rec["y"],
            marker='o', linestyle='-', linewidth=2, markersize=7,
            color=rec["color"], markerfacecolor=rec["color"], markeredgecolor='black',
            alpha=0.95, label=f'{freq} ms'
        )
    ax_comb.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax_comb.axvline(0.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax_comb.set_xlim(-0.5, 0.5)
    ax_comb.set_xlabel('Firefly - LED Phase (- firefly before, + firefly after)')
    ax_comb.set_ylabel('Period change (T_next / T_prev)')
    ax_comb.set_title('PRC (ratio) — Combined')
    ax_comb.grid(True, alpha=0.3)
    if len(all_results) <= 20:
        ax_comb.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    fig_comb.tight_layout()

    if SAVE_DIR:
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
        fig_comb.savefig(Path(SAVE_DIR) / "prc_ratio_combined.png", dpi=200)

    # ---------------- Figure B: per-frequency ----------------
    for freq in sorted(all_results.keys(), key=lambda x: float(x)):
        rec = all_results[freq]
        fig_i, ax_i = plt.subplots(figsize=figsize)
        ax_i.plot(
            rec["x"], rec["y"],
            marker='o', linestyle='-', linewidth=2, markersize=7,
            color=rec["color"], markerfacecolor=rec["color"], markeredgecolor='black',
            alpha=0.95
        )
        ax_i.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax_i.axvline(0.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax_i.set_xlim(-0.5, 0.5)
        ax_i.set_xlabel('Firefly - LED Phase (- firefly before, + firefly after)')
        ax_i.set_ylabel('Period change (T_next / T_prev)')
        ax_i.set_title(f'{freq} ms LED — PRC')
        ax_i.grid(True, alpha=0.3)
        fig_i.tight_layout()
        if SAVE_DIR:
            fig_i.savefig(Path(SAVE_DIR) / f"prc_ratio_{int(float(freq))}ms.png", dpi=200)

    # ---------------- Figure C: grand mean (weighted) ----------------
    weights = np.zeros(n_bins, float)
    weighted_sum = np.zeros(n_bins, float)

    for rec in all_results.values():
        # map rec["x"] back to global bin indices
        for xk, yk, nk in zip(rec["x"], rec["y"], rec["n"]):
            b = np.where(np.isclose(phase_centers_global, xk, atol=1e-12))[0]
            if b.size:
                bi = int(b[0])
                weighted_sum[bi] += yk * nk
                weights[bi] += nk

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_line = np.where(weights > 0, weighted_sum / weights, np.nan)

    valid_mean = np.isfinite(mean_line)

    fig_mean, ax_mean = plt.subplots(figsize=figsize)
    ax_mean.plot(
        phase_centers_global[valid_mean], mean_line[valid_mean],
        linewidth=3, linestyle='-', color='black', alpha=0.95, label='Mean PRC'
    )
    ax_mean.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax_mean.axvline(0.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax_mean.set_xlim(-0.5, 0.5)
    ax_mean.set_xlabel('Firefly - LED Phase (- firefly before, + firefly after)')
    ax_mean.set_ylabel('Period change (T_next / T_prev)')
    ax_mean.set_title('PRC, average over all LED frequencies')
    ax_mean.grid(True, alpha=0.3)
    fig_mean.tight_layout()
    if SAVE_DIR:
        fig_mean.savefig(Path(SAVE_DIR) / "prc_ratio_mean.png", dpi=200)

    if SAVE_DIR:
        out_dir = Path(SAVE_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        combined_path = out_dir / "prc_ratio_all.csv"
        with combined_path.open("w", newline="") as f_all:
            w_all = csv.writer(f_all)
            w_all.writerow(["led_ms", "phase", "ratio", "sem", "n"])
            for freq in sorted(all_results.keys(), key=lambda x: float(x)):
                rec = all_results[freq]
                per_path = out_dir / f"prc_ratio_{int(float(freq))}ms.csv"
                with per_path.open("w", newline="") as f_one:
                    w_one = csv.writer(f_one)
                    w_one.writerow(["phase", "ratio", "sem", "n"])
                    for p, r, s, nn in zip(rec["x"], rec["y"], rec["sem"], rec["n"]):
                        w_one.writerow([f"{p:.9f}", f"{r:.9f}", f"{s:.9f}", int(nn)])
                        w_all.writerow([int(float(freq)), f"{p:.9f}", f"{r:.9f}", f"{s:.9f}", int(nn)])

        print(f"[PRC] wrote CSVs to {out_dir}")

    return fig_mean, ax_mean, all_results


def print_sync_statistics(results):
    """Print summary statistics of synchronization analysis"""
    print("\nSynchronization Analysis Summary:")
    print("=" * 50)

    for freq, data in results.items():
        phase_diff = data['phase_diff']
        inter_flash = data['inter_flash']
        response_delay = data['response_delay']

        print(f"\n{freq} ms LED frequency:")
        print(f"  Data points: {len(phase_diff)}")
        print(f"  Phase difference range: {np.min(phase_diff):.1f} to {np.max(phase_diff):.1f} ms")
        print(f"  Mean inter-flash interval: {np.mean(inter_flash):.1f} ± {np.std(inter_flash):.1f} ms")
        print(f"  Mean response delay: {np.mean(response_delay):.1f} ± {np.std(response_delay):.1f} ms")
        print(f"  Mean flashes per bout: {data['flashes_per_bout']}")

        # Calculate correlation between phase and response
        correlation = np.corrcoef(phase_diff, response_delay)[0, 1]
        print(f"  Phase-response correlation: {correlation:.3f}")


def plot_hist_and_heatmap_per_led(
    LED_PERIODS_MS,
    heatmaps,                  # dict[p_ms] -> 3D array [len(K1), len(K2), len(BETA)]
    best_params,               # dict[p_ms] -> {"k1","k2","beta_scale","dist","hist"}
    exp_samples_by_led,        # dict[p_ms] -> 1D samples (ms)
    BINS_MS,                   # 1D array of bin edges (ms)
    K1_VALUES,                 # list/1D array of k1 grid (descending → 0)
    K2_VALUES,                 # list/1D array of k2 grid (increasing → 0)
    BETA_SCALES,               # list/1D array of β scales
    save_path=None,            # directory or None
    savefigs=False,            # save instead of show
):
    bin_edges = np.asarray(BINS_MS)
    bin_lefts = bin_edges[:-1]
    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for p_ms in LED_PERIODS_MS:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7.5, 7.8), sharex=False, constrained_layout=True
        )
        # -----------------------------
        # Top: Experimental vs Simulation histograms (bars)
        # -----------------------------
        # Experimental histogram (density)
        exp_ms = exp_samples_by_led[p_ms]
        # Clip to plotting window for fairness
        exp_ms_win = exp_ms[(exp_ms >= bin_edges[0]) & (exp_ms <= bin_edges[-1])]
        exp_hist, _ = np.histogram(exp_ms_win, bins=bin_edges, density=True)
        # Simulation histogram (already on same bins as density)
        bp = best_params[p_ms]
        sim_hist = bp.get("hist", None)
        # Plot experimental (blue) as bars
        ax_top.bar(
            bin_lefts, exp_hist, width=bin_widths,
            align="edge", alpha=0.5, color="C0", edgecolor="none",
            label="Experimental"
        )
        # Plot simulation (gray) as bars
        if sim_hist is not None and len(sim_hist) == len(exp_hist):
            ax_top.bar(
                bin_lefts, sim_hist, width=bin_widths,
                align="edge", alpha=0.5, color="0.4", edgecolor="none",
                label="Simulation (best)"
            )
        else:
            ax_top.text(
                0.5, 0.5, "No simulation histogram", ha="center", va="center",
                transform=ax_top.transAxes
            )
        ax_top.set_xlabel("ISI (ms)")
        ax_top.set_ylabel("Density")
        ax_top.set_title(
            f"LED {p_ms} ms — Best fit\n"
            f"k1={bp.get('k1', np.nan):.2f}, k2={bp.get('k2', np.nan):.2f}, "
            f"β={bp.get('beta_scale', np.nan):.2f}, Stat={bp.get('dist', np.nan):.3f}"
        )
        ax_top.legend(frameon=False)
        ax_top.grid(alpha=0.15, linestyle=":", linewidth=0.8)
        # -----------------------------
        # Bottom: Heatmap over (k1, k2) at the β that gives the overall best triple
        # -----------------------------
        cube = heatmaps[p_ms]  # shape [len(K1), len(K2), len(BETA)]
        if cube.ndim != 3 or not np.isfinite(cube).any():
            ax_bot.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_bot.transAxes)
            if savefigs and save_path:
                fname = f"{save_path}/led_{int(p_ms)}__hist_and_landscape.png"
                plt.savefig(fname, dpi=200)
                plt.close(fig)
            else:
                plt.show()
            continue

        # Find the single global best (k1*, k2*, β*)
        gi, gj, gk = np.unravel_index(np.nanargmin(cube), cube.shape)

        k1_star = float(K1_VALUES[gi])
        k2_star = float(K2_VALUES[gj])
        beta_star = float(BETA_SCALES[gk])

        # Take the β* slice for the whole heatmap
        slice_at_best_beta = cube[:, :, gk]  # distances at fixed β*

        im = ax_bot.imshow(
            slice_at_best_beta,
            origin="lower",
            aspect="auto",
            extent=[K2_VALUES[0], K2_VALUES[-1], K1_VALUES[0], K1_VALUES[-1]],
            cmap="viridis"
        )
        im.set_rasterized(True)
        cbar = fig.colorbar(im, ax=ax_bot, shrink=0.92, pad=0.01)
        cbar.set_label(f"Distance at β* = {beta_star:.2f}")

        # Mark the overall best point (k2*, k1*)
        ax_bot.plot(k2_star, k1_star, marker="*", ms=14, mfc="white", mec="black", mew=1.1)

        ax_bot.set_xlabel("k2")
        ax_bot.set_ylabel("k1")
        ax_bot.set_title(
            f"Fit landscape at β* (overall best): k1*={k1_star:.2f}, k2*={k2_star:.2f}, β*={beta_star:.2f}"
        )
        ax_bot.set_xticks(K2_VALUES)
        ax_bot.set_yticks(K1_VALUES)
        ax_bot.grid(alpha=0.12, linestyle=":", linewidth=0.8)
        # Save or show
        if savefigs and save_path:
            fname = f"{save_path}/led_{int(p_ms)}__hist_and_landscape.png"
            plt.savefig(fname, dpi=200)
            plt.close(fig)
        else:
            plt.show()


##### ALL R AND Z STUFF BELOW


# ----------------- Phase helpers -----------------
def wrap01(x):
    x = np.asarray(x, float)
    return x - np.floor(x)


def wrap_half(x):
    x = np.asarray(x, float)
    return ((x + 0.5) % 1.0) - 0.5


def ratio_to_Z_samples(phi_eval_bins, R_mean, w, clip_instant=True):
    """Invert R = (1 - (phi_pre + Z)) / (1 - phi_pre) at bin centers."""
    phi_pre = wrap01(phi_eval_bins + w)
    Z = (1.0 - phi_pre) * (1.0 - np.asarray(R_mean, float))
    if clip_instant:
        Z = np.maximum(Z, -phi_pre + (-1e-9))
    return Z, phi_pre


def hat_factory(name, **kw):
    if name == "cos":
        def hat(u):
            u = np.asarray(u, float)
            out = np.zeros_like(u)
            m = (np.abs(u) <= 1.0)
            out[m] = 0.5 * (1.0 + np.cos(np.pi * u[m]))
            return out

        return hat

    elif name == "tri":
        def hat(u):
            u = np.asarray(u, float)
            return np.clip(1.0 - np.abs(u), 0.0, 1.0)

        return hat

    elif name == "poly":
        p = float(kw.get("p", 2.0))
        assert p >= 1.0

        def hat(u):
            u = np.asarray(u, float)
            out = np.zeros_like(u)
            m = (np.abs(u) <= 1.0)
            out[m] = (1.0 - (u[m] ** 2)) ** p
            return out

        return hat

    elif name == "bump":
        def hat(u):
            u = np.asarray(u, float)
            out = np.zeros_like(u)
            m = (np.abs(u) < 1.0)
            z = 1.0 - (u[m] ** 2)
            out[m] = np.exp(-1.0 / z)
            out[m] /= out[m].max() if out[m].size else 1.0
            return out

        return hat


# ----------------- Z with pluggable hat -----------------
def Z_param_hat(phi, beta, rho, k1, k2, hat_fn):
    # Here phi should be the *unwrapped* phase in cycles (not wrap_half!)
    d = np.asarray(phi, float)
    right = hat_fn(d / k1) * (d > 0.0)
    left = hat_fn(d / abs(k2)) * (d < 0.0)
    return beta * (right - rho * left)


# ----------------- Fitting -----------------
@dataclass
class FitResultZHat:
    hat_name: str
    hat_params: dict
    beta: float;
    rho: float;
    k1: float;
    k2: float;
    w: float
    success: bool;
    cost: float
    bic: float


def fit_lobes_on_Z_with_hat(phi_eval_bins, R_mean, counts,
                            hat_name="cos", hat_params=None,
                            beta_init=0.06, rho_init=1.2, k1_init=0.10, k2_init=-0.12, w_init=0.0,
                            ridge_lambda=0.0):
    hat_params = hat_params or {}
    hat_fn = hat_factory(hat_name, **hat_params)

    phi_eval_bins = np.asarray(phi_eval_bins, float)
    weights = np.ones_like(phi_eval_bins) if counts is None else np.asarray(counts, float)

    # Bounds (adjust as needed)
    lb = np.array([0.0, 0.5, 0.05, -0.50, -0.25])  # beta, rho, k1, k2, w
    ub = np.array([1.0, 4.0, 0.300, -0.05, 0.25])
    x0 = np.array([beta_init, rho_init, k1_init, k2_init, w_init], float)

    def resid(x):
        beta, rho, k1, k2, w = x
        Z_obs, _ = ratio_to_Z_samples(phi_eval_bins, R_mean, w, clip_instant=True)
        Z_hat = Z_param_hat(wrap_half(phi_eval_bins), beta, rho, k1, k2, hat_fn)
        r = Z_hat - Z_obs
        if ridge_lambda > 0:
            reg = np.sqrt(ridge_lambda) * (x[:4] / np.array([0.1, 1.0, 0.1, 0.1]))
            return np.concatenate([np.sqrt(weights) * r, reg])
        return np.sqrt(weights) * r

    res = least_squares(resid, x0, bounds=(lb, ub), method="trf",
                        loss="soft_l1", f_scale=0.15,
                        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=5000)

    # Weighted RSS for BIC (use residuals w/o regularization part)
    r = resid(res.x)
    if r.ndim > 1:
        r = r[:phi_eval_bins.size]
    rss = float(np.sum(r ** 2))
    n = phi_eval_bins.size
    p = 5  # parameters: beta, rho, k1, k2, w
    bic = n * np.log(rss / max(n, 1)) + p * np.log(max(n, 1))

    beta, rho, k1, k2, w = res.x
    return FitResultZHat(hat_name, hat_params, beta, rho, k1, k2, w, res.success, res.cost, bic)


# ---- Wrapped step–tanh PRC  (type-0 style) ----
def Z_tanh_step(phi, A, y0, kappa, phi_c):
    """
    phi: evaluation phase in cycles (use wrap_half(phi) upstream)
    Returns Δφ(φ) = wrap_{1/2}( y0 + A * tanh(kappa*(φ - φ_c)) )
    where wrap_half(x) ∈ (-1/2, 1/2].
    A  : half jump height (total step = 2A)
    y0 : vertical midpoint of the unwrapped curve
    kappa > 0 : steepness
    phi_c : center of the step (in cycles)
    """
    return wrap_half(y0 + A * np.tanh(kappa * (phi - phi_c)))


from dataclasses import dataclass

@dataclass
class FitResultTanhStep:
    model: str   # "tanhstep"
    A: float; y0: float; kappa: float; phi_c: float
    success: bool; cost: float; bic: float

def fit_tanh_step(phi_eval_bins, R_mean, counts,
                  A_init=0.10, y0_init=0.00, kappa_init=12.0, phi_c_init=0.0,
                  ridge_lambda=0.0):
    """class FitResultTanhStep:
    model: str   # "tanhstep"
    A: float; y0: float; kappa: float; phi_c: float
    success: bool; cost: float; bic: float
    Least-squares fit in Z-space using the wrapped step–tanh PRC.
    Bounds are conservative; adjust if your species shows larger shifts.
    """
    phi_eval_bins = np.asarray(phi_eval_bins, float)
    weights = np.ones_like(phi_eval_bins) if counts is None else np.asarray(counts, float)

    # Parameter vector: x = [A, y0, kappa, phi_c, w]
    # Reasonable bounds for firefly PRCs
    lb = np.array([ 0.00, -0.20,   2.0, -0.15])  # A ∈ [0,0.3], y0 bias, kappa≥2, center/window
    ub = np.array([ 0.30,  0.20,  80.0,  0.15])
    x0 = np.array([A_init, y0_init, kappa_init, phi_c_init], float)

    def resid(x):
        A, y0, kappa, phi_c, w = x
        # observed Z from data given w (exactly like your other fits)
        Z_obs, _ = ratio_to_Z_samples(phi_eval_bins, R_mean, w=0.0, clip_instant=True)
        # model Z on wrapped phase
        Z_hat = Z_tanh_step(wrap_half(phi_eval_bins), A, y0, kappa, phi_c)
        r = Z_hat - Z_obs
        if ridge_lambda > 0:
            # tiny ridge on (A,y0,kappa,phi_c) to discourage extremes (scale roughly)
            scale = np.array([0.1, 0.1, 20.0, 0.05])
            reg = np.sqrt(ridge_lambda) * (x[:4] / scale)
            return np.concatenate([np.sqrt(weights)*r, reg])
        return np.sqrt(weights) * r

    res = least_squares(resid, x0, bounds=(lb, ub), method="trf",
                        loss="soft_l1", f_scale=0.15,
                        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=5000)

    # BIC using the residuals part only (exclude reg tail if present)
    r = resid(res.x)
    if r.ndim > 1:
        r = r[:phi_eval_bins.size]
    rss = float(np.sum(r**2))
    n = phi_eval_bins.size
    p = 5  # A, y0, kappa, phi_c, w
    bic = n*np.log(rss / max(n,1)) + p*np.log(max(n,1))

    A, y0, kappa, phi_c, w = res.x
    return FitResultTanhStep("tanhstep", A, y0, kappa, phi_c, res.success, res.cost, bic)


def Z_model_dispatch(model_name, params_dict_or_fit, phi_wrapped):
    """
    Returns Z(φ) for any model at wrapped phases phi_wrapped ∈ (-1/2, 1/2].
    - For "cos" / "tri" (hat-based), expects FitResultZHat (beta,rho,k1,k2) and hat_params.
    - For "tanhstep", expects FitResultTanhStep (A,y0,kappa,phi_c).
    """
    if model_name in ("cos", "tri"):
        fit = params_dict_or_fit
        hat_fn = hat_factory(fit.hat_name, **fit.hat_params)
        return Z_param_hat(phi_wrapped, fit.beta, fit.rho, fit.k1, fit.k2, hat_fn)
    elif model_name == "tanhstep":
        fit = params_dict_or_fit
        return Z_tanh_step(phi_wrapped, fit.A, fit.y0, fit.kappa, fit.phi_c)
    else:
        raise ValueError(f"Unknown model {model_name}")


def forward_R_binavg_any(fit, centers, r_max=2.0):
    """
    Bin-averaged forward model R(φ) for any fit type.
    - centers: φ grid in cycles
    """
    model_name = getattr(fit, "hat_name", getattr(fit, "model", None))
    if model_name is None:
        raise ValueError("Unrecognized fit object.")
    centers = np.asarray(centers, float)
    bw = np.median(np.diff(np.sort(centers)))
    half = 0.5*bw
    offs = np.linspace(-half, half, 21)
    out = np.empty_like(centers)
    for i, c in enumerate(centers):
        samp = c + offs
        phi_pre = wrap01(samp + fit.w)
        Z_vals  = Z_model_dispatch(model_name, fit, wrap_half(samp))
        num = 1.0 - (phi_pre + Z_vals)
        den = 1.0 - phi_pre
        eps = 1e-12
        R_s = np.maximum(num, 0.0) / np.maximum(den, eps)
        R_s = np.minimum(R_s, r_max)
        out[i] = R_s.mean()
    return out


def tanhjump_hat_factory(s=12.0, delta0=0.01):
    """
    Tanh 'jump' hat on normalized distance a=|u|, u=d/k.
    - dead zone |u|<=delta0 -> 0
    - peak = 1 just outside the dead zone (t=0)
    - tapers to 0 at |u|=1 (t=1)
    """

    def hat(u):
        u = np.asarray(u, float)
        a = np.abs(u)
        out = np.zeros_like(a)

        m = (a > delta0) & (a <= 1.0)
        if np.any(m):
            # map a in (delta0,1] -> t in (0,1]
            denom_range = max(1e-12, 1.0 - delta0)
            t = (a[m] - delta0) / denom_range
            # normalized tanh: h(0)=1, h(1)~0
            denom_tanh = np.tanh(s) if s > 1e-12 else 1.0
            base = 1.0 - (np.tanh(s * t) / denom_tanh)
            out[m] = np.clip(base, 0.0, 1.0)
        return out

    return hat


# Forward for R using bin-average (as in your plotting routine)
def forward_R_binavg(fit: FitResultZHat, centers, r_max=2.0):
    hat_fn = hat_factory(fit.hat_name, **fit.hat_params)

    def Z_func(phi_eval_wrapped):
        return Z_param_hat(phi_eval_wrapped, fit.beta, fit.rho, fit.k1, fit.k2, hat_fn)

    centers = np.asarray(centers, float)
    bw = np.median(np.diff(np.sort(centers)))
    half = 0.5 * bw
    offs = np.linspace(-half, half, 21)
    out = np.empty_like(centers)
    for i, c in enumerate(centers):
        samp = c + offs
        phi_pre = wrap01(samp + fit.w)
        Z_vals = Z_func(samp)
        num = 1.0 - (phi_pre + Z_vals)
        den = 1.0 - phi_pre
        eps = 1e-12
        R_s = np.maximum(num, 0.0) / np.maximum(den, eps)
        R_s = np.minimum(R_s, r_max)
        out[i] = R_s.mean()
    return out


# ----------------- Data loading (synthetic fallback) -----------------
def load_led_data_from_csv(save_dir,
                           led_periods_ms=None,
                           r_min=0.0, r_max=2.0,
                           phase_col="phase",
                           ratio_col="ratio",
                           count_col="n",
                           led_col_candidates=("led_ms", "led", "LED", "led_period_ms")):
    """
    Returns:
      led_periods_ms : list[int]
      data_per_led   : dict[int] -> (phi_bins, R_mean, counts)

    CSV expected columns:
      - phase bins      (default: 'phase')
      - ratio values    (default: 'ratio')
      - optional counts (default: 'n')
      - optional LED id (one of led_col_candidates); if present, groups by LED.

    Behavior:
      - If an LED column exists, group by it and sort each group by phase.
      - If no LED column exists, replicate same (phi, R, n) for each LED in `led_periods_ms`.
        If `led_periods_ms` is None, makes a single-key dict using 0 (or raise).
    """
    csv_path = Path(save_dir) / "prc_ratio_all.csv"
    df = pd.read_csv(csv_path)

    # Find LED column if present
    led_col = None
    for c in led_col_candidates:
        if c in df.columns:
            led_col = c
            break

    def _clean_triplet(frame):
        # Extract and clean
        phi_bins = frame[phase_col].to_numpy(float)
        R_mean = frame[ratio_col].to_numpy(float)
        if count_col in frame.columns:
            counts = frame[count_col].to_numpy(float)
        else:
            counts = np.ones_like(R_mean, dtype=float)

        # Replace non-finite ratios with 1.0 and clip range
        R_mean = np.where(np.isfinite(R_mean), R_mean, 1.0)
        R_mean = np.clip(R_mean, r_min, r_max)

        # Sort by phase (important for downstream plotting)
        order = np.argsort(phi_bins)
        phi_bins = phi_bins[order]
        R_mean = R_mean[order]
        counts = counts[order]

        # Ensure 1D contiguous arrays
        return np.ascontiguousarray(phi_bins, float), \
            np.ascontiguousarray(R_mean, float), \
            np.ascontiguousarray(counts, float)

    data_per_led = {}
    out_leds = []

    if led_col is not None:
        # Group by LED period from the CSV
        for led_val, g in df.groupby(led_col):
            try:
                ms = int(round(float(led_val)))
            except Exception:
                # If LED values aren't numeric, skip group
                continue
            phi_bins, R_mean, counts = _clean_triplet(g)
            data_per_led[ms] = (phi_bins, R_mean, counts)
            out_leds.append(ms)

        # Sort LED list for consistency
        led_periods_ms = sorted(out_leds)

    else:
        # No LED column: use provided list, or fall back to a single dummy key
        phi_bins, R_mean, counts = _clean_triplet(df)

        if led_periods_ms is None:
            # If you truly only have one dataset and no LED list, keep a single entry
            led_periods_ms = [0]  # or raise ValueError("Please pass led_periods_ms")
        for ms in led_periods_ms:
            data_per_led[int(ms)] = (phi_bins.copy(), R_mean.copy(), counts.copy())

    return led_periods_ms, data_per_led


# ----------------- Fitting across LEDs & pooled mean -----------------
def fit_models_across_leds(led_periods_ms, data_per_led, models):
    fits = {m: {} for m in models}
    # Per-LED
    for m in models:
        fits[m]['per_led'] = {}
        for ms in led_periods_ms:
            phi, Rm, cnt = data_per_led[ms]
            if m in ("tri", "cos"):
                fit = fit_lobes_on_Z_with_hat(phi, Rm, cnt, hat_name=m,
                                              hat_params={}, ridge_lambda=1e-6)
            elif m == "tanhstep":
                fit = fit_tanh_step(phi, Rm, cnt, ridge_lambda=1e-6)
            else:
                raise ValueError(f"Unknown model {m}")
            fits[m]['per_led'][ms] = fit
    # Pooled (stack bins; same approach as you had)
    for m in models:
        phis, Rs, Cs = [], [], []
        for ms in led_periods_ms:
            phi, Rm, cnt = data_per_led[ms]
            phis.append(phi); Rs.append(Rm); Cs.append(cnt)
        phi_all = np.concatenate(phis)
        R_all   = np.concatenate(Rs)
        C_all   = np.concatenate(Cs)
        if m in ("tri", "cos"):
            fits[m]['pooled'] = fit_lobes_on_Z_with_hat(phi_all, R_all, C_all,
                                                        hat_name=m, hat_params={}, ridge_lambda=1e-6)
        elif m == "tanhstep":
            fits[m]['pooled'] = fit_tanh_step(phi_all, R_all, C_all, ridge_lambda=1e-6)
    return fits


# ----------------- Visualization -----------------
def plot_R_and_Z_for_led(ms, data_per_led, fits_for_models, outdir):
    """
    Creates two figures for LED `ms`:
      1) R(phi): data vs model curves for each model
      2) Z(phi): model Z-curves for each model
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    phi, Rm, cnt = data_per_led[ms]
    grid = np.linspace(phi.min(), phi.max(), 600)

    # Figure 1: R(phi)
    plt.figure(figsize=(8, 5))
    plt.plot(phi, Rm, '.', alpha=0.7, label='Data')
    for model, fit in fits_for_models.items():
        y = forward_R_binavg_any(fit, grid)
        plt.plot(grid, y, lw=2, label=model)
    plt.axhline(1.0, ls='--')
    plt.xlabel(r'Phase $\phi$ (cycles)')
    plt.ylabel(r'$R(\phi)$')
    plt.title(f'R(phi) — LED {ms} ms')
    plt.legend()
    f1 = f'{outdir}/Rphi_LED{ms}.png'
    plt.tight_layout();
    plt.savefig(f1, dpi=300);
    plt.close()

    # Figure 2: Z(phi)
    plt.figure(figsize=(8, 5))
    for model, fit in fits_for_models.items():
        hat_fn = hat_factory(fit.hat_name, **fit.hat_params)
        Zg = Z_model_dispatch(getattr(fit, "hat_name", getattr(fit, "model", None)),
                              fit,
                              wrap_half(grid))
        plt.plot(grid, Zg, lw=2, label=model)
    plt.axhline(0.0, ls='--')
    plt.xlabel(r'Phase $\phi$ (cycles)')
    plt.ylabel(r'$Z(\phi)$')
    plt.title(f'Z(phi) — LED {ms} ms')
    plt.legend()
    f2 = f'{outdir}/Zphi_LED{ms}.png'
    plt.tight_layout();
    plt.savefig(f2, dpi=300);
    plt.close()

    return f1, f2


def plot_R_and_Z_pooled_mean(led_periods_ms, data_per_led, fits, outdir,
                             n_ref=600,  # points on the common reference grid
                             min_overlap_frac=0.8):  # require each LED to cover >=80% of common span
    """
    Build a common reference grid from the intersection of LED phase ranges,
    interpolate each LED's R(phi) onto it, average, and plot.
    No np.allclose calls; robust to differing bin counts/locations.
    """
    from pathlib import Path
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # 1) Collect per-LED phase ranges (after sorting & dedup)
    spans = []
    cleaned = {}
    for ms in led_periods_ms:
        phi, Rm, cnt = data_per_led[ms]
        phi = np.asarray(phi, float)
        Rm = np.asarray(Rm, float)

        # Drop NaNs/Infs
        good = np.isfinite(phi) & np.isfinite(Rm)
        phi, Rm = phi[good], Rm[good]

        # Sort and deduplicate phases (keep the first occurrence)
        order = np.argsort(phi)
        phi, Rm = phi[order], Rm[order]
        phi_u, idx = np.unique(phi, return_index=True)
        Rm = Rm[idx]
        phi = phi_u

        # Store cleaned series
        cleaned[ms] = (phi, Rm)
        if phi.size >= 2:
            spans.append((phi.min(), phi.max()))

    if not spans:
        raise ValueError("No usable LED series found for pooling.")

    # 2) Build common reference grid on the INTERSECTION of ranges
    common_min = max(s[0] for s in spans)
    common_max = min(s[1] for s in spans)
    if not np.isfinite(common_min) or not np.isfinite(common_max) or common_max <= common_min:
        raise ValueError("LED phase ranges do not overlap sufficiently to build a common grid.")

    ref_phi = np.linspace(common_min, common_max, n_ref)

    # 3) Interpolate each LED onto ref grid if it covers enough of the span
    R_stack = []
    used_leds = []
    span_len = common_max - common_min
    for ms in led_periods_ms:
        phi, Rm = cleaned[ms]
        # compute overlap with common span
        led_min, led_max = (phi.min(), phi.max()) if phi.size else (np.inf, -np.inf)
        overlap = max(0.0, min(led_max, common_max) - max(led_min, common_min))
        if phi.size >= 2 and overlap >= min_overlap_frac * span_len:
            # interpolate (clip endpoints to avoid extrapolation warnings)
            R_interp = np.interp(ref_phi, phi, Rm)
            R_stack.append(R_interp)
            used_leds.append(ms)
        # else: skip this LED for the pooled average

    if len(R_stack) == 0:
        raise ValueError("No LED series met the overlap requirement for pooling.")

    R_mean = np.mean(np.vstack(R_stack), axis=0)

    # 4) Plot pooled R(phi) vs. model curves (pooled fits)
    plt.figure(figsize=(8, 5))
    plt.plot(ref_phi, R_mean, '.', alpha=0.8, label=f'Data mean (n={len(R_stack)} LEDs)')
    grid = np.linspace(ref_phi.min(), ref_phi.max(), 600)
    for model, rec in fits.items():
        y = forward_R_binavg_any(rec['pooled'], grid)
        plt.plot(grid, y, lw=2, label=model)
    plt.axhline(1.0, ls='--')
    plt.xlabel(r'Phase $\phi$ (cycles)');
    plt.ylabel(r'$R(\phi)$')
    plt.title('R(φ): mean across LEDs (pooled)')
    plt.legend()
    fR = f'{outdir}/Rphi_mean_across_LEDs.png'
    plt.tight_layout();
    plt.savefig(fR, dpi=300);
    plt.close()

    # 5) Plot Z(phi) from pooled fits over the same domain
    plt.figure(figsize=(8, 5))
    for model, rec in fits.items():
        fit = rec['pooled']
        hat_fn = hat_factory(fit.hat_name, **fit.hat_params)
        Zg = Z_model_dispatch(getattr(fit, "hat_name", getattr(fit, "model", None)),
                              fit,
                              wrap_half(grid))

        plt.plot(grid, Zg, lw=2, label=model)
    plt.axhline(0.0, ls='--')
    plt.xlabel(r'Phase $\phi$ (cycles)');
    plt.ylabel(r'$Z(\phi)$')
    plt.title('Z(φ): pooled fits (on common domain)')
    plt.legend()
    fZ = f'{outdir}/Zphi_pooled.png'
    plt.tight_layout();
    plt.savefig(fZ, dpi=300);
    plt.close()

    # Optional: quick debug printout
    print(f"[pool] used LEDs: {used_leds} | common span = [{common_min:.4f}, {common_max:.4f}] | ref n={n_ref}")

    return fR, fZ


def r_and_z(datafile='figs'):
    led_periods_ms, data_per_led = load_led_data_from_csv(datafile)

    models = ["tri", "cos", "tanhjump"]

    fits = fit_models_across_leds(led_periods_ms, data_per_led, models=models)

    out_root = Path("sim_data/impulse_comparison")
    out_root.mkdir(parents=True, exist_ok=True)

    generated = []
    for ms in led_periods_ms:
        files = plot_R_and_Z_for_led(
            ms,
            data_per_led,
            {m: fits[m]['per_led'][ms] for m in models},
            outdir=str(out_root / f"LED_{ms}")
        )
        generated.extend(files)

    mean_files = plot_R_and_Z_pooled_mean(
        led_periods_ms, data_per_led, fits, outdir=str(out_root / "pooled")
    )
    generated.extend(mean_files)


def plot_R_and_Z_for_led_single(ms, data_per_led, fit, outdir):
    """Single-model R and Z plots for one LED."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    phi, Rm, _ = data_per_led[ms]
    grid = np.linspace(phi.min(), phi.max(), 600)

    # R(phi)
    plt.figure(figsize=(8, 5))
    plt.scatter(phi, Rm, c='k', s=20, alpha=0.2, label='Data')
    y = forward_R_binavg_any(fit, grid)
    plt.plot(grid, y, lw=2, label=f'{fit.hat_name}')
    plt.axhline(1.0, ls='--', color='gray')
    plt.xlabel(r'Phase $\phi$'); plt.ylabel(r'$R(\phi)$')
    plt.title(f'R(φ) — LED {ms} ms — {fit.hat_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'Rphi_LED{ms}_{fit.hat_name}.png', dpi=300)
    plt.close()

    # Z(phi)
    plt.figure(figsize=(8, 5))
    hat_fn = hat_factory(fit.hat_name, **fit.hat_params)
    Zg = Z_model_dispatch(getattr(fit, "hat_name", getattr(fit, "model", None)),
                          fit,
                          wrap_half(grid))
    plt.plot(grid, Zg, lw=2, label=f'{fit.hat_name}')
    plt.axhline(0.0, ls='--', color='gray')
    plt.xlabel(r'Phase $\phi$'); plt.ylabel(r'$Z(\phi)$')
    plt.title(f'Z(φ) — LED {ms} ms — {fit.hat_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'Zphi_LED{ms}_{fit.hat_name}.png', dpi=300)
    plt.close()


def plot_R_and_Z_pooled_single(led_periods_ms, data_per_led, fit, outdir):
    """Single-model pooled R and Z plots."""
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Build pooled mean
    phis, Rs = [], []
    for ms in led_periods_ms:
        phi, Rm, _ = data_per_led[ms]
        phis.append(phi); Rs.append(Rm)
    ref_phi = np.linspace(max([p.min() for p in phis]),
                          min([p.max() for p in phis]),
                          600)
    R_interp = [np.interp(ref_phi, p, r) for p, r in zip(phis, Rs)]
    R_mean = np.mean(R_interp, axis=0)

    # R(phi)
    plt.figure(figsize=(8,5))
    plt.scatter(np.concatenate(phis), np.concatenate(Rs),
                c='k', s=5, alpha=0.1, label='Data')
    y = forward_R_binavg_any(fit, ref_phi)
    plt.plot(ref_phi, y, lw=2, label=f'{fit.hat_name}')
    plt.axhline(1.0, ls='--', color='gray')
    plt.xlabel(r'Phase $\phi$'); plt.ylabel(r'$R(\phi)$')
    plt.title(f'Pooled R(φ) — {fit.hat_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'Rphi_pooled_{fit.hat_name}.png', dpi=300)
    plt.close()

    # Z(phi)
    plt.figure(figsize=(8,5))
    hat_fn = hat_factory(fit.hat_name, **fit.hat_params)
    Zg = Z_param_hat(wrap_half(ref_phi), fit.beta, fit.rho, fit.k1, fit.k2, hat_fn)
    plt.plot(ref_phi, Zg, lw=2, label=f'{fit.hat_name}')
    plt.axhline(0.0, ls='--', color='gray')
    plt.xlabel(r'Phase $\phi$'); plt.ylabel(r'$Z(\phi)$')
    plt.title(f'Pooled Z(φ) — {fit.hat_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'Zphi_pooled_{fit.hat_name}.png', dpi=300)
    plt.close()

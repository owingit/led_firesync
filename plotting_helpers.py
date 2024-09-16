import itertools
import pickle
import scipy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from scipy import stats


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
    colors = plt.cm.Spectral(np.linspace(0, 1, 24))  # Spectral colormap with 8 colors

    alphas = [0.6, 0.9]
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Key', y='Values', hue='State', data=df_melted, showfliers=False, hue_order=['before', 'induced'])
    plt.title('Firefly periods prior to and following LED introduction')

    for i, key in enumerate(sorted(data.keys(), key=lambda x: int(x))):
        key = int(key)
        state1_values = df_melted[(df_melted['Key'] == key) & (df_melted['State'] == 'before')]['Values']
        state2_values = df_melted[(df_melted['Key'] == key) & (df_melted['State'] == 'induced')]['Values']

        paired_ttest = stats.mannwhitneyu(state1_values, state2_values)
        print(paired_ttest)
        if paired_ttest.pvalue < 0.05:
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
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.legend([], [], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(plot_params.save_folder + '/boxplots.png')


def before_vs_mode_freq(rmses, plot_params):
    fig, ax = plt.subplots()
    colormap = cm.get_cmap('Spectral', 21)
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
    colormap = cm.get_cmap('Spectral', 21)
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



def plot_statistics(rmses, ks, plot_params):
    all_befores = {}
    all_afters = {}

    colormap = cm.get_cmap('Spectral', len(ks) * 3)

    if plot_params.do_delay_plot:
        fig, axes = plt.subplots(8)
        for i, k in enumerate(ks):
            all_delays = []
            all_responses = []
            for individual in rmses['led_ff_diffs'][k]:
                delays = [i[0] for i in individual]
                period_changes = [i[1] for i in individual]
                all_delays.extend(delays)
                all_responses.extend(period_changes)

            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            if i == 4:
                color = 'yellow'
            else:
                color = colormap.__call__(i * 3)
            axes[i].scatter(all_delays, all_responses, color=color)

            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(-0.5, 0.5)
            axes[i].set_ylim(0.0, 5.0)
        axes[int(len(ks) / 2)].set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/delay_distributions')
        plt.close(fig)

        fig, axes = plt.subplots(8)
        for i, k in enumerate(ks):
            all_delays = []
            for individual in rmses['led_ff_diffs'][k]:
                delays = [i[0] for i in individual]
                try:
                    all_delays.extend(delays)
                except TypeError:
                    all_delays.append(delays)

            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            if i == 4:
                color = 'yellow'
            else:
                color = colormap.__call__(i * 3)
            axes[i].hist(all_delays, density=True, bins=np.arange(-0.5, 0.5, 0.03), color=color)

            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(-0.5, 0.5)
            axes[i].set_ylim(0.0, 5.0)
        axes[int(len(ks) / 2)].set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/delay_distributions_hist')
        plt.close(fig)

    if plot_params.do_windowed_period_plot:
        fig, axes = plt.subplots(8)

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
                except TypeError:
                    all_befores[k].append(individual)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

            all_befores[k] = [x for x in all_befores[k] if not np.isnan(x)]
            all_afters[k] = [x for x in all_afters[k] if not np.isnan(x)]
            if i == 4:
                color = 'yellow'
            else:
                color = colormap.__call__(i * 3)

            #####

            #  here I want two plots:
            #  difference between induced mean/variance (all_afters) and natural mean/variance (all_befores)
            #  difference between induced mean/variance (all_afters) and k/1000

            #####

            axes[i].hist(all_befores[k], density=True, bins=np.arange(0.0, 3.0, 0.03), color='black', alpha=0.25)
            axes[i].hist(all_afters[k], density=True, bins=np.arange(0.0, 3.0, 0.03), color=color, alpha=0.75)
            if i != 0:
                axes[i].axvline(float(k) / 1000, color='black')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--')
            else:
                axes[i].axvline(float(k) / 1000, color='black', label='LED period')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--', label='0.5 * LED period')
                axes[i].legend()
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(0.0, 1.5)
            axes[i].set_ylim(0.0, 6.0)
        axes[len(ks)-1].set_xlabel('T[s]')
        axes[int(len(ks) / 2)].set_ylabel('pdf')
        if plot_params.save_data:
            with open(plot_params.save_folder+'all_befores_from_experiments.pickle', 'wb') as handle:
                pickle.dump(all_befores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(plot_params.save_folder+'all_afters_from_experiments.pickle', 'wb') as handle:
                pickle.dump(all_afters, handle, protocol=pickle.HIGHEST_PROTOCOL)
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_windowed_period_2024_beforeafter')
        plt.close(fig)
        if plot_params.do_boxplots:
            boxplots(all_befores, all_afters, plot_params)

        fig, axes = plt.subplots(8)
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
                except TypeError:
                    all_befores[k].append(individual)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            all_befores[k] = [x for x in all_befores[k] if not np.isnan(x)]
            all_afters[k] = [x for x in all_afters[k] if not np.isnan(x)]
            if i == 4:
                color = 'yellow'
            else:
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
                axes[i].legend()
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(0.0, 2.05)
            axes[i].set_ylim(0.0, 6.0)
        axes[len(ks) - 1].set_xlabel('T[s]')
        axes[int(len(ks) / 2)].set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_period_2024_windowed_beforeafter_w_lines')
        plt.close(fig)

        fig, axes = plt.subplots(len(ks), figsize=(8, 6))

        all_befores = {}
        all_afters = {}
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
                except TypeError:
                    all_befores[k].append(individual)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            if i == 4:
                color = 'yellow'
            else:
                color = colormap.__call__(i * 3)
            axes[i].hist(all_afters[k], density=True, bins=np.arange(0.0, 3.0, 0.03), color=color)
            if i != 0:
                axes[i].axvline(float(k) / 1000, color='black')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--')
            else:
                axes[i].axvline(float(k) / 1000, color='black', label='LED period')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--', label='0.5 * LED period')
                axes[i].legend()
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(0.0, 1.5)
            axes[i].set_ylim(0.0, 6.0)
        axes[len(ks) - 1].set_xlabel('T[s]')
        axes[int(len(ks) / 2)].set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_windowed_period_2024_after')
        plt.close(fig)

        fig, axes = plt.subplots(len(ks), figsize=(8, 6))

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
            if i == 4:
                color = 'yellow'
            else:
                color = colormap.__call__(i * 3)
            axes[i].hist(all_afters[k], density=True, bins=np.arange(0.0, 3.0, 0.03), color=color)
            if i != 0:
                axes[i].axvline(float(k) / 1000, color='black')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--')
            else:
                axes[i].axvline(float(k) / 1000, color='black', label='LED period')
                axes[i].axvline(0.5 * (float(k) / 1000), color='black', linestyle='--', label='0.5 * LED period')
                axes[i].legend()
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(0.0, 1.5)
            axes[i].set_ylim(0.0, 6.0)
        axes[len(ks) - 1].set_xlabel('T[s]')
        axes[int(len(ks) / 2)].set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_windowed_period_2024_w_aggregatebefore_after')
        plt.close(fig)

    if plot_params.do_recurrence_diagrams:
        for i, k in enumerate(ks):
            fig, ax = plt.subplots()
            all_recurrences_first = []
            all_recurrences_second = []
            all_recurrences_third = []
            for individual in rmses['windowed_period_after'][k]:
                try:
                    for j in range(len(individual) - 2):
                        all_recurrences_first.append(individual[j])
                        all_recurrences_second.append(individual[j + 1])
                        all_recurrences_third.append(individual[j + 2])
                except TypeError:
                    continue

            h = ax.hist2d(np.array(all_recurrences_first, dtype=np.float64),
                          np.array(all_recurrences_second, dtype=np.float64), norm=matplotlib.colors.LogNorm(),
                          bins=(np.arange(0.0, 2.0, (1 / 30)), np.arange(0.0, 2.0, (1 / 30))), density=True)

            ax.set_xlabel('Period time t')
            ax.set_ylabel('Period time t+1')
            ax.set_title(k)
            ax.set_facecolor(cm.get_cmap('viridis', 100).__call__(0))
            plt.colorbar(h[3])
            plt.savefig(plot_params.save_folder + '/recurrences2d_{}_2024'.format(k))
            plt.close(fig)

    if plot_params.do_period_over_time:
        for i, k in enumerate(ks):
            fig, axes = plt.subplots()
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)

            colormap_1 = cm.get_cmap('winter', len(rmses['windowed_period_after'][k]) * 2)
            for q, j in enumerate(range(len(rmses['windowed_period_after'][k]))):
                try:
                    axes.plot([q + x for x in rmses['windowed_period_after'][k][j] if x > 0.2][::2], lw=1,
                              color=colormap_1.__call__(j + j))
                except ValueError:
                    continue
                except TypeError:
                    axes.plot(q + np.array(rmses['windowed_period_after'][k][j]), lw=1, color=colormap_1.__call__(j))
            axes.set_xlabel('Flash instance')
            axes.set_ylabel('Period + firefly instance')
            plt.title('LED freq ' + k + 'ms')
            plt.savefig(plot_params.save_folder + '/period_lines_over_time_{}'.format(k))
            plt.close(fig)

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

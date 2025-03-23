import itertools
import pickle
import scipy
import os
import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import json
import pandas as pd
import seaborn as sns
import plotly.io as pio
from matplotlib import cm
from plotly.subplots import make_subplots
from scipy import stats
from scipy import signal, optimize
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import scipy.interpolate
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.stats import gaussian_kde

import helpers


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


def plot_dfa_results(scales, fluctuations, alpha, confidence_interval, title="DFA Analysis"):
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
    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.2)

    return fig, ax



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


def write_timeseries_figs(pargs):
    #
    # Write the timeseries figures path objects
    # Interactive timeseries plots for any given day - the raw data

    fpaths = os.listdir(pargs.data_path)
    for fp in fpaths:
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
                _, _, pairs = helpers.compute_phase_response_curve(time_series_led=led_xs_flashes,
                                                                   time_series_ff=ff_xs_flashes,
                                                                   epsilon=0.035,  # one frame
                                                                   do_responses_relative_to_ff=pargs.do_ffrt)

                times, phases, responses = zip(*[(t, p, r) for p, r, t in pairs])

                phase_derivative = helpers.sliding_time_window_derivative(times, phases, window_seconds=3.0)

                valid_indices = [i for i, val in enumerate(phase_derivative) if val is not None]
                valid_times = [times[i] for i in valid_indices]
                valid_derivatives = [phase_derivative[i] for i in valid_indices]
                phase_acceleration = helpers.sliding_time_window_derivative(valid_times, valid_derivatives,
                                                                            window_seconds=3.0)
                valid_indices_2nd = [i for i, val in enumerate(phase_acceleration) if val is not None]
                valid_times_2nd = [valid_times[i] for i in valid_indices_2nd]
                valid_accelerations = [phase_acceleration[i] for i in valid_indices_2nd]

                date_str = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
                title_text = f"<br>LED+FF Experiment<br>Date: {date_str}<br>LED Period: {key}ms<br>Temp: {temp}°C<br>Frame Rate: {round(1 / framerate, 3)} fps<br><br><br><br>"
                x_min = min(min(led_xs), min(ff_xs))
                x_max = max(max(led_xs), max(ff_xs))

                led_y_value = 0.505
                ff_y_value = 0.495

                trace1_led = go.Scatter(
                    x=led_xs[masked_led],
                    y=[led_y_value] * len(led_xs[masked_led]),
                    mode='markers',
                    name='LED',
                    marker=dict(color='orange', size=5)
                )

                trace1_ff = go.Scatter(
                    x=ff_xs[masked_ff],
                    y=[ff_y_value] * len(ff_xs[masked_ff]),
                    mode='markers',
                    name='Firefly',
                    marker=dict(color='green', size=5)
                )

                trace2 = go.Scatter(x=times, y=phases, mode='markers', name='Phase Differences',
                                    marker=dict(color='black', size=5))
                trace2_baseline = go.Scatter(x=[x_min, x_max], y=[0, 0], mode='lines',
                                             line=dict(color='blue', dash='dash', width=1), showlegend=False)
                trace3 = go.Scatter(x=times, y=responses, mode='markers', name='Firefly Period',
                                    marker=dict(color='red', size=5))
                trace3_baseline = go.Scatter(x=[x_min, x_max], y=[float(key) / 1000, float(key) / 1000], mode='lines',
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
                    name='Phase Second Derivative (Acceleration)',
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
                    width=800,
                    showlegend=False,
                    xaxis=dict(range=[x_min, x_max]),
                    xaxis2=dict(range=[x_min, x_max]),
                    xaxis3=dict(range=[x_min, x_max]),
                    xaxis4=dict(range=[x_min, x_max]),
                    yaxis=dict(
                        title="Signal presence",
                        tickmode='array',
                        tickvals=[ff_y_value, led_y_value],
                        ticktext=['Firefly', 'LED'],
                        range=[0.425, 0.575]
                    ),
                    yaxis2=dict(title="Phase difference"),
                    yaxis3=dict(title="Firefly period [s]"),
                    yaxis4=dict(title="Phase derivative"),
                    margin=dict(t=180, l=80, r=80, b=80),
                    title={
                        'text': title_text,
                        'y': 0.98,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    }
                )

                fig.update_xaxes(title_text="Time (s)", row=1, col=1, showticklabels=True)
                fig.update_xaxes(title_text="Time (s)", row=2, col=1, showticklabels=True)
                fig.update_xaxes(title_text="Time (s)", row=3, col=1, showticklabels=True)
                fig.update_xaxes(title_text="Time (s)", row=4, col=1, showticklabels=True)

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
                    output_path = f"{pargs.save_folder}/timeseries/LED_Period={key}ms/{date}_{key}_{index}_phase_ffrt.html"
                else:
                    output_path = f"{pargs.save_folder}/timeseries/LED_Period={key}ms/{date}_{key}_{index}_phase_rt.html"
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
                               marker=dict(color='orange')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(x=led_xs[masked_led], y=led_ys[masked_led] - 0.5, name='LED',
                           marker=dict(color='orange'), base=0.5),
                    row=1, col=1,
                )

                # Plot firefly
                fig.add_trace(
                    go.Scatter(x=ff_xs[masked_ff], y=ff_ys[masked_ff], name='FF', mode='markers',
                               marker=dict(color='green')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(x=ff_xs[masked_ff], y=ff_ys[masked_ff] - 0.498, name='FF',
                           marker=dict(color='green'), base=0.498),
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


def plot_statistics(rmses, ks, plot_params):
    all_befores = {}
    all_afters = {}

    colormap = cm.get_cmap('Spectral', len(ks) * 3)

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

            # Calculate mean and standard deviation
            mu = np.mean(all_delays)
            sigma = np.std(all_delays)

            # Generate Gaussian distribution
            gaussian = stats.norm.pdf(x, mu, sigma / 2)

            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

            # Plot the Gaussian
            if i == 4:
                color = 'yellow'
            else:
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

        colormap = cm.get_cmap('Spectral', len(ks) * 3)
        fig, ax = plt.subplots(8)

        for i, k in enumerate(ks):
            if i == 4:
                color = 'yellow'
            else:
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

        colormap = cm.get_cmap('Spectral', len(ks) * 3)
        fig, ax = plt.subplots(8)

        for i, k in enumerate(ks):
            if i == 4:
                color = 'xkcd:sun yellow'
            else:
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

            # # Add contour lines
            # xy = np.vstack([all_phases_, all_freqs_])
            # z = gaussian_kde(xy)(xy)
            #
            # # Sort points by density for better contour visualization
            # idx = z.argsort()
            # x, y, z = np.array(all_phases_)[idx], np.array(all_freqs_)[idx], z[idx]

            # Plot contours
            # ax[i].tricontour(x, y, z, levels=5, colors='black', linewidths=0.5, alpha=0.5)

            ax[i].set_ylim(0.0, 1.0)
            ax[i].tick_params(axis='y', colors='white')

        ax[len(ks) - 1].set_xlabel('Time delay between LED and forward firefly [s]', color='white')
        ax[int(len(ks) / 2) - 1].set_ylabel('Response period [s]', color='white')
        fig.set_facecolor(np.array([22, 22, 22]) / 255)

        plt.savefig(plot_params.save_folder + '/response_period_vs_delay')
        plt.close(fig)

        # delay and response time trajectories
        colormap = plt.cm.get_cmap('Spectral', len(ks)*3)

        # delay
        for i, k in enumerate(ks):
            fig, ax = plt.subplots(len(rmses['phase_time_diffs_instanced'][k]), figsize=(14, 16), sharex=True,
                                   gridspec_kw={'hspace': 0.4})
            color = colormap.__call__(i * 3)
            if i == 4:
                color = 'yellow'
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
                # ax[kk].set_ylim((-0.5 * (float(k)/1000)) - 0.02, ((float(k)/1000) * 0.5) + 0.02)

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

                if len(freqs) > 2:  # Ensure we have enough data points
                    times = np.array(times)
                    freqs = np.array(freqs)
                    ref_freq = float(k) / 1000

                    # Compute normalized differences
                    diffs = np.abs(freqs - ref_freq)
                    max_diff = np.max(diffs) if np.max(diffs) > 0 else 1
                    normalized_diffs = diffs / max_diff

                    alphas = 1.0 - 0.5 * normalized_diffs
                    ax[kk].scatter(times, freqs, color=color, alpha=alphas, s=10)
                ax[kk].axhline(float(k)/1000, color='white', linestyle='--', linewidth=1)
                # Axis formatting
                ax[kk].spines['top'].set_visible(False)
                ax[kk].spines['right'].set_visible(False)
                ax[kk].spines['bottom'].set_visible(True)
                ax[kk].spines['left'].set_visible(True)
                ax[kk].spines['bottom'].set_edgecolor('gray')
                ax[kk].spines['left'].set_edgecolor('gray')
                ax[kk].set_facecolor(np.array([22, 22, 22]) / 255)
                ax[kk].tick_params(axis='y', colors='gray')
                ax[kk].grid(False)
                # ax[kk].set_ylim(0.0, ((float(k)/1000) * 1.5) + 0.02)
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

            # Process the most common y-values for each x
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

            # Choose color
            color = 'yellow' if i == 4 else colormap.__call__(i * 3)

            # Plot the smooth envelope
            axes[i].plot(xx_, smooth_ys, color=color, linewidth=2)
            axes[i].fill_between(xx_, smooth_lower, smooth_upper, color='gray', alpha=0.25)

            # Hide x-axis except for the last subplot
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

            # Define bin edges
            bin_edges = np.arange(-0.5, 0.5 + bin_size, bin_size)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins

            # Compute mean and standard deviation in each bin
            binned_means = []
            binned_stds = []
            for j in range(len(bin_edges) - 1):
                mask = (xs >= bin_edges[j]) & (xs < bin_edges[j + 1])
                if np.sum(mask) > 0:
                    binned_means.append(np.mean(ys[mask]))
                    binned_stds.append(np.std(ys[mask]))
                else:
                    binned_means.append(np.nan)  # Empty bins remain NaN
                    binned_stds.append(np.nan)

            binned_means = np.array(binned_means)
            binned_stds = np.array(binned_stds)

            # Choose color
            color = 'yellow' if i == 4 else colormap.__call__(i * 3)

            # Adjust bar properties based on sign of mean shift
            for j in range(len(bin_centers)):
                if np.isnan(binned_means[j]):  # Skip empty bins
                    continue

                alpha = 1.0 if binned_means[j] > 0 else 0.6
                linewidth = 2.0 if binned_means[j] > 0 else 1.0  # Thicker outline for positive trend

                axes[i].bar(
                    bin_centers[j], binned_means[j], width=bin_size * 0.9,
                    color=color, edgecolor='black', alpha=alpha, linewidth=linewidth
                )

            # Formatting
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)

            axes[i].set_xlim(-0.5, 0.5)
            axes[i].set_ylim(-0.2, 0.2)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

        # Labels
        axes[int(len(ks) - 1)].set_xlabel('Normalized Time Delay')
        axes[int(len(ks) / 2)].set_ylabel('Average Period Shift [s]')

        # Save and close plot
        plt.savefig(plot_params.save_folder + '/binned_prc')
        plt.close(fig)
        fig, axes = plt.subplots()
        xs = np.array([a[0] for a in all_all_delay_responses])
        ys = np.array([a[1] for a in all_all_delay_responses])

        bin_edges = np.arange(-0.5, 0.5 + bin_size, bin_size)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins

        binned_means = []
        binned_stds = []
        for j in range(len(bin_edges) - 1):
            mask = (xs >= bin_edges[j]) & (xs < bin_edges[j + 1])
            if np.sum(mask) > 0:
                binned_means.append(np.mean(ys[mask]))
                binned_stds.append(np.std(ys[mask]))
            else:
                binned_means.append(np.nan)  # Empty bins remain NaN
                binned_stds.append(np.nan)

        binned_means = np.array(binned_means)
        binned_stds = np.array(binned_stds)

        color = 'purple'

        # Adjust bar properties based on sign of mean shift
        for j in range(len(bin_centers)):
            if np.isnan(binned_means[j]):  # Skip empty bins
                continue

            alpha = 1.0 if binned_means[j] > 0 else 0.6
            linewidth = 2.0 if binned_means[j] > 0 else 1.0  # Thicker outline for positive trend

            axes.bar(
                bin_centers[j], binned_means[j], width=bin_size * 0.9, color=color, edgecolor='black', alpha=alpha,
                linewidth=linewidth
            )

        # Formatting

        axes.set_xlim(-0.533, 0.533)
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
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_windowed_period_2024_before_nolines')
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
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_windowed_period_before_aggregate_all2024')
        plt.close(fig)

    if plot_params.do_windowed_period_plot:
        fig, axes = plt.subplots(len(ks))
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

        if plot_params.save_data:
            with open(plot_params.save_folder+'/all_befores_from_experiments.pickle', 'wb') as handle:
                pickle.dump(all_befores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(plot_params.save_folder+'/all_afters_from_experiments.pickle', 'wb') as handle:
                pickle.dump(all_afters, handle, protocol=pickle.HIGHEST_PROTOCOL)

        axes[len(ks) - 1].set_xlabel('T[s]')
        axes[int(len(ks) / 2)].set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_period_2024_windowed_beforeafter_w_lines')
        plt.close(fig)

        if plot_params.do_boxplots:
            # Fig 5
            boxplots(all_befores, all_afters, plot_params)

        fig, axes = plt.subplots(len(ks))
        for i, k in enumerate(ks):
            all_afters[k] = []
            for individual in rmses['windowed_period_after'][k]:
                try:
                    all_afters[k].extend(individual)
                except TypeError:
                    all_afters[k].append(individual)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            all_afters[k] = [x for x in all_afters[k] if not np.isnan(x)]
            if i == 4:
                color = 'yellow'
            else:
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
        axes[int(len(ks) / 2)].set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_period_2024_windowed_after_w_lines')
        plt.close(fig)

        fig, axes = plt.subplots(len(ks))
        for i, k in enumerate(ks):
            all_afters[k] = []
            for individual in rmses['windowed_period_after'][k]:
                try:
                    all_afters[k].extend(individual)
                except TypeError:
                    all_afters[k].append(individual)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            all_afters[k] = [x for x in all_afters[k] if not np.isnan(x)]
            if i == 4:
                color = 'yellow'
            else:
                color = colormap.__call__(i * 3)
            axes[i].hist(all_afters[k], density=True, bins=np.arange(0.0, 2.0, 0.03), color=color, alpha=0.75)
            if i != len(ks) - 1:
                axes[i].xaxis.set_visible(False)
            axes[i].set_xlim(0.0, 2.05)
            axes[i].set_ylim(0.0, 6.0)
        axes[len(ks) - 1].set_xlabel('T[s]')
        axes[int(len(ks) / 2)].set_ylabel('pdf')
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_period_2024_windowed_after')
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
        plt.savefig(plot_params.save_folder + '/LED_period_firefly_windowed_period_2024_w_aggregatebefore_after')
        plt.close(fig)

    ## Extraneous
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

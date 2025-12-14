import argparse
import csv
import numpy as np
import os

from simulation import prc_simulator
from helpers import plotting_helpers, exp_stat_agg_helpers, analysis_helpers


def aggregate_timeseries(path, pargs):
    all_data = {}
    date = path.split('_')[0]
    key = path.split('_')[1]

    if pargs.log:
        print('Loading timeseriesfrom {} with led freq {}'.format(date, key))
    if all_data.get(key) is None:
        all_data[key] = []
    with open(pargs.data_path + '/' + path, 'r') as data_file:
        ts = {'ff': [],
              'led': []
              }
        data = csv.reader(data_file)
        next(data)
        for line in data:
            try:
                ts['ff'].append(line[0])
                ts['led'].append(line[1])
            except IndexError:
                raise ('Problem loading data from {} with led freq {}'.format(f, key))

        all_data[key].append(ts)
    return all_data


def investigate_timeseries(pargs):
    final_dict = {
        '300': [],
        '400': [],
        '500': [],
        '600': [],
        '700': [],
        '770': [],
        '850': [],
        '1000': [],
    }
    ks = list(final_dict.keys())
    fpaths = os.listdir(pargs.data_path)
    for path in fpaths:
        if 'DS_Store' not in path and not os.path.isdir(pargs.data_path + '/' + path):
            d = aggregate_timeseries(path, pargs)
            for key in d:
                if final_dict.get(key):
                    final_dict[key].extend(d[key])
                else:
                    final_dict[key] = d[key]

    timeseries_stats, _ = exp_stat_agg_helpers.sliding_window_stats(final_dict, pargs)
    plotting_helpers.plot_statistics(timeseries_stats, ks, pargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Frontalis - LED interaction data analysis and plotting suite',
        description='Implements loading, cleansing, analysis, and visualization of timeseries data generated'
                    'via recordings of the interaction between fireflies and LEDs.',
    )

    parser.add_argument('-i', '--investigate', action='store_true', help='Whether to analyse timeseries')
    parser.add_argument('-a', '--do_nla', action='store_true', help='Whether to analyse with DFA')
    parser.add_argument('-w', '--write_timeseries', action='store_true', help='Whether to plot interactive timeseries')
    parser.add_argument('-s', '--simulate', action='store_true', help='Whether to simulate IF based on PRC')

    parser.add_argument('--with_stats', action='store_true',
                        help='Whether to write the phase and response time alongside the timeseries')
    parser.add_argument('--do_ffrt', action='store_true')

    parser.add_argument('--do_delay_plot', action='store_true', help='Whether to plot delay plots')
    parser.add_argument('--do_windowed_period_plot', action='store_true', help='Whether to plot period plots')
    parser.add_argument('--do_boxplots', action='store_true', help='Whether to plot boxplot comparisons')
    parser.add_argument('--save_data', action='store_true', help='Whether to save aggregate data for faster loading')
    parser.add_argument('--load_data_from_dists', action='store_true',
                        help='Whether to load data from processed distributions for speed')
    parser.add_argument('--do_initial_distribution', action='store_true',
                        help='Whether to plot initial distribution of periods for all sampled fireflies')
    parser.add_argument('--do_scatter_overall_stats', action='store_true', help='Whether to scatter @ aggregate-level')
    parser.add_argument('--do_means', action='store_true', help='Whether to bother with mean of means comparisons')
    parser.add_argument('--do_cc', action='store_true', help='Whether to bother trying connected component analysis')
    parser.add_argument('--do_prc', action='store_true', help='Whether to bother with phase response curve')
    parser.add_argument('--window_size_seconds', type=int, default=5, help='Window size, in seconds, for ts analysis')
    parser.add_argument('--re_norm', action='store_true', help='Whether to convert from -0.5 - 0.5 to 0.0 - 1.0')
    parser.add_argument('--plot_all_nla', action='store_true', help='Whether to attempt nla plots')
    parser.add_argument('--p', action='store_false', help='Whether to correct for bkgrd stack offset')
    parser.add_argument('--log', action='store_true', help='Whether to log data')
    parser.add_argument('--latex', action='store_true', help='Whether to write latex table')
    parser.add_argument('--test_synthetic', action='store_true',
                        help='Whether to test some synthetic data to see how we compare')
    parser.add_argument('--data_path', type=str, default='data_paths')
    parser.add_argument('--save_folder', type=str,
                        default='figs',
                        help='Folder path for saving')

    parser.add_argument('--prc_json_path', type=str, default='./saved_prc_normalized.json')
    parser.add_argument('--before_path', type=str, default='./simulation/all_befores_from_experiments.pickle')
    parser.add_argument('--after_path', type=str, default='./simulation/all_afters_from_experiments.pickle')
    parser.add_argument('--simulation_trials', type=int, default=20)

    # --- NEW: parameter grids for tanh PRC ---

    parser.add_argument(
        '--kappa_values', '--kappas', dest='kappa_values',
        nargs='+', type=float,
        default=list(np.arange(4.0, 12.0 + 1e-9, 2)),
        help='κ sweep (start=2, stop=24, step=2)'
    )
    parser.add_argument(
        '--phi_c_values', '--phic', dest='phi_c_values',
        nargs='+', type=float,
        default=list(np.arange(-0.3, 0.3 + 1e-9, 0.15)),
        help='φ_c sweep (start=-0.30, stop=0.30, step=0.02)'
    )
    parser.add_argument(
        '--ymax_values', '--ymax', dest='ymax_values',
        nargs='+', type=float,
        default=list(np.arange(0.6, 1.4 + 1e-9, 0.20)),
        help='Y_max sweep (start=0.20, stop=2.0, step=0.20)'
    )
    parser.add_argument(
        '--y0_values', '--y0', dest='y0_values',
        nargs='+', type=float,
        default=list(np.arange(1.0, 1.8 + 1e-9, 0.2)),
        help='y0 sweep (start=1.0, stop=1.6, step=0.1)'
    )
    parser.add_argument(
        '--led_periods_ms',
        nargs='+', type=int,
        default=[300, 400, 500, 600, 700, 770, 850, 1000],
        help='List of LED periods (ms)'
    )
    parser.add_argument('--use_prc', action='store_true')
    parser.add_argument('--show_heatmaps', action='store_true')
    parser.add_argument('--show_param_summary', action='store_true')
    parser.add_argument('--show_overlay', action='store_true')
    parser.add_argument('--save_sim_figs', action='store_true')
    parser.add_argument('--save_sim_data', action='store_true')
    parser.add_argument('--simulation_save_path', type=str, default='sim_data/together')
    args = parser.parse_args()

    if args.write_timeseries:
        plotting_helpers.write_timeseries_figs(args)
    if args.do_nla:
        data = analysis_helpers.do_nonlinear_analysis(args)
    if args.investigate:
        investigate_timeseries(args)

    if args.simulate:
        saved_manifest_path = prc_simulator.simulate_from_prc(args)
        print(saved_manifest_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import argparse
import re
import os

import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt



def get_bounding_box_interactive(points, title='Draw bounding box (2 clicks)'):
    """
    Plot the points and let the user click two corners of the bounding box.
    Returns ((x_min, y_min), (x_max, y_max))
    """
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=5)
    ax.set_title(title)
    coords = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            coords.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'rx')
            fig.canvas.draw()
            if len(coords) == 2:
                plt.close()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(coords) < 2:
        raise ValueError("You must click two corners to define the bounding box.")
    
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    return ((x_min, y_min), (x_max, y_max))


def process_trial_data(trial_number, input_csv_path, plot_vis=False):
    data = pd.read_csv(input_csv_path)
    required_columns = ['object time', 'object x position', 'object y position', 'frame time']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("CSV is missing one or more required columns.")

    detected = data.dropna(subset=['object time', 'object x position', 'object y position']).copy()
    
    positions = detected[['object x position', 'object y position']].values

    # Ask user to define LED bounding box
    print(f"\n--- Define LED bounding box for trial {trial_number} ---")
    led_bbox = get_bounding_box_interactive(positions, title='Click two corners for LED BBox')

    # Ask user to define Firefly bounding box
    print(f"\n--- Define Firefly bounding box for trial {trial_number} ---")
    ff_bbox = get_bounding_box_interactive(positions, title='Click two corners for Firefly BBox')

    # LED points
    led_mask = (
        (detected['object x position'] >= led_bbox[0][0]) &
        (detected['object x position'] <= led_bbox[1][0]) &
        (detected['object y position'] >= led_bbox[0][1]) &
        (detected['object y position'] <= led_bbox[1][1])
    )
    led_points = detected[led_mask].copy()
    led_points['cluster type'] = 'LED'

    # FF core points (right side of bounding box)
    ff_mask = (
        (detected['object x position'] >= ff_bbox[0][0]) &
        (detected['object x position'] <= ff_bbox[1][0]) &
        (detected['object y position'] >= ff_bbox[0][1]) &
        (detected['object y position'] <= ff_bbox[1][1])
    )
    ff_core = detected[ff_mask].copy()
    ff_core['cluster type'] = 'FF'
    led_points = led_points.sort_values('frame time').reset_index(drop=True)

    # Combine LED and FF
    detected = pd.concat([led_points, ff_core], ignore_index=True)

    # Build presence time series
    ff_times = defaultdict(lambda: 0.0)
    led_times_dict = defaultdict(lambda: 0.0)
    for _, row in detected.iterrows():
        t = float(row['object time'])
        if row['cluster type'] == 'FF':
            ff_times[round(t, 4)] = 1.0
        elif row['cluster type'] == 'LED':
            led_times_dict[round(t, 4)] = 2.0

    all_frame_times = sorted(data['frame time'].unique())
    combined_times = sorted(set(all_frame_times).union(set(led_times_dict.keys())))
    ff_series = []
    led_series = []

    for t in combined_times:
        t_rounded = round(t, 4)
        ff_value = float(ff_times.get(t_rounded, 0.0))
        led_value = float(led_times_dict.get(t_rounded, 1.0))
        ff_series.append((float(t), ff_value))
        led_series.append((float(t), led_value))

    # Save labeled CSV
    output_csv_path = input_csv_path.replace('.csv', '_labeled.csv')
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['FF times', 'LED times'])
        for ff, led in zip(ff_series, led_series):
            writer.writerow([f'{ff}', f'{led}'])

    print(f"Labeled CSV saved to: {output_csv_path}")

    if plot_vis:
        # Plot clusters
        plt.figure(figsize=(8, 6))
        colors = {'FF': 'green', 'LED': 'yellow'}
        for label, group in detected.groupby('cluster type'):
            plt.scatter(group['object x position'], group['object y position'],
                        s=10, label=label, color=colors.get(label, 'gray'))
        plt.legend()
        plt.title(f'Trial {trial_number} Clustered Positions')
        plt.xlabel('X Position (px)')
        plt.ylabel('Y Position (px)')
        plt.savefig(input_csv_path.replace('.csv', '_clusters.png'))
        plt.close()

        ff_times_plot, ff_values_plot = zip(*ff_series)
        led_times_plot, led_values_plot = zip(*led_series)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ff_times_plot, y=ff_values_plot,
                                mode='lines+markers', name='FF times', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=led_times_plot, y=led_values_plot,
                                mode='lines+markers', name='LED times', line=dict(color='yellow')))

        fig.update_layout(title='FF and LED Timeseries',
                        xaxis_title='Time (s)',
                        yaxis_title='Presence Value',
                        legend_title='Signal Type')

        html_path = input_csv_path.replace('.csv', '_ts.html')
        plot(fig, filename=html_path, auto_open=False)


def get_trial_folders(base_dir):
    trial_folders = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and re.match(r'Trial_\d+', name):
            trial_folders.append((int(re.findall(r'\d+', name)[0]), path))
    return sorted(trial_folders)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process trial data with DBSCAN clustering.")
    parser.add_argument("--base_dir", help="Base directory of the data files (e.g., '/mnt/e/Congaree_2025/20250518/Max850ms/mp4s/')",
                        default='/mnt/e/Congaree_2025/20250518/Max850ms/mp4s/')
    parser.add_argument("--plot", action="store_true", help="Whether to visualize the timeseries and clusterings you have applied")
    args = parser.parse_args()

    trial_folders = get_trial_folders(args.base_dir)
    print(f"Detected {len(trial_folders)} trial folders")

    for trial_number, trial_path in trial_folders:
        input_csv_path = os.path.join(trial_path, f'trial_{trial_number}_data.csv')
        process_trial_data(trial_number, input_csv_path, plot_vis=args.plot)

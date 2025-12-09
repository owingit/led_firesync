# led_firesync

This repository serves as a home for the data and analysis code for the paper:

**“Excitation–inhibition interactions mediate firefly flash synchronization”**  
*(doi incoming)*

---

## Motivation & Overview

Natural populations of synchronous fireflies are well-known examples of collective behavior. This project investigates whether and how individual fireflies of species *Photuris frontalis* respond to external periodic driving signals delivered via LEDs. The goal is to test for differential levels of **entrainment** or **phase-locking** as a response to LED perturbation, and to quantify how the frequency of the driving stimulus alters firefly flashing behavior.

Specifically, we analyze open-loop experiments in which fireflies receive a periodic visual stimulus but do not influence LED timing. For each experimental trial, we extract:

- Inter-flash interval (IFI) distributions
- Phase histograms relative to the LED
- Burst statistics
- Phase Response Curves (PRCs)

These metrics are compared against simulations using phenomenological PRC models and integrate-and-fire style oscillators to characterize excitation–inhibition dynamics governing flash timing.

---

## Repository Structure

led_firesync/

├── data_paths/   
├── simulation/    
├── figs/   
├── preprocessing/   
├── temp_data/  
├── helpers/   
├── led_analysis.py  
└── README.md 

### Description of key directories / files

- **data_paths/** — All experimental flash-timing data in CSV format. Each file is coded by the date and LED period as well as a unique index, of the form <MMDDYYYY_LEDPERIOD_INDEX>.csv. Within each file is a list of firefly time/state tuples and LED time/state tuples, representing the timeseries for the firefly (off=0, flash=1) and the timeseries for the LED (off=1, flash=2), respectively 
- **simulation/** — Code to simulate PRCs, entrainment, and frequency responses; includes saved model parameters.
- **preprocessing/** — Scripts for extracting flash timestamps from videos, cleaning raw event streams, etc.
- **temp_data/** — Contains environmental metadata (temperature, etc.) associated with trials.
- **figs/** — Saved static and interactive figures; `timeseries/` contains HTML representations for exploratory analysis.
- **helpers/** — Common plotting functions, period histogram utilities, statistical aggregation tools, file I/O helpers.
- **led_analysis.py** — Main script for reproducing all analytical steps and generating manuscript-ready figures.

---

## Dependencies / Built With

Developed in **Python 3.9** with the following packages:

```NumPy, SciPy, Pandas, Matplotlib, Seaborn, Plotly```

You can install dependencies via:

```pip install numpy scipy pandas matplotlib seaborn plotly  ```

## Usage
``` 
python led_analysis.py -w -i -a \
  --with_stats \
  --re_norm \
  --do_windowed_period_plot \
  --save_folder ./figs/
```

This will:

1. Load raw experimental data
2. Generate interactive timeseries
3. Generate PRCs and period ratio plots
4. Produce IFI distributions
5. Save all figures to ./figs/ and ./figs/timeseries

```
python led_analysis.py 
  -s \
  --led_periods_ms 300 400 500 600 700 770 850 1000 \
  --simulation_trials 50 \
  --save_sim_data
```

This will:

1. Run 50 trials of the response curve simulation 
2. Generate comparative plots
3. Save all figures to ./figs/



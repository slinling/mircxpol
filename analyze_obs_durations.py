"""
analyze_obs_durations.py

This script analyzes CHARA MIRC-X and MYSTIC interferometric data to determine 
the observing duration for each wavelength and baseline combination on specific dates. 
It reports the maximum and minimum duration of continuous observing segments by 
calculating the time span in decimal hours based on MJD timestamps.

Key Functionalities:
---------------------
1. Converts MJD (Modified Julian Date) to decimal UTC hours.
2. Iterates through baseline and wavelength combinations for MIRC-X and MYSTIC.
3. Identifies start/end MJDs and computes observing durations.
4. Prints max and min observing durations for each night.

Input Requirements:
---------------------
- CSV data files under:
    ./2022October_Wollaston_WAVELENGTH_DATA_MEGA/MJD_split_files/  (for MIRC-X)
    ./2022October_Wollaston_WAVELEN_TH_DATA/                       (for MYSTIC)
  Each file must contain an 'MJD' column.
-------
- Printed summary of max and min observing durations (per date/baseline/wavelength)

Author: [Your Name]
Date: [YYYY-MM-DD]
Affiliation: [Your Institution]
"""

# Script implementation begins below
import os
import pandas as pd
import numpy as np
from astropy.time import Time

def mjd_to_decimal_hours(mjd):
    """
    Convert Modified Julian Date (MJD) to decimal hours in UTC.

    Parameters:
        mjd (float or array-like): Modified Julian Date(s)

    Returns:
        float: Time in decimal hours (e.g., 13.25 for 13:15 UTC)
    """
    t = Time(mjd, format='mjd')
    utc = t.datetime
    return utc.hour + utc.minute / 60 + utc.second / 3600 + utc.microsecond / 3.6e9

# === Configuration ===
baseline_combinations = ['E1W2', 'E1W1', 'E1S2', 'E1S1', 'E1E2',
                         'W2W1', 'W2S2', 'W2S1', 'W2E2',
                         'W1S2', 'W1S1', 'W1E2',
                         'S2S1', 'S2E2', 'S1E2']

mircx_dates = [59871, 59873, 59874]
mystic_dates = ['2022Oct19', '2022Oct21', '2022Oct22']
mystic_wavelengths_by_date = {
    0: ['2.035', '2.075', '2.115', '2.154', '2.193', '2.232', '2.270', '2.308', '2.345', '2.374'],
    1: ['2.035', '2.075', '2.115', '2.155', '2.194', '2.232', '2.270', '2.308', '2.345', '2.375'],
    2: ['2.034', '2.074', '2.114', '2.154', '2.193', '2.231', '2.270', '2.308', '2.345', '2.375']
}

# === Part 1: MIRCX data ===
print("MIRCX Data Analysis:")
mircx_wavelengths = ['1.540', '1.571', '1.602', '1.631', '1.660', '1.688']

for k, date in enumerate(mircx_dates):
    durations = []
    info = []

    for baseline in baseline_combinations:
        for wavelength in mircx_wavelengths:
            filename = f'./2022October_Wollaston_WAVELENGTH_DATA_MEGA/MJD_split_files/MIRCX_2022Oct_{wavelength}mu_{baseline}_MJD_{date}.csv'
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                if not df.empty and 'MJD' in df.columns:
                    start = mjd_to_decimal_hours(df["MJD"].iloc[0])
                    end = mjd_to_decimal_hours(df["MJD"].iloc[-1])
                    duration = end - start if end >= start else end - start + 24
                    durations.append(duration)
                    info.append((duration, baseline, wavelength))

    if durations:
        max_dur, max_b, max_w = max(info, key=lambda x: x[0])
        min_dur, min_b, min_w = min(info, key=lambda x: x[0])
        print(f"Date: {date}")
        print(f"  Max observing duration: {max_dur:.2f} hours (Baseline: {max_b}, Wavelength: {max_w} μm)")
        print(f"  Min observing duration: {min_dur:.2f} hours (Baseline: {min_b}, Wavelength: {min_w} μm)\n")
    else:
        print(f"Date: {date} - No valid data files found.\n")

# === Part 2: MYSTIC data ===
print("\nMYSTIC Data Analysis:")

for k, date_str in enumerate(mystic_dates):
    durations = []
    info = []
    wavelengths = mystic_wavelengths_by_date[k]

    for baseline in baseline_combinations:
        for wavelength in wavelengths:
            filename = f'./2022October_Wollaston_WAVELEN_TH_DATA/MYSTIC_{date_str}_{wavelength}mu_{baseline}_fancy.csv'
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                if not df.empty and 'MJD' in df.columns:
                    start = mjd_to_decimal_hours(df["MJD"].iloc[0])
                    end = mjd_to_decimal_hours(df["MJD"].iloc[-1])
                    duration = end - start if end >= start else end - start + 24
                    durations.append(duration)
                    info.append((duration, baseline, wavelength))

    if durations:
        max_dur, max_b, max_w = max(info, key=lambda x: x[0])
        min_dur, min_b, min_w = min(info, key=lambda x: x[0])
        print(f"Date: {date_str}")
        print(f"  Max observing duration: {max_dur:.2f} hours (Baseline: {max_b}, Wavelength: {max_w} μm)")
        print(f"  Min observing duration: {min_dur:.2f} hours (Baseline: {min_b}, Wavelength: {min_w} μm)\n")
    else:
        print(f"Date: {date_str} - No valid data files found.\n")
   





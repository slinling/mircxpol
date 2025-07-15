"""
fit_sys_error_and_plot_mystic.py

This script fits a global polarization model to CHARA/MYSTIC interferometric data using 
systematic error estimates for normalized visibility ratios, differential phases, and 
flux ratios. It visualizes the model results and computes residuals for each observable.

Key Functionalities:
---------------------
1. Loads preprocessed data for 15 telescope baselines and 10 wavelength channels.
2. Applies a global model (defined in `mircxpol1`) using non-linear least squares fitting.
3. Incorporates three systematic error parameters: 
   - a1 for visibility ratio errors
   - a2 for differential phase errors
   - a3 for flux ratio errors
4. Generates plots of modeled vs. observed data across hour angle for:
   - VisRatio (VH/VV)
   - PhaseDiff (in degrees)
   - Flux ratios for Tel1 and Tel2
5. Saves fit results to `.npy` and residuals to `.npz` files.
6. Produces publication-quality plots for each night of observation.

Required Input:
---------------
- Systematic error values: a1, a2, a3 (from chi² minimization in step 1)
- CSV data files in ./2022October_Wollaston_WAVELEN_TH_DATA/
- Custom module `mircxpol1` for model equations

"""

# Script implementation begins below
# %%
import mircxpo as mp
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord, EarthLocation, AltAz
import pandas as pd 
from matplotlib import rc
import os
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize

wavel_combinations_22 = ['2.034', '2.074', '2.114','2.154','2.191', '2.232', '2.270', '2.308', '2.345', '2.375'] # MYSTIC 10-22
sns.set_palette("colorblind")
colors = sns.color_palette("colorblind", len(wavel_combinations_22))

# Set consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "mathtext.fontset": "cm"
})

# %%
times = ['2022-10-19T00:00:00', '2022-10-20T00:00:00', '2022-10-21T00:00:00','2022-10-22T00:00:00']
t = Time(times, format='isot', scale='utc')
t_mjd = t.mjd.astype(int)
t_mjd = np.array([59871, 59873, 59874])

# %%
def func_data(wavel, date, error_threshold=10, idx1 = 0, idx2 = -1):
    combinations = ['E1W2', 'E1W1', 'E1S2', 'E1S1', 'E1E2', 'W2W1', 'W2S2', 'W2S1', 'W2E2', 'W1S2', 'W1S1', 'W1E2', 'S2S1', 'S2E2', 'S1E2']
    
    data_dict = {}
    data_dict_tot = {}
    
    for i in range(len(combinations)):

        path_csv_i = './2022October_Wollaston_WAVELEN_TH_DATA/MYSTIC_2022Oct' + f'{date}_' + wavel + 'mu_' + combinations[i] + '_fancy.csv'
        df_i = pd.read_csv(path_csv_i)

        # Convert error columns to float, coercing errors to NaN
        df_i[' visratio_normed_err'] = pd.to_numeric(df_i[' visratio_normed_err'], errors='coerce')
        df_i[' Phase_err'] = pd.to_numeric(df_i[' Phase_err'], errors='coerce')
        df_i['tel1_ratio_err'] = pd.to_numeric(df_i['tel1_ratio_err'], errors='coerce')
        df_i['tel2_ratio_err'] = pd.to_numeric(df_i['tel2_ratio_err'], errors='coerce')

        # Apply error threshold to filter out bad data
        if date == '22':
            df_i['tel2_rati'] = pd.to_numeric(df_i['tel2_ratio'], errors='coerce')
            mask = (df_i[' visratio_normed_err'].values < error_threshold) & \
                (df_i[' Phase_err'].values < error_threshold) & \
                (df_i['tel1_ratio_err'].values < error_threshold) & \
                (df_i['tel2_ratio_err'].values < error_threshold) & \
                (df_i['tel2_ratio'].values > 0.9)
            # print(mask)
            
        else:
            mask = (df_i[' visratio_normed_err'].values < error_threshold) & \
               (df_i[' Phase_err'].values < error_threshold) & \
               (df_i['tel1_ratio_err'].values < error_threshold) & \
               (df_i['tel2_ratio_err'].values < error_threshold)

        df_i = df_i[mask]
        # Filter the DataFrame using the mask, excluding rows with NaN values
        df_i = df_i.dropna() 

        MJD_data_i = df_i['MJD'].values
        HA_data_i = df_i[' HA'].values
        # VisRatio
        VisRatio_data_i = df_i[' VisRatio_normed'].values
        VisRatio_err_data_i = df_i[' visratio_normed_err'].values
        # PhaseDiff in deg
        PD_data_i = df_i[' PhaseDiff'].values
        PD_err_data_i = df_i[' Phase_err'].values
        # Tel1_ratio
        T1_ratio_data_i = df_i[' Tel1_ratio'].values
        T1_ratio_err_data_i = df_i['tel1_ratio_err'].values
        # Tel2_ratio
        T2_ratio_data_i = df_i['tel2_ratio'].values
        T2_ratio_err_data_i = df_i['tel2_ratio_err'].values

        MJD_i = Time(MJD_data_i, format='mjd')
        tt_i = MJD_i.to_value('iso')
        upsandaltaz_i = upsand.transform_to(AltAz(obstime=tt_i, location=chara))
        alt_data_i = upsandaltaz_i.alt
        az_data_i = upsandaltaz_i.az

        if idx2 == -1:
            idx2 = len(VisRatio_data_i)
        
        # Store all data in dictionary
        data_dict_tot[combinations[i]] = {
            'MJD_data': MJD_data_i.astype(float),
            'HA_data': HA_data_i.astype(float),
            'VisRatio_data': VisRatio_data_i.astype(float),
            'VisRatio_err_data': VisRatio_err_data_i.astype(float),
            'PD_data': PD_data_i.astype(float),
            'PD_err_data': PD_err_data_i.astype(float),
            'T1_ratio_data': T1_ratio_data_i.astype(float),
            'T1_ratio_err_data': T1_ratio_err_data_i.astype(float),
            'T2_ratio_data': T2_ratio_data_i.astype(float),
            'T2_ratio_err_data': T2_ratio_err_data_i.astype(float),
            'MJD': MJD_i,
            'tt': tt_i,
            'alt_data': alt_data_i,
            'az_data': az_data_i
        }


        data_dict[combinations[i]] = {
            'MJD_data': MJD_data_i[idx1:idx2].astype(float),
            'HA_data': HA_data_i[idx1:idx2].astype(float),
            'VisRatio_data': VisRatio_data_i[idx1:idx2].astype(float),
            'VisRatio_err_data': VisRatio_err_data_i[idx1:idx2].astype(float),
            'PD_data': PD_data_i[idx1:idx2].astype(float),
            'PD_err_data': PD_err_data_i[idx1:idx2].astype(float),
            'T1_ratio_data': T1_ratio_data_i[idx1:idx2].astype(float),
            'T1_ratio_err_data': T1_ratio_err_data_i[idx1:idx2].astype(float),
            'T2_ratio_data': T2_ratio_data_i[idx1:idx2].astype(float),
            'T2_ratio_err_data': T2_ratio_err_data_i[idx1:idx2].astype(float),
            'MJD': MJD_i[idx1:idx2],
            'tt': tt_i[idx1:idx2],
            'alt_data': alt_data_i[idx1:idx2],
            'az_data': az_data_i[idx1:idx2],
        }

    return data_dict_tot, data_dict



# %%
def calc_altaz(HA_Data, MJD_Data):

    mjd = Time(MJD_Data, format='mjd')
    ptime_Data = mjd.iso

    upsandaltaz_Data = upsand.transform_to(AltAz(obstime=ptime_Data, location=chara))
    alt_Data = upsandaltaz_Data.alt
    az_Data = upsandaltaz_Data.az
    # Find the index of the maximum altitude
    zenith_idx_Data = np.argmax(alt_Data)
    zenith_time_Data = HA_Data[zenith_idx_Data]
    HA_Data = HA_Data - zenith_time_Data

    return alt_Data, az_Data, HA_Data

# %%
## E1-0, W2-5, W1-9, S2-11， S1-14, E2-14
beam_combinations = ['E1W2', 'E1W1', 'E1S2', 'E1S1', 'E1E2', 'W2W1', 'W2S2', 'W2S1', 'W2E2', 'W1S2', 'W1S1', 'W1E2', 'S2S1', 'S2E2', 'S1E2']
date_combinations = ['2022Oct19', '2022Oct21', '2022Oct22']
time_combinations = ["2022-10-19 0:00:00", "2022-10-21 0:00:00", "2022-10-22 0:00:00"]
# wavl for MYSTIC
wavel_combinations_22 = ['2.034', '2.074', '2.114','2.154','2.191', '2.232', '2.270', '2.308', '2.345', '2.375'] # MYSTIC 10-22
color_combinations = ['deeppink', 'pink','mediumpurple', 'turquoise','dodgerblue','limegreen','yellowgreen', 'gold','orange','tomato'] # MYSTIC
index_combinations = [0, 5, 9, 11, 14]
tel_combinations = ['E1', 'W2', 'W1', 'S2', 'S1', 'E2']
date = ['19', '21', '22']

upsand = SkyCoord.from_name("ups and")
chara = EarthLocation.of_site("CHARA")

#48 params
var = np.array([1., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
bound = ((-np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
    (np.inf, np.inf, np.inf, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)) 

t_mjd = np.array([59871, 59873, 59874])

# %%
def plot_mircx(sys):
    
    a1, a2, a3 = sys

    Delta_norm_vis = 0
    Delta_diff_phase = 0
    Delta_flux_ratio = 0
    len_data = 0
    len_data_flux_ratio = 0

    ERR_norm_vis = []
    ERR_diff_phase = []
    ERR_flux_ratio = []

    for ii in range(3):
        popt_x = {}

        if ii == 0:
        ##########################################################################################################################################################################################################
            time1 = np.linspace(4.5, 12.01, 100) * u.hour
            wavel_combinations = ['2.035', '2.075', '2.115','2.154','2.193', '2.232', '2.270', '2.308', '2.345', '2.374'] # MYSTIC 10-19
        elif ii == 1:
            time1 = np.linspace(1.5, 12.01, 100) * u.hour
            wavel_combinations = ['2.035', '2.075', '2.115','2.155','2.194', '2.232', '2.270', '2.308', '2.345', '2.375'] # MYSTIC 10-21
        else:
            time1 = np.linspace(1.5, 12.01, 100) * u.hour
            wavel_combinations = ['2.034', '2.074', '2.114','2.154','2.193', '2.231', '2.270', '2.308', '2.345', '2.375'] # MYSTIC 10-22
        ##########################################################################################################################################################################################################
        ptime = Time(time_combinations[1]) + time1
        HA = Angle(time1).value
        upsandaltaz = upsand.transform_to(AltAz(obstime=ptime, location=chara))
        alt = upsandaltaz.alt
        az = upsandaltaz.az
        # Find the index of the maximum altitude
        zenith_idx = np.argmax(alt)
        zenith_time = HA[zenith_idx]
        HA = HA - zenith_time

        fig1, axes1 = plt.subplots(nrows=5, ncols=3, figsize=(12, 14), sharex=True, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
        fig2, axes2 = plt.subplots(nrows=5, ncols=3, figsize=(12, 14), sharex=True, gridspec_kw={'hspace': 0.05})
        fig3, axes3 = plt.subplots(nrows=2, ncols=3, figsize=(12, 5), sharex=True, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

        axes1 = axes1.flatten()
        axes2 = axes2.flatten()
        axes3 = axes3.flatten()

        for i in range(len(wavel_combinations_22)):

            k = 0
            popt = {}

            if ii == 1: #only substract first 3 data dots for Oct21
                data_dict_tot, data_dict = func_data(wavel_combinations[i], date[ii])
            else:
                data_dict_tot = func_data(wavel_combinations[i], date[ii])[0]
                data_dict = data_dict_tot

            Delta_norm_vis_err = []
            Delta_diff_phase_err = []
            Delta_flux_ratio_err = []

            popt = least_squares(
                mp.err_global, var + np.random.uniform(-1, 1, size=(len(var))) * 0.3, bounds=bound,
                args=(
                    data_dict['E1W2']['alt_data'], data_dict['E1W2']['az_data'], data_dict['E1W2']['VisRatio_data'], data_dict['E1W2']['PD_data'], data_dict['E1W2']['VisRatio_err_data']+ a1**2, data_dict['E1W2']['PD_err_data'] + a2**2,
                    data_dict['E1W1']['alt_data'], data_dict['E1W1']['az_data'], data_dict['E1W1']['VisRatio_data'], data_dict['E1W1']['PD_data'], data_dict['E1W1']['VisRatio_err_data']+ a1**2, data_dict['E1W1']['PD_err_data'] + a2**2,
                    data_dict['E1S2']['alt_data'], data_dict['E1S2']['az_data'], data_dict['E1S2']['VisRatio_data'], data_dict['E1S2']['PD_data'], data_dict['E1S2']['VisRatio_err_data']+ a1**2, data_dict['E1S2']['PD_err_data'] + a2**2, 
                    data_dict['E1S1']['alt_data'], data_dict['E1S1']['az_data'], data_dict['E1S1']['VisRatio_data'], data_dict['E1S1']['PD_data'], data_dict['E1S1']['VisRatio_err_data']+ a1**2, data_dict['E1S1']['PD_err_data'] + a2**2,
                    data_dict['E1E2']['alt_data'], data_dict['E1E2']['az_data'], data_dict['E1E2']['VisRatio_data'], data_dict['E1E2']['PD_data'], data_dict['E1E2']['VisRatio_err_data']+ a1**2, data_dict['E1E2']['PD_err_data'] + a2**2,
                    data_dict['W2W1']['alt_data'], data_dict['W2W1']['az_data'], data_dict['W2W1']['VisRatio_data'], data_dict['W2W1']['PD_data'], data_dict['W2W1']['VisRatio_err_data']+ a1**2, data_dict['W2W1']['PD_err_data'] + a2**2, 
                    data_dict['W2S2']['alt_data'], data_dict['W2S2']['az_data'], data_dict['W2S2']['VisRatio_data'], data_dict['W2S2']['PD_data'], data_dict['W2S2']['VisRatio_err_data']+ a1**2, data_dict['W2S2']['PD_err_data'] + a2**2, 
                    data_dict['W2S1']['alt_data'], data_dict['W2S1']['az_data'], data_dict['W2S1']['VisRatio_data'], data_dict['W2S1']['PD_data'], data_dict['W2S1']['VisRatio_err_data']+ a1**2, data_dict['W2S1']['PD_err_data'] + a2**2,
                    data_dict['W2E2']['alt_data'], data_dict['W2E2']['az_data'], data_dict['W2E2']['VisRatio_data'], data_dict['W2E2']['PD_data'], data_dict['W2E2']['VisRatio_err_data']+ a1**2, data_dict['W2E2']['PD_err_data'] + a2**2, 
                    data_dict['W1S2']['alt_data'], data_dict['W1S2']['az_data'], data_dict['W1S2']['VisRatio_data'], data_dict['W1S2']['PD_data'], data_dict['W1S2']['VisRatio_err_data']+ a1**2, data_dict['W1S2']['PD_err_data'] + a2**2,
                    data_dict['W1S1']['alt_data'], data_dict['W1S1']['az_data'], data_dict['W1S1']['VisRatio_data'], data_dict['W1S1']['PD_data'], data_dict['W1S1']['VisRatio_err_data']+ a1**2, data_dict['W1S1']['PD_err_data'] + a2**2, 
                    data_dict['W1E2']['alt_data'], data_dict['W1E2']['az_data'], data_dict['W1E2']['VisRatio_data'], data_dict['W1E2']['PD_data'], data_dict['W1E2']['VisRatio_err_data']+ a1**2, data_dict['W1E2']['PD_err_data'] + a2**2, 
                    data_dict['S2S1']['alt_data'], data_dict['S2S1']['az_data'], data_dict['S2S1']['VisRatio_data'], data_dict['S2S1']['PD_data'], data_dict['S2S1']['VisRatio_err_data']+ a1**2, data_dict['S2S1']['PD_err_data'] + a2**2, 
                    data_dict['S2E2']['alt_data'], data_dict['S2E2']['az_data'], data_dict['S2E2']['VisRatio_data'], data_dict['S2E2']['PD_data'], data_dict['S2E2']['VisRatio_err_data']+ a1**2, data_dict['S2E2']['PD_err_data'] + a2**2, 
                    data_dict['S1E2']['alt_data'], data_dict['S1E2']['az_data'], data_dict['S1E2']['VisRatio_data'], data_dict['S1E2']['PD_data'], data_dict['S1E2']['VisRatio_err_data']+ a1**2, data_dict['S1E2']['PD_err_data'] + a2**2, 
                    data_dict['E1W2']['T1_ratio_data'], data_dict['E1W2']['T2_ratio_data'], data_dict['E1W1']['T2_ratio_data'], data_dict['E1S2']['T2_ratio_data'], data_dict['E1S1']['T2_ratio_data'], data_dict['E1E2']['T2_ratio_data'],
                    data_dict['E1W2']['T1_ratio_err_data']+ a3**2, data_dict['E1W2']['T2_ratio_err_data']+ a3**2, data_dict['E1W1']['T2_ratio_err_data']+ a3**2, data_dict['E1S2']['T2_ratio_err_data']+ a3**2, data_dict['E1S1']['T2_ratio_err_data']+ a3**2, data_dict['E1E2']['T2_ratio_err_data']+ a3**2
                )
            )

            popt_x[wavel_combinations[i]] = {'popt.x': popt.x}

            for j in range(len(beam_combinations)):

                k = j
                alt_k = data_dict[beam_combinations[k]]['alt_data']
                az_k = data_dict[beam_combinations[k]]['az_data']

                delta_vis_ratio = np.array((mp.func1(alt_k, az_k, popt.x)[k] - data_dict[beam_combinations[k]]['VisRatio_data'])**2 / ((data_dict[beam_combinations[k]]['VisRatio_err_data'])**2 + a1**2)) 
                Delta_norm_vis +=  np.sum(delta_vis_ratio)

                delta_diff_phase = np.array((mp.func2(alt_k, az_k, popt.x)[k] - data_dict[beam_combinations[k]]['PD_data'])**2 / ((data_dict[beam_combinations[k]]['PD_err_data'])**2 + a2**2))
                Delta_diff_phase +=  np.sum(delta_diff_phase)

                err_vis_ratio = np.array(mp.func1(alt_k, az_k, popt.x)[k] - data_dict[beam_combinations[k]]['VisRatio_data']) 
                Delta_norm_vis_err = np.append(Delta_norm_vis_err, err_vis_ratio, axis = 0)

                err_diff_phase = np.array(mp.func2(alt_k, az_k, popt.x)[k] - data_dict[beam_combinations[k]]['PD_data'])
                Delta_diff_phase_err = np.append(Delta_diff_phase_err, err_diff_phase, axis = 0)

                len_data += len(delta_diff_phase)

                if k == 0:
                    delta_flux_ratio = np.array((mp.func3(alt_k, az_k, popt.x)[k] - data_dict[beam_combinations[k]]['T1_ratio_data'])**2 / ((data_dict[beam_combinations[k]]['T1_ratio_err_data'])**2 + a3**2))
                    Delta_flux_ratio +=  np.sum(delta_flux_ratio)
                    len_data_flux_ratio += len(delta_flux_ratio)

                    err_flux_ratio = np.array(mp.func3(alt_k, az_k, popt.x)[k] - data_dict[beam_combinations[k]]['T1_ratio_data'])
                    Delta_flux_ratio_err = np.append(Delta_flux_ratio_err, err_flux_ratio, axis = 0)

                if k < 5:
                    delta_flux_ratio = np.array((mp.func4(alt_k, az_k, popt.x)[k] - data_dict[beam_combinations[k]]['T2_ratio_data'])**2 / ((data_dict[beam_combinations[k]]['T2_ratio_err_data'])**2 + a3**2))
                    Delta_flux_ratio +=  np.sum(delta_flux_ratio)
                    len_data_flux_ratio += len(delta_flux_ratio)

                    err_flux_ratio = np.array(mp.func3(alt_k, az_k, popt.x)[k] - data_dict[beam_combinations[k]]['T2_ratio_data']) 
                    Delta_flux_ratio_err = np.append(Delta_flux_ratio_err, err_flux_ratio, axis = 0)

        

                axes1[j].plot(HA.flatten(), mp.func1(alt, az, popt.x)[j], color = color_combinations[i], alpha = 0.7, label='_nolegend_', lw=2)
                axes1[j].errorbar(data_dict_tot[beam_combinations[j]]['HA_data'], data_dict_tot[beam_combinations[j]]['VisRatio_data'], yerr = np.sqrt(data_dict_tot[beam_combinations[j]]['VisRatio_err_data']**2 + a1**2), fmt='o', color = color_combinations[i], alpha = 0.7, capsize=2, markersize=4)
                # axes1[j].set_yticks(yticks)  # Set y-ticks to fixed positions
                # axes1[j].set_yticklabels([f"{tick:.2f}" for tick in yticks])
                axes1[j].set_ylim(0.8, 1.15)

                if i == 0:  # Only add the label once for the first wavelength
                    custom_legend = plt.Line2D([0], [0], color='none', marker=None, linestyle='None', label=beam_combinations[j])
                    axes1[j].legend(handles=[custom_legend], loc='upper right', fontsize=14, handlelength=0, handletextpad=0, framealpha=0.1)
                    if j > 11:
                        axes1[j].set_xlabel("Hour Angle", fontsize=15)
                    if j %3 == 0:
                        axes1[j].set_ylabel(r"${V}_H / {V}_V$", fontsize=15)
                    else:
                        axes1[j].set_yticklabels([])
                        axes1[j].tick_params(labelleft=False)

                axes2[j].plot(HA, mp.func2(alt, az, popt.x)[j], color = color_combinations[i], alpha = 0.7, label='_nolegend_', lw=2)
                axes2[j].errorbar(data_dict_tot[beam_combinations[j]]['HA_data'], data_dict_tot[beam_combinations[j]]['PD_data'], yerr = np.sqrt(data_dict_tot[beam_combinations[j]]['PD_err_data']**2 + a2**2), fmt='o', color = color_combinations[i], alpha = 0.7, capsize=2, markersize=4)
                if i == 0:
                    custom_legend = plt.Line2D([0], [0], color='none', marker=None, linestyle='None', label=beam_combinations[j])
                    axes2[j].legend(handles=[custom_legend], loc='upper right', fontsize=14, handlelength=0, handletextpad=0, framealpha=0.1)
                    if j > 11:
                        axes2[j].set_xlabel("Hour Angle", fontsize=15)
                    if j % 3 == 0:
                        axes2[j].set_ylabel(r"$\delta \psi_{H-V} (deg)$", fontsize=15)

                if j == 0:

                    axes3[0].plot(HA, mp.func3(alt, az, popt.x)[j], color = color_combinations[i], alpha = 0.7, label='_nolegend_', lw=2)
                    axes3[0].errorbar(data_dict_tot[beam_combinations[j]]['HA_data'], data_dict_tot[beam_combinations[j]]['T1_ratio_data'], yerr = np.sqrt(data_dict_tot[beam_combinations[j]]['T1_ratio_err_data']**2 + a3**2), fmt = 'o', color = color_combinations[i], alpha = 0.7, capsize=2, markersize=4)
                    if i == 0:
                        custom_legend = plt.Line2D([0], [0], color='none', marker=None, linestyle='None', label=tel_combinations[j])
                        axes3[j].legend(handles=[custom_legend], loc='upper right', fontsize=14, handlelength=0, handletextpad=0, framealpha=0.1)
                        axes3[j].set_ylabel(r"$f_H / f_V$", fontsize=15)
                        axes3[j].axvline(0, linestyle='--', color='gray', linewidth=1)


                if j < 5: 
                    axes3[j+1].plot(HA, mp.func4(alt, az, popt.x)[j], color = color_combinations[i], alpha = 0.7, label='_nolegend_', lw=2)
                    axes3[j+1].errorbar(data_dict_tot[beam_combinations[j]]['HA_data'], data_dict_tot[beam_combinations[j]]['T2_ratio_data'], yerr = np.sqrt(data_dict_tot[beam_combinations[j]]['T2_ratio_err_data']**2 + a3**2), fmt = 'o', color = color_combinations[i], alpha = 0.7, capsize=2, markersize=4)

                    if i == 0:
                        custom_legend = plt.Line2D([0], [0], color='none', marker=None, linestyle='None', label=tel_combinations[j+1])
                        axes3[j+1].legend(handles=[custom_legend], loc='upper right', fontsize=14, handlelength=0, handletextpad=0, framealpha=0.1)
                        axes3[j+1].axvline(0, linestyle='--', color='gray', linewidth=1)

                        if j > 1:
                            axes3[j+1].set_xlabel("Hour Angle (hr)", fontsize = 15)
                        if j == 2:
                            axes3[j+1].set_ylabel(r"$f_H / f_V$", fontsize=15)
                        else:
                            axes3[j+1].set_yticklabels([])
                            axes3[j+1].tick_params(labelleft=False)
            
            if ii == 0:
                for jj in range(6):
                    axes3[jj].set_ylim(0.8,1.2)
            elif ii == 1:
                for jj in range(6):
                    axes3[jj].set_ylim(0.88, 1.3)
            elif ii == 2:
                for jj in range(6):
                    axes3[jj].set_ylim(0.88, 1.3)

        fig1.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.06)
        fig2.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.06)
        fig3.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.06)
                
        for ax in axes1:
            ax.label_outer()  # This hides x labels on inner subplots and keeps outer ones visible

        handles = [plt.Line2D([0], [0], color = color_combinations[i], lw=2) for i in range(len(wavel_combinations))]
        labels = [f'{wavel_combinations[i]}' for i in range(len(wavel_combinations))]
        
        legend1 = fig1.legend(handles, labels, loc='lower center', title='Wavelength ($\mu m$)', ncol=len(wavel_combinations), bbox_to_anchor=(0.5, -0.05), title_fontsize=14, prop={'size': 12})
        legend2 = fig2.legend(handles, labels, loc='lower center', title='Wavelength ($\mu m$)', ncol=len(wavel_combinations), bbox_to_anchor=(0.5, -0.05), title_fontsize=14, prop={'size': 12})
        legend3 = fig3.legend(handles, labels, loc='lower center', title='Wavelength ($\mu m$)', ncol=len(wavel_combinations), bbox_to_anchor=(0.5, -0.25), title_fontsize=14, prop={'size': 12})
        ############################################################################################################################################################################################
        fig1.savefig('data_file/fig_MYSTIC_2022_10_' + date[ii] + '_vis_ratios.pdf', dpi=300, bbox_extra_artists=(legend1,), bbox_inches='tight')
        fig2.savefig('data_file/fig_MYSTIC2022_10_' + date[ii] + '_diff_phases.pdf', dpi=300, bbox_extra_artists=(legend2,), bbox_inches='tight')
        fig3.savefig('data_file/fig_MYSTIC_2022_10_' + date[ii] + '_tel_ratios.pdf', dpi=300, bbox_extra_artists=(legend3,), bbox_inches='tight')

        np.save('data_file/MYSTIC_2022_10_' + date[ii] + '.npy', popt_x)

        ERR_norm_vis.append(Delta_norm_vis_err)
        ERR_diff_phase.append(Delta_diff_phase_err)
        ERR_flux_ratio.append(Delta_flux_ratio_err)

    norm_vis_residuals = np.concatenate(ERR_norm_vis)
    diff_phase_residuals = np.concatenate(ERR_diff_phase)
    flux_ratio_residuals = np.concatenate(ERR_flux_ratio)

    tot_chi2_vis = (Delta_norm_vis)/(len_data - 3*len(popt.x)) 
    tot_chi2_phase = (Delta_diff_phase)/(len_data - 3*len(popt.x))
    tot_chi2_flux = (Delta_flux_ratio)/(len_data_flux_ratio - 3*len(popt.x))

    print('-----------------------------------------------------------------------------------')
    print("sys errors:", a1, a2, a3)
    print('total chi2 for Norm Vis Ratio =', tot_chi2_vis)
    print('total chi2 for Diff Pahse =', tot_chi2_phase)
    print('total chi2 for Flux Ratio =', tot_chi2_flux)

    np.savez("data_file/residuals_MYSTIC_2022_10.npz",
         norm_vis_res=norm_vis_residuals,
         diff_phase_res=diff_phase_residuals,
         flux_ratio_res=flux_ratio_residuals)

    return None

plot_mircx(input)

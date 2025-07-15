"""
plot_chi2_median_difference_heatmap.py

This script compares the model fitting performance between the *global* and *individual* 
polarization models applied to CHARA/MIRC-X data across all wavelength channels and 
baseline combinations. It visualizes the difference in reduced chi-squared values 
(Δχ²_ν = χ²_ν,global − χ²_ν,individual) using a heatmap.

Key Functionalities:
---------------------
1. Loads previously saved χ² residuals for:
   - Normalized visibility ratio: Δχ²_ν(V_H/V_V)
   - Differential phase: Δχ²_ν(Δψ_{H−V})
   - Flux ratio (T1 and T2): Δχ²_ν(f_H/f_V)_{T1,T2}
2. For each baseline, computes the median Δχ²_ν between global and individual models.
3. Plots a heatmap:
   - x-axis: Observable types
   - y-axis: Beam combinations
   - Cell values: Median Δχ²_ν difference
   - Color: Red → individual model better; Blue → global model better

Required Input:
---------------
- Systematic error values: c1, c2, c3

Output:
-------
- data_file/fig_chi2_median_difference_heatmap.pdf
- NPY files per beam in `data_file/final/`:
    - chi2_overall<BEAM>.npy
    - chi2_individual<BEAM>.npy
"""

# Script implementation begins below
# %%
import mircxpol as mp
import mircxpol2 as mp_one
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord, EarthLocation, AltAz
import pandas as pd 
import os
import seaborn as sns


##
beam_combinations = ['E1W2', 'E1W1', 'E1S2', 'E1S1', 'E1E2', 'W2W1', 'W2S2', 'W2S1', 'W2E2', 'W1S2', 'W1S1', 'W1E2', 'S2S1', 'S2E2', 'S1E2']
date_combinations = ['2022Oct19', '2022Oct21', '2022Oct22']
time_combinations = ["2022-10-19 0:00:00", "2022-10-21 0:00:00", "2022-10-22 0:00:00"]
wavel_combinations = ['1.540', '1.571','1.602','1.631', '1.660', '1.688']
color_combinations = ['mediumpurple', 'turquoise','dodgerblue','limegreen','gold','orange','tomato']
index_combinations = [0, 5, 9, 11, 14]
tel_combinations = ['E1', 'W2', 'W1', 'S2', 'S1', 'E2']
date = ['19', '21', '22']

upsand = SkyCoord.from_name("ups and")
chara = EarthLocation.of_site("CHARA")

#48 params
var = np.array([1., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
bound = ((-np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf),#0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
    (np.inf, np.inf, np.inf, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)) 


var_one = np.array([1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.])
bound_one = ((-np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf),
       (np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf)) 


t_mjd = np.array([59871, 59873, 59874])

# %%
def func_data(wavel, beam, error_threshold=5):
    combinations = ['E1W2', 'E1W1', 'E1S2', 'E1S1', 'E1E2', 'W2W1', 'W2S2', 'W2S1', 'W2E2', 'W1S2', 'W1S1', 'W1E2', 'S2S1', 'S2E2', 'S1E2']
    
    data_dict = {}
    data_dict_tot = {}
    
    for i in range(len(combinations)):

        path_csv_i = './2022October_Wollaston_WAVELENGTH_DATA_MEGA/MJD_split_files/MIRCX_2022Oct_' + wavel + 'mu_' + beam + f'_MJD_59874.csv'
        df_i = pd.read_csv(path_csv_i)

        # Apply error threshold to filter out bad data
        mask = (df_i[' visratio_normed_err'].values < error_threshold) & \
               (df_i[' Phase_err'].values < error_threshold) & \
               (df_i['tel1_ratio_err'].values < error_threshold) & \
               (df_i['tel2_ratio_err'].values < error_threshold)

        df_i = df_i[mask]

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
        
        # Store all data in dictionary
        data_dict_tot[combinations[i]] = {
            'MJD_data': MJD_data_i,
            'HA_data': HA_data_i,
            'VisRatio_data': VisRatio_data_i,
            'VisRatio_err_data': VisRatio_err_data_i,
            'PD_data': PD_data_i,
            'PD_err_data': PD_err_data_i,
            'T1_ratio_data': T1_ratio_data_i,
            'T1_ratio_err_data': T1_ratio_err_data_i,
            'T2_ratio_data': T2_ratio_data_i,
            'T2_ratio_err_data': T2_ratio_err_data_i,
            'MJD': MJD_i,
            'tt': tt_i,
            'alt_data': alt_data_i,
            'az_data': az_data_i
        }

    return data_dict_tot


def func_data_one(wavel, beam, error_threshold=5):
    combinations = ['E1W2', 'E1W1', 'E1S2', 'E1S1', 'E1E2', 'W2W1', 'W2S2', 'W2S1', 'W2E2', 'W1S2', 'W1S1', 'W1E2', 'S2S1', 'S2E2', 'S1E2']
    
    data_dict = {}
    data_dict_tot = {}

    path_csv_i = './2022October_Wollaston_WAVELENGTH_DATA_MEGA/MJD_split_files/MIRCX_2022Oct_' + wavel + 'mu_' + beam + f'_MJD_59874.csv'
    df_i = pd.read_csv(path_csv_i)

    # Apply error threshold to filter out bad data
    mask = (df_i[' visratio_normed_err'].values < error_threshold) & \
            (df_i[' Phase_err'].values < error_threshold) & \
            (df_i['tel1_ratio_err'].values < error_threshold) & \
            (df_i['tel2_ratio_err'].values < error_threshold)

    df_i = df_i[mask]

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
    
    # Store all data in dictionary
    data_dict_tot = {
        'MJD_data': MJD_data_i,
        'HA_data': HA_data_i,
        'VisRatio_data': VisRatio_data_i,
        'VisRatio_err_data': VisRatio_err_data_i,
        'PD_data': PD_data_i,
        'PD_err_data': PD_err_data_i,
        'T1_ratio_data': T1_ratio_data_i,
        'T1_ratio_err_data': T1_ratio_err_data_i,
        'T2_ratio_data': T2_ratio_data_i,
        'T2_ratio_err_data': T2_ratio_err_data_i,
        'MJD': MJD_i,
        'tt': tt_i,
        'alt_data': alt_data_i,
        'az_data': az_data_i
    }

    return data_dict_tot

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


c1 = 0.0166
c2 = 1.8286
c3 = 0.0127

a1 = 0.0166
a2 = 1.8286
a3 = 0.0127

date = ['19', '21', '22']
ii = 2  # Choose the date index 

filename = f"data_file/MIRCX_2022_10_{date[ii]}.npy"
popt_x_loaded = np.load(filename, allow_pickle=True).item()

for k in range(len(beam_combinations)):

    beams = beam_combinations[k]

    popt_x = {}
    popt_x_one = {}

    chi2_overall = {}
    chi2_individual = {}

    tot_chi2_overall = {}
    tot_chi2_individual = {}

    Delta_norm_vis = []
    Delta_diff_phase = []
    Delta_flux_ratio_T1 = []
    Delta_flux_ratio_T2 = []

    Delta_norm_vis1 = []
    Delta_diff_phase1 = []
    Delta_flux_ratio1_T1 = []
    Delta_flux_ratio1_T2 = []

    for i in range(len(wavel_combinations)):

        data_dict = func_data(wavel_combinations[i],beams)
        popt = popt_x_loaded[wavel_combinations[i]]['popt.x']

    ##################################################################################################################################

        path_csv2 =  './2022October_Wollaston_WAVELENGTH_DATA_MEGA/MJD_split_files/MIRCX_2022Oct_' + wavel_combinations[i] + 'mu_' + beams + f'_MJD_59874.csv'
        df2 = pd.read_csv(path_csv2)
        data_dict_one = func_data_one(wavel_combinations[i],beams)

        MJD_data2 = df2['MJD'].values
        HA_data2 = df2[' HA'].values
        # VisRatio
        VisRatio_data2 = df2[' VisRatio_normed'].values
        VisRatio_err_data2 = df2[' visratio_normed_err'].values
        # PhaseDiff
        PD_data2 = df2[' PhaseDiff'].values
        PD_err_data2 = df2[' Phase_err'].values
        # Tel1_ratio
        T1_ratio_data2 = df2[' Tel1_ratio'].values
        T1_ratio_err_data2 = df2['tel1_ratio_err'].values
        # Tel2_ratio
        T2_ratio_data2 = df2['tel2_ratio'].values
        T2_ratio_err_data2 = df2['tel2_ratio_err'].values

        MJD2 = Time(MJD_data2, format='mjd')
        tt2 = MJD2.to_value('iso')
        upsandaltaz2 = upsand.transform_to(AltAz(obstime = tt2, location=chara))
        alt_data2 = upsandaltaz2.alt
        az_data2 = upsandaltaz2.az

        popt_one = least_squares(mp_one.err_global, var_one + np.random.uniform(-1, 1, size=(len(var_one))) * 0.3, args=(alt_data2, az_data2, VisRatio_data2, PD_data2, T1_ratio_data2, T2_ratio_data2, VisRatio_err_data2, PD_err_data2, T1_ratio_err_data2, T2_ratio_err_data2), 
                        bounds = bound_one)  

        popt_x_one[wavel_combinations[i]] = {'popt.x': popt_one.x}
    ##################################################################################################################################   

        alt_k = data_dict[beam_combinations[k]]['alt_data']
        az_k = data_dict[beam_combinations[k]]['az_data']

        delta_vis_ratio = np.array((mp.func1(alt_k, az_k, popt)[k] - data_dict[beam_combinations[k]]['VisRatio_data'])**2 / ((data_dict[beam_combinations[k]]['VisRatio_err_data'])**2 + c1**2))
        Delta_norm_vis = np.append(Delta_norm_vis, delta_vis_ratio, axis = 0)

        delta_diff_phase = np.array((mp.func2(alt_k, az_k, popt)[k] - data_dict[beam_combinations[k]]['PD_data'])**2 / ((data_dict[beam_combinations[k]]['PD_err_data'])**2 + c2**2))
        Delta_diff_phase = np.append(Delta_diff_phase, delta_diff_phase, axis = 0)

        delta_flux_ratio_T1 = np.array((mp.func3(alt_k, az_k, popt)[k] - data_dict[beam_combinations[k]]['T1_ratio_data'])**2 / ((data_dict[beam_combinations[k]]['T1_ratio_err_data'])**2 + c3**2))
        Delta_flux_ratio_T1 = np.append(Delta_flux_ratio_T1, delta_flux_ratio_T1, axis = 0)

        delta_flux_ratio_T2 = np.array((mp.func4(alt_k, az_k, popt)[k] - data_dict[beam_combinations[k]]['T2_ratio_data'])**2 / ((data_dict[beam_combinations[k]]['T2_ratio_err_data'])**2 + c3**2))
        Delta_flux_ratio_T2 = np.append(Delta_flux_ratio_T2, delta_flux_ratio_T2, axis = 0)
    ##################################################################################################################################  

        delta_vis_ratio1 = np.array((mp_one.func1(alt_k, az_k, popt_one.x) - data_dict[beam_combinations[k]]['VisRatio_data'])**2 / ((data_dict[beam_combinations[k]]['VisRatio_err_data'])**2 + c1**2))
        Delta_norm_vis1 = np.append(Delta_norm_vis1, delta_vis_ratio1, axis = 0)

        delta_diff_phase1 = np.array((mp_one.func2(alt_k, az_k, popt_one.x) - data_dict[beam_combinations[k]]['PD_data'])**2 / ((data_dict[beam_combinations[k]]['PD_err_data'])**2 + c2**2))
        Delta_diff_phase1 = np.append(Delta_diff_phase1, delta_diff_phase1, axis = 0)

        delta_flux_ratio1_T1 = np.array((mp_one.func3(alt_k, az_k, popt_one.x) - data_dict[beam_combinations[k]]['T1_ratio_data'])**2 / ((data_dict[beam_combinations[k]]['T1_ratio_err_data'])**2 + c3**2))
        Delta_flux_ratio1_T1 = np.append(Delta_flux_ratio1_T1, delta_flux_ratio1_T1, axis = 0)

        delta_flux_ratio1_T2 = np.array((mp_one.func4(alt_k, az_k, popt_one.x) - data_dict[beam_combinations[k]]['T2_ratio_data'])**2 / ((data_dict[beam_combinations[k]]['T2_ratio_err_data'])**2 + c3**2))
        Delta_flux_ratio1_T2 = np.append(Delta_flux_ratio1_T2, delta_flux_ratio1_T2, axis = 0)
    ##################################################################################################################################  

    # Compute chi2 values for overall case
        tot_chi2_overall[wavel_combinations[i]] = {
        "wavelength": wavel_combinations[i],
        "chi2_norm_vis": Delta_norm_vis,
        "chi2_diff_phase": Delta_diff_phase,
        "chi2_flux_ratio_T1": (Delta_flux_ratio_T1),
        "chi2_flux_ratio_T2": (Delta_flux_ratio_T2),
        "len_Delta_norm_vis": len(Delta_norm_vis),
        "len_Delta_flux_ratio_T1": len(Delta_flux_ratio_T1),
        "len_Delta_flux_ratio_T2": len(Delta_flux_ratio_T2),
        "len_popt": len(popt)
    }

    # Compute chi2 values for individual case
        tot_chi2_individual[wavel_combinations[i]] = {
        "wavelength": wavel_combinations[i],
        "chi2_norm_vis": (Delta_norm_vis1),
        "chi2_diff_phase": (Delta_diff_phase1),
        "chi2_flux_ratio_T1": (Delta_flux_ratio1_T1),
        "chi2_flux_ratio_T2": (Delta_flux_ratio1_T2),
        "len_Delta_norm_vis": len(Delta_norm_vis1),
        "len_Delta_flux_ratio_T1": len(Delta_flux_ratio1_T1),
        "len_Delta_flux_ratio_T2": len(Delta_flux_ratio1_T2),
        "len_popt": len(popt_one.x)
    }
 ##################################################################################################################################  

    np.save('data_file/final/chi2_overall'+ beams + '.npy', tot_chi2_overall)
    np.save('data_file/final/chi2_individual'+ beams + '.npy', tot_chi2_individual)

# %%
file_path = "./data_file/final" 
beam_combinations = [
    'E1W2', 'E1W1', 'E1S2', 'E1S1', 'E1E2',
    'W2W1', 'W2S2', 'W2S1', 'W2E2',
    'W1S2', 'W1S1', 'W1E2',
    'S2S1', 'S2E2', 'S1E2'
]
param_keys = {
    "chi2_norm_vis": ("len_Delta_norm_vis", r"$\chi^2(V_H/V_V)$"),
    "chi2_diff_phase": ("len_Delta_norm_vis", r"$\chi^2(\Delta(\psi_H - \psi_V))$"),
    "chi2_flux_ratio_T1": ("len_Delta_flux_ratio_T1", r"$\chi^2(f_H/f_V) T1$"),
    "chi2_flux_ratio_T2": ("len_Delta_flux_ratio_T2", r"$\chi^2(f_H/f_V) T2$")
}
# --------------------------------

median_diff = {key: [] for key in param_keys}

for beam in beam_combinations:
    try:
        f_o = os.path.join(file_path, f"chi2_overall{beam}.npy")
        f_i = os.path.join(file_path, f"chi2_individual{beam}.npy")
        d_o = np.load(f_o, allow_pickle=True).item()
        d_i = np.load(f_i, allow_pickle=True).item()

        for key, (len_key, label) in param_keys.items():
            diffs = []
            for wave in set(d_o).intersection(d_i):
                try:
                    chi_o = np.sum(d_o[wave][key]) / (np.sum(d_o[wave][len_key]) - 48)
                    chi_i = np.sum(d_i[wave][key]) / (np.sum(d_o[wave][len_key]) - 13)
                    dif = chi_o - chi_i
                    if (np.sum(d_o[wave][len_key]) - 48) > 0:
                        diffs.append(dif)
                except:
                    continue
            median_diff[key].append(np.median(diffs) if diffs else np.nan)
    except Exception as e:
        for key in param_keys:
            median_diff[key].append(np.nan)

# Build dataframe and plot
df = pd.DataFrame(median_diff, index=beam_combinations)
custom_x_labels = [v[1] for v in param_keys.values()]

# Custom x-axis labels with LaTeX formatting
custom_x_labels = [
    r"$\Delta \chi_{\nu}^2(V_H/V_V)$", 
    r"$\Delta \chi_{\nu}^2(\Delta\psi_{H-V})$", 
    r"$\Delta \chi_{\nu}^2(f_H/f_V)_{T1}$", 
    r"$\Delta \chi_{\nu}^2(f_H/f_V)_{T2}$"
]

# Plot setup
fig, ax = plt.subplots(figsize=(10, 9))
sns.set(font="serif", style="whitegrid")

# Heatmap
heatmap = sns.heatmap(
    df,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.4,
    linecolor='gray',
    center=0,
    annot_kws={"size": 14},
    cbar_kws={"label": r"$\Delta\chi_{\nu}^2$"}
)

# Title and labels
plt.title(r"$\chi^2$ Median Difference Heatmap (Overall $-$ Individual)", fontsize=20, pad=20)
ax.set_xticklabels(custom_x_labels, rotation=0, fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

# Adjust colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=13)
cbar.set_label(r"$\Delta\chi_{\nu}^2$", fontsize=16)

# Remove spines and keep clean axis
sns.despine(left=True, bottom=True)

# Add tighter layout and save
fig.tight_layout()
fig.savefig("data_file/fig_chi2_median_difference_heatmap.pdf", dpi=300, bbox_inches="tight")


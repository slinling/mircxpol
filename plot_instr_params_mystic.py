"""
plot_instr_params_mystic.py

This script visualizes the wavelength-dependent instrumental polarization parameters 
fitted for each CHARA/MYSTIC telescope beam (E1, W2, W1, S2, S1, E2) across three nights 
of observation (Oct 19, 21, 22, 2022).

Key Functionalities:
---------------------
1. Loads parameter fit results saved from the global model fitting script (`fit_sys_error_and_plot_mystic.py`).
2. Aligns phase and amplitude parameters by removing night-to-night average offsets.
3. Separates amplitude and phase components for:
   - AT: Beam train from primary mirror
   - Coudé: Intermediate optics
   - Lab: Final detection optics
4. Produces 6 subplots (amplitude and phase per stage) showing variation across wavelength.
5. Adds error bars based on standard deviation across nights.
6. Saves the combined figure as a high-quality PDF.

Required Input:
---------------
- `MYSTIC_2022_10_19.npy`
- `MYSTIC_2022_10_21.npy`
- `MYSTIC_2022_10_22.npy`
These files must contain dictionaries with key `'popt.x'` for each wavelength.
"""

# Script implementation begins below
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.time import Time


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
# %%
params1 = np.load('data_file/MYSTIC_2022_10_19.npy', allow_pickle=True).item()
params2 = np.load('data_file/MYSTIC_2022_10_21.npy', allow_pickle=True).item()
params3 = np.load('data_file/MYSTIC_2022_10_22.npy', allow_pickle=True).item()
############################################################################################################################################################################################
params1_add_zeros = params1
params2_add_zeros = params2
params3_add_zeros = params3

positions = [1, 3, 5]
# Function to insert zeros at specific positions
def insert_zeros(array, positions):
    # Convert array to a list for easy insertion
    array_list = array.tolist()
    for pos in sorted(positions):
        array_list.insert(pos, 0)
    return np.array(array_list)

# Iterate over each key-value pair and insert zeros
for key in params1_add_zeros:
    params1_add_zeros[key]['popt.x'] = insert_zeros(params1_add_zeros[key]['popt.x'], positions)
    
for key in params2_add_zeros:
    params2_add_zeros[key]['popt.x'] = insert_zeros(params2_add_zeros[key]['popt.x'], positions)

for key in params3_add_zeros:
    params3_add_zeros[key]['popt.x'] = insert_zeros(params3_add_zeros[key]['popt.x'], positions)

params1_arrays = [v['popt.x'] for v in params1_add_zeros.values()]
params2_arrays = [v['popt.x'] for v in params2_add_zeros.values()]
params3_arrays = [v['popt.x'] for v in params3_add_zeros.values()]
params1 = np.stack(params1_arrays)
params2 = np.stack(params2_arrays)
params3 = np.stack(params3_arrays)

# Stack the arrays into a single 3D array with shape (3, 6, 51)
stacked_arrays = np.stack([params1, params2, params3])

# Calculate the mean along the first axis (which corresponds to the 3 arrays)
mean_array = np.mean(stacked_arrays, axis=0)

delta_1 = params1 - mean_array
delta_2 = params2 - mean_array
delta_3 = params3 - mean_array

delta1 = np.zeros(51)
delta2 = np.zeros(51)
delta3 = np.zeros(51)
for i in range(51):

    delta1[i] = np.mean(delta_1[:, i])
    delta2[i] = np.mean(delta_2[:, i])
    delta3[i] = np.mean(delta_3[:, i])

params1_new = params1 - delta1
params2_new = params2 - delta2
params3_new = params3 - delta3

# %%
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.03, wspace=0.2)
axes = []
tel_list = ['E1', 'W2', 'W1', 'S2', 'S1', 'E2']
sns.set_palette("colorblind")
colors = sns.color_palette("colorblind", len(wavel_combinations_22))

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
x = np.array(['2.035', '2.075', '2.115','2.154','2.193', '2.232', '2.270', '2.308', '2.345', '2.374'])

params_list1 = [
    r'$\Delta A_{AT}^2$', r'$\Delta \psi_{AT}\,[\mathrm{deg}]$', 
    r'$\Delta A_{\mathrm{Coud\acute{e}}}^2$', r'$\Delta \psi_{\mathrm{Coud\acute{e}}}\,[\mathrm{deg}]$', 
    r'$\Delta A_{Lab}^2$', r'$\Delta \psi_{Lab}\,[\mathrm{deg}]$'
]

for j in range(6):
    ax = fig.add_subplot(gs[j // 2, j % 2])
    axes.append(ax)

    for k in range(len(tel_list)):
        idx = 6 * k + j
        y1 = np.degrees(params1_new[:, idx]) if (idx + 1) % 2 == 0 else (params1_new[:, idx])**2
        y2 = np.degrees(params2_new[:, idx]) if (idx + 1) % 2 == 0 else (params2_new[:, idx])**2
        y3 = np.degrees(params3_new[:, idx]) if (idx + 1) % 2 == 0 else (params3_new[:, idx])**2

        y_mean = np.mean([y1, y2, y3], axis=0)
        y_std = np.std([y1, y2, y3], axis=0)

        color = color_cycle[k % len(color_cycle)]

        # Main line
        ax.plot(x, y_mean, linestyle='-', linewidth=2, color=color)

        # Error bars with transparency
        ax.errorbar(x, y_mean, yerr=y_std, fmt='o', color=color, alpha=0.6, capsize=3, markersize=4)

    # Labeling
    ax.set_ylabel(params_list1[j], fontsize=20)
    if j >= 4:
        ax.set_xlabel(r'$\lambda\ (\mu \mathrm{m})$', fontsize=20)

    # Ticks
    ax.tick_params(axis='both', labelsize=12)

# Adjust layout
plt.subplots_adjust(left=0.1, right=0.92, top=0.96, bottom=0.13)

# Legend
handles = [plt.Line2D([0], [0], color=color_cycle[i], lw=2) for i in range(len(tel_list))]
labels = tel_list
fig.legend(
    handles=handles,
    labels=labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.015),
    ncol=6,
    fontsize=15,
    frameon=False  # <-- this removes the surrounding box!
)

plt.savefig("data_file/fig_params_MYSTIC.pdf", dpi=300, bbox_inches='tight')
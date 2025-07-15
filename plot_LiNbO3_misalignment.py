"""
plot_differential_phase_vs_wavelength.py

This script plots the differential phase delay (Δφ) as a function of wavelength 
for two CHARA beam combiners: MIRC-X (H band) and MYSTIC (K band). It compares 
observed instrumental parameters against a model prediction based on a dual 
LiNbO₃ plate setup.

Key Functionalities:
---------------------
1. Calculates theoretical phase delays using a Sellmeier model for LiNbO₃ birefringence.
2. Loads wavelength-dependent fitted instrumental parameters (ψ_lab) for:
   - S1 in MIRC-X (H band)
   - W2 in MYSTIC (K band)
3. Aligns the model curves to observed values by removing wavelength-independent offsets.
4. Overlays model predictions with measured data for Oct 19, 21, and 22.
5. Saves a 2-panel PDF showing Δφ(λ) for MIRC-X and MYSTIC.

Required Input:
---------------
- Numpy `.npy` files with instrumental parameters:
    - MIRCX_2022_10_19.npy, _21.npy, _22.npy
    - MYSTIC_2022_10_19.npy, _21.npy, _22.npy
- LiNbO₃ Sellmeier coefficients (hardcoded)
- Custom formatting for CHARA publication

"""

# Script implementation begins below
# %%
import mircxpol as mp
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
thick = 4.0  # mm
angle1_H = np.deg2rad(19.94)  # radians
angle2_H = np.deg2rad(22.09)  # radians
angle1_K = np.deg2rad(19.94)  # radians
angle2_K = np.deg2rad(22.79)  # radians

coef_no = [2.6734, 1.2290, 12.614, 0.01764, 0.05914, 474.60]
coef_ne = [2.9804, 0.5981, 8.9543, 0.02047, 0.0666, 416.08]

# Wavelengths for H and K bands (microns)
l_H = np.linspace(1.54, 1.688, 5000)
l_K = np.linspace(2.035, 2.374, 5000)

def sellmeier(l, coef):
    term = coef[0] * l**2 / (l**2 - coef[3]) + \
           coef[1] * l**2 / (l**2 - coef[4]) + \
           coef[2] * l**2 / (l**2 - coef[5])
    return np.sqrt(1.0 + term)

def compute_dphi(l, coef_no, coef_ne, angle1, angle2, thick):
    n_no = sellmeier(l, coef_no)
    n_ne = sellmeier(l, coef_ne)
    
    phi_o_1 = thick * 2 * np.pi / (l * 1e-3) * np.sqrt(n_no**2 - np.sin(angle1)**2)
    phi_e_1 = thick * 2 * np.pi / (l * 1e-3) * (n_no / n_ne) * np.sqrt(n_ne**2 - np.sin(angle1)**2)

    phi_o_2 = thick * 2 * np.pi / (l * 1e-3) * np.sqrt(n_no**2 - np.sin(angle2)**2)
    phi_e_2 = thick * 2 * np.pi / (l * 1e-3) * (n_no / n_ne) * np.sqrt(n_ne**2 - np.sin(angle2)**2)

    air = (n_no / np.sqrt(1.0 - (np.sin(angle1) / n_no)**2) -
           n_no / np.sqrt(1.0 - (np.sin(angle2) / n_no)**2)) * thick  # mm

    air = np.mean(air)
    print(air)

    dphi_o = np.degrees((phi_o_2 - phi_o_1) - 2.0 * np.pi * air / (l * 1e-3)) 
    dphi_e = np.degrees((phi_e_2 - phi_e_1) - 2.0 * np.pi * air / (l * 1e-3)) 
    
    return dphi_o, dphi_e, -dphi_o + dphi_e

# Compute phase differences
dphi_o_H, dphi_e_H, x1 = compute_dphi(l_H, coef_no, coef_ne, angle1_H, angle2_H, thick)
dphi_o_K, dphi_e_K, y1 = compute_dphi(l_K, coef_no, coef_ne, angle1_K, angle2_K, thick)

dphi_o_e_H = -dphi_o_H + dphi_e_H
dphi_o_e_K = -dphi_o_K + dphi_e_K

# %%
params1_H = np.load('data_file/MIRCX_2022_10_19.npy', allow_pickle=True).item()
params2_H= np.load('data_file/MIRCX_2022_10_21.npy', allow_pickle=True).item()
params3_H = np.load('data_file/MIRCX_2022_10_22.npy', allow_pickle=True).item()
############################################################################################################################################################################################
params1_add_zeros_H = params1_H
params2_add_zeros_H = params2_H
params3_add_zeros_H= params3_H

positions = [1, 3, 5]
# Function to insert zeros at specific positions
def insert_zeros(array, positions):
    # Convert array to a list for easy insertion
    array_list = array.tolist()
    for pos in sorted(positions):
        array_list.insert(pos, 0)
    return np.array(array_list)

# Iterate over each key-value pair and insert zeros
for key in params1_add_zeros_H:
    params1_add_zeros_H[key]['popt.x'] = insert_zeros(params1_add_zeros_H[key]['popt.x'], positions)
    
for key in params2_add_zeros_H:
    params2_add_zeros_H[key]['popt.x'] = insert_zeros(params2_add_zeros_H[key]['popt.x'], positions)

for key in params3_add_zeros_H:
    params3_add_zeros_H[key]['popt.x'] = insert_zeros(params3_add_zeros_H[key]['popt.x'], positions)

params1_arrays_H = [v['popt.x'] for v in params1_add_zeros_H.values()]
params2_arrays_H = [v['popt.x'] for v in params2_add_zeros_H.values()]
params3_arrays_H= [v['popt.x'] for v in params3_add_zeros_H.values()]
params1_H = np.stack(params1_arrays_H)
params2_H = np.stack(params2_arrays_H)
params3_H = np.stack(params3_arrays_H) #  excludes the first row (i.e., the first (1, 51))
# params3.shape

# Stack the arrays into a single 3D array with shape (3, 6, 51)
stacked_arrays_H = np.stack([params1_H, params2_H, params3_H])

# Calculate the mean along the first axis (which corresponds to the 3 arrays)
mean_array_H = np.mean(stacked_arrays_H, axis=0)
# mean_array[0]

delta_1_H = params1_H - mean_array_H
delta_2_H = params2_H - mean_array_H
delta_3_H = params3_H - mean_array_H

delta1_H = np.zeros(51)
delta2_H = np.zeros(51)
delta3_H = np.zeros(51)
for i in range(51):

    delta1_H[i] = np.mean(delta_1_H[:, i])
    delta2_H[i] = np.mean(delta_2_H[:, i])
    delta3_H[i] = np.mean(delta_3_H[:, i])

params1_new_H = params1_H - delta1_H
params2_new_H= params2_H - delta2_H
params3_new_H = params3_H - delta3_H

x_H = np.array([1.540, 1.571, 1.602, 1.631, 1.660, 1.688])
# match_wavelength_H = 1.658
match_wavelength_H = 1.655
idx_model_H = np.argmin(np.abs(l_H - match_wavelength_H))  # closest index in l_H
idx_data_H = np.argmin(np.abs(x_H - match_wavelength_H))     # index in x

# S1, ψ_lab = column index 29 = 6 * 4 + 5
observed_value_H = np.degrees(params1_new_H[idx_data_H, 6 * 4 + 5])
model_value_H = dphi_o_e_H[idx_model_H]

# Compute offset and align model
offset_H = model_value_H - observed_value_H
model_aligned_H = dphi_o_e_H - offset_H

# %%
##########################################################################################################################################################################################################
params1_K = np.load('data_file/MYSTIC_2022_10_19.npy', allow_pickle=True).item()
params2_K = np.load('data_file/MYSTIC_2022_10_21.npy', allow_pickle=True).item()
params3_K = np.load('data_file/MYSTIC_2022_10_22.npy', allow_pickle=True).item()


params1_add_zeros_K = params1_K
params2_add_zeros_K = params2_K
params3_add_zeros_K = params3_K

positions = [1, 3, 5]

def insert_zeros(array, positions):
    array_list = array.tolist()
    for pos in sorted(positions):
        array_list.insert(pos, 0)
    return np.array(array_list)

for key in params1_add_zeros_K:
    params1_add_zeros_K[key]['popt.x'] = insert_zeros(params1_add_zeros_K[key]['popt.x'], positions)

for key in params2_add_zeros_K:
    params2_add_zeros_K[key]['popt.x'] = insert_zeros(params2_add_zeros_K[key]['popt.x'], positions)

for key in params3_add_zeros_K:
    params3_add_zeros_K[key]['popt.x'] = insert_zeros(params3_add_zeros_K[key]['popt.x'], positions)

params1_arrays_K = [v['popt.x'] for v in params1_add_zeros_K.values()]
params2_arrays_K = [v['popt.x'] for v in params2_add_zeros_K.values()]
params3_arrays_K = [v['popt.x'] for v in params3_add_zeros_K.values()]

params1_K = np.stack(params1_arrays_K)
params2_K = np.stack(params2_arrays_K)
params3_K = np.stack(params3_arrays_K)

stacked_arrays_K = np.stack([params1_K, params2_K, params3_K])
mean_array_K = np.mean(stacked_arrays_K, axis=0)

delta_1_K = params1_K - mean_array_K
delta_2_K = params2_K - mean_array_K
delta_3_K = params3_K - mean_array_K

delta1_K = np.zeros(51)
delta2_K = np.zeros(51)
delta3_K = np.zeros(51)

for i in range(51):
    delta1_K[i] = np.mean(delta_1_K[:, i])
    delta2_K[i] = np.mean(delta_2_K[:, i])
    delta3_K[i] = np.mean(delta_3_K[:, i])

params1_new_K = params1_K - delta1_K
params2_new_K = params2_K - delta2_K
params3_new_K = params3_K - delta3_K

x_K = np.array([2.035, 2.075, 2.115,2.154,2.193, 2.232, 2.270, 2.308, 2.345, 2.374])
# match_wavelength_K = 2.121
match_wavelength_K = 2.12
idx_model_K = np.argmin(np.abs(l_K - match_wavelength_K))
idx_data_K = np.argmin(np.abs(x_K - match_wavelength_K))

observed_value_K = np.degrees(params1_new_K[idx_data_K, 6 * 1 + 5])
model_value_K = dphi_o_e_K[idx_model_K]

offset_K = model_value_K - observed_value_K
model_aligned_K = dphi_o_e_K - offset_K


# %%

# Use serif fonts and adjust global formatting for publication
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
})

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

# Define colorblind-friendly palette
colors = {
    "Oct 19": "#1f77b4",  # blue
    "Oct 21": "#ff7f0e",  # orange
    "Oct 22": "#2ca02c",  # green
}

# ----------------------------
# LEFT: MIRC-X S1 (H band)
# ----------------------------
axs[0].plot(l_H, model_aligned_H, label="LiNbO₃ Plate Model",
            color="black", linestyle="--", linewidth=2.5)
axs[0].plot(x_H, np.degrees(params1_new_H[:, 6 * 4 + 5]), label="Oct 19",
            color=colors["Oct 19"], marker='o', linestyle='-', linewidth=1.5)
axs[0].plot(x_H, np.degrees(params2_new_H[:, 6 * 4 + 5]), label="Oct 21",
            color=colors["Oct 21"], marker='s', linestyle='-', linewidth=1.5)
axs[0].plot(x_H, np.degrees(params3_new_H[:, 6 * 4 + 5]), label="Oct 22",
            color=colors["Oct 22"], marker='^', linestyle='-', linewidth=1.5)

axs[0].set_title("MIRC-X S1 (H band)")
axs[0].set_xlabel(r"Wavelength $\lambda$ [$\mu$m]")
axs[0].set_ylabel(r"$\Delta \phi$ [deg]")
axs[0].legend(loc='upper left', frameon=False)

# ----------------------------
# RIGHT: MYSTIC W2 (K band)
# ----------------------------
axs[1].plot(l_K, model_aligned_K, label="LiNbO₃ Plate Model",
            color="black", linestyle="--", linewidth=2.5)
axs[1].plot(x_K, np.degrees(params1_new_K[:, 6 * 1 + 5]), label="Oct 19",
            color=colors["Oct 19"], marker='o', linestyle='-', linewidth=1.5)
axs[1].plot(x_K, np.degrees(params2_new_K[:, 6 * 1 + 5]), label="Oct 21",
            color=colors["Oct 21"], marker='s', linestyle='-', linewidth=1.5)
axs[1].plot(x_K, np.degrees(params3_new_K[:, 6 * 1 + 5]), label="Oct 22",
            color=colors["Oct 22"], marker='^', linestyle='-', linewidth=1.5)

axs[1].set_title("MYSTIC W2 (K band)")
axs[1].set_xlabel(r"Wavelength $\lambda$ [$\mu$m]")
axs[1].legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig("data_file/fig_sys.pdf", dpi=300, bbox_inches="tight")
plt.show()


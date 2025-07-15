"""
plot_joint_residual_histograms.py

This script visualizes the residual distributions from global polarization model fits 
for CHARA/MIRCX and CHARA/MYSTIC interferometric instruments. It generates a 2×3 panel 
of histograms comparing model residuals for:
  - Normalized visibility ratio
  - Differential phase
  - Flux ratio

Key Functionalities:
---------------------
1. Loads residuals from two `.npz` files, each containing arrays for:
   - vis: Normalized visibility ratio residuals
   - phase: Differential phase residuals (in degrees)
   - flux: Flux ratio residuals
2. Computes basic statistics: mean, median, RMS, standard deviation.
3. Generates histograms with ±1σ markers for:
   - MIRC-X (top row)
   - MYSTIC (bottom row)
4. Annotates each plot with RMS value and standard deviation lines.
5. Saves a high-resolution summary figure as PDF.

Required Input:
---------------
- data_file/mircx_residuals.npz
- data_file/mystic_residuals.npz

Each file should contain:
  - 'vis': array of visibility residuals
  - 'phase': array of differential phase residuals
  - 'flux': array of flux ratio residuals
Output:
-------
- data_file/fig_residual_histograms.pdf
"""

# Script implementation begins below
# %%

import numpy as np
import matplotlib.pyplot as plt

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

# %%
def compute_stats(data):
    mean = np.mean(data)
    median = np.median(data)
    rms = np.sqrt(np.mean(data**2))
    std = np.std(data)
    return mean, median, rms, std

def add_sigma_lines(ax, mean, std, ymax, label_color='gray'):
    for sigma in [-1, 1]:
        x = mean + sigma * std
        ax.axvline(x, color=label_color, linestyle='--', linewidth=1)
        label = f'{sigma:+d}σ'  # formats as '+1σ' or '-1σ'
        ha = 'right' if sigma == -1 else 'left'
        ax.text(x, ymax * 0.65, label,
                rotation=0, va='top', ha=ha, fontsize=15, color=label_color)


# %%
data_mircx = np.load("data_file/mircx_residuals.npz")
data_mystic = np.load("data_file/mystic_residuals.npz")

# Compute stats for both
residual_sets = [
    (data_mircx['vis'], data_mircx['phase'], data_mircx['flux']),
    (data_mystic['vis'], data_mystic['phase'], data_mystic['flux'])
]
titles = ["Normalized Visibility", "Differential Phase", "Flux Ratio"]
colors = ["skyblue", "salmon", "lightgreen"]
instrument_labels = ["MIRC-X", "MYSTIC"]

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex='col')

for row, (residuals, instrument) in enumerate(zip(residual_sets, instrument_labels)):
    for col, (ax, data, title, color) in enumerate(zip(axes[row], residuals, titles, colors)):
        mean, median, rms, std = compute_stats(data)
        bin_width = 0.02 if col == 0 else (0.8 if col == 1 else 0.01)
        data_min, data_max = np.min(data), np.max(data)
        bins = np.arange(data_min, data_max + bin_width, bin_width)
        counts, bins, _ = ax.hist(data, bins=bins, color=color, edgecolor='black')
        ymax = counts.max()

        ax.set_title(f"{instrument}: {title}")
        if row == 1 and col == 1:
            ax.set_xlabel("Residual (deg)")
        elif row == 1:
            ax.set_xlabel("Residual")
            
        if col == 0:
            ax.set_ylabel("Count")  # Only the leftmost column gets y-axis labels

        # Set custom x-axis limits for better visual comparison
        if col == 0:  # Normalized Vis
            ax.set_xlim(-0.5, 0.5)
        elif col == 1:  # Differential Phase
            ax.set_xlim(-15, 15)
        elif col == 2:  # Flux Ratio
            ax.set_xlim(-0.1, 0.2)


        add_sigma_lines(ax, mean, std, ymax)
        ax.text(0.98, 0.95,
                f"RMS = {rms:.3f}",
                transform=ax.transAxes,
                ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

plt.tight_layout()
plt.savefig("data_file/fig_residual_histograms.pdf", dpi=300, bbox_inches="tight")
plt.show()





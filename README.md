# CHARA Polarization Modeling and Systematics Analysis

This repository contains Python scripts and modules used to analyze polarization effects 
in interferometric observations with the CHARA Array, focusing on the MIRC-X and MYSTIC instruments. 
The analysis compares global and individual polarization models, estimates systematic errors, 
and visualizes model performance across wavelengths and baselines.

---

## Repository Structure

```
.
├── analyze_obs_durations.py       # Prints max/min observing durations for MIRCX & MYSTIC
├── analyze_sys_error_mircx.py               # Systematic error fitting for MIRCX
├── analyze_sys_error_mystic.py              # Systematic error fitting for MYSTIC
├── plot_sys_error_fit_mircx.py         # Apply sys err and plot modeled vs observed data (MIRC-X)
├── plot_sys_error_fit_mystic.py         # Apply sys err and plot modeled vs observed data (MYSTIC)
├── mircxpol.py                    # Polarization modeling core module
├── plot_chi2_median_difference_heatmap.py   # Heatmap comparing global vs individual χ² fits
├── plot_LiNbO3_misalignment.py # Δφ(λ) plots for MIRCX & MYSTIC
├── plot_instr_params_mircx.py               # Instrumental parameter trends (MIRCX)
├── plot_instr_params_mystic.py              # Instrumental parameter trends (MYSTIC)
├── plot_joint_residual_histograms.py        # Histogram of residuals for MIRCX and MYSTIC
├── data_file/                               # Directory for .npy/.npz residual and fit results
```

---

##  Main Results

- Systematic Error Estimation: Estimates additive systematic error terms for each observable
  (vis ratio, diff phase, flux ratio).
- Global vs. Individual Models: Visualizes chi-square improvement for each baseline and parameter.
- Instrumental Parameters: Extracted as a function of wavelength and telescope arm.
- Differential Phase Modeling: Compared to birefringent plate model using Sellmeier equation.
- Residual Distributions: Compared across instruments and observables.

---

##  Requirements

- Python ≥ 3.8  
- numpy  
- scipy  
- pandas  
- matplotlib  
- seaborn  
- astropy  

Custom module:
- `mircxpol.py`: Defines polarization model (matrix formalism)

---

## Run Guide: 
This section outlines the full sequence for running the CHARA polarization model fitting and analysis.

1. Estimate Systematic Errors
```bash
python analyze_sys_error_mircx.py     # For MIRC-X (H-band)
python analyze_sys_error_mystic.py    # For MYSTIC (K-band)
```

2. Apply Systematic Errors and Plot Fits
```bash
python plot_sys_error_fit_mircx.py     # Visualize model fits for MIRC-X
python plot_sys_error_fit_mystic.py    # Visualize model fits for MYSTIC
```

3. Plot Instrumental Parameters
```bash
python plot_instr_params_mircx.py     # Wavelength-dependent ΔA² and Δψ (MIRC-X)
python plot_instr_params_mystic.py    # Same for MYSTIC
```

4. Compare Residuals
```bash
python plot_joint_residual_histograms.py
```

---

###  Optional Steps

#### Compare Global vs Individual Fits
```bash
python plot_chi2_median_difference_heatmap.py
```

####  Plot Phase Model: Dual-Plate LiNbO₃
```bash
python plot_LiNbO3_misalignment.py
```

####  Print Observing Duration Coverage
```bash
python analyze_obs_durations.py
```

---

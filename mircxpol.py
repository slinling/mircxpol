#!/usr/bin/env python
# coding: utf-8

# Inputs from command line:
#   sys.argv[1]: file name for your data // e.g. "2022Oct19_W1S1.csv"
#   sys.argv[2]: name of the system // e.g. "ups and"
#   sys.argv[3]: observing date // e.g. "2022-10-19"

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord, EarthLocation, AltAz
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider
from IPython.display import display, HTML
import pandas as pd 

def J(alt, az, M13, M47, M819):

    alt_rad = np.radians(alt)
    az_rad = np.radians(az)

    matrix1 = np.array([[1, 0], [0, M819]])
    matrix2 = np.array([[np.cos(az_rad), -np.sin(az_rad)], [np.sin(az_rad), np.cos(az_rad)]])
    matrix3 = np.array([[1, 0], [0, M47]])
    # matrix4 = np.array([[np.sin(alt_rad), -np.cos(alt_rad)], [np.cos(alt_rad), np.sin(alt_rad)]])
    matrix4 = np.array([[np.cos(1/2 * np.pi - alt_rad), -np.sin(1/2 * np.pi- alt_rad)], [np.sin(1/2 * np.pi - alt_rad), np.cos(1/2 * np.pi - alt_rad)]])
    matrix5 = np.array([[1, 0], [0, M13]])
    theta = np.radians(39.85)
    matrix6 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    result = np.matmul(matrix1, np.matmul(matrix6, np.matmul(matrix2, np.matmul(matrix3, np.matmul(matrix4, matrix5)))))
    return result

def Jall(alt, az, m13, p13, m47, p47, m819, p819):
    
    return J(alt, az, m13 * np.exp(-1j * p13 * np.pi), m47 * np.exp(-1j * p47 * np.pi), m819 * np.exp(-1j * p819 * np.pi))


def C(I, Q, U, V):
    CC = np.array([[I + Q, U - V * 1j], [U + V * 1j, I - Q]])
    # if I**2 < (Q**2 + U**2 + V**2):
    #     raise ValueError(f"I^2 < (Q^2 + U^2 + V^2)")
    # else:
    return CC / 2

def C_(I, Q, U, V):
    CC = np.array([[I + Q, U - V * 1j], [U + V * 1j, I - Q]])
    return CC / 2

def Vis(J1, J2, c12, c11, c22, p): # J1, J2 and C are Matrix

    param_12 = J1 @ c12 @ J2.T.conj() * 2
    V_12_HH, V_12_VV = np.diagonal(param_12)

    param_11 = J1 @ c11 @ J1.T.conj()
    V_11_HH, V_11_VV = np.diagonal(param_11)

    param_22 = J2 @ c22 @ J2.T.conj()
    V_22_HH, V_22_VV = np.diagonal(param_22)

    if p == 1:
        VH = V_12_HH / (V_11_HH + V_22_HH)
        PH = np.angle(VH, deg=True) # Return the angle of the complex argument in degree.
        return VH, PH
    elif p == 2:
        VV = V_12_VV / (V_11_VV + V_22_VV)
        PV = np.angle(VV, deg=True)
        return VV, PV
    else:
        V = (V_12_VV + V_12_HH) / (V_11_VV + V_22_VV + V_11_HH + V_22_HH)
        P = np.angle(V, deg=True) 
        return V, P

def Vis_diff(J1, J2, c12, c11, c22): # J1, J2 and C are Matrix

    param_12 = J1 @ c12 @ J2.T.conj() * 2
    V_12_HH, V_12_VV = np.diagonal(param_12)

    param_11 = J1 @ c11 @ J1.T.conj()
    V_11_HH, V_11_VV = np.diagonal(param_11)

    param_22 = J2 @ c22 @ J2.T.conj()
    V_22_HH, V_22_VV = np.diagonal(param_22)

    VH = V_12_HH / (V_11_HH + V_22_HH)
    PH = np.angle(VH, deg=True)

    VV = V_12_VV / (V_11_VV + V_22_VV)
    PV = np.angle(VV, deg=True)
    return VH, VV, PH, PV

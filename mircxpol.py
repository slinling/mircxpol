"""
mircxpol.py

This module implements the polarization model for the CHARA/MIRC-X and MYSTIC interferometers.
It defines matrix-based transformations and Stokes parameter handling to model the effects of 
instrumental polarization across multiple telescope arms and optical elements.

Key Functionalities:
---------------------
1. Jones Matrix modeling for individual telescope beam trains using optical element parameters (J, Jall).
2. Stokes matrix generator for defining input light polarization (C).
3. Visibility and polarization computation routines:
   - Vis_comp: full complex HH and VV visibilities
   - Vis_norm: normalized visibility ratio and differential phase
   - Vis_tel: flux ratio proxies per telescope arm
4. Model functions func1–func4 for computing:
   - Visibility ratio (func1)
   - Differential phase (func2)
   - Telescope 1 flux ratio (func3)
   - Telescope 2 flux ratio (func4)
5. err_global: Computes normalized residual vector (used for chi-squared minimization).
   It supports multiple baselines, hour angle input, and includes systematic error scaling.

Usage:
------
This module is called by analysis scripts to evaluate and fit visibility ratio, phase difference, 
and flux ratio data as a function of baseline geometry, parallactic angle, and hour angle.

Dependencies:
-------------
- numpy
- pandas

Author: [Your Name]
Date: [YYYY-MM-DD]
Affiliation: [Your Institution]
"""

# Implementation begins below
#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
import pandas as pd 


def J(alt, az, M13, M47, M819):

    alt_rad = np.radians(alt)
    az_rad = np.radians(az)

    matrix1 = np.array([[1, 0], [0, M819]])
    matrix2 = np.moveaxis(np.array([[np.cos(az_rad), -np.sin(az_rad)], [np.sin(az_rad), np.cos(az_rad)]]), 2, 0) # form (2,2,5) to (5,2,2)
    matrix3 = np.array([[1, 0], [0, M47]])
    matrix4 = np.moveaxis(np.array([[np.sin(alt_rad), -np.cos(alt_rad)], [np.cos(alt_rad), np.sin(alt_rad)]]), 2, 0)
    matrix5 = np.array([[1, 0], [0, M13]])
    theta = np.radians(39.85)
    matrix6 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return np.matmul(matrix1, np.matmul(matrix6, np.matmul(matrix2, np.matmul(matrix3, np.matmul(matrix4, matrix5)))))

def Jall(alt, az, m13, p13, m47, p47, m819, p819): # p in radians
    
    return J(alt, az, m13 * np.exp(-1j * p13), m47 * np.exp(-1j * p47), m819 * np.exp(-1j * p819))


def C(I, Q, U, V):
    CC = np.array([[I + Q, U - V * 1j], [U + V * 1j, I - Q]])
    return CC / 2

def Vis_comp(J1, J2, c12, c11, c22): # J1, J2 and C are Matrix

    param_12 = J1 @ c12 @ np.moveaxis(J2, 2, 1).conj() * 2
    V_12_HH = np.diagonal(param_12, axis1=1, axis2=2)[:, 0]
    V_12_VV = np.diagonal(param_12, axis1=1, axis2=2)[:, 1]

    param_11 = J1 @ c11 @ np.moveaxis(J1, 2, 1).conj() 
    V_11_HH = np.diagonal(param_11, axis1=1, axis2=2)[:, 0]
    V_11_VV = np.diagonal(param_11, axis1=1, axis2=2)[:, 1]

    param_22 = J2 @ c22 @ np.moveaxis(J2, 2, 1).conj() 
    V_22_HH = np.diagonal(param_22, axis1=1, axis2=2)[:, 0]
    V_22_VV = np.diagonal(param_22, axis1=1, axis2=2)[:, 1]

    V_H = V_12_HH / np.sqrt(V_11_HH * V_22_HH)
    V_V = V_12_VV / np.sqrt(V_11_VV * V_22_VV)

    # V_H = V_12_HH / (V_11_HH + V_22_HH)
    # V_V = V_12_VV / (V_11_VV + V_22_VV)

    return V_H, V_V

def Vis_norm(J1, J2, c12, c11, c22): # J1, J2 and C are Matrix

    param_12 = J1 @ c12 @ np.moveaxis(J2, 2, 1).conj() * 2 
    V_12_HH = np.diagonal(param_12, axis1=1, axis2=2)[:, 0]
    V_12_VV = np.diagonal(param_12, axis1=1, axis2=2)[:, 1]

    param_11 = J1 @ c11 @ np.moveaxis(J1, 2, 1).conj() 
    V_11_HH = np.diagonal(param_11, axis1=1, axis2=2)[:, 0]
    V_11_VV = np.diagonal(param_11, axis1=1, axis2=2)[:, 1]

    param_22 = J2 @ c22 @ np.moveaxis(J2, 2, 1).conj() 
    V_22_HH = np.diagonal(param_22, axis1=1, axis2=2)[:, 0]
    V_22_VV = np.diagonal(param_22, axis1=1, axis2=2)[:, 1]

    V_norm = (V_12_HH / V_12_VV) * (V_11_VV / V_11_HH)**0.5 * (V_22_VV / V_22_HH)**0.5 ####* f
    P_norm = np.angle(V_norm, deg=True) # Return the angle of the complex argument in degree.

    return V_norm, P_norm

def Vis_tel(J1, J2, c12, c11, c22): # J1, J2 and C are Matrix

    param_11 = J1 @ c11 @ np.moveaxis(J1, 2, 1).conj() 
    V_11_HH = np.diagonal(param_11, axis1=1, axis2=2)[:, 0]
    V_11_VV = np.diagonal(param_11, axis1=1, axis2=2)[:, 1]

    param_22 = J2 @ c22 @ np.moveaxis(J2, 2, 1).conj()
    V_22_HH = np.diagonal(param_22, axis1=1, axis2=2)[:, 0]
    V_22_VV = np.diagonal(param_22, axis1=1, axis2=2)[:, 1]

    V_tel_1 = abs(V_11_HH / (V_11_VV))
    V_tel_2 = abs(V_22_HH / (V_22_VV))

    return V_tel_1, V_tel_2

##################################################################################################

def func1(alt, az, v, pol = False):

    t1_r_m3, t1_r_m4, t1_r_m8, t2_r_m3, t2_c_m3, t2_r_m4, t2_c_m4, t2_r_m8, t2_c_m8, t3_r_m3, t3_c_m3, t3_r_m4, t3_c_m4, t3_r_m8, t3_c_m8, t4_r_m3, t4_c_m3, t4_r_m4, t4_c_m4, t4_r_m8, t4_c_m8, t5_r_m3, t5_c_m3, t5_r_m4, t5_c_m4, t5_r_m8, t5_c_m8, t6_r_m3, t6_c_m3, t6_r_m4, t6_c_m4, t6_r_m8, t6_c_m8, k12, k13, k14, k15, k16, k23, k24, k25, k26, k34, k35, k36, k45, k46, k56 = v 
    Q12, U12, V12, Qii, Uii, Vii = 0., 0., 0., 0., 0., 0.
    I12 = 1.
    t1_c_m3, t1_c_m4, t1_c_m8 = 0., 0., 0.
    
    C11 = C(1, Qii, Uii, Vii)
    C22 = C(1, Qii, Uii, Vii)
    C12 = C(I12, Q12, U12, V12)

    J1 = Jall(alt, az, t1_r_m3, t1_c_m3, t1_r_m4, t1_c_m4, t1_r_m8, t1_c_m8) # E1
    J2 = Jall(alt, az, t2_r_m3, t2_c_m3, t2_r_m4, t2_c_m4, t2_r_m8, t2_c_m8) # W2
    J3 = Jall(alt, az, t3_r_m3, t3_c_m3, t3_r_m4, t3_c_m4, t3_r_m8, t3_c_m8) # W1
    J4 = Jall(alt, az, t4_r_m3, t4_c_m3, t4_r_m4, t4_c_m4, t4_r_m8, t4_c_m8) # S2
    J5 = Jall(alt, az, t5_r_m3, t5_c_m3, t5_r_m4, t5_c_m4, t5_r_m8, t5_c_m8) # S1
    J6 = Jall(alt, az, t6_r_m3, t6_c_m3, t6_r_m4, t6_c_m4, t6_r_m8, t6_c_m8) # E2

    VR_12 = abs(Vis_norm(J1, J2, C12, C11, C22)[0]) * k12
    VR_13 = abs(Vis_norm(J1, J3, C12, C11, C22)[0]) * k13
    VR_14 = abs(Vis_norm(J1, J4, C12, C11, C22)[0]) * k14
    VR_15 = abs(Vis_norm(J1, J5, C12, C11, C22)[0]) * k15
    VR_16 = abs(Vis_norm(J1, J6, C12, C11, C22)[0]) * k16
    VR_23 = abs(Vis_norm(J2, J3, C12, C11, C22)[0]) * k23
    VR_24 = abs(Vis_norm(J2, J4, C12, C11, C22)[0]) * k24
    VR_25 = abs(Vis_norm(J2, J5, C12, C11, C22)[0]) * k25
    VR_26 = abs(Vis_norm(J2, J6, C12, C11, C22)[0]) * k26
    VR_34 = abs(Vis_norm(J3, J4, C12, C11, C22)[0]) * k34
    VR_35 = abs(Vis_norm(J3, J5, C12, C11, C22)[0]) * k35
    VR_36 = abs(Vis_norm(J3, J6, C12, C11, C22)[0]) * k36
    VR_45 = abs(Vis_norm(J4, J5, C12, C11, C22)[0]) * k45
    VR_46 = abs(Vis_norm(J4, J6, C12, C11, C22)[0]) * k46
    VR_56 = abs(Vis_norm(J5, J6, C12, C11, C22)[0]) * k56

    return VR_12, VR_13, VR_14, VR_15, VR_16, VR_23, VR_24, VR_25, VR_26, VR_34, VR_35, VR_36, VR_45, VR_46, VR_56

def func2(alt, az, v, pol = False):

    t1_r_m3, t1_r_m4, t1_r_m8, t2_r_m3, t2_c_m3, t2_r_m4, t2_c_m4, t2_r_m8, t2_c_m8, t3_r_m3, t3_c_m3, t3_r_m4, t3_c_m4, t3_r_m8, t3_c_m8, t4_r_m3, t4_c_m3, t4_r_m4, t4_c_m4, t4_r_m8, t4_c_m8, t5_r_m3, t5_c_m3, t5_r_m4, t5_c_m4, t5_r_m8, t5_c_m8, t6_r_m3, t6_c_m3, t6_r_m4, t6_c_m4, t6_r_m8, t6_c_m8, k12, k13, k14, k15, k16, k23, k24, k25, k26, k34, k35, k36, k45, k46, k56 = v 
    Q12, U12, V12, Qii, Uii, Vii = 0., 0., 0., 0., 0., 0.
    I12 = 1.
    t1_c_m3, t1_c_m4, t1_c_m8 = 0., 0., 0.

    C11 = C(1, Qii, Uii, Vii)
    C22 = C(1, Qii, Uii, Vii)
    C12 = C(I12, Q12, U12, V12)

    J1 = Jall(alt, az, t1_r_m3, t1_c_m3, t1_r_m4, t1_c_m4, t1_r_m8, t1_c_m8) # E1
    J2 = Jall(alt, az, t2_r_m3, t2_c_m3, t2_r_m4, t2_c_m4, t2_r_m8, t2_c_m8) # W2
    J3 = Jall(alt, az, t3_r_m3, t3_c_m3, t3_r_m4, t3_c_m4, t3_r_m8, t3_c_m8) # W1
    J4 = Jall(alt, az, t4_r_m3, t4_c_m3, t4_r_m4, t4_c_m4, t4_r_m8, t4_c_m8) # S2
    J5 = Jall(alt, az, t5_r_m3, t5_c_m3, t5_r_m4, t5_c_m4, t5_r_m8, t5_c_m8) # S1
    J6 = Jall(alt, az, t6_r_m3, t6_c_m3, t6_r_m4, t6_c_m4, t6_r_m8, t6_c_m8) # E2

    DP_12 = Vis_norm(J1, J2, C12, C11, C22)[1]
    DP_13 = Vis_norm(J1, J3, C12, C11, C22)[1]
    DP_14 = Vis_norm(J1, J4, C12, C11, C22)[1]
    DP_15 = Vis_norm(J1, J5, C12, C11, C22)[1]
    DP_16 = Vis_norm(J1, J6, C12, C11, C22)[1]
    DP_23 = Vis_norm(J2, J3, C12, C11, C22)[1]
    DP_24 = Vis_norm(J2, J4, C12, C11, C22)[1]
    DP_25 = Vis_norm(J2, J5, C12, C11, C22)[1]
    DP_26 = Vis_norm(J2, J6, C12, C11, C22)[1]
    DP_34 = Vis_norm(J3, J4, C12, C11, C22)[1]
    DP_35 = Vis_norm(J3, J5, C12, C11, C22)[1]
    DP_36 = Vis_norm(J3, J6, C12, C11, C22)[1]
    DP_45 = Vis_norm(J4, J5, C12, C11, C22)[1]
    DP_46 = Vis_norm(J4, J6, C12, C11, C22)[1]
    DP_56 = Vis_norm(J5, J6, C12, C11, C22)[1]

    return DP_12, DP_13, DP_14, DP_15, DP_16, DP_23, DP_24, DP_25, DP_26, DP_34, DP_35, DP_36, DP_45, DP_46, DP_56

def func3(alt, az, v, pol = False):

    t1_r_m3, t1_r_m4, t1_r_m8, t2_r_m3, t2_c_m3, t2_r_m4, t2_c_m4, t2_r_m8, t2_c_m8, t3_r_m3, t3_c_m3, t3_r_m4, t3_c_m4, t3_r_m8, t3_c_m8, t4_r_m3, t4_c_m3, t4_r_m4, t4_c_m4, t4_r_m8, t4_c_m8, t5_r_m3, t5_c_m3, t5_r_m4, t5_c_m4, t5_r_m8, t5_c_m8, t6_r_m3, t6_c_m3, t6_r_m4, t6_c_m4, t6_r_m8, t6_c_m8, k12, k13, k14, k15, k16, k23, k24, k25, k26, k34, k35, k36, k45, k46, k56 = v 
    Q12, U12, V12, Qii, Uii, Vii = 0., 0., 0., 0., 0., 0.
    I12 = 1.
    t1_c_m3, t1_c_m4, t1_c_m8 = 0., 0., 0.

    C11 = C(1, Qii, Uii, Vii)
    C22 = C(1, Qii, Uii, Vii)
    C12 = C(I12, Q12, U12, V12)

    J1 = Jall(alt, az, t1_r_m3, t1_c_m3, t1_r_m4, t1_c_m4, t1_r_m8, t1_c_m8) # E1
    J2 = Jall(alt, az, t2_r_m3, t2_c_m3, t2_r_m4, t2_c_m4, t2_r_m8, t2_c_m8) # W2
    J3 = Jall(alt, az, t3_r_m3, t3_c_m3, t3_r_m4, t3_c_m4, t3_r_m8, t3_c_m8) # W1
    J4 = Jall(alt, az, t4_r_m3, t4_c_m3, t4_r_m4, t4_c_m4, t4_r_m8, t4_c_m8) # S2
    J5 = Jall(alt, az, t5_r_m3, t5_c_m3, t5_r_m4, t5_c_m4, t5_r_m8, t5_c_m8) # S1
    J6 = Jall(alt, az, t6_r_m3, t6_c_m3, t6_r_m4, t6_c_m4, t6_r_m8, t6_c_m8) # E2

    T1_12 = Vis_tel(J1, J2, C12, C11, C22)[0]
    T1_13 = Vis_tel(J1, J3, C12, C11, C22)[0]
    T1_14 = Vis_tel(J1, J4, C12, C11, C22)[0]
    T1_15 = Vis_tel(J1, J5, C12, C11, C22)[0]
    T1_16 = Vis_tel(J1, J6, C12, C11, C22)[0]
    T1_23 = Vis_tel(J2, J3, C12, C11, C22)[0]
    T1_24 = Vis_tel(J2, J4, C12, C11, C22)[0]
    T1_25 = Vis_tel(J2, J5, C12, C11, C22)[0]
    T1_26 = Vis_tel(J2, J6, C12, C11, C22)[0]
    T1_34 = Vis_tel(J3, J4, C12, C11, C22)[0]
    T1_35 = Vis_tel(J3, J5, C12, C11, C22)[0]
    T1_36 = Vis_tel(J3, J6, C12, C11, C22)[0]
    T1_45 = Vis_tel(J4, J5, C12, C11, C22)[0]
    T1_46 = Vis_tel(J4, J6, C12, C11, C22)[0]
    T1_56 = Vis_tel(J5, J6, C12, C11, C22)[0]

    return T1_12, T1_13, T1_14, T1_15, T1_16, T1_23, T1_24, T1_25, T1_26, T1_34, T1_35, T1_36, T1_45, T1_46, T1_56

def func4(alt, az, v, pol = False):

    t1_r_m3, t1_r_m4, t1_r_m8, t2_r_m3, t2_c_m3, t2_r_m4, t2_c_m4, t2_r_m8, t2_c_m8, t3_r_m3, t3_c_m3, t3_r_m4, t3_c_m4, t3_r_m8, t3_c_m8, t4_r_m3, t4_c_m3, t4_r_m4, t4_c_m4, t4_r_m8, t4_c_m8, t5_r_m3, t5_c_m3, t5_r_m4, t5_c_m4, t5_r_m8, t5_c_m8, t6_r_m3, t6_c_m3, t6_r_m4, t6_c_m4, t6_r_m8, t6_c_m8, k12, k13, k14, k15, k16, k23, k24, k25, k26, k34, k35, k36, k45, k46, k56 = v 
    Q12, U12, V12, Qii, Uii, Vii = 0., 0., 0., 0., 0., 0.
    I12 = 1.
    t1_c_m3, t1_c_m4, t1_c_m8 = 0., 0., 0.

    C11 = C(1, Qii, Uii, Vii)
    C22 = C(1, Qii, Uii, Vii)
    C12 = C(I12, Q12, U12, V12)

    J1 = Jall(alt, az, t1_r_m3, t1_c_m3, t1_r_m4, t1_c_m4, t1_r_m8, t1_c_m8) # E1
    J2 = Jall(alt, az, t2_r_m3, t2_c_m3, t2_r_m4, t2_c_m4, t2_r_m8, t2_c_m8) # W2
    J3 = Jall(alt, az, t3_r_m3, t3_c_m3, t3_r_m4, t3_c_m4, t3_r_m8, t3_c_m8) # W1
    J4 = Jall(alt, az, t4_r_m3, t4_c_m3, t4_r_m4, t4_c_m4, t4_r_m8, t4_c_m8) # S2
    J5 = Jall(alt, az, t5_r_m3, t5_c_m3, t5_r_m4, t5_c_m4, t5_r_m8, t5_c_m8) # S1
    J6 = Jall(alt, az, t6_r_m3, t6_c_m3, t6_r_m4, t6_c_m4, t6_r_m8, t6_c_m8) # E2

    T2_12 = Vis_tel(J1, J2, C12, C11, C22)[1]
    T2_13 = Vis_tel(J1, J3, C12, C11, C22)[1]
    T2_14 = Vis_tel(J1, J4, C12, C11, C22)[1]
    T2_15 = Vis_tel(J1, J5, C12, C11, C22)[1]
    T2_16 = Vis_tel(J1, J6, C12, C11, C22)[1]
    T2_23 = Vis_tel(J2, J3, C12, C11, C22)[1]
    T2_24 = Vis_tel(J2, J4, C12, C11, C22)[1]
    T2_25 = Vis_tel(J2, J5, C12, C11, C22)[1]
    T2_26 = Vis_tel(J2, J6, C12, C11, C22)[1]
    T2_34 = Vis_tel(J3, J4, C12, C11, C22)[1]
    T2_35 = Vis_tel(J3, J5, C12, C11, C22)[1]
    T2_36 = Vis_tel(J3, J6, C12, C11, C22)[1]
    T2_45 = Vis_tel(J4, J5, C12, C11, C22)[1]
    T2_46 = Vis_tel(J4, J6, C12, C11, C22)[1]
    T2_56 = Vis_tel(J5, J6, C12, C11, C22)[1]

    return T2_12, T2_13, T2_14, T2_15, T2_16, T2_23, T2_24, T2_25, T2_26, T2_34, T2_35, T2_36, T2_45, T2_46, T2_56


def err_global(var,
               alt_12, az_12, Mat_VR_12, Mat_DP_12, Mat_VR_err_12, Mat_DP_err_12, 
               alt_13, az_13, Mat_VR_13, Mat_DP_13, Mat_VR_err_13, Mat_DP_err_13, 
               alt_14, az_14, Mat_VR_14, Mat_DP_14, Mat_VR_err_14, Mat_DP_err_14,
               alt_15, az_15, Mat_VR_15, Mat_DP_15, Mat_VR_err_15, Mat_DP_err_15, 
               alt_16, az_16, Mat_VR_16, Mat_DP_16, Mat_VR_err_16, Mat_DP_err_16, 
               alt_23, az_23, Mat_VR_23, Mat_DP_23, Mat_VR_err_23, Mat_DP_err_23, 
               alt_24, az_24, Mat_VR_24, Mat_DP_24, Mat_VR_err_24, Mat_DP_err_24, 
               alt_25, az_25, Mat_VR_25, Mat_DP_25, Mat_VR_err_25, Mat_DP_err_25, 
               alt_26, az_26, Mat_VR_26, Mat_DP_26, Mat_VR_err_26, Mat_DP_err_26, 
               alt_34, az_34, Mat_VR_34, Mat_DP_34, Mat_VR_err_34, Mat_DP_err_34, 
               alt_35, az_35, Mat_VR_35, Mat_DP_35, Mat_VR_err_35, Mat_DP_err_35, 
               alt_36, az_36, Mat_VR_36, Mat_DP_36, Mat_VR_err_36, Mat_DP_err_36, 
               alt_45, az_45, Mat_VR_45, Mat_DP_45, Mat_VR_err_45, Mat_DP_err_45, 
               alt_46, az_46, Mat_VR_46, Mat_DP_46, Mat_VR_err_46, Mat_DP_err_46, 
               alt_56, az_56, Mat_VR_56, Mat_DP_56, Mat_VR_err_56, Mat_DP_err_56,
               Mat_T1_12, Mat_T2_12, Mat_T2_13, Mat_T2_14, Mat_T2_15, Mat_T2_16,
               Mat_T1_err_12, Mat_T2_err_12, Mat_T2_err_13, Mat_T2_err_14, Mat_T2_err_15, Mat_T2_err_16):

    
    def calc_errors(alt, az, Mat_VR, Mat_DP, Mat_VR_err, Mat_DP_err, idx):
        Mat_VR = pd.to_numeric(Mat_VR, errors='coerce')
        Mat_DP = pd.to_numeric(Mat_DP, errors='coerce')
    
        err_VR = (Mat_VR - func1(alt, az, var)[idx]) / np.sqrt(Mat_VR_err**2)
        err_DP = (Mat_DP - func2(alt, az, var)[idx]) / np.sqrt(Mat_DP_err**2)

        return err_VR, err_DP

    err_12_VR = calc_errors(alt_12, az_12, Mat_VR_12, Mat_DP_12, Mat_VR_err_12, Mat_DP_err_12, 0)[0]
    err_13_VR = calc_errors(alt_13, az_13, Mat_VR_13, Mat_DP_13, Mat_VR_err_13, Mat_DP_err_13, 1)[0]
    err_14_VR = calc_errors(alt_14, az_14, Mat_VR_14, Mat_DP_14, Mat_VR_err_14, Mat_DP_err_14, 2)[0]
    err_15_VR = calc_errors(alt_15, az_15, Mat_VR_15, Mat_DP_15, Mat_VR_err_15, Mat_DP_err_15, 3)[0]
    err_16_VR = calc_errors(alt_16, az_16, Mat_VR_16, Mat_DP_16, Mat_VR_err_16, Mat_DP_err_16, 4)[0]
    err_23_VR = calc_errors(alt_23, az_23, Mat_VR_23, Mat_DP_23, Mat_VR_err_23, Mat_DP_err_23, 5)[0]
    err_24_VR = calc_errors(alt_24, az_24, Mat_VR_24, Mat_DP_24, Mat_VR_err_24, Mat_DP_err_24, 6)[0]
    err_25_VR = calc_errors(alt_25, az_25, Mat_VR_25, Mat_DP_25, Mat_VR_err_25, Mat_DP_err_25, 7)[0]
    err_26_VR = calc_errors(alt_26, az_26, Mat_VR_26, Mat_DP_26, Mat_VR_err_26, Mat_DP_err_26, 8)[0]
    err_34_VR = calc_errors(alt_34, az_34, Mat_VR_34, Mat_DP_34, Mat_VR_err_34, Mat_DP_err_34, 9)[0]
    err_35_VR = calc_errors(alt_35, az_35, Mat_VR_35, Mat_DP_35, Mat_VR_err_35, Mat_DP_err_35, 10)[0]
    err_36_VR = calc_errors(alt_36, az_36, Mat_VR_36, Mat_DP_36, Mat_VR_err_36, Mat_DP_err_36, 11)[0]
    err_45_VR = calc_errors(alt_45, az_45, Mat_VR_45, Mat_DP_45, Mat_VR_err_45, Mat_DP_err_45, 12)[0]
    err_46_VR = calc_errors(alt_46, az_46, Mat_VR_46, Mat_DP_46, Mat_VR_err_46, Mat_DP_err_46, 13)[0]
    err_56_VR = calc_errors(alt_56, az_56, Mat_VR_56, Mat_DP_56, Mat_VR_err_56, Mat_DP_err_56, 14)[0]
    errors_VR = np.concatenate((err_12_VR, err_13_VR, err_14_VR, err_15_VR, err_16_VR, err_23_VR, err_24_VR, err_25_VR, err_26_VR, err_34_VR, err_35_VR, err_36_VR, err_45_VR, err_46_VR, err_56_VR), axis=None)

    err_12_DP = calc_errors(alt_12, az_12, Mat_VR_12, Mat_DP_12, Mat_VR_err_12, Mat_DP_err_12, 0)[1]
    err_13_DP = calc_errors(alt_13, az_13, Mat_VR_13, Mat_DP_13, Mat_VR_err_13, Mat_DP_err_13, 1)[1]
    err_14_DP = calc_errors(alt_14, az_14, Mat_VR_14, Mat_DP_14, Mat_VR_err_14, Mat_DP_err_14, 2)[1]
    err_15_DP = calc_errors(alt_15, az_15, Mat_VR_15, Mat_DP_15, Mat_VR_err_15, Mat_DP_err_15, 3)[1]
    err_16_DP = calc_errors(alt_16, az_16, Mat_VR_16, Mat_DP_16, Mat_VR_err_16, Mat_DP_err_16, 4)[1]
    err_23_DP = calc_errors(alt_23, az_23, Mat_VR_23, Mat_DP_23, Mat_VR_err_23, Mat_DP_err_23, 5)[1]
    err_24_DP = calc_errors(alt_24, az_24, Mat_VR_24, Mat_DP_24, Mat_VR_err_24, Mat_DP_err_24, 6)[1]
    err_25_DP = calc_errors(alt_25, az_25, Mat_VR_25, Mat_DP_25, Mat_VR_err_25, Mat_DP_err_25, 7)[1]
    err_26_DP = calc_errors(alt_26, az_26, Mat_VR_26, Mat_DP_26, Mat_VR_err_26, Mat_DP_err_26, 8)[1]
    err_34_DP = calc_errors(alt_34, az_34, Mat_VR_34, Mat_DP_34, Mat_VR_err_34, Mat_DP_err_34, 9)[1]
    err_35_DP = calc_errors(alt_35, az_35, Mat_VR_35, Mat_DP_35, Mat_VR_err_35, Mat_DP_err_35, 10)[1]
    err_36_DP = calc_errors(alt_36, az_36, Mat_VR_36, Mat_DP_36, Mat_VR_err_36, Mat_DP_err_36, 11)[1]
    err_45_DP = calc_errors(alt_45, az_45, Mat_VR_45, Mat_DP_45, Mat_VR_err_45, Mat_DP_err_45, 12)[1]
    err_46_DP = calc_errors(alt_46, az_46, Mat_VR_46, Mat_DP_46, Mat_VR_err_46, Mat_DP_err_46, 13)[1]
    err_56_DP = calc_errors(alt_56, az_56, Mat_VR_56, Mat_DP_56, Mat_VR_err_56, Mat_DP_err_56, 14)[1]
    errors_DP = np.concatenate((err_12_DP, err_13_DP, err_14_DP, err_15_DP, err_16_DP, err_23_DP, err_24_DP, err_25_DP, err_26_DP, err_34_DP, err_35_DP, err_36_DP, err_45_DP, err_46_DP, err_56_DP), axis=None)

    # Convert to numeric and filter NaNs
    Mat_T1_12 = pd.to_numeric(Mat_T1_12, errors='coerce')
    Mat_T2_12 = pd.to_numeric(Mat_T2_12, errors='coerce')
    Mat_T2_13 = pd.to_numeric(Mat_T2_13, errors='coerce')
    Mat_T2_14 = pd.to_numeric(Mat_T2_14, errors='coerce')
    Mat_T2_15 = pd.to_numeric(Mat_T2_15, errors='coerce')
    Mat_T2_16 = pd.to_numeric(Mat_T2_16, errors='coerce')
    err_T1_12 = (Mat_T1_12 - func3(alt_12, az_12, var)[0]) / np.sqrt(Mat_T1_err_12**2)
    err_T2_12 = (Mat_T2_12 - func4(alt_12, az_12, var)[0]) / np.sqrt(Mat_T2_err_12**2)
    err_T2_13 = (Mat_T2_13 - func4(alt_13, az_13, var)[1]) / np.sqrt(Mat_T2_err_13**2)
    err_T2_14 = (Mat_T2_14 - func4(alt_14, az_14, var)[2]) / np.sqrt(Mat_T2_err_14**2)
    err_T2_15 = (Mat_T2_15 - func4(alt_15, az_15, var)[3]) / np.sqrt(Mat_T2_err_15**2)
    err_T2_16 = (Mat_T2_16 - func4(alt_16, az_16, var)[4]) / np.sqrt(Mat_T2_err_16**2)

    # Filter out NaNs
    err_T1_12 = err_T1_12[~np.isnan(err_T1_12)]
    err_T2_12 = err_T2_12[~np.isnan(err_T2_12)]
    err_T2_13 = err_T2_13[~np.isnan(err_T2_13)]
    err_T2_14 = err_T2_14[~np.isnan(err_T2_14)]
    err_T2_15 = err_T2_15[~np.isnan(err_T2_15)]
    err_T2_16 = err_T2_16[~np.isnan(err_T2_16)]

    err_tel = np.concatenate((err_T1_12, err_T2_12, err_T2_13, err_T2_14, err_T2_15, err_T2_16), axis=None)

    err = np.concatenate((errors_VR, errors_DP, err_tel), axis=None)
    return err

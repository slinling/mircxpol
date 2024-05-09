#!/usr/bin/env python
# coding: utf-8

# Inputs from command line:
#   sys.argv[1]: file name for your data // e.g. "2022Oct19_W1S1.csv"
#   sys.argv[2]: name of the system // e.g. "ups and"
#   sys.argv[3]: observing date // e.g. "2022-10-19"

import sys
import numpy as np

var = np.array([1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.])
bound=((-np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi),
       (np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi)) 
list = ['r_m31', 'c_m31', 'r_m41', 'c_m41', 'r_m81', 'c_m81', 'r_m32', 'c_m32', 'r_m42', 'c_m42', 'r_m82', 'c_m82']
list_index = [5]

#############################################################################################

var_free = np.array([1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0, 0, 0, 0, 0, 0])
bound_free =((-np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf, -np.pi, 0, -1, -1, -1, -1, -1, -1),
       (np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, np.inf, np.pi, 1, 1, 1, 1, 1, 1, 1)) # 1, 1, 1, 1, 1, 1
list_free = ['r_m31', 'c_m31', 'r_m41', 'c_m41', 'r_m81', 'c_m81', 'r_m32', 'c_m32', 'r_m42', 'c_m42', 'r_m82', 'c_m82', 'I12', 'Q12', 'U12', 'V12', 'Qii', 'Uii', 'Vii']
list_index_free = [5, 11, 15]

def J(alt, az, M13, M47, M819):

    alt_rad = np.radians(alt)
    az_rad = np.radians(az)

    matrix1 = np.array([[1, 0], [0, M819]])
    matrix2 = np.moveaxis(np.array([[np.cos(az_rad), -np.sin(az_rad)], [np.sin(az_rad), np.cos(az_rad)]]), 2, 0) # form (2,2,5) to (5,2,2)
    matrix3 = np.array([[1, 0], [0, M47]])
    matrix4 = np.moveaxis(np.array([[np.sin(alt_rad), -np.cos(alt_rad)], [np.cos(alt_rad), np.sin(alt_rad)]]), 2, 0)
    # matrix4 = np.array([[np.cos(1/2 * np.pi - alt_rad), -np.sin(1/2 * np.pi- alt_rad)], [np.sin(1/2 * np.pi - alt_rad), np.cos(1/2 * np.pi - alt_rad)]])
    matrix5 = np.array([[1, 0], [0, M13]])
    theta = np.radians(39.85)
    matrix6 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return np.matmul(matrix1, np.matmul(matrix6, np.matmul(matrix2, np.matmul(matrix3, np.matmul(matrix4, matrix5)))))

def Jall(alt, az, m13, p13, m47, p47, m819, p819):
    
    return J(alt, az, m13 * np.exp(-1j * p13), m47 * np.exp(-1j * p47), m819 * np.exp(-1j * p819))


def C(I, Q, U, V):
    CC = np.array([[I + Q, U - V * 1j], [U + V * 1j, I - Q]])
    # if I**2 < (Q**2 + U**2 + V**2):
    #     raise ValueError(f"I^2 < (Q^2 + U^2 + V^2)")
    # else:
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

    V_norm = (V_12_HH / V_12_VV) * np.sqrt(V_11_VV / V_11_HH) * np.sqrt(V_22_VV / V_22_HH)
    P_norm = np.angle(V_norm, deg=True) # Return the angle of the complex argument in degree.

    # V_norm = (V_12_HH / V_12_VV) * (V_11_VV + V_22_VV) / (V_11_HH + V_22_HH)
    # P_norm = np.angle(V_norm, deg=True) # Return the angle of the complex argument in degree.
    
    return V_norm, P_norm

def Vis_tel(J1, J2, c12, c11, c22): # J1, J2 and C are Matrix

    param_11 = J1 @ c11 @ np.moveaxis(J1, 2, 1).conj()
    V_11_HH = np.diagonal(param_11, axis1=1, axis2=2)[:, 0]
    V_11_VV = np.diagonal(param_11, axis1=1, axis2=2)[:, 1]

    param_22 = J2 @ c22 @ np.moveaxis(J2, 2, 1).conj()
    V_22_HH = np.diagonal(param_22, axis1=1, axis2=2)[:, 0]
    V_22_VV = np.diagonal(param_22, axis1=1, axis2=2)[:, 1]

    V_tel_1 = V_11_HH / (V_11_VV)
    V_tel_2 = V_22_HH / (V_22_VV)

    return V_tel_1, V_tel_2

##################################################################################################

def func1(alt, az, v, pol = False):

    if pol == False:
        r_m31, c_m31, r_m41, c_m41, r_m81, c_m81, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82 = v 
        Q12, U12, V12, Qii, Uii, Vii = 0., 0., 0., 0., 0., 0.
        I12 = 1.
    else:
        r_m31, c_m31, r_m41, c_m41, r_m81, c_m81, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82, I12, Q12, U12, V12, Qii, Uii, Vii = v 
    
    C11 = C(1, Qii, Uii, Vii)
    C22 = C(1, Qii, Uii, Vii)
    C12 = C(I12, Q12, U12, V12)

    J1 = Jall(alt, az, r_m31, c_m31, r_m41, c_m41, r_m81, c_m81)
    J2 = Jall(alt, az, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82)

    VR = abs(Vis_norm(J1, J2, C12, C11, C22)[0])
    return VR.flatten()

def func2(alt, az, v, pol = False):


    if pol == False:
        r_m31, c_m31, r_m41, c_m41, r_m81, c_m81, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82 = v 
        Q12, U12, V12, Qii, Uii, Vii = 0., 0., 0., 0., 0., 0.
        I12 = 1.
    else:
        r_m31, c_m31, r_m41, c_m41, r_m81, c_m81, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82, I12, Q12, U12, V12, Qii, Uii, Vii = v 
    

    C11 = C(1, Qii, Uii, Vii)
    C22 = C(1, Qii, Uii, Vii)
    C12 = C(I12, Q12, U12, V12)

    J1 = Jall(alt, az, r_m31, c_m31, r_m41, c_m41, r_m81, c_m81)
    J2 = Jall(alt, az, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82)

    DP = Vis_norm(J1, J2, C12, C11, C22)[1] # Diff_phase_norm
    return DP.flatten()

def func3(alt, az, v, pol = False):


    if pol == False:
        r_m31, c_m31, r_m41, c_m41, r_m81, c_m81, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82 = v 
        Q12, U12, V12, Qii, Uii, Vii = 0., 0., 0., 0., 0., 0.
        I12 = 1.
    else:
        r_m31, c_m31, r_m41, c_m41, r_m81, c_m81, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82, I12, Q12, U12, V12, Qii, Uii, Vii = v 
    
    
    C11 = C(1, Qii, Uii, Vii)
    C22 = C(1, Qii, Uii, Vii)
    C12 = C(I12, Q12, U12, V12)

    J1 = Jall(alt, az, r_m31, c_m31, r_m41, c_m41, r_m81, c_m81)
    J2 = Jall(alt, az, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82)

    T1 = abs(Vis_tel(J1, J2, C12, C11, C22)[0]) # Vis_ratio_tel1
    
    return T1.flatten()


def func4(alt, az, v, pol = False):


    if pol == False:
        r_m31, c_m31, r_m41, c_m41, r_m81, c_m81, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82 = v 
        Q12, U12, V12, Qii, Uii, Vii = 0., 0., 0., 0., 0., 0.
        I12 = 1.
    else:
        r_m31, c_m31, r_m41, c_m41, r_m81, c_m81, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82, I12, Q12, U12, V12, Qii, Uii, Vii = v 
    
    
    C11 = C(1, Qii, Uii, Vii)
    C22 = C(1, Qii, Uii, Vii)
    C12 = C(I12, Q12, U12, V12)

    J1 = Jall(alt, az, r_m31, c_m31, r_m41, c_m41, r_m81, c_m81)
    J2 = Jall(alt, az, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82)

    T2 = abs(Vis_tel(J1, J2, C12, C11, C22)[1]) # Vis_ratio_tel1
    return T2.flatten()

def func_comp(alt, az, v, pol = False):


    if pol == False:
        r_m31, c_m31, r_m41, c_m41, r_m81, c_m81, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82 = v 
        Q12, U12, V12, Qii, Uii, Vii = 0., 0., 0., 0., 0., 0.
        I12 = 1.
    else:
        r_m31, c_m31, r_m41, c_m41, r_m81, c_m81, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82, I12, Q12, U12, V12, Qii, Uii, Vii = v 
    
    
    C11 = C(1, Qii, Uii, Vii)
    C22 = C(1, Qii, Uii, Vii)
    C12 = C(I12, Q12, U12, V12)

    J1 = Jall(alt, az, r_m31, c_m31, r_m41, c_m41, r_m81, c_m81)
    J2 = Jall(alt, az, r_m32, c_m32, r_m42, c_m42, r_m82, c_m82)

    VH = abs(Vis_comp(J1, J2, C12, C11, C22)[0]) 
    VV = abs(Vis_comp(J1, J2, C12, C11, C22)[1])

    PH = np.angle(Vis_comp(J1, J2, C12, C11, C22)[0], deg=True)
    PV= np.angle(Vis_comp(J1, J2, C12, C11, C22)[1], deg=True)
    
    return VH, VV, PH, PV

def err_global(var, a1, a2, Mat_VR, Mat_DP, Mat_T1, Mat_T2, Mat_VR_err, Mat_DP_err, Mat_T1_err, Mat_T2_err):

    err_VR = (Mat_VR - func1(a1, a2, var)) / Mat_VR_err
    err_DP = (Mat_DP - func2(a1, a2, var)) / Mat_DP_err
    err_T1 = (Mat_T1 - func3(a1, a2, var)) / Mat_T1_err
    err_T2 = (Mat_T2 - func4(a1, a2, var)) / Mat_T2_err
    err = np.concatenate((err_VR, err_DP, err_T1, err_T2), axis = None)

    return err
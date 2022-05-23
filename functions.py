#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions needed to run the inversion accomplished on:
    "Plate-locking, uncertainty estimation and spatial correlations revealed
      with a Bayesian model selection method:
      Application to the Central Chile subduction zone"
        
Author: V. Becerra-Carreño
        
last version: May 22 2022
"""

## Needed libraries:
import numpy as np
import scipy.special as sp

## FUNCTIONS

def fg(x):
    """
    Gaussian correlation function
    """
    return np.exp(-x*x)

def fe(x):
    """
    exponential correlation function
    """
    return np.exp(-x)

def fVK(x, H):
    """
    Von Karman correlation function
    """
    return ((x**H)*sp.kv(H,x))/(2**(H-1)*sp.gamma(H))

def G_rotation(dip, strike, g_dip, g_stk, conv):
    """
    To permorm a rotation of G matrix, since the typical Dip-Strike system,
    to a X-Y system centered 45 degrees from plate convergence direction.
    
    Inputs:
        dip: array with dip direction in degrees of each subfault
        strike: array with strike direction in degrees of each subfault
        g_dip: Green matrix of dip slip
        g_stk: Green matrix of strike slip
        conv: angle of plate convergence direction
        
    Outputs:
        G: the green matrix rotated in X-Y coordinate  system.
        theta: array of angles by which each subfault rotated
    """

    an_inter = conv-strike # Angle between strike line and plate convergence.
    proj_stk = np.cos(an_inter*np.pi/180) # Projection of the plate convergence angle on the fault plane.
    proj_dip = np.sin(an_inter*np.pi/180)*np.cos(dip*np.pi/180)

    theta_g = np.arctan2(proj_stk,proj_dip)*180/np.pi - 45 # Angle in degrees it should rotate
    theta = -theta_g*np.pi/180 # radians
    
    # Apply rotation
    g_X = g_dip*np.cos(theta) - g_stk*np.sin(theta)
    g_Y = g_dip*np.sin(theta) + g_stk*np.cos(theta)

    G = np.concatenate((g_X,g_Y),axis=1)
    
    return G, theta

def m_return_rotation(m_X, m_Y, theta):
    """
    Function to go back from the rotated system to a dip-strike domain.
    
    Inputs:
        m_X: backslip in X direction
        m_Y: backslip in Y direction
        theta: array of angles by which each subfault rotated, generated in
              G_rotation function     
              
    Output:
        m: backslip vector in Dip-Strike domain.
    """
    
    slip_dip = m_X*np.cos(theta) + m_Y*np.sin(theta)
    slip_stk = -m_X*np.sin(theta) + m_Y*np.cos(theta)
    m = np.concatenate((slip_dip,slip_stk),axis=0)
    return m



def v_euler_pole(omega, lon_pe, lat_pe, lon, lat):
    """
    Velocity estimation by Euler pole rotation, following the 
    Goudarzi et al, 2014 methodology.
    
    Inputs:
        omega: angular velocity in degrees/yrs
        lon_pe: longitude of Euler Pole in degrees
        lat_pe: latitude of Euler Pole in degrees
        lon: array with longitudes of data points (degrees)
        lon: array with latitudes of data points (degrees)
        
    Outpus:
        ve: array with velocities in East-West direction on each data point [mm/yr]
        vn: array with velocities in North-South direction on each data point [mm/yr]
    """
    
    r1 = 6378137e3; # Equitorial axis of geoid [mm]
    r2 = 6356752e3; # Polar axis of geoid [mm]
    
    rX = r1*np.cos(np.deg2rad(lat))
    rY = r2*np.sin(np.deg2rad(lat))
    re = np.sqrt(rX**2 + rY**2) # Radius from ellipsoid center to surface
    
    # Follow ecuation 5 of Goudarzi et al, 2014
    wx = np.deg2rad(omega)*np.cos(np.deg2rad(lat_pe))*np.cos(np.deg2rad(lon_pe))
    wy = np.deg2rad(omega)*np.cos(np.deg2rad(lat_pe))*np.sin(np.deg2rad(lon_pe))
    wz = np.deg2rad(omega)*np.sin(np.deg2rad(lat_pe))
    
    # Follow ecuation 11 of Goudarzi et al, 2014
    ve = re*(-np.sin(np.deg2rad(lat))*np.cos(np.deg2rad(lon))*wx \
            -np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(lon))*wy \
            + np.cos(np.deg2rad(lat))*wz)
    
    vn = re*(np.sin(np.deg2rad(lon))*wx \
            -np.cos(np.deg2rad(lon))*wy)
    
    return ve, vn

def covariance_matrix_Rg(xyz, parameters, corr_DsSs = 0):
    """
    Generate the covariance matrix for a Gaussian correlation function, given
    a correlation length and standard deviation.
    
    Inputs:
        xyz: a list of 3 arrays that contains the UTM locations in X, Y, Z
    directions as:
            xyz = [X, Y, Z]
        parameters: a list of two objects that contains the standard deviation
    value, and the correlation length value as:
            parameters = [sigma_s, Lc]
        corr_DsSs: a correlation value between Dip direction and strike
    direction in a subfault. Set just in case this value will be known.
    
    Outputs:
        M: the covariance matrix
    
    """
    n_m = int(len(xyz[0]))
    X = xyz[0]
    Y = xyz[1]
    Z = xyz[2]
    
    R = np.zeros([2*n_m,2*n_m])
    
    Lc = parameters[1]
    
    for k in range(n_m):
        rx = X[k] - X
        ry = Y[k] - Y
        rz = Z[k] - Z
        r = np.sqrt(rx**2 + ry**2 + rz**2)

        R[k,:] = np.concatenate((fg(r/Lc),np.zeros([n_m,])),axis=0) 
        R[n_m+k,:] = np.concatenate((np.zeros([n_m,]),fg(r/Lc)),axis=0)
        R[k,k] = 1
        R[n_m+k,n_m+k] = 1
        R[k,n_m+k] = corr_DsSs
        R[n_m+k,k] = corr_DsSs
            
    #Constructing covariance matrix:
    sigma = parameters[0]
    M = (sigma**2)*R
    return M

def covariance_matrix_Re(xyz, parameters, corr_DsSs = 0):
    """
    Generate the covariance matrix for a exponential correlation function, given
    a correlation length and standard deviation.
    
    Inputs:
        xyz: a list of 3 arrays that contains the UTM locations in X, Y, Z
    directions as:
            xyz = [X, Y, Z]
        parameters: a list of two objects that contains the standard deviation
    value, and the correlation length value as:
            parameters = [sigma_s, Lc]
        corr_DsSs: a correlation value between Dip direction and strike
    direction in a subfault. Set just in case this value will be known.
    
    Outputs:
        M: the covariance matrix
    
    """
    n_m = int(len(xyz[0]))
    X = xyz[0]
    Y = xyz[1]
    Z = xyz[2]
    
    R = np.zeros([2*n_m,2*n_m])
    
    Lc = parameters[1]
    
    for k in range(n_m):
        rx = X[k] - X
        ry = Y[k] - Y
        rz = Z[k] - Z
        r = np.sqrt(rx**2 + ry**2 + rz**2)

        R[k,:] = np.concatenate((fe(r/Lc),np.zeros([n_m,])),axis=0) 
        R[n_m+k,:] = np.concatenate((np.zeros([n_m,]),fe(r/Lc)),axis=0)
        R[k,k] = 1
        R[n_m+k,n_m+k] = 1
        R[k,n_m+k] = corr_DsSs
        R[n_m+k,k] = corr_DsSs
            
    #Constructing covariance matrix:
    sigma = parameters[0]
    M = (sigma**2)*R
    return M

def covariance_matrix_RVK(xyz, parameters, corr_DsSs = 0):
    """
    Generate the covariance matrix for a von Kármán correlation function, given
    a correlation length and standard deviation.
    
    Inputs:
        xyz: a list of 3 arrays that contains the UTM locations in X, Y, Z
    directions as:
            xyz = [X, Y, Z]
        parameters: a list of tree objects that contains the standard deviation
    value, and the correlation length value and the Hurs parameter as:
            parameters = [sigma_s, Lc, H]
        corr_DsSs: a correlation value between Dip direction and strike
    direction in a subfault. Set just in case this value will be known.
    
    Outputs:
        M: the covariance matrix
    
    """
    n_m = int(len(xyz[0]))
    X = xyz[0]
    Y = xyz[1]
    Z = xyz[2]
    
    R = np.zeros([2*n_m,2*n_m])
    
    Lc = parameters[1]
    H = parameters[2]
    for k in range(n_m):
        rx = X[k] - X
        ry = Y[k] - Y
        rz = Z[k] - Z
        r = np.sqrt(rx**2 + ry**2 + rz**2)

        R[k,:] = np.concatenate((fVK(r/Lc, H),np.zeros([n_m,])),axis=0) 
        R[n_m+k,:] = np.concatenate((np.zeros([n_m,]),fVK(r/Lc, H)),axis=0)
        R[k,k] = 1
        R[n_m+k,n_m+k] = 1
        R[k,n_m+k] = corr_DsSs
        R[n_m+k,k] = corr_DsSs
            
    #Constructing covariance matrix:
    sigma = parameters[0]
    M = (sigma**2)*R
    return M

def bayesian_positive_inversion(G, d, s_p, S_inv, D_inv, maxiter=30, tol_res=0.05):
    """
    This function performs the Matsu’ura and Hasegawa (1987) algorithm, according 
    the formulation proposed on Becerra-Carreño et al. (2022) research, for a
    positive constraint inversion.
    
    Inputs:
        G: green matrix of the problem to solve
        d: data set
        s_p: array with the prior mean of the Gaussian variable  s
        S_inv: Inverse prior covariance matrix of parameters s
        D_inv: inverse likellihood covariance matrix
        maxiter: maximum aceptable value for iterations
        tol_res: trheshold to minimization of residual
        
    Outputs:
        s0: Posterior solution of Gaussian variable s
        C: Posterior covariance matrix
        C_inv: Inverse  of C
        phi_s: objective function evaluated on the posterior solution s0
    """
    
    n_m = len(s_p)
    
    # Defining Objective function
    def phi(s):
        m = np.abs(s)
        A = d - G@m
        return A.T@D_inv@A + (s_p - s).T@S_inv@(s_p - s)
    
    # Defining Jacobian function
    def jacA(s):
        I = s/np.abs(s)
        return G*I.T
    
    # Starting the algorithm (Matsu’ura and Hasegawa, 1987). 
    # Here we ensure that algorith go through almost more than 1 iteration
    s0 = s_p*1
    m = np.abs(s0)
    A = jacA(s0)
    C_inv = A.T@D_inv@A + S_inv
    C = np.linalg.inv(C_inv)
    
    rk = A.T@D_inv@(d - G@m) + S_inv@(s_p - s0)

    alpha = 1
    sk = s0 + alpha*(C@rk)
            
    phi0 = phi(s0)
    phik = phi(sk)
            
    if phi0 <= phik:
        while phi0 <= phik:   
            alpha -= .1
            sk = s0 + alpha*(C@rk)
            phik = phi(sk)
            if alpha <= 0.01:
                alpha = 0.1
                sk = s0 + alpha*(C@rk)
                break
     
    # Continuating the algorithm (Matsu’ura and Hasegawa, 1987).     
    for k in range(maxiter):
                    
        s0 = sk*1.0
        m = np.abs(s0)
        A = jacA(s0)
        C_inv = A.T@D_inv@A + S_inv
        C = np.linalg.inv(C_inv)
            
        rk = A.T@D_inv@(d - G@m) + S_inv@(s_p - s0)
        
        norm_res = np.linalg.norm(rk)/(n_m)

        if norm_res < tol_res:
            break
        else:
            alpha = 1
            sk = s0 + alpha*(C@rk)
            
            phi0 = phi(s0)
            phik = phi(sk)
            
            if phi0 <= phik:
                while phi0 <= phik:   
                    alpha -= .1
                    sk = s0 + alpha*(C@rk)
                    phik = phi(sk)
                    if alpha <= 0.01:
                        alpha = 0.1
                        sk = s0 + alpha*(C@rk)
                        break        
        
        if k == maxiter - 1:
            print('Warning: The algorithm reached the iterations threshold')
    
    phi_s = phi(s0)
    
    return s0, C, C_inv, phi_s

def log_evidence(D, S, C, phi_s, n_d):
    """
    To perform the logarithm of evidence.
    
    Input:
        D: likellihood covariance matrix
        S: prior covariance matrix
        C: posterior covariance matrix
        phi_s: objective function evaluated on the posterior solution
        n_d: length of data vector
        
    Output:
        ln_evi: logarithm of evidence
    """
    (signC, ln_detC)=np.linalg.slogdet(C)
    (signS, ln_detS)=np.linalg.slogdet(S)
    (signD, ln_detD)=np.linalg.slogdet(D)
    
    ln_evi = -(n_d/2)*np.log(2*np.pi) + (ln_detC - ln_detS - ln_detD - phi_s)/2
    
    return ln_evi
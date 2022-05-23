#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main code created by execute the inversions accomplished on:
    "Plate-locking, uncertainty estimation and spatial correlations revealed
      with a Bayesian model selection method:
      Application to the Central Chile subduction zone"
        
Author: V. Becerra-Carreño
        
last version: May 22 2022
"""


## Needed libraries:
import numpy as np
from os import mkdir, path

import functions as fn
import folded_norm as fdn

#%%############################################################################
# PARAMETERS TO CHANGE
###############################################################################

## You can change this parameters an run different inversions

s_prior =  35 # prior mean s_p in [mm/yr]
sigma_s = 8 #  prior standard deviation \sigma_s in [mm/yr]
sigma_d = 4.0 # fator for adjust the data error \sigma_d in [mm/yr]
Lc = 80 # correlation length L_c in [km]
H = 0.3 # Hurst parameter. It wil be used just in case of von Kármán
        # correlation function. Otherwise, it will be ignored
        
limit_depth = 65.0 # The limitdepth h of seismogenic zone in [km]

Correlation = 'Rg'  # Gaussian: 'Rg'
                    # Exponential: 'Re'
                    # Von Karman: 'RVK'

# Euler pole for Andean Sliver correction (Metois et al., 2016)            
lat_pe = -56.37 # latitude in degress
lon_pe = -41.27 # longitude in degrees
omega = -0.12e-6 # angular velocity in radians

# Plate converge in degrees (Baker et al., 2013)
conv = 79

#%%############################################################################
# DATA READING AND MATRIX/VECTOR CREATION
###############################################################################

# First, reading the slab (subduction interface) parameters                    
slab = np.loadtxt('geometry_data/slab_la.txt') # Geometry in longitude-latitud-depth
slab_XYZ = np.loadtxt('geometry_data/slab_XYZ.txt') # Geometry in UTM system

lon_slab = slab[:,0]
lat_slab = slab[:,1]
dep_slab = slab[:,2]

X = slab_XYZ[:,0]/1000 # in [km]
Y = slab_XYZ[:,1]/1000 # in [km]
Z = slab_XYZ[:,2]/1000 # in [km]
xyz = [X, Y, Z]

n_m = len(slab)

# Strike y Dip of each subfault
St_Dp = np.loadtxt('geometry_data/strike_dip.txt')
strike = St_Dp[:,0]
dip = St_Dp[:,1]


# Second, reading the GPS velocitie vector
surf_H = np.loadtxt('data/gps_inter_h.txt') # reading file with horizontal vectors
n_d_h = len(surf_H)

lon_H = surf_H[:,0]
lat_H = surf_H[:,1]
d_ew = surf_H[:,2] # in [mm/yr]
d_ns = surf_H[:,3] # in [mm/yr]
sigma_ew = surf_H[:,4] # in [mm/yr]
sigma_ns = surf_H[:,5] # in [mm/yr]

# vertical data
surf_U = np.loadtxt('data/gps_inter_v.txt') # reading file with vertical vectors
n_d_up = len(surf_U)


lon_up = surf_U[:,0]
lat_up = surf_U[:,1]
d_up = surf_U[:,2] # in [mm/yr]
sigma_up = surf_U[:,3] # in [mm/yr]


# Last, reading de Greeen functions matrices

 # Horizontals
gf_ew_dip = np.loadtxt('GFs/greenMatrix_ew_gps_dip.txt')
gf_ns_dip = np.loadtxt('GFs/greenMatrix_ns_gps_dip.txt')
gf_ew_stk = np.loadtxt('GFs/greenMatrix_ew_gps_stk.txt')
gf_ns_stk = np.loadtxt('GFs/greenMatrix_ns_gps_stk.txt')

# verticals
gf_up_dip = np.loadtxt('GFs/greenMatrix_up_gps_dip.txt')
gf_up_stk = np.loadtxt('GFs/greenMatrix_up_gps_stk.txt')


# Reading the index file that contain the GPS stations located in the backarc 
pto_blq = np.loadtxt('geometry_data/index_AS.txt')
pto_blq = pto_blq.astype(int).tolist()

#%%  CREATION OF MATRIX/VECTORS ##############################################

#constructing the G matrix, such that Gm=d
g_dip = np.concatenate((gf_ew_dip,gf_ns_dip, gf_up_dip),axis=0)
g_stk = np.concatenate((gf_ew_stk,gf_ns_stk, gf_up_stk),axis=0)

# Rotation of G matrix according to Becerra-Carreño et al., 2022:
G, theta = fn.G_rotation(dip, strike, g_dip, g_stk, conv)

### Velocity estimation by the "Andean Sliver" Euler pole
# (Goudarzi et al, 2014; Metois et al., 2016)
ve, vn = fn.v_euler_pole(omega, lon_pe, lat_pe, lon_H[pto_blq], lat_H[pto_blq])

d_euler_e = np.zeros(n_d_h)
d_euler_e[pto_blq] = ve

d_euler_n = np.zeros(n_d_h)
d_euler_n[pto_blq] = vn

d_euler_u = np.zeros(n_d_up)

d_euler = np.concatenate((d_euler_e, d_euler_n, d_euler_u), axis=0)


# Constructing the d vetor:
d_hor = np.concatenate((d_ew,d_ns),axis=0)
d = np.concatenate((d_hor,d_up),axis=0) - d_euler

N_d = len(d)

# Data covariance matrix D
d_sigma_hor = np.concatenate((sigma_ew,sigma_ns), axis=0)
d_sigma_hor[np.where(d_sigma_hor == 0)] = 1 
sigma_up[np.where(sigma_up == 0)] = 4 # when data error is set to 0, 
                                      # it was not estimated. For instance, 
                                      # we set the mean data error of horizontal
                                      # and vertical velocities.
d_sigma = np.concatenate((d_sigma_hor,sigma_up), axis=0) # data error vector

D_data = np.diag((d_sigma**2))
D = D_data*(sigma_d**2)
D_inv = np.diag(1/np.diag(D)) # due to is a diagonal matrix

# Backslip a priori
   # As we invert to coordinates X and Y, we divided by sqrt(2) to ensure
   # a mean amplitude of s_prior
Ds_p = s_prior*np.ones(n_m).T/np.sqrt(2)
Ss_p = s_prior*np.ones(n_m).T/np.sqrt(2)

   # We impose the condition of the seismogenic depth
Ds_p[dep_slab < -limit_depth ] = 0.01/np.sqrt(2)
Ss_p[dep_slab < -limit_depth ] = 0.01/np.sqrt(2)

m_prior = np.concatenate((Ds_p,Ss_p),axis=0)


# PRIOR COVARIANCE MATRIX    
if Correlation == 'RVK':
    Nh = 6
    parameters = [sigma_s, Lc, H]
    S = fn.covariance_matrix_RVK(xyz, parameters)
elif Correlation == 'Rg':
    Nh = 5
    parameters = [sigma_s, Lc]
    S = fn.covariance_matrix_Rg(xyz, parameters)
elif Correlation == 'Re':
    Nh = 5
    parameters = [sigma_s, Lc]
    S = fn.covariance_matrix_Re(xyz, parameters)
    
S_inv = np.linalg.inv(S)
s_p = np.abs(m_prior)



#%% ##########################################################################
# BAYESIAN POSITIVE INVERSION
##############################################################################

s, C, C_inv, phi_s = fn.bayesian_positive_inversion(G, d, s_p, S_inv, D_inv)
ln_evi = fn.log_evidence(D, S, C, phi_s, N_d)
abic = -2*ln_evi + 2*Nh

print('ln_evi = ', ln_evi)
print('abic = ', abic)

#%% ##########################################################################
# BACK TO THE ORIGINAL BACKSLIP DOMAIN
#  Kan and Robotti, 2017
##############################################################################
m_mean, Cm = fdn.fnormmom(s,C)

#%% ##########################################################################
# RESULTS
##############################################################################

# Results for s variable
slip_s = np.abs(s)

s_X = slip_s[0:n_m]
s_Y = slip_s[n_m:]
s_abs = np.sqrt(s_X**2 + s_Y**2)

sigma2_s = np.diag(C) # Variance
sigma_s = np.sqrt(sigma2_s) # standard deviation
sigma_sx = sigma_s[0:n_m]
sigma_sy = sigma_s[n_m:]


# Results of m variable (backslip)
m_X= m_mean[0:n_m]
m_Y = m_mean[n_m:]
m_abs = np.sqrt(m_X**2 + m_Y**2)

sigma2_m = np.diag(Cm) # Variance
sigma_m = np.sqrt(sigma2_m) # standard deviation
sigma_mx = sigma_m[0:n_m]
sigma_my = sigma_m[n_m:]

# DATA PREDICTIONS
d_pred = G@m_mean

d_pred_E = d_pred[:n_d_h]
d_pred_N = d_pred[n_d_h:2*n_d_h]
d_pred_U = d_pred[2*n_d_h:]

#%% SAVE RESULTS (IN A FORMAT TO EASILY GRAPH)
folder = 'Results/'

if not path.exists(folder):
    mkdir(folder)
    
    
# Saving a file for s variable
file = folder + 's.lonlat'
f = open(file,'w')
f.write('#lon     lat    Amp S [mm/yr]    S_X [mm/yr]   S_Y [mm/yr]   Sigma_X [mm/yr]   Sigma_Y [mm/yr]\n')
for k in range(n_m):
    f.write('%f %f %f %f %f %f %f\n' % (lon_slab[k], lat_slab[k], s_abs[k], s_X[k], s_Y[k], sigma_sx[k], sigma_sy[k]))
f.close()


# Saving a file for m variable (The backslip result)
file = folder + 'm.lonlat'
f = open(file,'w')
f.write('#lon     lat    Amp m [mm/yr]    m_X [mm/yr]   m_Y [mm/yr]   Sigma_X [mm/yr]   Sigma_Y [mm/yr]\n')
for k in range(n_m):
    f.write('%f %f %f %f %f %f %f\n' % (lon_slab[k], lat_slab[k], m_abs[k], m_X[k], m_Y[k], sigma_mx[k], sigma_my[k]))
f.close()


# Saving a file for horizontal data (real and predicted by our solution  model)
file = folder + 'data_h.lonlat'
f = open(file,'w')
f.write('#lon     lat   dE [mm/yr]   dN [mm/yr]  dE_pred [mm/yr]   dN_pred [mm/yr]\n')
for k in range(n_d_h):
    f.write('%f %f %f %f %f %f\n' % (lon_H[k], lat_H[k], d_ew[k], d_ns[k], d_pred_E[k], d_pred_N[k]))
f.close()


# Saving a file for vertical data (real and predicted by our solution  model)
file = folder + 'data_v.lonlat'
f = open(file,'w')
f.write('#lon     lat   dU [mm/yr]   dU_pred [mm/yr]\n')
for k in range(n_d_up):
    f.write('%f %f %f %f\n' % (lon_up[k], lat_up[k], d_up[k], d_pred_U[k]))
f.close()

### SAVING THE SOLUTION S VECTOR AND C MATRIX AS NPY FILES
# YOU WILL NEED THESE FOR OBTAINING SAMPLING
# using example:  C = np.load('C.npy')
#                 s = np.load('s.npy')
#                 n_samples = 100000
#                 samples = np.random.multivariate_normal(mean=s, cov=C, size=n_samples)

file = folder + 'C'
np.save(file, C)
file = folder + 's'
np.save(file, s)
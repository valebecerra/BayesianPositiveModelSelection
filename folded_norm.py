#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 18:54:34 2020

@author: Valeria Becerra-Carreño

Package of funtions to perform Folded multivariate Distributions, created by
 "Kan & Robotti, 2017" and modified in a Python code
 
Source: Raymond Kan and Cesare Robotti. On Moments of Folded and
        Truncated Multivariate Normal Distributions, 2017. ISSN 15372715.

"""

import numpy as np
import scipy.special as sp
import scipy.stats as st

rroot2 = 0.70710678118654752440

def tf1(h, a, ah):
    rtwopi = 0.15915494309189533577
    rrtpi = 0.39894228040143267794
    
    c2= [0.99999999999999987510e+00, -0.99999999999988796462e+00, 0.99999999998290743652e+00,
     -0.99999999896282500134e+00, 0.99999996660459362918e+00, -0.99999933986272476760e+00,
     0.99999125611136965852e+00, -0.99991777624463387686e+00, 0.99942835555870132569e+00,
     -0.99697311720723000295e+00, 0.98751448037275303682e+00, -0.95915857980572882813e+00,
     0.89246305511006708555e+00, -0.76893425990463999675e+00, 0.58893528468484693250e+00,
     -0.38380345160440256652e+00, 0.20317601701045299653e+00, -0.82813631607004984866e-01,
     0.24167984735759576523e-01, -0.44676566663971825242e-02, 0.39141169402373836468e-03]
    
    pts = [0.35082039676451715489e-02, 0.31279042338030753740e-01, 0.85266826283219451090e-01,
       0.16245071730812277011e+00, 0.25851196049125434828e+00, 0.36807553840697533536e+00,
       0.48501092905604697475e+00, 0.60277514152618576821e+00, 0.71477884217753226516e+00,
       0.81475510988760098605e+00, 0.89711029755948965867e+00, 0.95723808085944261843e+00,
       0.99178832974629703586e+00]
    
    wts = [0.18831438115323502887e-01, 0.18567086243977649478e-01, 0.18042093461223385584e-01,
       0.17263829606398753364e-01, 0.16243219975989856730e-01, 0.14994592034116704829e-01,
       0.13535474469662088392e-01, 0.11886351605820165233e-01, 0.10070377242777431897e-01,
       0.81130545742299586629e-02, 0.60419009528470238773e-02, 0.38862217010742057883e-02,
       0.16793031084546090448e-02]
    
    meth = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6]
    Ord = [2, 3, 4, 5, 7, 10, 12, 18, 10, 20, 30, 20, 4, 7, 8, 20, 13, 0]
    h_range = [0.02, 0.06, 0.09, 0.125, 0.26, 0.4, 0.6, 1.6, 1.7, 2.33, 2.4, 3.36, 3.4, 4.8, np.inf]
    a_range = [0.025, 0.09, 0.15, 0.36, 0.5, 0.9, 0.99999, 1]
    
    select = [[1, 1, 2, 13, 13, 13, 13, 13, 13, 13, 13, 16, 16, 16, 9], [1, 2, 2, 3, 3, 5, 5, 14, 14, 15, 15, 16, 16, 16, 9],
          [2, 2, 3, 3, 3, 5, 5, 15, 15, 15, 15, 16, 16, 16, 10], [2, 2, 3, 5, 5, 5, 5, 7, 7, 16, 16, 16, 16, 16, 10],
          [2, 3, 3, 5, 5, 6, 6, 8, 8, 17, 17, 17, 12, 12, 11], [2, 3, 5, 5, 5, 6, 6, 8, 8, 17, 17, 17, 12, 12, 12],
          [2, 3, 4, 4, 6, 6, 8, 8, 17, 17, 17, 17, 17, 12, 12], [2, 3, 4, 4, 6, 6, 18, 18, 18, 18, 17, 17, 17, 12, 12]]
    
    #  determine appropriate method from t1...t6
    
    for ihint in range(len(h_range)):
        if h <= h_range[ihint]:
            break
        
    for iaint in range(8):
        if a <= a_range[iaint]:
            break
    
    icode = select[iaint][ihint]
    m = Ord[icode - 1]
    
    if icode <= 8:
        """
        t1(h,a,m);  m = 2, 3, 4, 5, 7, 10, 12 or 18
        jj = 2j - 1; gj = exp(-h*h/2)*(-h*h/2)**j/j!
        aj = a**(2j-1)/(2*pi)
        """
        hs = -0.5*h*h
        dhs = np.exp(hs)
        a2 = a*a
        aj = rtwopi*a
        dj = dhs - 1
        gj = hs*dhs
        tf = rtwopi*np.arctan(a) + dj*aj
        for j in range(m-1):
            jj = j + 2
            aj = aj*a2
            dj = gj - dj
            gj = gj*hs/jj
            tf = tf + dj*aj/(2*jj-1)
            
    elif icode <= 11:
        """
        t2(h,a,m) ; m = 10, 20 or 30
        z = (-1)**(i-1)*zi ; ii = 2i - 1
        vi = (-1)**(i-1)*a**(2i-1)*exp[-(a*h)**2/2]/sqrt(2*pi)
        """
        hs = h*h
        a2 = -a*a
        vi = rrtpi*a*np.exp(-0.5*ah*ah)
        z = 0.5*sp.erf(ah*rroot2)/h
        y = 1/hs
        tf = z
        for i in range(m):
            ii = 2*(i+1) - 1
            z = y*(vi-ii*z)
            vi = a2*vi
            tf = tf+z
        tf = tf*rrtpi*np.exp(-0.5*hs)
        
    elif icode == 12:
        """
        t3(h,a,m) ; m = 20
        ii = 2i - 1 
        vi = a**(2i-1)*exp[-0.5*(a*h)^2]/sqrt(2*pi)
        """
        hs = h*h
        a2 = a*a
        vi = rrtpi*a*np.exp(-0.5*ah*ah)
        zi = 0.5*sp.erf(ah*rroot2)/h
        y = 1/hs
        tf = zi*c2[0]
        for i in range(m):
            ii = i + 2
            zi = y*((2*ii-3)*zi-vi)
            vi = a2*vi
            tf = tf + zi*c2[i-1]
        tf = tf*rrtpi*np.exp(-0.5*hs)
        
    elif icode <= 16:
        """
        t4(h,a,m) ; m = 4, 7, 8 or 20;  ii = 2i + 1
        ai = a*exp[-h*h*(1+a*a)/2]*(-a*a)**i/(2*pi)
        """
        hs = h*h
        a2 = -a*a
        ai = rtwopi*a*np.exp(-0.5*hs*(1-a2))
        tf = ai
        yi = 1
        for i in range(m):
            ii = 2*i + 3
            yi = (1-hs*yi)/ii
            ai = ai*a2
            tf = tf + ai*yi
            
    elif icode == 17:
        """
        t5(h,a,m) ; m = 13
        2m - point gaussian quadrature
        """
        tf = 0
        a2 = a*a
        hs = -0.5*h*h
        for i in range(m):
            r = 1 + a2*pts[i]
            tf = tf + wts[i]*np.exp(hs*r)/r
        tf = a*tf
        
    else:
        """
        t6(h,a);  approximation for a near 1 (a<=1)
        """
        normh = 0.5*sp.erfc(h*rroot2)
        tf = 0.5*normh*(1-normh)
        y = 1 - a
        r = np.arctan(y/(1+a))
        if r != 0:
            tf = tf - rtwopi*r*np.exp(-0.5*y*h*h/r)
            
    return tf


def tfun(h,a):
    """
       Program to compute the Owen's T function
    """
    if a == 0:
        t = 0
        return t
     
    if h == 0 and np.isinf(a):
        t = np.sign(a)/4
        return t
    
    absh = abs(h)
    absa = abs(a)
    ah = absa*absh
    if absa <= 1:
        t = tf1(absh,absa,ah)
    else:
        if absh <= 0.67:
            t = 0.25 - 0.25*sp.erf(absh*rroot2)*sp.erf(ah*rroot2) - tf1(ah,1/absa,absh)
        else:
            normh = 0.5*sp.erfc(absh*rroot2)
            normah = 0.5*sp.erfc(ah*rroot2)
            t = 0.5*(normh+normah) - normh*normah - tf1(ah,1/absa,absh)
            
    if a < 0.0:
        t = -1*t
        
    return t

def bnorm(x, y, r):
    """
    A function to compute the cumulative bivariate normal density
    with parameter r.
    """
    if np.isinf(x) or np.isinf(y):
        if x==-np.inf or y==-np.inf:
            z = 0
        else:
            if np.isinf(x):
                z = st.norm.cdf(y)
            else:
                z = st.norm.cdf(x)
        return z
    
    if abs(r) != 1:
        if x == 0 and y == 0:
            z = 0.25 + np.arcsin(r)/(2*np.pi)
            return z
        
        temp = 1/np.sqrt(np.abs(1-r*r))
        if x != 0:
            t1 = tfun(x, y*temp/x - r*temp)
        else: # tfun(0,(-)inf) = (-)1/4
            if y < 0:
                t1 = -1/4
            else:
                t1 = 1/4
                
        if y != 0:
            t2 = tfun(y, x*temp/y - r*temp)
        else:
            if x < 0:
                t2 = -1/4
            else:
                t2 = 1/4
                
        if x*y < 0 or ((x*y==0) and (x+y<0)):
            z = max(0, 0.5*(st.norm.cdf(x) + st.norm.cdf(y) - 1.0) - t1 - t2)
        else:
            z = max(0, 0.5*(st.norm.cdf(x) + st.norm.cdf(y)) - t1 - t2)
            
    else:
        if r == 1.0:
            z = st.norm.cdf(min(x, y)) # P[X<x;X<y]=P[X<min(x,y)]
        else:
            z = max(st.norm.cdf(x) - st.norm.cdf(-y), 0) # P[X<x;-X<y]=P[-y<X<x]
    
    return z


def fnormmom(mu,S):
    """
    Computes the mean and covariance matrix of Y=|X|,
    where X ~ N(mu,S).
    Input:
      mu: an nx1 vector of mean of X
      S: an nxn covariance matrix of X
    Output
      muY: E[Y] = E[|X|]
      varY: Var[Y] = Var[|X|]
    """
    
    n = len(mu)
    muY = np.zeros([n,1])
    varY = np.zeros([n,n])
    s = np.sqrt(np.diag(S))
    h = mu/s
    pdfh = st.norm.pdf(h)
    cdfh = st.norm.cdf(h)
    
    muY = s*(2*pdfh + h*sp.erf(h/np.sqrt(2)))
    
    R = S/(np.outer(s,s));   # correlation matrix
    np.fill_diagonal(R, 0.0) # some cleaning up
    
    h1 = np.outer(h, np.ones([n]))
    A = (h1 - R*h1.T)/np.sqrt(2*(1-R*R))
    np.fill_diagonal(A, 0.0) # some cleaning up
    gam = (h@pdfh.T)*sp.erf(A)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                varY[i,j] = mu[i]**2 + s[i]**2
            elif i > j:
                varY[i,j] = varY[j,i]*1
            else:
                r = R[i,j]
                eta = np.sqrt(np.abs(1 - r**2))
                p = 4*bnorm(h[i],h[j], r) - 2*cdfh[i] - 2*cdfh[j] + 1
                c = np.sqrt(h[i]**2 + h[j]**2 - 2*r*h[i]*h[j])/eta
                varY[i,j] = s[i]*s[j]*(p*(h[i]*h[j] + r) + 2*gam[i,j] + 2*gam[j,i] + 4*eta/np.sqrt(2*np.pi)*st.norm.pdf(c))
    
    varY = varY - np.outer(muY, muY)
    
    return muY, varY

    

            
            
            
            
        
        
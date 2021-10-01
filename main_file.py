# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:49:21 2021

@author: Sergio
"""

import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat, unumpy as unp
from uncertainties.unumpy import nominal_values as nv, std_devs as sd
from uncertainties.umath import *
from scipy.optimize import curve_fit as cf
from scipy.optimize import curve_fit as cf
from scipy.odr import ODR, Model, RealData
import pandas as pd
from functions_relativity_1 import *


#Define constants
Ms = 2e30 #kg, mass of the Sun
c = 3e8 #m/s, speed of light
G = 6.67e-11 #SI, gravitational constant
Rs = 7e5*1e3 #m, radius of the Sun
rg_sun = G*Ms/c**2 #m, Gravitational radius of the Sun

def line(eta, re, rp, rg):
    
    """
    Function to determine the light delay due to gravitational effects
    of a mass considering a linear trayectory when it is send to a planet and
    it comes back

    Parameters
    ----------
    eta : closest approach distance to the massive object
    
    re : Distance from the observer to the massive object
    
    rp : Distance from the target planet to the massive object
    
    rg : Gravitational radius of the massive object (rg = GM/c^2)

    Returns
    -------
    Time delay. Units are the same as the speed of light ones

    """
    
    def t(r):
        return r + np.sqrt((r*r) - (eta*eta))
    
    
    return 4*rg/c * np.log( ( (t(rp) * t(re)) / (eta*eta) ) )



def shapiro(eta, re, rp, rg):
    
    """
    Function to determine the light delay due to gravitational effects
    of a mass considering the curvature of the trayectory when it is send
    to a planet and it comes back

    Parameters
    ----------
    eta : closest approach distance to the massive object
    
    re : Distance from the observer to the massive object
    
    rp : Distance from the target planet to the massive object
    
    rg : Gravitational radius of the massive object (rg = GM/c^2)

    Returns
    -------
    Time delay. Units are the same as the speed of light ones

    """
    
    def sq(r):
        return np.sqrt( (r*r) - (eta*eta) )
    
    return 4*rg/c * np.log((re + sq(re) + rp + sq(rp)) / 
                           (re - sq(re) + rp - sq(rp)) )



def get_time(exp, teo):
    """
    

    Parameters
    ----------
    exp : experimental data (brute, 4 columns)
    teo : theoretical data (plotted, 2 columns)

    Returns
    -------
    array with the corresponding theoretical delay for the exp data

    """
    x = exp[:,0]
    x_t = teo[:,0]
    t_t = teo[:,1]
    
    t_f = np.array([])
    for xe in x:
        mini = np.inf
        for i, xt in enumerate(x_t):
            aux = abs(xt-xe)
            if aux < mini:
                mini = aux
            else: 
                break
        
        t_f = np.append(t_f, t_t[i-1])
    return t_f


def get_M(exp, teo, M_min, M_max, N):
    """
    

    Parameters
    ----------
    exp : experimental data
    teo : theoretical data
    M_min : left limit
    M_max : right limit
    N : number of iterations

    Returns
    -------
    Mass with error (1 sigma)

    """
    #Define step
    fact = (M_max-M_min)/N
    
    m = (M_min-fact)
    
    #Initialize arrays
    M = np.array([])
    Chi = np.array([])
    
    #Loop for all the masses
    while m < M_max:
        #Initialize data in the loop
        m = m+fact
        rg = G*m/c**2
        
        #Add the mass to the Mass array
        M = np.append(M, m)
        
        #Get Shapiro's prediction for mass m
        dt = shapiro(teo[:,3], teo[:,2], teo[:,1], rg)*1e6#micros
        
        #Clean data to use in the functions
        shap_plot = np.array([teo[:,0], dt]).T
        
        #Calculate chi for that mass
        dt_s = get_time(exp, shap_plot) #Times equivalent for the exp data to 
        #be able to calculate chi
        
        #Obtain chi 
        xi = chi_s(dt_s, exp[:,1], exp[:,2], exp[:,3])
        
        #Add chi to the Chi array
        Chi = np.append(Chi, xi) 
    #Get minimum value of chi to compare with
    chi_min = min(Chi)
    m_chi_min = M[np.argmin(Chi)]
    
    #Initialize validation arrays
    M_valid = np.array([])
    Chi_valid = np.array([])
    
    #Loop to Chi
    for i, j in enumerate(Chi):
        #Check for validity condition
        if j <= chi_min+1:
            M_valid = np.append(M_valid, M[i])
            Chi_valid = np.append(Chi_valid, j)
    return M_valid, m_chi_min
    

    
def main():
    #Open files
    are = np.genfromtxt('arecibo.dat', delimiter=',', skip_header=2)
    hay = np.genfromtxt('haystack.dat', delimiter=',', skip_header=2)
    geo = np.genfromtxt('geom.dat', delimiter=' ')
    
    #Plot the experimental data
    graph(are, 'b', '.', 'Arecibo')
    graph(hay, 'r', '.', 'Haystack')
    
    #Calculate the predictions of both models
    rv = geo[:,1]
    re = geo[:,2]
    eta = geo[:,3]
    
    #Linear
    dt_l = line(eta, re, rv, rg_sun)*1e6#micros
    
    #Curve (Shapiro's model)
    dt_s = shapiro(eta, re, rv, rg_sun)*1e6#micros
    
    #Compress data
    linear = np.array([geo[:,0], dt_l]).T
    shap = np.array([geo[:,0], dt_s]).T 
    
    #Plot both predictions
    graph(linear, 'k', ':', 'rect')
    graph(shap, 'g', '-.', 'Shap')
    
    
    #Plot final settings
    plt.xlabel('Días desde la conjunción superior')
    plt.ylabel(r'$\Delta t$/$\mu$s')
    plt.ylim(0, 200)
    show_graph()
    
    #Check for differnece between models
    dif = dt_l-dt_s
    ddif = np.array([geo[:,0], np.abs(dif)]).T
    graph(ddif, 'k', '.', None)
    plt.ylabel(r'$\Delta(\Delta t)$/$\mu$s')
    plt.xlabel('Días desde la conjunción superior')
    show_graph()
    
    #Max deflection
    phi = rg_sun/Rs
    # print(phi*180/np.pi*3600)
    
    
    #Obtain theoretical delay corresponding to each point
    
    #Arecibo
    t_ar = get_time(are, shap)
    
    chi_are = chi_s(t_ar, are[:,1], are[:,2], are[:,3])
    print('Chi for Areibo:', chi_are)
    
    #Haystack
    t_hay = get_time(hay, shap)
    
    chi_hay = chi_s(t_hay, hay[:,1], hay[:,2], hay[:,3])
    print('Chi for haystack:', chi_hay)
    
    #All points together
    
    fin = np.concatenate((are, hay))
    t_tot = get_time(fin, shap)
    
    chi_tot = chi_s(t_tot, fin[:,1], fin[:,2], fin[:,3])
    print('Chi for all the data:', chi_tot)
    
    #For the last part, uncomment from the next block of lines till the end, 
    #it takes some time to perform
    
    # #Estimate mass of the Sun

    # # Width of the interval
    # w = 1e30
    # #Central value
    # c = Ms
    # #Number of iterations
    # N = 100
    
    # #Get valid values of the mass of the Sun
    # M, m_opt = get_M(fin, geo, c-w, c+w, N)
    
    # #Get final result
    # error = (M[-1]-M[0])/2
    # m_sun = ufloat(m_opt, error)
    # print('Mass of the Sun:', m_sun)
    
    
   
    

    
    
    
    
main()
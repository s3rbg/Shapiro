# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:58:51 2021

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

def graph(data, color, f, label):
    """
    Plots data either experimental (4 parameters) or theoretical (2 parameters)

    """
    #Check if the data is experimental or theoretical predictions
    if len(data.T)==4:
        #Define the error in the y axis from the upper and lower error data
        dy = [data[:,2], data[:,3]]
        #Plot experimental data
        plt.errorbar(data[:,0], data[:,1], dy, fmt=f, color=color, label=label)
    elif len(data.T)==2:
        #Plot theoretical predictions
        plt.errorbar(data[:,0], data[:,1], fmt=f, color=color, label=label)
    else:
        print('Not valid data')

def show_graph():
    plt.legend()
    plt.grid()
    plt.show()
    


def chi_s(t_teor, t_exp, s_up, s_down):
    
    return np.sum( (t_teor-t_exp)**2 / ((s_up+s_down)/2)**2)/len(s_up)
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:07:04 2021

@author: samvirthandi
"""

import numpy as np
import matplotlib.pyplot as plt
from highspeed_fft import velocity_calculator
from scipy.special import i0, i1


def weber_velocity(weber_number, reynolds_number):
    """works out velocity from weber number
    """
    d = 2e-3
    sigma = 0.07
    rho_g = 1.225
    u_l = velocity_calculator(reynolds_number)
    u_g = np.sqrt((weber_number*sigma)/(d*rho_g))+u_l
    return u_g



def rayleigh():
    """
    Produces rayleigh dispersion relation

    Returns
    -------
    None.

    """
    k = np.linspace(0, 1000, 10000)
    sigma = 0.07
    a = 1e-3
    rho = 1000
    w_squared = ((sigma*k)/(rho*a**2))*(1-k**2*a**2)*(i1(k*a)/i0(k*a))
    normaliser = np.sqrt(sigma/rho*a**3)
    normal_w = w_squared**0.5/normaliser
    sqrt_w = np.sqrt(w_squared)

    fig, ax = plt.subplots()
    ax.plot(k*a, normal_w)
    ax.set_xlim(0, 1.2)

    wavelength = 2*np.pi/k
    u_g = weber_velocity(5.22, 1551)
    u_l = velocity_calculator(1551)
    u_avg = (u_l+u_g)/2
    freq = u_avg/wavelength

    fig1, ax1 = plt.subplots()
    ax1.plot(freq, sqrt_w)


def plotting()
    

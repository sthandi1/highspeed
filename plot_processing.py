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
from scipy.signal import savgol_filter

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


def plotting(file1, file2, file3):
    """
    Main plotting function
    """

    freqs, _, thresh_800_w, _, _, _, _ = np.loadtxt(file1, delimiter=',',
                                                    unpack=True)
    freqs, _, thresh_1000_w, _, _, _, _ = np.loadtxt(file2, delimiter=',',
                                                     unpack=True)
    freqs, _, thresh_1400_w, _, _, _, _ = np.loadtxt(file3, delimiter=',',
                                                     unpack=True)

    k = np.linspace(0, 1000, 10000)
    sigma = 0.07
    a = 1e-3
    rho = 1000
    w_squared = ((sigma*k)/(rho*a**2))*(1-k**2*a**2)*(i1(k*a)/i0(k*a))
    normaliser = np.sqrt(sigma/rho*a**3)
    normal_w = w_squared**0.5/normaliser
    sqrt_w = np.sqrt(w_squared)
    
    wavelength = 2*np.pi/k
    u_g = weber_velocity(5.22, 1551)
    u_l = velocity_calculator(1551)
    u_avg = (u_l+u_g)/2 - 4
    freq_ra = u_avg/wavelength
    print(u_avg)
    print(u_l)
    print(u_g)

    fig, ax = plt.subplots()
    ax.plot(freqs, thresh_800_w)
    ax.plot(freqs, thresh_1000_w)
    ax.plot(freqs, thresh_1400_w)
    ax.plot(freq_ra, sqrt_w)
    ax.set_xlim(0, 1250)
    ax.set_ylim(0, 100)

    savgol_800 = savgol_filter(thresh_800_w, 11, 2)
    savgol_1000 = savgol_filter(thresh_1000_w, 11, 2)
    savgol_1400 = savgol_filter(thresh_1400_w, 11, 2)

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs, savgol_800)
    ax1.plot(freqs, savgol_1000)
    ax1.plot(freqs, savgol_1400)
    ax1.plot(freq_ra, sqrt_w)
    ax1.set_xlim(0, 1250)
    ax1.set_ylim(0, 100)
    
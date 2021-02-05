#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:07:04 2021

@author: samvirthandi
"""

import numpy as np
import matplotlib.pyplot as plt
from highspeed_fft import velocity_calculator, weber_velocity
from scipy.special import i0, i1
from scipy.signal import savgol_filter


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


def moro_re_calc(Re):
    """

    Parameters
    ----------
    Re : int
        Reynolds number

    Returns
    -------
    u : float
        velocity.

    """
    # viscosity
    mu = 8.9e-4
    # density
    rho = 1000
    # diameter
    d = 4/1000
    # reynolds number standard equation
    u = Re*mu/(rho*d)
    return u


def rayleigh_moro():
    k = np.linspace(0, 1000, 10000)
    sigma = 0.07
    a = 2e-3
    rho = 1000
    w_squared = ((sigma*k)/(rho*a**2))*(1-k**2*a**2)*(i1(k*a)/i0(k*a))
    normaliser = np.sqrt(sigma/rho*a**3)
    normal_w = w_squared**0.5/normaliser
    sqrt_w = np.sqrt(w_squared)

    fig, ax = plt.subplots()
    ax.plot(k*a, normal_w)
    ax.set_xlim(0, 1.2)

    wavelength = 2*np.pi/k
    u_g = weber_velocity(5.22, 2940)
    u_l = moro_re_calc(2940)
    u_avg = (u_l+u_g)/2
    freq = u_l/wavelength

    fig1, ax1 = plt.subplots()
    ax1.plot(freq, sqrt_w)
    ax1.set_xlabel('Frequencies (Hz)')
    ax1.set_ylabel('Growth rate (1/s)')
    ax1.set_title('Rayleigh with Morozumi parameters')


def plotting_generic(file1, file2, file3):
    """
    Main plotting function
    """

    freqs, _, morozumi_time, _, _, _, _ = np.loadtxt(file1, delimiter=',',
                                                     unpack=True)
    freqs, _, constant_vel, _, _, _, _ = np.loadtxt(file2, delimiter=',',
                                                     unpack=True)
    freqs, _, avg_vel, _, _, _, _ = np.loadtxt(file3, delimiter=',',
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
    u_avg = (u_l+u_g)/2 -2.5
    freq_ra = u_avg/wavelength
    print(u_avg)
    print(u_l)
    print(u_g)

    fig, ax = plt.subplots()
    ax.plot(freqs, morozumi_time, label='Morozumi time')
    ax.plot(freqs, constant_vel, label='constant velocity')
    ax.plot(freqs, avg_vel, label='averaged velocity')
    ax.plot(freq_ra, sqrt_w, label='Rayleigh theoretical')
    ax.set_xlim(0, 1250)
    ax.set_ylim(0, 600)
    ax.set_title('Unfiltered data')
    ax.set_xlabel('Frequencies (Hz)')
    ax.set_ylabel('Growth rate (1/s)')
    ax.legend()

    savgol_moro= savgol_filter(morozumi_time, 1001, 2)
    savgol_const = savgol_filter(constant_vel, 1001, 2)
    savgol_avg = savgol_filter(avg_vel, 1001, 2)

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs, savgol_moro, label='Morozumi time')
    ax1.plot(freqs, savgol_const, label='Constant velocity')
    ax1.plot(freqs, savgol_avg, label='Averaged velocity')
    ax1.plot(freq_ra, sqrt_w, label='Rayleigh theoretical')
    ax1.set_xlim(0, 700)
    ax1.set_ylim(0, 600)
    ax1.set_title('Savgol filtering (window size = 1001) data')
    ax1.set_xlabel('Frequencies (Hz)')
    ax1.set_ylabel('Growth rate (1/s)')
    ax1.legend()


def plotting_ra(file1):
    """
    Main plotting function
    """

    freqs, _, rayleigh_1300, _, _, _, _ = np.loadtxt(file1, delimiter=',',
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
    freq_ra = u_l/wavelength

    fig, ax = plt.subplots()
    ax.plot(freqs, rayleigh_1300, label='Experimental Rayleigh')
    ax.plot(freq_ra, sqrt_w, label='Theoretical Rayleigh')
    ax.set_xlim(0, 1250)
    ax.set_ylim(0, 100)
    ax.set_title('Unfiltered data')
    ax.set_xlabel('Frequencies (Hz)')
    ax.set_ylabel('Growth rate (1/s)')
    ax.legend()

    savgol_ra = savgol_filter(rayleigh_1300, 1001, 2)

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs, savgol_ra, label='Experimental Rayleigh')
    ax1.plot(freq_ra, sqrt_w, label='Theoretical Rayleigh')
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 100)
    ax1.set_title('Savgol filtering (window size = 1001) data')
    ax1.set_xlabel('Frequencies (Hz)')
    ax1.set_ylabel('Growth rate (1/s)')
    ax1.legend()


def plotting_moro(file1):
    """
    Main plotting function
    """

    freqs, _, morozumi, _, _, _, _ = np.loadtxt(file1, delimiter=',',
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
    freq_ra = u_l/wavelength

    fig, ax = plt.subplots()
    ax.plot(freqs, morozumi, label='Morozumi replica Re=2940, We=5.22')
    ax.set_xlim(0, 1250)
    ax.set_ylim(0, 100)
    ax.set_title('Unfiltered data')
    ax.set_xlabel('Frequencies (Hz)')
    ax.set_ylabel('Growth rate (1/s)')
    ax.legend()

    savgol_ra = savgol_filter(morozumi, 101, 2)

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs, savgol_ra, label='Morozumi replica Re=2940, We=5.22')
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 100)
    ax1.set_title('Savgol filtering (window size = 11) data')
    ax1.set_xlabel('Frequencies (Hz)')
    ax1.set_ylabel('Growth rate (1/s)')
    ax1.legend()
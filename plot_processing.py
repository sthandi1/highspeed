#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:07:04 2021

@author: samvirthandi
"""

# locating latex installation
import os
import numpy as np
import matplotlib.pyplot as plt
from highspeed_fft import velocity_calculator, weber_velocity, file_id
from scipy.special import i0, i1
from scipy.signal import savgol_filter


os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'


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
    u_arbitrary = (u_l+u_g)/2 - 2.5
    freq = u_avg/wavelength

    fig1, ax1 = plt.subplots()
    ax1.plot(freq, sqrt_w)
    print(u_arbitrary)
    print(u_g)
    print(u_l)
    print(u_avg)


def moro_re_calc(Re):
    """
    Function to calculate velocity for Morozumi's data
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


def plotting_generic(file1, file2, file3, file4):
    """
    Main plotting function
    """

    freqs, _, morozumi_time, _, _, _, _ = np.loadtxt(file1, delimiter=',',
                                                     unpack=True)
    freqs, _, constant_vel, _, _, _, _ = np.loadtxt(file2, delimiter=',',
                                                    unpack=True)
    freqs, _, avg_vel, _, _, _, _ = np.loadtxt(file3, delimiter=',',
                                               unpack=True)
    freqs, _, aero_vel, _, _, _, _ = np.loadtxt(file4, delimiter=',',
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
    u_avg = (u_l+u_g)/2 - 2.5
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

    savgol_moro = savgol_filter(morozumi_time, 1001, 2)
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

    peak_freq = freqs[np.where(savgol_ra == np.max(savgol_ra))]
    print(peak_freq)
    return peak_freq


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


def arai_velocity(Re):
    g = 9.81
    z_pixels = np.array([50, 100, 150, 250, 350, 450, 550, 650, 750, 800])
    v = velocity_calculator(Re)
    print(v)
    z = z_pixels*0.02/1000

    z_star = 2*g*z/(v**2)

    wave_vel = v*(1+z_star)**0.5
    avg = np.mean(wave_vel)
    return avg


def plotting_arai(file1):
    """
    Main plotting function
    """

    freqs, _, control, _, _, _, _ = np.loadtxt(file1, delimiter=',',
                                                unpack=True)

    k = np.linspace(0, 3000, 10000)
    sigma = 0.07
    a = (2e-3)/2
    rho = 1000
    w_squared = ((sigma*k)/(rho*a**2))*(1-k**2*a**2)*(i1(k*a)/i0(k*a))
    sqrt_w = np.sqrt(w_squared)

    v = arai_velocity(1551)
    wavelength = v/freqs
    wavenumber = 2*np.pi/wavelength
    savgol_control = savgol_filter(control, 1001, 2)

    fig, ax = plt.subplots()
    ax.plot(k*a, sqrt_w, label='Rayleigh')
    ax.plot(wavenumber*a, savgol_control, label='Experimental (average velocity)')
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_xlabel('ka', fontsize=16)
    ax.set_ylabel('$\omega$', fontsize=16)


def plotting_measured_wavelength(file1, file2, file3):
    """
    Main plotting function
    """

    freqs, _, control, _, _, _, _ = np.loadtxt(file1, delimiter=',',
                                               unpack=True)
    freqs, _, control_2, _, _, _, _ = np.loadtxt(file2, delimiter=',',
                                                 unpack=True)
    freqs, _, control_3, _, _, _, _ = np.loadtxt(file3, delimiter=',',
                                                 unpack=True)

    k = np.linspace(0, 7000, 10000000)
    sigma = 0.07
    a = 1e-3
    rho = 1000
    w_squared = ((sigma*k)/(rho*a**2))*(1-k**2*a**2)*(i1(k*a)/i0(k*a))
    sqrt_w = np.sqrt(w_squared)

    v = 1.2668032087199999
    wavelength = v/freqs
    wavenumber = 2*np.pi/wavelength
    savgol_control = savgol_filter(control, 101, 2)
    savgol_control_2 = savgol_filter(control_2, 101, 2)
    savgol_control_3 = savgol_filter(control_3, 101, 2)

    fig, ax = plt.subplots()
    ax.plot(k*a, sqrt_w, label='Rayleigh', color='black', linestyle='solid')
    ax.plot(wavenumber*a, savgol_control, label='Morozumi and Fukai model',
            marker='o', markevery=100)
    ax.plot(wavenumber*a, savgol_control_2, label='Aerodynamic model',
            marker='s', markevery=100)
    ax.plot(wavenumber*a, savgol_control_3, label='Arai and Amagai model',
            marker='^', markevery=100)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_xlabel('ka', fontsize=16)
    ax.set_ylabel('$\omega$', fontsize=16)
    fig.set_size_inches(6.5, 4.5)
    fig.savefig(fname='time_models.pgf', bbox_inches='tight')


def plotting_2file(file1, file2):
    # load the two files
    freqs, _, file1_axi, _, _, file1_as, _ = np.loadtxt(file1, delimiter=',',
                                                        unpack=True)
    freqs, _, file2_axi, _, _, file2_as, _ = np.loadtxt(file2, delimiter=',',
                                                        unpack=True)

    file1_axi_savgol = savgol_filter(file1_axi, 101, 2)
    file2_axi_savgol = savgol_filter(file2_axi, 101, 2)

    fig, ax = plt.subplots()
    ax.plot(freqs, file1_axi, label='file1')
    ax.plot(freqs, file2_axi, label='file2')
    ax.set_title('Standard data')
    ax.legend()
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 80)

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs, file1_axi_savgol, label='file1')
    ax1.plot(freqs, file2_axi_savgol, label='file2')
    ax1.set_title('Savgol')
    ax1.legend()
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 80)

    fig2, ax2 = plt.subplots()
    file1_as_savgol = savgol_filter(file1_as, 101, 2)
    file2_as_savgol = savgol_filter(file2_as, 101, 2)

    ax2.plot(freqs, file1_as_savgol, label='file1')
    ax2.plot(freqs, file2_as_savgol, label='file2')
    ax2.legend()
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 60)


def plotting_3file(file1, file2, file3):
    # load the two files
    freqs, _, file1_axi, _, _, _, _ = np.loadtxt(file1, delimiter=',',
                                                 unpack=True)
    freqs, _, file2_axi, _, _, _, _ = np.loadtxt(file2, delimiter=',',
                                                 unpack=True)
    freqs, _, file3_axi, _, _, _, _ = np.loadtxt(file3, delimiter=',',
                                                 unpack=True)

    file1_axi_savgol = savgol_filter(file1_axi, 101, 2)
    file2_axi_savgol = savgol_filter(file2_axi, 101, 2)
    file3_axi_savgol = savgol_filter(file3_axi, 101, 2)

    fig, ax = plt.subplots()
    ax.plot(freqs, file1_axi_savgol, label='Threshold=800')
    ax.plot(freqs, file2_axi_savgol, label='Threshold=1000')
    ax.plot(freqs, file3_axi_savgol, label='Threshold=1400')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('$\omega$')
    ax.legend()
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 65)

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs, file1_axi_savgol, label='Threshold=800')
    ax1.plot(freqs, file2_axi_savgol, label='Threshold=1000')
    ax1.plot(freqs, file3_axi_savgol, label='Threshold=1400')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('$\omega$')
    ax1.legend()
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 25)
    fig.set_size_inches(6, 4)
    fig1.set_size_inches(6, 4)
    fig.savefig(fname='threshold_comparison_savgol_101.pgf',
                bbox_inches='tight')
    fig1.savefig(fname='threshold_comparison_savgol_101_zoom.pgf',
                 bbox_inches='tight')

    thresh_800_diff = 100*(file2_axi_savgol - file1_axi_savgol)/file2_axi_savgol
    thresh_1400_diff = 100*(file2_axi_savgol - file3_axi_savgol)/file2_axi_savgol

    fig2, ax2 = plt.subplots()
    ax2.plot(freqs, thresh_800_diff)
    ax2.plot(freqs, thresh_1400_diff)
    ax2.set_xlim(0, 700)
    ax2.set_ylim(0, 100)


def plotting_4file(file1, file2, file3, file4):
    """
    Main plotting function
    """

    freqs, _, file1_axi, _, _, file1_as, _ = np.loadtxt(file1, delimiter=',',
                                                        unpack=True)
    freqs, _, file2_axi, _, _, file2_as, _ = np.loadtxt(file2, delimiter=',',
                                                        unpack=True)
    freqs, _, file3_axi, _, _, file3_as, _ = np.loadtxt(file3, delimiter=',',
                                                        unpack=True)
    freqs, _, file4_axi, _, _, file4_as, _ = np.loadtxt(file4, delimiter=',',
                                                        unpack=True)

    casename, Re, We = file_id(file1)

    fig, ax = plt.subplots()
    ax.plot(freqs, file1_axi, label='$RR_G=1.5$')
    ax.plot(freqs, file2_axi, label='$RR_G=3.9$')
    ax.plot(freqs, file3_axi, label='$RR_G=6.3$')
    ax.plot(freqs, file4_axi, label='$RR_G=8.7$')
    ax.set_xlim(0, 1250)
    ax.set_ylim(0, 400)
    ax.set_title('Unfiltered data')
    ax.set_xlabel('Frequencies (Hz)')
    ax.set_ylabel('Growth rate (1/s)')
    ax.legend()

    savgol_file1_axi = savgol_filter(file1_axi, 1001, 2)
    savgol_file2_axi = savgol_filter(file2_axi, 1001, 2)
    savgol_file3_axi = savgol_filter(file3_axi, 1001, 2)
    savgol_file4_axi = savgol_filter(file4_axi, 1001, 2)

    savgol_file1_as = savgol_filter(file1_as, 1001, 2)
    savgol_file2_as = savgol_filter(file2_as, 1001, 2)
    savgol_file3_as = savgol_filter(file3_as, 1001, 2)
    savgol_file4_as = savgol_filter(file4_as, 1001, 2)

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs, savgol_file1_axi, label='$RR_G=1.5$', marker='o',
             markevery=100, markersize=8)
    ax1.plot(freqs, savgol_file2_axi, label='$RR_G=3.9$', marker='s',
             markevery=100, markersize=8)
    ax1.plot(freqs, savgol_file3_axi, label='$RR_G=6.3$', marker='^',
             markevery=100, markersize=8)
    ax1.plot(freqs, savgol_file4_axi, label='$RR_G=8.7$', marker='P',
             markevery=100, markersize=8)
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 110)
    ax1.set_xlabel('$f$ (Hz)')
    ax1.set_ylabel('$\omega$')
    ax1.legend()
    We_underscored = We.split('.') + '_' + We.split('.')[1]
    print(We_underscored)
    fig1filename = str(Re) + '_' + str(We_underscored) + '_' + 'axi.pgf'
    fig1.savefig(fname=fig1filename, bbox_inches='tight')


    fig2, ax2 = plt.subplots()
    ax2.plot(freqs, savgol_file1_as, label='$RR_G=1.5$', marker='o',
             markevery=100, markersize=8)
    ax2.plot(freqs, savgol_file2_as, label='$RR_G=3.9$', marker='s',
             markevery=100, markersize=8)
    ax2.plot(freqs, savgol_file3_as, label='$RR_G=6.3$', marker='^',
             markevery=100, markersize=8)
    ax2.plot(freqs, savgol_file4_as, label='$RR_G=8.7$', marker='P',
             markevery=100, markersize=8)
    ax2.set_xlim(0, 1200)
    ax2.set_ylim(0, 60)
    ax2.set_xlabel('$f$ (Hz)')
    ax2.set_ylabel('$\omega$')
    ax2.legend()


def plotting_1file(file1):
    # load the two files
    freqs, _, file1_axi, _, _, file1_as, _ = np.loadtxt(file1, delimiter=',',
                                                        unpack=True)

    file1_axi_savgol = savgol_filter(file1_axi, 11, 2)
    file1_axi_savgol_1001 = savgol_filter(file1_axi, 101, 2)

    fig, ax = plt.subplots()
    ax.plot(freqs, file1_axi, color='black')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 90)
    ax.set_xlabel('$f$ (Hz)')
    ax.set_ylabel('$\omega$')
    fig.set_size_inches(6,4)
    fig.savefig(fname='unfiltered_example.pgf', bbox_inches='tight')

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs, file1_axi_savgol, color='black')
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 90)
    ax1.set_xlabel('$f$ (Hz)')
    ax1.set_ylabel('$\omega$')
    fig1.set_size_inches(6,4)
    fig1.savefig(fname='savgol_11_example.pgf', bbox_inches='tight')

    fig2, ax2 = plt.subplots()

    ax2.plot(freqs, file1_axi_savgol_1001, color='black')
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 90)
    ax2.set_xlabel('$f$ (Hz)')
    ax2.set_ylabel('$\omega$')
    fig2.set_size_inches(6,4)
    fig2.savefig(fname='savgol_101_example.pgf', bbox_inches='tight')

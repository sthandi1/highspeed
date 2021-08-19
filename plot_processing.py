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
from highspeed_fft import (model_growth_rate, velocity_calculator,
                           weber_velocity, file_id)
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
    u_l = moro_re_calc(2940)
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
    sqrt_w = np.sqrt(w_squared)

    wavelength = 2*np.pi/k
    u_l = velocity_calculator(1551)
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
    ax.plot(wavenumber*a, savgol_control,
            label='Experimental (average velocity)')
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_xlabel('ka', fontsize=16)
    ax.set_ylabel('$\\omega$', fontsize=16)


def plotting_measured_wavelength(file1, file2, file3, file4):
    """
    Main plotting function
    """

    freqs, _, control, _, _, _, _ = np.loadtxt(file1, delimiter=',',
                                               unpack=True)
    freqs, _, control_2, _, _, _, _ = np.loadtxt(file2, delimiter=',',
                                                 unpack=True)
    freqs, _, control_3, _, _, _, _ = np.loadtxt(file3, delimiter=',',
                                                 unpack=True)
    freqs, _, control_4, _, _, _, _ = np.loadtxt(file4, delimiter=',',
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
    savgol_control_3 = savgol_filter(control_3, 101, 2)

    fig, ax = plt.subplots()
    ax.plot(k*a, sqrt_w, label='Rayleigh', color='black', linestyle='solid')
    ax.plot(wavenumber*a, savgol_control, label='Morozumi and Fukai model')
    ax.plot(wavenumber*a, savgol_control_3, label='Arai and Amagai model')
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_xlabel('ka', fontsize=16)
    ax.set_ylabel('$\\omega$ (1/s)', fontsize=16)
    ax.grid()
    fig.set_size_inches(6, 4)
    fig.savefig(fname='time_models.pgf', bbox_inches='tight')


def plotting_2file(file1, file2):
    # load the two files
    freqs, _, file1_axi, _, _, file1_as, _ = np.loadtxt(file1, delimiter=',',
                                                        unpack=True)
    freqs, _, file2_axi, _, _, file2_as, _ = np.loadtxt(file2, delimiter=',',
                                                        unpack=True)

    file1_axi_savgol = savgol_filter(file1_axi, 1001, 2)
    file2_axi_savgol = savgol_filter(file2_axi, 1001, 2)

    fig, ax = plt.subplots()
    ax.plot(freqs, file1_axi, label='file1')
    ax.plot(freqs, file2_axi, label='file2')
    ax.set_title('Standard data')
    ax.legend()
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 250)

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs, file1_axi_savgol, label='file1')
    ax1.plot(freqs, file2_axi_savgol, label='file2')
    ax1.set_title('Savgol')
    ax1.legend()
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 140)

    fig2, ax2 = plt.subplots()
    file1_as_savgol = savgol_filter(file1_as, 101, 2)
    file2_as_savgol = savgol_filter(file2_as, 101, 2)

    ax2.plot(freqs, file1_as_savgol, label='file1')
    ax2.plot(freqs, file2_as_savgol, label='file2')
    ax2.legend()
    ax2.set_xlim(0, 15000)
    ax2.set_ylim(0, 300)


def plotting_3file(file1, file2, file3):
    # load the two files
    freqs, _, file1_axi, _, _, _, _ = np.loadtxt(file1, delimiter=',',
                                                 unpack=True)
    freqs, _, file2_axi, _, _, _, _ = np.loadtxt(file2, delimiter=',',
                                                 unpack=True)
    freqs, _, file3_axi, _, _, _, _ = np.loadtxt(file3, delimiter=',',
                                                 unpack=True)

    file1_axi_savgol = savgol_filter(file1_axi, 1001, 2)
    file2_axi_savgol = savgol_filter(file2_axi, 1001, 2)
    file3_axi_savgol = savgol_filter(file3_axi, 1001, 2)

    fig, ax = plt.subplots()
    ax.plot(freqs, file1_axi_savgol, label='file1')
    ax.plot(freqs, file2_axi_savgol, label='file2')
    ax.plot(freqs, file3_axi_savgol, label='file3')
    ax.set_xlabel('$f$ (Hz)', fontsize=16)
    ax.set_ylabel('$\\omega$ (1/s)', fontsize=16)
    ax.grid()
    ax.tick_params(axis='both', labelsize=12)
    ax.legend()
    ax.set_xlim(0, 4000)
    ax.set_ylim(0, 140)


def plotting_4file(file1, file2, file3, file4):
    """
    Main plotting function
    """

    freqs1, _, file1_axi, _, _, file1_as, _ = np.loadtxt(file1, delimiter=',',
                                                         unpack=True)
    freqs2, _, file2_axi, _, _, file2_as, _ = np.loadtxt(file2, delimiter=',',
                                                         unpack=True)
    freqs3, _, file3_axi, _, _, file3_as, _ = np.loadtxt(file3, delimiter=',',
                                                         unpack=True)
    freqs4, _, file4_axi, _, _, file4_as, _ = np.loadtxt(file4, delimiter=',',
                                                         unpack=True)

    casename, Re, We = file_id(file1)

    fig, ax = plt.subplots()
    ax.plot(freqs1, file1_axi, label='$RR_G=1.5$')
    ax.plot(freqs2, file2_axi, label='$RR_G=3.9$')
    ax.plot(freqs3, file3_axi, label='$RR_G=6.3$')
    ax.plot(freqs4, file4_axi, label='$RR_G=8.7$')
    ax.set_xlim(0, 1250)
    ax.set_ylim(0, 400)
    ax.set_title('Unfiltered data')
    ax.set_xlabel('Frequencies (Hz)')
    ax.set_ylabel('Growth rate (1/s)')
    ax.legend()

    savgol_file1_axi = savgol_filter(file1_axi, 701, 2)
    savgol_file2_axi = savgol_filter(file2_axi, 701, 2)
    savgol_file3_axi = savgol_filter(file3_axi, 701, 2)
    savgol_file4_axi = savgol_filter(file4_axi, 701, 2)

    savgol_file1_as = savgol_filter(file1_as, 801, 2)
    savgol_file2_as = savgol_filter(file2_as, 801, 2)
    savgol_file3_as = savgol_filter(file3_as, 801, 2)
    savgol_file4_as = savgol_filter(file4_as, 801, 2)

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs1, savgol_file1_axi,
             label='$RR_G=1.5$')
    ax1.plot(freqs2, savgol_file2_axi,
             label='$RR_G=3.9$')
    ax1.plot(freqs3, savgol_file3_axi,
             label='$RR_G=6.3$')
    ax1.plot(freqs4, savgol_file4_axi,
             label='$RR_G=8.7$')
    ax1.set_xlim(0, 4000)
    ax1.set_ylim(0, 180)
    ax1.set_xlabel('$f$ (Hz)', fontsize=16)
    ax1.set_ylabel('$\\omega$ (1/s)', fontsize=16)
    ax1.grid()
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend()
    We_underscored = We.split('.')[0] + '_' + We.split('.')[1]
    fig1filename = str(Re) + '_' + str(We_underscored) + '_' + 'axi.pgf'
    print(fig1filename)
    fig1.set_size_inches(6, 4)
    fig1.savefig(fname=fig1filename, bbox_inches='tight')

    # ASYMMETRIC PLOTS

    fig2, ax2 = plt.subplots()
    ax2.plot(freqs1, savgol_file1_as, label='$RR_G=1.5$')
    ax2.plot(freqs2, savgol_file2_as, label='$RR_G=3.9$')
    ax2.plot(freqs3, savgol_file3_as, label='$RR_G=6.3$')
    ax2.plot(freqs4, savgol_file4_as, label='$RR_G=8.7$')
    ax2.set_xlim(0, 4000)
    ax2.set_ylim(0, 180)
    ax2.set_xlabel('$f$ (Hz)', fontsize=16)
    ax2.set_ylabel('$\\omega$ (1/s)', fontsize=16)
    ax2.grid()
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend()
    fig2filename = str(Re) + '_' + str(We_underscored) + '_' + 'as.pgf'
    print(fig2filename)
    fig2.set_size_inches(6, 4)
    fig2.savefig(fname=fig2filename, bbox_inches='tight')


def plotting_1file(file1):
    # load the file
    freqs, _, file1_axi, _, _, file1_as, _ = np.loadtxt(file1, delimiter=',',
                                                        unpack=True)

    file1_axi_savgol = savgol_filter(file1_axi, 11, 2)
    file1_axi_savgol_101 = savgol_filter(file1_axi, 101, 2)
    file1_axi_savgol_1001 = savgol_filter(file1_axi, 1001, 2)

    fig, ax = plt.subplots()
    ax.plot(freqs, file1_axi, color='black')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 90)
    ax.set_xlabel('$f$ (Hz)', fontsize=16)
    ax.set_ylabel('$\\omega$ (1/s)', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    fig.set_size_inches(6, 4)
    fig.savefig(fname='unfiltered_example.pgf', bbox_inches='tight')

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs, file1_axi_savgol, color='black')
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 90)
    ax1.set_xlabel('$f$ (Hz)', fontsize=16)
    ax1.set_ylabel('$\\omega$ (1/s)', fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    fig1.set_size_inches(6, 4)
    fig1.savefig(fname='savgol_11_example.pgf', bbox_inches='tight')

    fig2, ax2 = plt.subplots()

    ax2.plot(freqs, file1_axi_savgol_101, color='black')
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 90)
    ax2.set_xlabel('$f$ (Hz)', fontsize=16)
    ax2.set_ylabel('$\\omega$ (1/s)', fontsize=16)
    ax2.tick_params(axis='both', labelsize=12)
    fig2.set_size_inches(6, 4)
    fig2.savefig(fname='savgol_101_example.pgf', bbox_inches='tight')

    fig3, ax3 = plt.subplots()
    ax3.plot(freqs, file1_axi_savgol_1001, color='black')
    ax3.set_xlim(0, 1000)
    ax3.set_ylim(0, 90)
    ax3.set_xlabel('$f$ (Hz)', fontsize=16)
    ax3.set_ylabel('$\\omega$ (1/s)', fontsize=16)
    ax3.tick_params(axis='both', labelsize=12)
    fig3.set_size_inches(6, 4)
    fig3.savefig(fname='savgol_1001_example.pgf', bbox_inches='tight')

    print(len(freqs))
    print(len(file1_axi_savgol_1001))


def example_curve_fit(filename):
    (freqs, diameter_a0, diameter_growth_rates, diameter_errs, centroid_a0,
     centroid_growth_rates, centroid_errs) = np.loadtxt(filename,
                                                        delimiter=',',
                                                        unpack=True)
    modelling_ts = np.linspace(0, 0.02, 1000)
    modelling_amps = model_growth_rate(modelling_ts, diameter_a0[600],
                                       diameter_growth_rates[600])

    fig, ax = plt.subplots()
    ax.plot(modelling_ts, modelling_amps)
    ax.set_xlabel('$\\frac{a}{a}$')


def morozumi_comparison(morozumi_axi_5_22, morozumi_as_5_22,
                        morozumi_axi_22_9, morozumi_as_22_9,
                        morozumi_axi_52_7, morozumi_as_52_7,
                        experimental_file_5_22, experimental_file_22_9):
    # loading the 5.22 experimental file
    (freqs_5_22, _, exp_5_22_axi, _, _,
     exp_5_22_as, _) = np.loadtxt(experimental_file_5_22,
                                  delimiter=',', unpack=True)
    # loading the 22.9 experimental file
    (freqs_22_9, _, exp_22_9_axi, _, _,
     exp_22_9_as, _) = np.loadtxt(experimental_file_22_9,
                                  delimiter=',', unpack=True)
    # loading the morozumi axisymmetric 5.22 data file
    (morozumi_axi_5_22_freqs,
     moro_axi_5_22_growth_rates) = np.loadtxt(morozumi_axi_5_22,
                                              delimiter=',', unpack=True)
    # loading the morozumi asymmteric 5.22 data file
    (morozumi_as_5_22_freqs,
     moro_as_5_22_growth_rates) = np.loadtxt(morozumi_as_5_22,
                                             delimiter=',', unpack=True)
    # loading the morozumi axisymmetric 22.9 data file
    (morozumi_axi_22_9_freqs,
     moro_axi_22_9_growth_rates) = np.loadtxt(morozumi_axi_22_9,
                                              delimiter=',', unpack=True)
    # loading the morozumi asymmteric 22.9 data file
    (morozumi_as_22_9_freqs,
     moro_as_22_9_growth_rates) = np.loadtxt(morozumi_as_22_9,
                                             delimiter=',', unpack=True)
    # loading the morozumi axisymmetric 52.7 data file
    (morozumi_axi_52_7_freqs,
     moro_axi_52_7_growth_rates) = np.loadtxt(morozumi_axi_52_7,
                                              delimiter=',', unpack=True)
    # loading the morozumi asymmteric 52.7 data file
    (morozumi_as_52_7_freqs,
     moro_as_52_7_growth_rates) = np.loadtxt(morozumi_as_52_7,
                                             delimiter=',', unpack=True)
    # performing savgol filtering on both the axisymmetric and asymmetric 5.22
    # experimental files
    savgol_axi_5_22 = savgol_filter(exp_5_22_axi, 101, 2)
    savgol_as_5_22 = savgol_filter(exp_5_22_as, 101, 2)

    # performing savgol filtering on both the axisymmetric and asymmetric 22.9
    # experimental files
    savgol_axi_22_9 = savgol_filter(exp_22_9_axi, 101, 2)
    savgol_as_22_9 = savgol_filter(exp_22_9_as, 101, 2)

    # setting up the plots
    fig, ax = plt.subplots()
    ax.plot(freqs_5_22, savgol_axi_5_22,
            label='Experimental data $\\mathrm{We}_\\mathrm{g}=5.22$',
            color='black', linewidth=1)
    ax.plot(freqs_22_9, savgol_axi_22_9,
            label='Experimental data $\\mathrm{We}_\\mathrm{g}=22.9$',
            color='black', linestyle='dotted', linewidth=1)
    ax.plot(morozumi_axi_5_22_freqs, moro_axi_5_22_growth_rates,
            label='Morozumi and Fukai $\\mathrm{We}_\\mathrm{g}=5.22$',
            marker='o', markevery=10, markersize=7, color='black')
    ax.plot(morozumi_axi_22_9_freqs, moro_axi_22_9_growth_rates,
            label='Morozumi and Fukai $\\mathrm{We}_\\mathrm{g}=22.9$',
            marker='s', markevery=10, markersize=7, color='black')
    ax.plot(morozumi_axi_52_7_freqs, moro_axi_52_7_growth_rates,
            label='Morozumi and Fukai $\\mathrm{We}_\\mathrm{g}=52.7$',
            marker='^', markevery=10, markersize=7, color='black')
    ax.set_xlim(0, 1000)
    ax.set_ylim(-20, 220)
    ax.legend(fontsize=8)
    ax.set_xlabel('$f$ (Hz)', fontsize=14)
    ax.set_ylabel('$\\omega$ (1/s)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    fig.set_size_inches(6, 6)
    fig.savefig(fname='morozumi_axi_comparison.pgf', bbox_inches='tight')

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs_5_22, savgol_as_5_22,
             label='Experimental data $\\mathrm{We}_\\mathrm{g}=5.22$',
             color='black', linewidth=1)
    ax1.plot(freqs_22_9, savgol_as_22_9,
             label='Experimental data $\\mathrm{We}_\\mathrm{g}=22.9$',
             color='black', linestyle='dotted', linewidth=1)
    ax1.plot(morozumi_as_5_22_freqs, moro_as_5_22_growth_rates,
             label='Morozumi and Fukai $\\mathrm{We}_\\mathrm{g}=5.22$',
             marker='o', markevery=10, markersize=7, color='black')
    ax1.plot(morozumi_as_22_9_freqs, moro_as_22_9_growth_rates,
             label='Morozumi and Fukai $\\mathrm{We}_\\mathrm{g}=22.9$',
             marker='s', markevery=10, markersize=7, color='black')
    ax1.plot(morozumi_as_52_7_freqs, moro_as_52_7_growth_rates,
             label='Morozumi and Fukai $\\mathrm{We}_\\mathrm{g}=52.7$',
             marker='^', markevery=10, markersize=7, color='black')
    ax1.legend()
    ax1.set_xlim(0, 800)
    ax1.set_ylim(0, 200)
    ax1.legend(fontsize=8)
    ax1.set_xlabel('$f$ (Hz)', fontsize=16)
    ax1.set_ylabel('$\\omega$ (1/s)', fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    fig1.set_size_inches(6, 6)
    fig1.savefig(fname='morozumi_as_comparison.pgf', bbox_inches='tight')

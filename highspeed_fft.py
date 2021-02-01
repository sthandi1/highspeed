#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:08:38 2021

@author: samvirthandi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import time


def file_id(filename):
    dirs = filename.split('/')
    experiment = dirs[-1]
    casename = experiment.split('.')[0]
    params = experiment.split('_')
    recess = params[0]
    Re = params[2]
    We = params[3] + '.' + params[4].split('.')[0]
    print('Recess length is: ', recess)
    print('Reynolds number is: ', Re)
    print('Weber number is:', We)
    print('The casename is:', casename)
    return casename, Re


def model_growth_rate(t, a_0, omega):
    """This is the growth rate model

    Args:
        t (list): this corresponds to the z positions
        a_0 (float): initial disturbance
        omega (float): [description]

    Returns:
        float: amplitude
    """
    a = a_0 * np.exp(omega * t)
    return a


def param_extractor(ts, amps):
    """Works out a_0 and omega from the model

    Args:
        ts (array): array of the time values
        amps (array): amplitude values

    Returns:
        a_0: initial disturbance
        omega: growth rate
        omega_err: error in growth rate
    """
    # using scipy's curve fit model, p_cov is accuracy
    p, pcov = curve_fit(model_growth_rate, ts, amps)
    a_0, omega = p
    # calculates the standard deviation and returns an array
    # first value is the error of a_0, second value is error of omega
    perr = np.sqrt(np.diag(pcov))
    omega_err = perr[1]
    return a_0, omega, omega_err


def velocity_calculator(Re):
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
    d = 2/1000
    # reynolds number standard equation
    u = Re*mu/(rho*d)
    return u


def morozumi_time(u, z_locations):
    """


    Parameters
    ----------
    u : float
        velocity
    z_locations : array
        z locations in metres

    Returns
    -------
    t : array
        time values using the morozumi model

    """
    g = 9.81
    t = (-u+np.sqrt(u**2+2*g*z_locations))/g
    return t


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def fft_checking(filename):
    """This function will check the file and ensure reasonable data has
    been captured and produce fft graphs to be checked.

    Args:
        filename (str): file to be analysed
    """
    # print out experimental parameters and store the case name
    casename, Re = file_id(filename)

    # load the data into three numpy arrays
    frames, left_edges, right_edges = np.loadtxt(filename, delimiter=',',
                                                 unpack=True)
    # create a new numpy array which converts the frames into time by using
    # the frame rate
    frame_time = frames/27000
    # calculating the jet diameter (numpy array)
    jet_diameter = 0.02*(right_edges-left_edges)
    # calculating the jet centroid (numpy array)
    jet_centroid = 0.02*0.5*(right_edges+left_edges)

    # setting up plots for left and right edges
    fig, ax = plt.subplots()
    # plotting left edge
    ax.plot(frame_time, left_edges)
    # plotting right edge
    ax.plot(frame_time, right_edges, '.')
    ax.set_xlabel('frame_time (seconds)')
    ax.set_ylabel('Pixels')
    ax.set_title('Edges plot')

    # plotting jet diameter
    fig1, ax1 = plt.subplots()
    ax1.plot(frame_time, jet_diameter)
    ax1.set_xlabel('frame_time (seconds)')
    ax1.set_ylabel('Jet diameter (mm)')
    ax1.set_title('Jet diameter plot')

    # plotting jet centroid
    fig2, ax2 = plt.subplots()
    ax2.plot(frame_time, jet_centroid)
    ax2.set_xlabel('frame_time (seconds)')
    ax2.set_ylabel('Jet centroid location (mm)')
    ax2.set_title('Jet centroid plot')

    # fft of jet centroid (non shifted)
    centroid_fft = rfft(jet_centroid)
    centroid_freqs = rfftfreq(len(jet_centroid), 1/27000)

    fig3, ax3 = plt.subplots()
    ax3.stem(centroid_freqs, np.abs(centroid_fft))
    ax3.set_title("centroid fft")
    ax3.set_xlabel('Frequencies')
    ax3.set_ylabel('Amplitude')

    # fft of jet diameter (non shifted)
    loc1_diameter_fft = rfft(jet_diameter)
    loc1_diameter_freqs = rfftfreq(len(jet_diameter), 1/27000)

    fig4, ax4 = plt.subplots()
    ax4.stem(loc1_diameter_freqs, np.abs(loc1_diameter_fft))
    ax4.set_title('Jet diameter frequencies')
    ax4.set_xlabel('Frequencies')
    ax4.set_ylabel('Amplitude')

    # shifted FFTs
    # Shifted jet diameter plotting
    shifted_jet_diameter = jet_diameter - np.mean(jet_diameter)
    fig5, ax5 = plt.subplots()
    ax5.plot(frame_time, shifted_jet_diameter)
    ax5.set_title("Shifted jet diameter")
    ax5.set_xlabel('frame_time (seconds)')
    ax5.set_ylabel('Shifted jet diameter (mm)')

    # shifted jet diameter fft
    shifted_jet_diameter_fft = rfft(shifted_jet_diameter)
    shifted_jet_diameter_freqs = rfftfreq(len(shifted_jet_diameter), 1/27000)

    fig6, ax6 = plt.subplots()
    ax6.stem(shifted_jet_diameter_freqs, np.abs(shifted_jet_diameter_fft))
    ax6.set_title('Shifted jet diameter FFT')
    ax6.set_xlabel('Frequencies')
    ax6.set_ylabel('Amplitude')

    # shifted jet centroid plotting
    shifted_jet_centroid = jet_centroid - np.mean(jet_centroid)
    fig7, ax7 = plt.subplots()
    ax7.plot(frame_time, shifted_jet_centroid)
    ax7.set_title("Shifted jet centroid")
    ax7.set_xlabel('frame_time (seconds)')
    ax7.set_ylabel('Shifted jet centroid (mm)')

    # shifted jet centroid fft
    shifted_jet_centroid_fft = rfft(shifted_jet_centroid)
    shifted_jet_centroid_freqs = rfftfreq(len(shifted_jet_centroid), 1/27000)

    fig8, ax8 = plt.subplots()
    ax8.stem(shifted_jet_centroid_freqs, np.abs(shifted_jet_centroid_fft))
    ax8.set_title('Shifted jet centroid FFT')
    ax8.set_xlabel('Frequencies')
    ax8.set_ylabel('Amplitude')

    print(len(shifted_jet_centroid_freqs))
    print(len(shifted_jet_diameter))
    print(len(shifted_jet_centroid_fft))


def fft_output(filename):
    """This is a backend function which produces the fft data for the given
    filename

    Args:
        filename (str): csv file to be processed
    """
    # print out experimental parameters and store the case name
    file_id(filename)

    # load the data into three numpy arrays
    frames, left_edges, right_edges = np.loadtxt(filename, delimiter=',',
                                                 unpack=True)
    # calculating the jet diameter (numpy array) in mm
    jet_diameter = 0.02*(right_edges-left_edges)
    # calculating the jet centroid (numpy array) in mm
    jet_centroid = 0.02*0.5*(right_edges+left_edges)

    # Shifted jet diameter
    shifted_jet_diameter = jet_diameter - np.mean(jet_diameter)
    # shifted jet centroid
    shifted_jet_centroid = jet_centroid - np.mean(jet_centroid)

    # calculating frequencies
    freqs = rfftfreq(len(shifted_jet_centroid), 1/27000)

    # shifted jet diameter fft
    shifted_jet_diameter_fft = rfft(shifted_jet_diameter)

    # shifted jet centroid fft
    shifted_jet_centroid_fft = rfft(shifted_jet_centroid)

    # finding the modulus of the fft to allow the amplitude to be real
    abs_jet_diameter_fft = np.abs(shifted_jet_diameter_fft)
    abs_jet_centroid_fft = np.abs(shifted_jet_centroid_fft)

    # total time
    total_time = len(shifted_jet_centroid)/27000

    return total_time, freqs, abs_jet_diameter_fft, abs_jet_centroid_fft


def growth_rate(filenames):
    """Main growth rate calculator

    Args:
        filenames (str): 10 csv files to be analysed
    """
    # file ID

    print("storing casename and Reynolds number\n\n")
    casename, Re = file_id(filenames[0])

    print("\nNow calculating FFTs\n\n")
    # calculating ffts

    t, freqs, loc0_diameter_fft, loc0_centroid_fft = fft_output(filenames[0])
    t, freqs, loc1_diameter_fft, loc1_centroid_fft = fft_output(filenames[1])
    t, freqs, loc2_diameter_fft, loc2_centroid_fft = fft_output(filenames[2])
    t, freqs, loc3_diameter_fft, loc3_centroid_fft = fft_output(filenames[3])
    t, freqs, loc4_diameter_fft, loc4_centroid_fft = fft_output(filenames[4])
    t, freqs, loc5_diameter_fft, loc5_centroid_fft = fft_output(filenames[5])
    t, freqs, loc6_diameter_fft, loc6_centroid_fft = fft_output(filenames[6])
    t, freqs, loc7_diameter_fft, loc7_centroid_fft = fft_output(filenames[7])
    t, freqs, loc8_diameter_fft, loc8_centroid_fft = fft_output(filenames[8])
    t, freqs, loc9_diameter_fft, loc9_centroid_fft = fft_output(filenames[9])

    # real amplitudes from morozumi equation

    loc0_diameter_amp = np.sqrt((4/t)*loc0_diameter_fft)
    loc0_centroid_amp = np.sqrt((4/t)*loc0_centroid_fft)

    loc1_diameter_amp = np.sqrt((4/t)*loc1_diameter_fft)
    loc1_centroid_amp = np.sqrt((4/t)*loc1_centroid_fft)

    loc2_diameter_amp = np.sqrt((4/t)*loc2_diameter_fft)
    loc2_centroid_amp = np.sqrt((4/t)*loc2_centroid_fft)

    loc3_diameter_amp = np.sqrt((4/t)*loc3_diameter_fft)
    loc3_centroid_amp = np.sqrt((4/t)*loc3_centroid_fft)

    loc4_diameter_amp = np.sqrt((4/t)*loc4_diameter_fft)
    loc4_centroid_amp = np.sqrt((4/t)*loc4_centroid_fft)

    loc5_diameter_amp = np.sqrt((4/t)*loc5_diameter_fft)
    loc5_centroid_amp = np.sqrt((4/t)*loc5_centroid_fft)

    loc6_diameter_amp = np.sqrt((4/t)*loc6_diameter_fft)
    loc6_centroid_amp = np.sqrt((4/t)*loc6_centroid_fft)

    loc7_diameter_amp = np.sqrt((4/t)*loc7_diameter_fft)
    loc7_centroid_amp = np.sqrt((4/t)*loc7_centroid_fft)

    loc8_diameter_amp = np.sqrt((4/t)*loc8_diameter_fft)
    loc8_centroid_amp = np.sqrt((4/t)*loc8_centroid_fft)

    loc9_diameter_amp = np.sqrt((4/t)*loc9_diameter_fft)
    loc9_centroid_amp = np.sqrt((4/t)*loc9_centroid_fft)

    # setting up storage array for the z_locations
    z_locations = np.zeros(10)

    # using filenames to ID z locations
    for i in range(len(filenames)):
        # separate into the paramaters
        underscore_split = filenames[i].split('_')
        # identify the last parameter, split by the . and then take the first
        # value as this will be the z_location
        z_loc = underscore_split[-1].split('.')[0]
        z_locations[i] = int(z_loc)

    # calculating velocity
    u = velocity_calculator(int(Re))

    # converting z_locations into real distances
    zs_metres = 0.02*z_locations/1000

    # time model can be changed as needed
    z_times = morozumi_time(u, zs_metres)

    # initialising storage arrays for growth rates
    diameter_growth_rates = np.zeros((len(loc0_diameter_amp)))
    diameter_a0 = np.zeros((len(loc0_diameter_amp)))
    diameter_errs = np.zeros((len(loc0_diameter_amp)))

    centroid_growth_rates = np.zeros((len(loc0_centroid_amp)))
    centroid_a0 = np.zeros((len(loc0_centroid_amp)))
    centroid_errs = np.zeros((len(loc0_centroid_amp)))

    # performing loop to work out growth rates of diameter from curve fitting
    # various z locations (z times)

    print("\n\nNow calculating the diameter growth rates:\n\n")
    # i is an indexer for the length of the array, equal to the frame number
    for i in range(len(loc0_diameter_amp)):
        # progress calculator
        if (i % 1000) == 0:
            print("Progress: {:.1f}%".format(i*100/len(loc0_diameter_amp)))
        # assign a local array which takes the diameter amp at the current
        # index across the 10 z locations
        local_amps = np.array((loc0_diameter_amp[i], loc1_diameter_amp[i],
                              loc2_diameter_amp[i], loc3_diameter_amp[i],
                              loc4_diameter_amp[i], loc5_diameter_amp[i],
                              loc6_diameter_amp[i], loc7_diameter_amp[i],
                              loc8_diameter_amp[i], loc9_diameter_amp[i]))
        # work out the local a_0, growth rate, and error in curve fit
        # using the curve fit function defined earlier
        loc_a_0, loc_omega, loc_err = param_extractor(z_times, local_amps)
        # assign local variables to global array
        diameter_a0[i] = loc_a_0
        diameter_growth_rates[i] = loc_omega
        diameter_errs[i] = loc_err

    print('diameter growth rate calculation complete')

    # stack the arrays together to make a single array for file output
    diameter_output = np.stack((freqs, diameter_growth_rates, diameter_a0,
                                diameter_errs))

    print("\n\nNow calculating the centroid growth rates:\n\n")
    for i in range(len(loc0_centroid_amp)):
        # progress calculator
        if (i % 1000) == 0:
            print("Progress: {:.1f}%".format(i*100/len(loc0_centroid_amp)))
        # assign a local array which takes the centroid amp at the current
        # index across the 10 z locations
        local_amps = np.array((loc0_centroid_amp[i], loc1_centroid_amp[i],
                              loc2_centroid_amp[i], loc3_centroid_amp[i],
                              loc4_centroid_amp[i], loc5_centroid_amp[i],
                              loc6_centroid_amp[i], loc7_centroid_amp[i],
                              loc8_centroid_amp[i], loc9_centroid_amp[i]))
        # work out the local a_0, growth rate, and error in curve fit
        # using the curve fit function defined earlier
        loc_a_0, loc_omega, loc_err = param_extractor(z_times, local_amps)
        # assign local variables to global array
        centroid_a0[i] = loc_a_0
        centroid_growth_rates[i] = loc_omega
        centroid_errs[i] = loc_err

    # POST PROCESSING TEXTING, NOT FOR DEPLOYMENT

    figfft, axfft = plt.subplots()
    axfft.stem(freqs, loc1_diameter_fft)
    axfft.set_title("FFT data for loc1")
    axfft.set_xlabel("Frequencies")
    axfft.set_ylabel("Amplitude")

    fig, ax = plt.subplots()
    ax.plot(freqs, diameter_growth_rates, '.', color='yellow')
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 90)
    ax.set_title("Growth rates vs frequencies")
    ax.set_xlabel("Frequencies")
    ax.set_ylabel("Growth rates")

    print("minimum error is:", diameter_errs.min())

    minimum_location = diameter_errs.argmin()
    print(minimum_location)
    print("minimum error frequency:", freqs[minimum_location])

    # 1253 is the location of 290.04 Hz
    amps = [loc0_diameter_amp[600], loc1_diameter_amp[600],
            loc2_diameter_amp[600], loc3_diameter_amp[600],
            loc4_diameter_amp[600], loc5_diameter_amp[600],
            loc6_diameter_amp[600], loc7_diameter_amp[600],
            loc8_diameter_amp[600], loc9_diameter_amp[600]]

    fig1, ax1 = plt.subplots()
    ax1.plot(z_times, amps, 'o')

    modelling_ts = np.linspace(0, 0.02, 1000)
    modelling_amps = model_growth_rate(modelling_ts, diameter_a0[600],
                                       diameter_growth_rates[600])

    ax1.plot(modelling_ts, modelling_amps)
    ax1.set_xlabel("z time (seconds)")
    ax1.set_ylabel("amplitude (m)?")
    ax1.set_title("An example plot")

    fig2, ax2 = plt.subplots()
    ax2.plot(freqs, diameter_errs, '.')
    ax2.set_xlim(0, 1000)
    ax2.set_title('Errors')
    ax2.set_xlabel("Frequencies")
    ax2.set_ylabel("Standard deviation of curve fit")

    freqs_1000 = freqs[4315]

    avg_err_1000 = diameter_errs[0:4315].mean()

    fig3, ax3 = plt.subplots()
    ax3.plot(freqs, loc0_diameter_amp)

    print(freqs[600])

    mov_avg = movingaverage(diameter_growth_rates, 1000)

    fig4, ax4 = plt.subplots()
    ax4.plot(freqs, mov_avg)
    ax4.set_xlim(0, 5000)
    ax4.set_title('Moving average')
    ax4.set_xlabel('Frequencies')
    ax4.set_ylabel('Growth rate')

    ax.plot(freqs, mov_avg, label='moving average', color='blue')

    w = savgol_filter(diameter_growth_rates, 1001, 2)
    fig5, ax5 = plt.subplots()
    ax5.plot(freqs, w)
    ax5.set_title('Savitzky-Golay filter')
    ax5.set_xlim(0, 5000)
    ax5.set_xlabel('Frequencies')
    ax5.set_ylabel('Growth rate')

    ax.plot(freqs, w, label='Savitzky-Golay', color='red')
    ax.legend()

    fig6, ax6 = plt.subplots()
    ax6.plot(freqs, w, label='Savitzky-Golay')
    ax6.plot(freqs, mov_avg, label='Moving average')
    ax6.legend()
    ax6.set_xlim(0, 710)
    ax6.set_ylim(0, 70)
    ax6.set_title("Moving average vs Savitzky-Golay")
    ax6.set_xlabel("Frequency")
    ax6.set_ylabel("Growth rate")

    zero_crossings_mov_avg = np.where(np.diff(np.signbit(mov_avg)))[0]
    zero_crossings_w = np.where(np.diff(np.signbit(w)))[0]

    print("Zeros avg:", freqs[zero_crossings_mov_avg])
    print("Zeros savgol", freqs[zero_crossings_w])

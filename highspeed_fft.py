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
import csv
from statistics import mean


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
    time = frames/27000
    # calculating the jet diameter (numpy array)
    jet_diameter = 0.02*(right_edges-left_edges)
    # calculating the jet centroid (numpy array)
    jet_centroid = 0.02*0.5*(right_edges+left_edges)

    # setting up plots for left and right edges
    fig, ax = plt.subplots()
    # plotting left edge
    ax.plot(time, left_edges)
    # plotting right edge
    ax.plot(time, right_edges, '.')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Pixels')
    ax.set_title('Edges plot')

    # plotting jet diameter
    fig1, ax1 = plt.subplots()
    ax1.plot(time, jet_diameter)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Jet diameter (mm)')
    ax1.set_title('Jet diameter plot')

    # plotting jet centroid
    fig2, ax2 = plt.subplots()
    ax2.plot(time, jet_centroid)
    ax2.set_xlabel('Time (seconds)')
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
    ax5.plot(time, shifted_jet_diameter)
    ax5.set_title("Shifted jet diameter")
    ax5.set_xlabel('Time (seconds)')
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
    ax7.plot(time, shifted_jet_centroid)
    ax7.set_title("Shifted jet centroid")
    ax7.set_xlabel('Time (seconds)')
    ax7.set_ylabel('Shifted jet centroid (mm)')

    # shifted jet centroid fft
    shifted_jet_centroid_fft = rfft(shifted_jet_centroid)
    shifted_jet_centroid_freqs = rfftfreq(len(shifted_jet_centroid), 1/27000)

    fig8, ax8 = plt.subplots()
    ax8.stem(shifted_jet_centroid_freqs, np.abs(shifted_jet_centroid_fft))
    ax8.set_title('Shifted jet centroid FFT')
    ax8.set_xlabel('Frequencies')
    ax8.set_ylabel('Amplitude')

    # storing fft data
    # jet diameter results
    abs_jet_diameter_fft = np.abs(shifted_jet_diameter_fft)
    jet_diameter_results = np.stack([shifted_jet_diameter_freqs,
                                     abs_jet_diameter_fft])
    # jet centroid results
    abs_jet_centroid_fft = np.abs(shifted_jet_centroid_fft)
    jet_centroid_results = np.stack([shifted_jet_centroid_freqs,
                                     abs_jet_centroid_fft])

    print(len(shifted_jet_centroid_freqs))
    print(len(shifted_jet_diameter))
    print(len(shifted_jet_centroid_fft))


def fft_output(filename):
    """This is a backend function which produces the fft data for the given filename

    Args:
        filename (str): csv file to be processed
    """
    # print out experimental parameters and store the case name
    file_id(filename)

    # load the data into three numpy arrays
    frames, left_edges, right_edges = np.loadtxt(filename, delimiter=',',
                                                 unpack=True)
    # calculating the jet diameter (numpy array)
    jet_diameter = 0.02*(right_edges-left_edges)
    # calculating the jet centroid (numpy array)
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
    time = len(shifted_jet_centroid)/27000

    return time, freqs, abs_jet_diameter_fft, abs_jet_centroid_fft


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

    amps = [loc0_diameter_amp[700], loc1_diameter_amp[700], loc2_diameter_amp[700],
            loc3_diameter_amp[700], loc4_diameter_amp[700], loc5_diameter_amp[700],
            loc6_diameter_amp[700], loc7_diameter_amp[700], loc8_diameter_amp[700],
            loc9_diameter_amp[700]]

    ind = np.arange(0, 10, 1)

    fig, ax = plt.subplots()
    ax.plot(ind, amps)

    fig1, ax1 = plt.subplots()
    ax1.plot(freqs, loc1_diameter_fft)

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
    t = morozumi_time(u, zs_metres)
    
    

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
        pcov: convergence
    """
    p, pcov = curve_fit(model_growth_rate, ts, amps)
    a_0, omega = p
    print("Fitted values a_0={}, omega={}".format(a_0, omega))
    print("Accuracy=",pcov)
    return a_0, omega, pcov


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
    mu = 8.9e-4
    rho = 1000
    d = 2/1000
    u = Re*mu/(rho*d)
    return u


def morozumi_time(u, z_locations):
    """
    

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    z_locations : TYPE
        DESCRIPTION.

    Returns
    -------
    t : TYPE
        DESCRIPTION.

    """
    g = 9.81
    t = (-u+np.sqrt(u**2+2*g*z_locations))/g
    return t
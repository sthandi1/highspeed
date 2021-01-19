#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:08:38 2021

@author: samvirthandi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def synthetic():
    t = np.linspace(0, 2*np.pi, 100)
    a0 = 100
    w = 55.6
    f = 5
    a = a0*np.exp(w*t)
    upper = a*np.sin(f*t)+2
    lower = -a*np.sin(f*t)-2
    fig, ax = plt.subplots()
    ax.plot(t, upper)
    ax.plot(t, lower)


def fft_test():
    f = 10  # Frequency, in cycles per second, or Hertz
    f_s = 100  # Sampling rate, or number of measurements per second
    t = np.linspace(0, 2, 2 * f_s, endpoint=False)
    x = np.sin(f * 2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(t, x)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal amplitude');
    
    transform = fft(x)
    freqs = fftfreq(len(x)) * f_s
    
    fig1, ax1 = plt.subplots()

    ax1.stem(freqs, np.abs(transform))
    ax1.set_xlabel('Frequency in Hertz [Hz]')
    ax1.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax1.set_xlim(-f_s / 2, f_s / 2)
    ax1.set_ylim(-5, 110)


def fft_test_2():
    x = np.linspace(0, 2*np.pi, 1000)
    f = 10
    y = np.sin(f*x)
    
    fourier = fft(y)
    freqs = fftfreq(len(y), 1/len(y))
    fig, ax = plt.subplots()
    ax.plot(x,y)
    fig1, ax1 = plt.subplots()
    ax1.stem(freqs, fourier)
    ax1.set_xlim(-20,20)

def fft_testing(filename):
    """
    Plotting the csv results from the previous function    

    Returns
    -------
    None.

    """
    frames = []
    left_edges = []
    right_edges = []

    # open csv file loc1
    with open(filename) as csvfile:
        # read csv file
        readCSV = csv.reader(csvfile, delimiter=',')
        # iterate along rows
        for row in readCSV:
            # assign row values to a local operator
            # frame number is first column
            frame = int(row[0])
            # left edge is second column
            left_edge = int(row[1])
            # right edge is third column
            right_edge = int(row[2])
            # append local frame to global list
            frames.append(frame)
            # append local left edge to global list
            left_edges.append(left_edge)
            # append local right edge to global list
            right_edges.append(right_edge)

            
    # converting frames into real time
    
    time = []
    
    for frame in frames:
        time.append(frames[frame]/27000)
        
    # converting pixels to mm edges
    
    mm_left_edge = []
    mm_right_edge = []
 
    
    for edge in left_edges:
        mm_left_edge.append(left_edges[edge]*0.02)
    
    for edge in right_edges:
        mm_right_edge.append(right_edges[edge]*0.02)
                        
    # jet diameter
    jet_diameter = []
    
    for i in frames:
        jet_diameter.append(0.02*(right_edges[i]-left_edges[i]))
    
    # setting up plots for left and right edges
    fig, ax = plt.subplots()
    # plotting left edge
    ax.plot(time, left_edges)
    # plotting right edge
    ax.plot(time, right_edges, '.')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Pixels')

    # plotting jet diameter
    fig1, ax1 = plt.subplots()
    ax1.plot(time, jet_diameter)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Jet diameter (mm)')

    sample_rate = len(loc1_time)/loc1_time[-1]
    print('Sample rate:',sample_rate)
    print('Final value:',loc1_time[-1])
    loc1_diameter_fft = fft(loc1_jet_diameter*10000)
    loc1_diameter_freqs = fftfreq(len(loc1_jet_diameter), 27000)
    fig4, ax4 = plt.subplots()
    ax4.stem(loc1_diameter_freqs, np.abs(loc1_diameter_fft))
    
    fig5, ax5 = plt.subplots()
    ax5.plot(loc1_time[:100], loc1_left_edges[:100], linestyle='none', marker='.')
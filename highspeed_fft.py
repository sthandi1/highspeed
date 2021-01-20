#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:08:38 2021

@author: samvirthandi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import csv
from statistics import mean


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
    
    # jet centroid
    jet_centroid = []
    for i in frames:
        jet_centroid.append(0.02*0.5*(right_edges[i]+left_edges[i]))    
    
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

    # fft of jet centroid
    centroid_fft = rfft(jet_centroid)
    centroid_freqs = rfftfreq(len(jet_centroid), 1/27000)
    fig3, ax3 = plt.subplots()
    ax3.stem(centroid_freqs, np.abs(centroid_fft))
    ax3.set_title("centroid fft")
    ax3.set_xlabel('Frequencies')
    ax3.set_ylabel('Amplitude')

    # fft of jet diameter
    loc1_diameter_fft = rfft(jet_diameter)
    loc1_diameter_freqs = rfftfreq(len(jet_diameter), 1/27000)
    fig4, ax4 = plt.subplots()
    ax4.stem(loc1_diameter_freqs, np.abs(loc1_diameter_fft))
    ax4.set_xlim(2, 500)
    ax4.set_ylim(0, 150)
    ax4.set_title('Jet diameter change frequencies')
    ax4.set_xlabel('Frequencies')
    ax4.set_ylabel('Amplitude')

    # trying normalisation for jet diameter
    normalised_jet_diameter = []
    jet_diameter_mean = mean(jet_diameter)
    for diameter in jet_diameter:
        normalised_jet_diameter.append(diameter - jet_diameter_mean)
    fig5, ax5 = plt.subplots()
    ax5.plot(time, normalised_jet_diameter)
    ax5.set_title("Normalised jet diameter")
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Normalised jet diameter (mm)')

    # normalisation fft for jet diameter
    
    normalised_jet_diameter_fft = rfft(normalised_jet_diameter)
    normalised_jet_diameter_freqs = rfftfreq(len(normalised_jet_diameter), 1/27000)
    fig6, ax6 = plt.subplots()
    ax6.stem(normalised_jet_diameter_freqs, np.abs(normalised_jet_diameter_fft))
    ax6.set_title('Normalised jet diameter FFT')
    ax6.set_xlabel('Frequencies')
    ax6.set_ylabel('Amplitude')

    # trying normalisation for jet centroid
    normalised_jet_centroid = []
    jet_centroid_mean = mean(jet_centroid)
    for centroid in jet_centroid:
        normalised_jet_centroid.append(centroid - jet_centroid_mean)
    fig7, ax7 = plt.subplots()
    ax7.plot(time, normalised_jet_centroid)
    ax7.set_title("Normalised jet centroid")
    ax7.set_xlabel('Time (seconds)')
    ax7.set_ylabel('Normalised jet centroid (mm)')

    # normalisation fft for jet centroid
    
    normalised_jet_centroid_fft = rfft(normalised_jet_centroid)
    normalised_jet_centroid_freqs = rfftfreq(len(normalised_jet_centroid), 1/27000)
    fig8, ax8 = plt.subplots()
    ax8.stem(normalised_jet_centroid_freqs, np.abs(normalised_jet_centroid_fft))
    ax8.set_title('Normalised jet centroid FFT')
    ax8.set_xlabel('Frequencies')
    ax8.set_ylabel('Amplitude')
    ax8.set_xlim(0, 250)

    # identifying peak frequencies
    # jet diameter

    for i in range(len(np.abs(normalised_jet_diameter_fft))):
        if np.abs(normalised_jet_diameter_fft)[i] > 100:
            print("Amplitudes:",np.abs(normalised_jet_diameter_fft)[i])
            print("Frequencies:", normalised_jet_centroid_freqs[i])



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
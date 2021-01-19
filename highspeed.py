# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:38:34 2020

@author: st8g14
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mraw_v2 import mraw
import csv
from scipy.fft import fft, fftfreq


def nothing(x):
    # this function is used by the trackbars
    print(x)


def single_image_thresh_value(inputFile, thresh):
    """
    This function is used to determine the threshold value to be used in the
    processing

    Returns
    -------
    None.

    """
    # make a new window called thresholded image
    cv2.namedWindow('threshold trackbar')
    # make a trackbar which will govern the threshold 
    # first argument is the name of the trackbar
    # second argument is the window name that the trackbar should be in
    # third argument is the starting value
    # fourth argument is the maximum value
    # fifth argument is the callback function
    cv2.createTrackbar('threshold', 'threshold trackbar', 0, 4096, nothing)

    movie = mraw(inputFile)
    image = movie[0]
    # while loop allows image to be dynamically updated
    while True:
        # assign the threshold value to a parameter
        thresh = cv2.getTrackbarPos('threshold', 'threshold trackbar')
        # apply binary threshold where below threshold is zero and above is max
        _, th1 = cv2.threshold(image, thresh, 4096, cv2.THRESH_BINARY)
        # show thresholded image
        cv2.imshow('threshold trackbar', th1)
        # how long to wait before closing window
        key = cv2.waitKey(1)
        if key == 27:
            break
        # stops crashes and destroys windows
    cv2.destroyAllWindows()



def single_image_thresh_data(inputFile, thresh):
    """
    This function gets the data for a single image and plots it on a graph
    it compares the normal image and thresholded image

    Returns
    -------
    None.

    """

    movie = mraw(inputFile)
    width = movie.width
    image = movie[0]

    # apply binary threshold where below threshold is zero and above is max
    _, th1 = cv2.threshold(image, thresh, 4096, cv2.THRESH_BINARY)

    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax.imshow(image, cmap=plt.cm.gray)
    ax1.imshow(th1, cmap=plt.cm.gray)
    ax2.imshow(th1, cmap=plt.cm.gray)

    # LEFT EDGES
    for pixel in range(width):
        # for loop goes through the threshold array at the given
        # downstream 'y' position (loc1) and then cycles across in the 'x'
        # direction until it finds a zero. This zero indicate the edge of
        # the jet.
        if th1[loc1, pixel] == 0:
            # assign the iterator to a variable
            loc1_left_edge = pixel
            # exit the for loop once edge has been found
            break

    for pixel in range(width):
        # This for loop finds the left edge at loc2
        if th1[loc2, pixel] == 0:
            # assign the pixel to a variable
            loc2_left_edge = pixel
            # exit the for loop once edge has been found
            break

    for pixel in range(width):
        # This for loop finds the left edge at loc3
        if th1[loc3, pixel] == 0:
            # assign the pixel to a variable
            loc3_left_edge = pixel
            # exit the for loop once edge has been found
            break

    # RIGHT EDGES
    for pixel in range(width):
        # invert the loop so it counts down instead of up
        inv = width - pixel - 1
        # This for loop finds the rihgt edge at loc1 and assings it to a
        # variable
        if th1[loc1, inv] == 0:
            # assigning the pixel to a variable
            loc1_right_edge = inv
            # exit the for loop once edge has been found
            break
    
    for pixel in range(width):
        # invert the loop so it counts down instead of up
        inv = width - pixel - 1
        # This for loop finds the right edge at loc2 and assigns it to a
        # variable
        if th1[loc2, inv] == 0:
            # assigning the pixel to a variable
            loc2_right_edge = inv
        # exit the for loop once edge has been found
            break

    for pixel in range(width):
        # invert the loop so it counts down instead of up
        inv = width - pixel - 1
        # This for loop finds the right edge at loc3 and assigns it to a
        # variable
        if th1[loc3, inv] == 0:
            # assigning the pixel to a variable
            loc3_right_edge = inv
            # exit the for loop once edge has been found
            break
    
    # plot the left edges point on the thresholded image
    ax1.plot(loc1_left_edge, loc1, linestyle='none', marker='x', markersize=12)
    ax1.plot(loc2_left_edge, loc2, linestyle='none', marker='x', markersize=12)
    ax1.plot(loc3_left_edge, loc3, linestyle='none', marker='x', markersize=12)
    # plt the right edges point on the thresholded image
    ax1.plot(loc1_right_edge, loc1, linestyle='none', marker='x',
             markersize=12)
    ax1.plot(loc2_right_edge, loc2, linestyle='none', marker='x',
             markersize=12)
    ax1.plot(loc3_right_edge, loc3, linestyle='none', marker='x',
             markersize=12)
    fig.set_size_inches(4, 8)
    fig1.set_size_inches(4, 8)
    fig2.set_size_inches(4, 8)
    fig.savefig(fname='normal.png', format='png')
    fig1.savefig(fname='thresholded_edges.png', format='png')
    fig2.savefig(fname='thresholded.png', format='png')

def multi_image(y_loc, inputFile, thresh):
    """
    This is the main code. It runs through all the photos in the file to
    determine edge locations and then it saves it in a csv file to be 
    processed later

    Returns
    -------
    None.

    """
    # load all images into a class called movie from Ivo's mraw code
    movie = mraw(inputFile)
    # defines the width of the image
    width = movie.width
    # defines the numnber of images in the movie
    frames = movie.image_count
    print("Number of images in movie: {:d}".format(frames))
    # set up a storage arrays which has the same number of rows as the number of
    # images in the movie (frames) and 3 columns. First column is the frame
    # number, second column is the left edge, third column is the right edge

    edges = np.zeros((frames, 3))

    # Now looping over individual frames in the file
    for frame in range(frames):
        # print every 100 iterations so progress is monitored
        if (frame % 1000) == 0:
            print("Progress: {:.1f}%".format(frame*100/frames))
        # load the image from the movie. Index the movie based on the frame
        # iterator
        image = movie[frame]
        # define the frame number in the storage array first column. The frame
        # is the y location in the storage array and also the value. Frame is
        # the iterator too. e.g. edges[frame, 0] = frame will give the first
        # column value and then the row number is the frame number. This
        # is given the value of the frame. Therefore the array of zeros
        # becomes an array where the first column counts up from 0 to
        # the number of frames in the video.
        edges[frame, 0] = frame
       
        # apply binary threshold where below threshold is zero and above is
        # 4096 which is 2^12 since the image is a 12 bit image
        _, th1 = cv2.threshold(image, thresh, 4096, cv2.THRESH_BINARY)

        # LEFT EDGES
        for pixel in range(width):
            # for loop goes through the threshold array at the given
            # downstream 'y' position (y_loc) and then cycles across in the 'x'
            # direction until it finds a zero. This zero indicate the edge of
            # the jet.

            # This for loop finds the left edge at y_loc and assigns it to the
            # second column, which is indexed as 1
            if th1[y_loc, pixel] == 0:
                edges[frame, 1] = pixel
                # exit the for loop once edge has been found
                break
    
        # RIGHT EDGES
        for pixel in range(width):
            # invert the loop so it counts down instead of up
            inv = width - pixel - 1
            # This for loop finds the right edge at y_loc and assigns it to the
            # third column, which is indexed as 2
            if th1[y_loc, inv] == 0:
                # assigning the pixel to the storage array for the given frame
                edges[frame, 2] = inv
                # exit the for loop once edge has been found
                break
    


    filename = 'edges_results_' + str(y_loc) + '.csv'

    # save the edges storage matrix into a text file
    np.savetxt(filename, edges, fmt='%d', delimiter=',')


def plotting():
    """
    Plotting the csv results from the previous function    

    Returns
    -------
    None.

    """

    loc1_csv = 'edges_results_200.csv'
    loc2_csv = 'edges_results_512.csv'
    loc3_csv = 'edges_results_840.csv'

    # setting up storage arrays
    loc1_frames = []
    loc1_left_edges = []
    loc1_right_edges = []

    loc2_frames = []
    loc2_left_edges = []
    loc2_right_edges = []

    loc3_frames = []
    loc3_left_edges = []
    loc3_right_edges = []

    # open csv file loc1
    with open(loc1_csv) as csvfile_loc1:
        # read csv file
        readCSV_loc1 = csv.reader(csvfile_loc1, delimiter=',')
        # iterate along rows
        for row in readCSV_loc1:
            # assign row values to a local operator
            # frame number is first column
            frame = int(row[0])
            # left edge is second column
            left_edge = int(row[1])
            # right edge is third column
            right_edge = int(row[2])
            # append local frame to global list
            loc1_frames.append(frame)
            # append local left edge to global list
            loc1_left_edges.append(left_edge)
            # append local right edge to global list
            loc1_right_edges.append(right_edge)

    # open csv file loc2
    with open(loc2_csv) as csvfile_loc2:
        # read csv file
        readCSV_loc2 = csv.reader(csvfile_loc2, delimiter=',')
        # iterate along rows
        for row in readCSV_loc2:
            # assign row values to a local operator
            # frame number is first column
            frame = int(row[0])
            # left edge is second column
            left_edge = int(row[1])
            # right edge is third column
            right_edge = int(row[2])
            # append local frame to global list
            loc2_frames.append(frame)
            # append local left edge to global list
            loc2_left_edges.append(left_edge)
            # append local right edge to global list
            loc2_right_edges.append(right_edge)
   
    # open csv file loc3
    with open(loc3_csv) as csvfile_loc3:
        # read csv file
        readCSV_loc3 = csv.reader(csvfile_loc3, delimiter=',')
        # iterate along rows
        for row in readCSV_loc3:
            # assign row values to a local operator
            # frame number is first column
            frame = int(row[0])
            # left edge is second column
            left_edge = int(row[1])
            # right edge is third column
            right_edge = int(row[2])
            # append local frame to global list
            loc3_frames.append(frame)
            # append local left edge to global list
            loc3_left_edges.append(left_edge)
            # append local right edge to global list
            loc3_right_edges.append(right_edge)
            
    # converting frames into real time
    
    loc1_time = []
    loc2_time = []
    loc3_time = []
    
    for frame in loc1_frames:
        loc1_time.append(loc1_frames[frame]/27000)

    for frame in loc2_frames:
        loc2_time.append(loc2_frames[frame]/27000)

    for frame in loc3_frames:
        loc3_time.append(loc3_frames[frame]/27000)
        
    # converting pixels to mm edges
    
    loc1_mm_left_edge = []
    loc2_mm_left_edge = []
    loc3_mm_left_edge = []
    
    loc1_mm_right_edge = []
    loc2_mm_right_edge = []
    loc3_mm_right_edge = []
    
    for edge in loc1_left_edges:
        loc1_mm_left_edge.append(loc1_left_edges[edge]*0.02)
    
    for edge in loc2_left_edges:
        loc2_mm_left_edge.append(loc2_left_edges[edge]*0.02)

    for edge in loc3_left_edges:
        loc3_mm_left_edge.append(loc3_left_edges[edge]*0.02)
        
    for edge in loc1_right_edges:
        loc1_mm_right_edge.append(loc1_right_edges[edge]*0.02)
        
    for edge in loc2_right_edges:
        loc2_mm_right_edge.append(loc2_right_edges[edge]*0.02)

    for edge in loc3_right_edges:
        loc3_mm_right_edge.append(loc3_right_edges[edge]*0.02)
                        
    # jet diameter
    loc1_jet_diameter = []
    loc2_jet_diameter = []
    loc3_jet_diameter = []
    
    for i in loc1_frames:
        loc1_jet_diameter.append(0.02*(loc1_right_edges[i]-loc1_left_edges[i]))
    
    for i in loc2_frames:
        loc2_jet_diameter.append(0.02*(loc2_right_edges[i]-loc2_left_edges[i]))

    for i in loc3_frames:
        loc3_jet_diameter.append(0.02*(loc3_right_edges[i]-loc3_left_edges[i]))
    
    # setting up plots for left and right edges
    fig, ax = plt.subplots()
    # plotting left edge
    ax.plot(loc1_time, loc1_left_edges)
    # plotting right edge
    ax.plot(loc1_time, loc1_right_edges, '.')
    ax.set_title('4mm downstream left and right edges')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Pixels')


    fig1, ax1 = plt.subplots()
    ax1.plot(loc1_time, loc1_jet_diameter)
    ax1.set_title('Jet diameter 4mm downstream')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Jet diameter (mm)')
    
    fig2, ax2 = plt.subplots()
    ax2.plot(loc2_time, loc2_jet_diameter)
    ax2.set_title('Jet diameter 13mm downstream')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Jet diameter (mm)')
    
    fig3, ax3 = plt.subplots()
    ax3.plot(loc2_time, loc2_left_edges)
    ax3.plot(loc2_time, loc2_right_edges)
    ax3.set_title('13mm downstream left and right edges')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Pixels')
    
    sample_rate = len(loc1_time)/loc1_time[-1]
    print('Sample rate:',sample_rate)
    print('Final value:',loc1_time[-1])
    loc1_diameter_fft = fft(loc1_jet_diameter*10000)
    loc1_diameter_freqs = fftfreq(len(loc1_jet_diameter), 27000)
    fig4, ax4 = plt.subplots()
    ax4.stem(loc1_diameter_freqs, np.abs(loc1_diameter_fft))
    
    fig5, ax5 = plt.subplots()
    ax5.plot(loc1_time[:100], loc1_left_edges[:100], linestyle='none', marker='.')

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
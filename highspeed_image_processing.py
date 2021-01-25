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


def change(x):
    # this function is used by the trackbars
    pass


def single_image_thresh_value(inputFile):
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
    cv2.createTrackbar('threshold', 'threshold trackbar', 0, 4096, change)

    movie = mraw(inputFile)
    image = movie[0]
    # while loop allows image to be dynamically updated
    while(True):
        # assign the threshold value to a parameter
        thresh = cv2.getTrackbarPos('threshold', 'threshold trackbar')
        # apply binary threshold where below threshold is zero and above is max
        _, th1 = cv2.threshold(image, thresh, 4096, cv2.THRESH_BINARY)
        # show thresholded image
        cv2.imshow('thresh', th1)
        # how long to wait before closing window
        if cv2.waitKey(1) == 27:
            break
        # stops crashes and destroys windows
    cv2.destroyAllWindows()




def single_image_thresh_data(inputFile, thresh, z_location):
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
        # downstream 'z' position (z_location) and then cycles across in the
        # 'x' direction until it finds a zero. This zero indicate the edge of
        # the jet.
        if th1[z_location, pixel] == 0:
            # assign the iterator to a variable
            left_edge = pixel
            # exit the for loop once edge has been found
            break

    # RIGHT EDGES
    for pixel in range(width):
        # invert the loop so it counts down instead of up
        inv = width - pixel - 1
        # This for loop finds the rihgt edge at z_location and assings it to a
        # variable
        if th1[z_location, inv] == 0:
            # assigning the pixel to a variable
            right_edge = inv
            # exit the for loop once edge has been found
            break

    # plot the left edges point on the thresholded image
    ax1.plot(left_edge, z_location, linestyle='none', marker='x',
             markersize=12)
    # plt the right edges point on the thresholded image
    ax1.plot(right_edge, z_location, linestyle='none', marker='x',
             markersize=12)
    fig.set_size_inches(4, 8)
    fig1.set_size_inches(4, 8)
    fig2.set_size_inches(4, 8)
    fig.savefig(fname='normal.png', format='png')
    fig1.savefig(fname='thresholded_edges.png', format='png')
    fig2.savefig(fname='thresholded.png', format='png')


def multi_image(z_locations, inputFile, thresh):
    """
    This is the main code. It runs through all the photos in the file to
    determine edge locations and then it saves it in a csv file to be
    processed later. It takes a list or numpy array of z locations

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
    # set up a storage arrays which has the same number of rows as the number
    # of images in the movie (frames) and 3 columns. First column is the frame
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
        # is the z location in the storage array and also the value. Frame is
        # the iterator too. e.g. edges[frame, 0] = frame will give the first
        # column value and then the row number is the frame number. This
        # is given the value of the frame. Therefore the array of zeros
        # becomes an array where the first column counts up from 0 to
        # the number of frames in the video.
        edges[frame, 0] = frame

        # apply binary threshold where below threshold is zero and above is
        # 4096 which is 2^12 since the image is a 12 bit image
        _, th1 = cv2.threshold(image, thresh, 4096, cv2.THRESH_BINARY)

        edges_zloc0 = edge_detector(edges, z_loc[0], frame, th1, width)
        edges_zloc1 = edge_detector(edges, z_loc[0], frame, th1, width)
        edges_zloc2 = edge_detector(edges, z_loc[0], frame, th1, width)
        edges_zloc3 = edge_detector(edges, z_loc[0], frame, th1, width)
        edges_zloc4 = edge_detector(edges, z_loc[0], frame, th1, width)
        edges_zloc5 = edge_detector(edges, z_loc[0], frame, th1, width)
        edges_zloc6 = edge_detector(edges, z_loc[0], frame, th1, width)
        edges_zloc7 = edge_detector(edges, z_loc[0], frame, th1, width)
        edges_zloc8 = edge_detector(edges, z_loc[0], frame, th1, width)
        edges_zloc9 = edge_detector(edges, z_loc[0], frame, th1, width)

    for z_loc in z_locations:
        output_filename = 'edges_results_' + str(z_loc) + '.csv'
        # save the edges storage matrix into a text file
        np.savetxt(output_filename, edges, fmt='%d', delimiter=',')


def edge_detector(edges, z_loc, frame, th1, width):
    for pixel in range(width):
        # for loop goes through the threshold array at the given
        # downstream 'z' position (z_loc) and then cycles across in the 'x'
        # direction until it finds a zero. This zero indicate the edge of
        # the jet.

        # This for loop finds the left edge at z_loc and assigns it to the
        # second column, which is indexed as 1
        if th1[z_loc, pixel] == 0:
            edges[frame, 1] = pixel
            # exit the for loop once edge has been found
            break

    # RIGHT EDGES
    for pixel in range(width):
        # invert the loop so it counts down instead of up
        inv = width - pixel - 1
        # This for loop finds the right edge at z_loc and assigns it to the
        # third column, which is indexed as 2
        if th1[z_loc, inv] == 0:
            # assigning the pixel to the storage array for the given frame
            edges[frame, 2] = inv
            # exit the for loop once edge has been found
            break
    return edges

single_image_thresh_value("/Volumes/My Passport/Experiments/2020-02 High speed camera experiments/0_cowl/Re_1551/We_5_22/0_cowl_1551_522.cihx")
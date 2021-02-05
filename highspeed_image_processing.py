# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:38:34 2020

@author: st8g14
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mraw_v2 import mraw


def change(x):
    """
    Test function for trackbars

    Parameters
    ----------
    x : int
        input

    Returns
    -------
    None.

    """
    # this function is used by the trackbars
    print(x)


def single_image_thresh_value(inputFile):
    """
    This function is used to determine the threshold value to be used in the
    processing. Not very stable currently, do not use

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

    Parameters
    ----------
    inputFile : str
        file to be tested
    thresh : int
        Threshold value
    z_location : int
        location to perform edge detection

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

    Parameters
    ----------
    z_locations : list or array
        z locations to detect edges at, length must be 10
    inputFile : str
        file to process
    thresh : int
        threshold value

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

    file_id(inputFile)

    edges_zloc0 = np.zeros((frames, 3))
    edges_zloc1 = np.zeros((frames, 3))
    edges_zloc2 = np.zeros((frames, 3))
    edges_zloc3 = np.zeros((frames, 3))
    edges_zloc4 = np.zeros((frames, 3))
    edges_zloc5 = np.zeros((frames, 3))
    edges_zloc6 = np.zeros((frames, 3))
    edges_zloc7 = np.zeros((frames, 3))
    edges_zloc8 = np.zeros((frames, 3))
    edges_zloc9 = np.zeros((frames, 3))

    # Now looping over individual frames in the file
    for frame in range(frames):
        # print every 100 iterations so progress is monitored
        if (frame % 1000) == 0:
            print("Progress: {:.1f}%".format(frame*100/frames))
            print(frame)
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
        edges_zloc0[frame, 0] = frame
        edges_zloc1[frame, 0] = frame
        edges_zloc2[frame, 0] = frame
        edges_zloc3[frame, 0] = frame
        edges_zloc4[frame, 0] = frame
        edges_zloc5[frame, 0] = frame
        edges_zloc6[frame, 0] = frame
        edges_zloc7[frame, 0] = frame
        edges_zloc8[frame, 0] = frame
        edges_zloc9[frame, 0] = frame

        # apply binary threshold where below threshold is zero and above is
        # 4096 which is 2^12 since the image is a 12 bit image
        _, th1 = cv2.threshold(image, thresh, 4096, cv2.THRESH_BINARY)

        #######################################################################
        """
        .##........#######...######.......#####..
        .##.......##.....##.##....##.....##...##.
        .##.......##.....##.##..........##.....##
        .##.......##.....##.##..........##.....##
        .##.......##.....##.##..........##.....##
        .##.......##.....##.##....##.....##...##.
        .########..#######...######.......#####..
        """
        #######################################################################
        # LEFT EDGES
        for pixel in range(width):
            # for loop goes through the threshold array at the given
            # downstream 'z' position (z_loc) and then cycles across in the 'x'
            # direction until it finds a zero. This zero indicate the edge of
            # the jet.

            # This for loop finds the left edge at z_loc and assigns it to the
            # second column, which is indexed as 1
            if th1[z_locations[0], pixel] == 0:
                edges_zloc0[frame, 1] = pixel
                # exit the for loop once edge has been found
                break

        # RIGHT EDGES
        for pixel in range(width):
            # invert the loop so it counts down instead of up
            inv = width - pixel - 1
            # This for loop finds the right edge at z_loc and assigns it to the
            # third column, which is indexed as 2
            if th1[z_locations[0], inv] == 0:
                # assigning the pixel to the storage array for the given frame
                edges_zloc0[frame, 2] = inv
                # exit the for loop once edge has been found
                break

        #######################################################################
        """
        .##........#######...######........##..
        .##.......##.....##.##....##.....####..
        .##.......##.....##.##.............##..
        .##.......##.....##.##.............##..
        .##.......##.....##.##.............##..
        .##.......##.....##.##....##.......##..
        .########..#######...######......######
        """
        #######################################################################
        # LEFT EDGES
        for pixel in range(width):
            # for loop goes through the threshold array at the given
            # downstream 'z' position (z_loc) and then cycles across in the 'x'
            # direction until it finds a zero. This zero indicate the edge of
            # the jet.

            # This for loop finds the left edge at z_loc and assigns it to the
            # second column, which is indexed as 1
            if th1[z_locations[1], pixel] == 0:
                edges_zloc1[frame, 1] = pixel
                # exit the for loop once edge has been found
                break

        # RIGHT EDGES
        for pixel in range(width):
            # invert the loop so it counts down instead of up
            inv = width - pixel - 1
            # This for loop finds the right edge at z_loc and assigns it to the
            # third column, which is indexed as 2
            if th1[z_locations[1], inv] == 0:
                # assigning the pixel to the storage array for the given frame
                edges_zloc1[frame, 2] = inv
                # exit the for loop once edge has been found
                break

        #######################################################################
        """
        .##........#######...######......#######.
        .##.......##.....##.##....##....##.....##
        .##.......##.....##.##.................##
        .##.......##.....##.##...........#######.
        .##.......##.....##.##..........##.......
        .##.......##.....##.##....##....##.......
        .########..#######...######.....#########
        """
        #######################################################################
        # LEFT EDGES
        for pixel in range(width):
            # for loop goes through the threshold array at the given
            # downstream 'z' position (z_loc) and then cycles across in the 'x'
            # direction until it finds a zero. This zero indicate the edge of
            # the jet.

            # This for loop finds the left edge at z_loc and assigns it to the
            # second column, which is indexed as 1
            if th1[z_locations[2], pixel] == 0:
                edges_zloc2[frame, 1] = pixel
                # exit the for loop once edge has been found
                break

        # RIGHT EDGES
        for pixel in range(width):
            # invert the loop so it counts down instead of up
            inv = width - pixel - 1
            # This for loop finds the right edge at z_loc and assigns it to the
            # third column, which is indexed as 2
            if th1[z_locations[2], inv] == 0:
                # assigning the pixel to the storage array for the given frame
                edges_zloc2[frame, 2] = inv
                # exit the for loop once edge has been found
                break
        #######################################################################
        """
        .##........#######...######......#######.
        .##.......##.....##.##....##....##.....##
        .##.......##.....##.##.................##
        .##.......##.....##.##...........#######.
        .##.......##.....##.##.................##
        .##.......##.....##.##....##....##.....##
        .########..#######...######......#######.
        """
        #######################################################################
        # LEFT EDGES
        for pixel in range(width):
            # for loop goes through the threshold array at the given
            # downstream 'z' position (z_loc) and then cycles across in the 'x'
            # direction until it finds a zero. This zero indicate the edge of
            # the jet.

            # This for loop finds the left edge at z_loc and assigns it to the
            # second column, which is indexed as 1
            if th1[z_locations[3], pixel] == 0:
                edges_zloc3[frame, 1] = pixel
                # exit the for loop once edge has been found
                break

        # RIGHT EDGES
        for pixel in range(width):
            # invert the loop so it counts down instead of up
            inv = width - pixel - 1
            # This for loop finds the right edge at z_loc and assigns it to the
            # third column, which is indexed as 2
            if th1[z_locations[3], inv] == 0:
                # assigning the pixel to the storage array for the given frame
                edges_zloc3[frame, 2] = inv
                # exit the for loop once edge has been found
                break

        #######################################################################
        """
        .##........#######...######.....##.......
        .##.......##.....##.##....##....##....##.
        .##.......##.....##.##..........##....##.
        .##.......##.....##.##..........##....##.
        .##.......##.....##.##..........#########
        .##.......##.....##.##....##..........##.
        .########..#######...######...........##.
        """
        #######################################################################
        # LEFT EDGES
        for pixel in range(width):
            # for loop goes through the threshold array at the given
            # downstream 'z' position (z_loc) and then cycles across in the 'x'
            # direction until it finds a zero. This zero indicate the edge of
            # the jet.

            # This for loop finds the left edge at z_loc and assigns it to the
            # second column, which is indexed as 1
            if th1[z_locations[4], pixel] == 0:
                edges_zloc4[frame, 1] = pixel
                # exit the for loop once edge has been found
                break

        # RIGHT EDGES
        for pixel in range(width):
            # invert the loop so it counts down instead of up
            inv = width - pixel - 1
            # This for loop finds the right edge at z_loc and assigns it to the
            # third column, which is indexed as 2
            if th1[z_locations[4], inv] == 0:
                # assigning the pixel to the storage array for the given frame
                edges_zloc4[frame, 2] = inv
                # exit the for loop once edge has been found
                break

        #######################################################################
        """
        .##........#######...######.....########
        .##.......##.....##.##....##....##......
        .##.......##.....##.##..........##......
        .##.......##.....##.##..........#######.
        .##.......##.....##.##................##
        .##.......##.....##.##....##....##....##
        .########..#######...######......######.
        """
        #######################################################################
        # LEFT EDGES
        for pixel in range(width):
            # for loop goes through the threshold array at the given
            # downstream 'z' position (z_loc) and then cycles across in the 'x'
            # direction until it finds a zero. This zero indicate the edge of
            # the jet.

            # This for loop finds the left edge at z_loc and assigns it to the
            # second column, which is indexed as 1
            if th1[z_locations[5], pixel] == 0:
                edges_zloc5[frame, 1] = pixel
                # exit the for loop once edge has been found
                break

        # RIGHT EDGES
        for pixel in range(width):
            # invert the loop so it counts down instead of up
            inv = width - pixel - 1
            # This for loop finds the right edge at z_loc and assigns it to the
            # third column, which is indexed as 2
            if th1[z_locations[5], inv] == 0:
                # assigning the pixel to the storage array for the given frame
                edges_zloc5[frame, 2] = inv
                # exit the for loop once edge has been found
                break
        #######################################################################
        """
        .##........#######...######......#######.
        .##.......##.....##.##....##....##.....##
        .##.......##.....##.##..........##.......
        .##.......##.....##.##..........########.
        .##.......##.....##.##..........##.....##
        .##.......##.....##.##....##....##.....##
        .########..#######...######......#######.
        """
        #######################################################################
        # LEFT EDGES
        for pixel in range(width):
            # for loop goes through the threshold array at the given
            # downstream 'z' position (z_loc) and then cycles across in the 'x'
            # direction until it finds a zero. This zero indicate the edge of
            # the jet.

            # This for loop finds the left edge at z_loc and assigns it to the
            # second column, which is indexed as 1
            if th1[z_locations[6], pixel] == 0:
                edges_zloc6[frame, 1] = pixel
                # exit the for loop once edge has been found
                break

        # RIGHT EDGES
        for pixel in range(width):
            # invert the loop so it counts down instead of up
            inv = width - pixel - 1
            # This for loop finds the right edge at z_loc and assigns it to the
            # third column, which is indexed as 2
            if th1[z_locations[6], inv] == 0:
                # assigning the pixel to the storage array for the given frame
                edges_zloc6[frame, 2] = inv
                # exit the for loop once edge has been found
                break

        #######################################################################
        """
        .##........#######...######.....########
        .##.......##.....##.##....##....##....##
        .##.......##.....##.##..............##..
        .##.......##.....##.##.............##...
        .##.......##.....##.##............##....
        .##.......##.....##.##....##......##....
        .########..#######...######.......##....
        """
        #######################################################################
        # LEFT EDGES
        for pixel in range(width):
            # for loop goes through the threshold array at the given
            # downstream 'z' position (z_loc) and then cycles across in the 'x'
            # direction until it finds a zero. This zero indicate the edge of
            # the jet.

            # This for loop finds the left edge at z_loc and assigns it to the
            # second column, which is indexed as 1
            if th1[z_locations[7], pixel] == 0:
                edges_zloc7[frame, 1] = pixel
                # exit the for loop once edge has been found
                break

        # RIGHT EDGES
        for pixel in range(width):
            # invert the loop so it counts down instead of up
            inv = width - pixel - 1
            # This for loop finds the right edge at z_loc and assigns it to the
            # third column, which is indexed as 2
            if th1[z_locations[7], inv] == 0:
                # assigning the pixel to the storage array for the given frame
                edges_zloc7[frame, 2] = inv
                # exit the for loop once edge has been found
                break

        #######################################################################
        """
        .##........#######...######......#######.
        .##.......##.....##.##....##....##.....##
        .##.......##.....##.##..........##.....##
        .##.......##.....##.##...........#######.
        .##.......##.....##.##..........##.....##
        .##.......##.....##.##....##....##.....##
        .########..#######...######......#######.
        """
        #######################################################################
        # LEFT EDGES
        for pixel in range(width):
            # for loop goes through the threshold array at the given
            # downstream 'z' position (z_loc) and then cycles across in the 'x'
            # direction until it finds a zero. This zero indicate the edge of
            # the jet.

            # This for loop finds the left edge at z_loc and assigns it to the
            # second column, which is indexed as 1
            if th1[z_locations[8], pixel] == 0:
                edges_zloc8[frame, 1] = pixel
                # exit the for loop once edge has been found
                break

        # RIGHT EDGES
        for pixel in range(width):
            # invert the loop so it counts down instead of up
            inv = width - pixel - 1
            # This for loop finds the right edge at z_loc and assigns it to the
            # third column, which is indexed as 2
            if th1[z_locations[8], inv] == 0:
                # assigning the pixel to the storage array for the given frame
                edges_zloc8[frame, 2] = inv
                # exit the for loop once edge has been found
                break
        #######################################################################
        """
        .##........#######...######......#######.
        .##.......##.....##.##....##....##.....##
        .##.......##.....##.##..........##.....##
        .##.......##.....##.##...........########
        .##.......##.....##.##.................##
        .##.......##.....##.##....##....##.....##
        .########..#######...######......#######.
        """
        #######################################################################
        # LEFT EDGES
        for pixel in range(width):
            # for loop goes through the threshold array at the given
            # downstream 'z' position (z_loc) and then cycles across in the 'x'
            # direction until it finds a zero. This zero indicate the edge of
            # the jet.

            # This for loop finds the left edge at z_loc and assigns it to the
            # second column, which is indexed as 1
            if th1[z_locations[9], pixel] == 0:
                edges_zloc9[frame, 1] = pixel
                # exit the for loop once edge has been found
                break

        # RIGHT EDGES
        for pixel in range(width):
            # invert the loop so it counts down instead of up
            inv = width - pixel - 1
            # This for loop finds the right edge at z_loc and assigns it to the
            # third column, which is indexed as 2
            if th1[z_locations[9], inv] == 0:
                # assigning the pixel to the storage array for the given frame
                edges_zloc9[frame, 2] = inv
                # exit the for loop once edge has been found
                break

    dirs = inputFile.split('/')
    experiment = dirs[-1]
    casename = experiment.split('.')[0]

    output_filename = []
    for z_loc in z_locations:
        output = str(casename) + '_' + 'results_' + str(z_loc) + '.csv'
        output_filename.append(output)

    print('Saving data...')
    # save the edges storage matrix into a text file
    np.savetxt(output_filename[0], edges_zloc0, fmt='%d', delimiter=',')
    np.savetxt(output_filename[1], edges_zloc1, fmt='%d', delimiter=',')
    np.savetxt(output_filename[2], edges_zloc2, fmt='%d', delimiter=',')
    np.savetxt(output_filename[3], edges_zloc3, fmt='%d', delimiter=',')
    np.savetxt(output_filename[4], edges_zloc4, fmt='%d', delimiter=',')
    np.savetxt(output_filename[5], edges_zloc5, fmt='%d', delimiter=',')
    np.savetxt(output_filename[6], edges_zloc6, fmt='%d', delimiter=',')
    np.savetxt(output_filename[7], edges_zloc7, fmt='%d', delimiter=',')
    np.savetxt(output_filename[8], edges_zloc8, fmt='%d', delimiter=',')
    np.savetxt(output_filename[9], edges_zloc9, fmt='%d', delimiter=',')
    print('data saved successfully')



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

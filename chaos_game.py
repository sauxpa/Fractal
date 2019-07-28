#!/usr/bin/env python
# coding: utf-8

# importing necessary modules 
import numpy as np
from PIL import Image
from random import random, randint

def midpoint(p, q, w=0.5):
    """Find the weighted midpoint of two cartesian points.
    """
    return tuple(w*c1 + (1-w)*c2 for c1, c2 in zip(p, q))

def play_game(N, w=0.5, corners=[(0, 0), (0.5, np.sqrt(3)/2), (1, 0)]):
    """Play turn of the chaos game.
    w : weight to use when computing the average between last random point and the guider point
    corners : determines the guider points
    """
    x = np.zeros(N)
    y = np.zeros(N)

    x[0] = random()
    y[0] = random()
    
    n_corners = len(corners)
    
    for i in range(1, N):
        k = randint(0, n_corners-1) # random triangle vertex
        x[i], y[i] = midpoint(corners[k], (x[i-1], y[i-1]), w)
    return x, y

def render_image(x, y, size):
    x_im = np.floor(x*size).astype(int)
    y_im = np.floor(y*size).astype(int)
    square = np.empty([size, size, 3], dtype = np.uint8) 
    color = np.array([255, 255, 255], dtype = np.uint8) 
    square.fill(0) 

    for x_coord, y_coord in zip(x_im, y_im):
        # transpose and flip to render the image with the correct orientation
        square[size-y_coord-1, x_coord, :] = color

    im = Image.fromarray(square)
    return im
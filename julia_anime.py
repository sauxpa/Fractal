#!/usr/bin/env python
# coding: utf-8

# need to plot in a separate window to see the animation
#get_ipython().run_line_magic('matplotlib', 'qt')

import argparse, sys

from julia import julia_set
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
plt.style.use('dark_background')

# matplotlib.use('TkAgg')  # or 'GTK3Cairo'
# matplotlib.use('GTK3Cairo')  # or 'GTK3Cairo'

# Get arguments
parser=argparse.ArgumentParser()
parser.add_argument('--xmin', default=-2.0)
parser.add_argument('--xmax', default=2.0)
parser.add_argument('--ymin', default=-2.0)
parser.add_argument('--ymax', default=2.0)
parser.add_argument('--width', default=1920)
parser.add_argument('--height', default=1080)
parser.add_argument('--frames', default=5, type=int)
parser.add_argument('--maxiter', default=1024, type=int)
parser.add_argument('--interval', default=10, help='In milliseconds', type=int)
parser.add_argument('--method', default='parallel')
parser.add_argument('--c0', default=0.7885, help='Initial c in Julia iteration z -> z**2+c', type=float)
parser.add_argument('--filename', '-f', default='', help='File to save the animation')
parser.add_argument('--fps', default=15, help='Frames per second, works only for mp4')
parser.add_argument('--verbose', '-v', default=False)

# Set arguments
args = parser.parse_args()
xmin = args.xmin
xmax = args.xmax
ymin = args.ymin
ymax = args.ymax
width = args.width
height = args.height
frames = args.frames
maxiter = args.maxiter
interval = args.interval
method = args.method
c0 = args.c0
filename = args.filename
fps = args.fps
verbose = args.verbose

julia = julia_set(xmin, xmax, ymin, ymax, width, height, maxiter, c=c0, method=method)

fig = plt.figure()
im = plt.imshow(julia, interpolation='none', animated=True)

def init():
    julia = julia_set(xmin, xmax, ymin, ymax, width, height, maxiter, c=c0, method=method)
    im.set_array(julia)
    return [im]

def animate(i):
    theta = 2*np.pi*i/frames
    c = c0*np.exp(theta*1j)
    if verbose:
        print('{}/{}'.format(i, frames))
    julia = julia_set(xmin, xmax, ymin, ymax, width, height, maxiter, c=c, method=method)
    im.set_array(julia)
    return [im]

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True, repeat=True)
plt.axis('off')

if len(filename):
    if '.mp4' in filename:
        writer = animation.FFMpegWriter(fps=fps, bitrate=-1)
        anim.save(filename, writer=writer)
    else:
        anim.save(filename, writer='imagemagick')
else:
    plt.show(fig)

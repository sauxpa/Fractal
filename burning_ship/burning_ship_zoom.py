#!/usr/bin/env python
# coding: utf-8

import argparse, sys
from burning_ship import burning_ship_set
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
plt.style.use('dark_background')

# Get arguments
parser=argparse.ArgumentParser()
parser.add_argument('--xmin', default=-2.25, type=float)
parser.add_argument('--xmax', default=1.25, type=float)
parser.add_argument('--ymin', default=-2.25, type=float)
parser.add_argument('--ymax', default=2.25, type=float)
parser.add_argument('--xtarget', default=0.0, help='x coordinate of the attractor', type=float)
parser.add_argument('--ytarget', default=0.0, help='y coordinate of the attractor', type=float)
parser.add_argument('--width', default=1920)
parser.add_argument('--height', default=1080)
parser.add_argument('--frames', default=20, type=int)
parser.add_argument('--zoom', '-z', default=0.9, type=float, help='Zoom factor at each iteration')
parser.add_argument('--maxiter', default=1024, type=int)
parser.add_argument('--interval', default=100, help='In milliseconds', type=int)
parser.add_argument('--method', default='parallel')
parser.add_argument('--filename', '-f', default='', help='File to save the animation')
parser.add_argument('--fps', default=15, help='Frames per second, works only for mp4')
parser.add_argument('--verbose', '-v', default=False)
parser.add_argument('--colormap', default='nipy_spectral', type=str)

# Set arguments
args = parser.parse_args()
xmin = args.xmin
xmax = args.xmax
ymin = args.ymin
ymax = args.ymax
xtarget = args.xtarget
ytarget = -args.ytarget # need to take the opposite to flip it correctly when rendering
width = args.width
height = args.height
frames = args.frames
zoom = args.zoom
maxiter = args.maxiter
interval = args.interval
method = args.method
filename = args.filename
fps = args.fps
verbose = args.verbose
colormap = args.colormap

fig = plt.figure()

burning_ship = burning_ship_set(xmin, xmax, ymin, ymax, width, height, maxiter, method=method)
im = plt.imshow(burning_ship, interpolation='none', animated=True)

def init():
    burning_ship = burning_ship_set(xmin, xmax, ymin, ymax, width, height, maxiter, method=method)
    im.set_data(burning_ship)
    return [im]

def animate(i):
    xmini = (xmin-xtarget)*zoom**i+xtarget
    xmaxi = (xmax-xtarget)*zoom**i+xtarget
    ymini = (ymin-ytarget)*zoom**i+ytarget
    ymaxi = (ymax-ytarget)*zoom**i+ytarget
    burning_ship = burning_ship_set(xmini, xmaxi, ymini, ymaxi, width, height, maxiter, method=method)
    im.set_array(getattr(cm, colormap)(burning_ship))
    return [im]

plt.axis('off')
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames,
                                interval=interval, blit=True, repeat=True)

if len(filename):
    if '.mp4' in filename:
        writer = animation.FFMpegWriter(fps=fps, bitrate=-1)
        anim.save(filename, writer=writer)
    else:
        anim.save(filename, writer='imagemagick')
else:
    plt.show(fig)

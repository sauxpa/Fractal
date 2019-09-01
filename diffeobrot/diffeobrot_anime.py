#!/usr/bin/env python
# coding: utf-8

import argparse, sys
from diffeobrot import diffeobrot_set
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
plt.style.use('dark_background')

# Get arguments
parser=argparse.ArgumentParser()
parser.add_argument('--xmin', default=-2.0, type=float)
parser.add_argument('--xmax', default=0.5, type=float)
parser.add_argument('--ymin', default=-1.5, type=float)
parser.add_argument('--ymax', default=1.5, type=float)
parser.add_argument('--width', default=512, type=int)
parser.add_argument('--height', default=256, type=int)
parser.add_argument('--t0min', default=0.05, help='min range for t0', type=float)
parser.add_argument('--t0max', default=10.0, help='max range for t0', type=float)
parser.add_argument('--dt', default=0.1, help='step size in the discretization scheme', type=float)
parser.add_argument('--frames', default=20, type=int)
parser.add_argument('--maxiter', default=128, type=int)
parser.add_argument('--interval', default=100, help='In milliseconds', type=int)
parser.add_argument('--method', default='parallel')
parser.add_argument('--filename', '-f', default='', help='File to save the animation')
parser.add_argument('--fps', default=15, help='Frames per second, works only for mp4')
parser.add_argument('--verbose', '-v', default=False)
parser.add_argument('--colormap', default='gist_earth', type=str)

# Set arguments
args = parser.parse_args()
xmin = args.xmin
ymin = args.ymin
xmax = args.xmax
ymax = args.ymax
width = args.width
height = args.height
t0min = args.t0min
t0max = args.t0max
dt = args.dt
frames = args.frames
maxiter = args.maxiter
interval = args.interval
method = args.method
filename = args.filename
fps = args.fps
verbose = args.verbose
colormap = args.colormap

fig = plt.figure()

diffeobrot = diffeobrot_set(xmin, xmax, ymin, ymax, width, height, t0max, dt, maxiter, method=method)
im = plt.imshow(diffeobrot, interpolation='none', animated=True)

t0s = np.linspace(t0min, t0max, frames)

def init():
    diffeobrot = diffeobrot_set(xmin, xmax, ymin, ymax, width, height, t0max, dt, maxiter, method=method)
    im.set_data(diffeobrot)
    return [im]

def animate(i):
    t0 = t0s[-i+1]
    diffeobrot = diffeobrot_set(xmin, xmax, ymin, ymax, width, height, t0, dt, maxiter, method=method)
    im.set_array(getattr(cm, colormap)(diffeobrot))
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

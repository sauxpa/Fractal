#!/usr/bin/env python
# coding: utf-8

# need to plot in a separate window to see the animation
#get_ipython().run_line_magic('matplotlib', 'qt')

import argparse, sys

from julia_quaternion import julia_set
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
parser.add_argument('--xmax', default=2.0, type=float)
parser.add_argument('--res', default=100, type=int)
parser.add_argument('--frames', default=100, type=int)
parser.add_argument('--maxiter', default=128, type=int)
parser.add_argument('--interval', default=100, help='In milliseconds', type=int)
parser.add_argument('--ca', default=0.285, help='Real part of c where Julia iteration z -> z**2+c', type=float)
parser.add_argument('--cb', default=0.01, help='First imaginary part of c where Julia iteration z -> z**2+c', type=float)
parser.add_argument('--cc', default=0.0, help='Second imaginary part of c where Julia iteration z -> z**2+c', type=float)
parser.add_argument('--cd', default=0.0, help='Third imaginary part of c where Julia iteration z -> z**2+c', type=float)
parser.add_argument('--axis', default=0, help='On the three remaining dimensions, slice along this axis', type=int)
parser.add_argument('--sliceidx', default=50, help='Where to slice the fourth dimension', type=int)
parser.add_argument('--slicedim', default=3, help='Along which dimension to slice', type=int)
parser.add_argument('--filename', '-f', default='', help='File to save the animation')
parser.add_argument('--fps', default=15, help='Frames per second, works only for mp4')
parser.add_argument('--verbose', '-v', default=False)
parser.add_argument('--colormap', default='gist_earth', type=str)

# Set arguments
args = parser.parse_args()
xmin = args.xmin
xmax = args.xmax
res = args.res
frames = args.frames
maxiter = args.maxiter
interval = args.interval
ca = args.ca
cb = args.cb
cc = args.cc
cd = args.cd
axis = args.axis
slice_idx = args.sliceidx
slice_dim = args.slicedim
filename = args.filename
fps = args.fps
verbose = args.verbose
colormap = args.colormap

c = [ca, cb, cc, cd]
xmins = [xmin,]*4
xmaxs = [xmax,]*4
dims = [res,]*4

julia = julia_set(xmins, xmaxs, dims, maxiter, c=c, slice_dim=slice_dim, slice_idx=slice_idx)

fig = plt.figure()

julia_slice = np.take(julia, slice_idx, axis=axis)
im = plt.imshow(julia_slice, interpolation='none', animated=True)

def init():
    julia_slice = np.take(julia, slice_idx, axis=axis)
    im.set_array(getattr(cm, colormap)(julia_slice))
    return [im]

def animate(i):
    julia_slice = np.take(julia, (slice_idx + i) % frames, axis=axis)
    im.set_array(getattr(cm, colormap)(julia_slice))
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
    plt.show()

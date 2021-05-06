import numpy as np
from scipy.stats import norm, laplace, triang, semicircular, arcsine
import matplotlib.pyplot as plt
import argparse


plt.style.use('dark_background')

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--N', '-N', default=250, type=int)
parser.add_argument('--n', '-n', default=20, type=int)
parser.add_argument('--nmin', default=20, type=int)
parser.add_argument('--nmax', default=80, type=int)
parser.add_argument('--width', default=7, type=int)
parser.add_argument('--height', default=7, type=int)
parser.add_argument('--alpha', '-a', default=0.9, type=float)
parser.add_argument('--distr', '-d', default='norm', type=str)
parser.add_argument('--filename', '-f', default='', type=str)
parser.add_argument('--colormap', '-c', default='cividis', type=str)

# Set arguments
args = parser.parse_args()
N = args.N
n = args.n
nmin = args.nmin
nmax = args.nmax
width = args.width
height = args.height
alpha = args.alpha
distr = args.distr
filename = args.filename
colormap = args.colormap

if distr == 'norm':
    sample = norm.rvs(size=N)
elif distr == 'exp':
    sample = laplace.rvs(size=N)
elif distr == 'triang':
    sample = triang(0.5).rvs(size=N)
elif distr == 'semicircular':
    sample = semicircular.rvs(size=N)
elif distr == 'arcsine':
    sample = arcsine.rvs(size=N)
else:
    raise ValueError('{:s} is not an available distribution'.format(distr))

bins = np.floor(np.linspace(nmin, nmax, n)).astype(int)

cm = plt.get_cmap(colormap)
colors = cm.colors

fig, ax = plt.subplots(figsize=(width, height), nrows=1, ncols=1)

for i in range(n):
    ax.hist(sample, bins=bins[i], color=colors[i * cm.N // n], alpha=alpha)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

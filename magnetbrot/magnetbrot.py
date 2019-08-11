#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import jit, guvectorize, complex128, int32

# magnetbrot set

@jit(int32(complex128, int32))
def magnetbrot_iter(c, maxiter):
    z = 0.+0.j
    
    for n in range(maxiter):
        zreal = z.real
        zimag = z.imag
        zreal2 = zreal * zreal
        zimag2 = zimag * zimag
        if zreal2 + zimag2 > 4.0:
            return n
        # use complex operations here
        # would be faster breaking it down to real operations
        z = ((z**2 + c -1)/(2*z+c-2)) ** 2
    return 0

@guvectorize([(complex128[:], int32[:], int32[:])], '(n),()->(n)', target='parallel')
def magnetbrot_numpy(z, maxit, output):
    maxiter = maxit[0]
    for i in range(z.shape[0]):
        output[i] = magnetbrot_iter(z[i], maxiter)

def magnetbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter, method='parallel'):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    z = r1 + r2[:,None]*1j
    magnetbrot = np.empty(z.shape, int)
    maxit = np.ones(z.shape, int) * maxiter

    if method == 'parallel':
        magnetbrot = magnetbrot_numpy(z, maxiter)
    elif method == 'single':
        for i in range(width):
            for j in range(height):
                magnetbrot[j,i] = magnetbrot_iter(r1[i]+r2[j]*1j, maxiter)
    else:
        raise NameError('Unknown method: {}'.format(method))

    return magnetbrot

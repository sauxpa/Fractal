#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import jit, guvectorize, float64, float32, complex128, complex64, int32, uint8

# Julia set

@jit(int32(complex64, complex64, int32))
def julia_iter(z, c, maxiter):
    creal = c.real
    cimag = c.imag

    zreal = z.real
    zimag = z.imag
    zreal2 = zreal*zreal
    zimag2 = zimag*zimag

    output = 0
    while zimag2 + zreal2 <= 4 and output < maxiter:
        zimag = 2*zreal*zimag + cimag
        zreal = zreal2 - zimag2 + creal
        zreal2 = zreal*zreal
        zimag2 = zimag*zimag
        output += 1

    return output

@guvectorize([(complex64[:], complex64[:], int32[:], int32[:])], '(n),(),()->(n)', target='parallel')
def julia_numpy(z, c, maxit, output):
    maxiter = maxit[0]
    creal = c[0].real
    cimag = c[0].imag
        
    for i in range(z.shape[0]):
        zreal = z[i].real
        zimag = z[i].imag
        zreal2 = zreal*zreal
        zimag2 = zimag*zimag

        output[i] = 0
        while zimag2 + zreal2 <= 4 and output[i] < maxiter:
            zimag = 2*zreal*zimag + cimag
            zreal = zreal2 - zimag2 + creal
            zreal2 = zreal*zreal
            zimag2 = zimag*zimag
            output[i] += 1

@guvectorize([(complex64[:], complex64[:], int32[:], int32[:])], '(n),(n),(n)->(n)', target='cuda')
def julia_numpy_cuda(z, c, maxit, output):
    maxiter = maxit[0]
    c = c[0]
    for i in range(z.shape[0]):
        creal = c.real
        cimag = c.imag

        zreal = z[i].real
        zimag = z[i].imag
        zreal2 = zreal*zreal
        zimag2 = zimag*zimag

        output_i = 0
        while zimag2 + zreal2 <= 4 and output_i < maxiter:
            zimag = 2*zreal*zimag + cimag
            zreal = zreal2 - zimag2 + creal
            zreal2 = zreal*zreal
            zimag2 = zimag*zimag
            output_i += 1

        output[i] = output_i

def julia_set(xmin, xmax, ymin, ymax, width, height, maxiter, method='parallel', c=0+0j):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    z = r1 + r2[:,None]*1j
    julia = np.empty(z.shape, int)
    maxit = np.ones(z.shape, int) * maxiter
    c_grid = np.ones(z.shape, int) * c
    c_grid = c_grid.astype(np.dtype('complex64'))

    if method == 'parallel':
        julia = julia_numpy(z, c, maxiter)
    elif method == 'cuda':
        julia = julia_numpy_cuda(z, c_grid, maxit)
    elif method == 'single':
        for i in range(width):
            for j in range(height):
                julia[j,i] = julia_iter(r1[i]+r2[j]*1j, c, maxiter)
    else:
        raise NameError('Unknown method: {}'.format(method))

    return julia

#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import jit, guvectorize, float64, float32, complex128, complex64, int32, uint8

# Mandelbrot set

@jit(int32(complex64, int32))
def mandelbrot_iter(z, maxiter):
    nreal = 0
    real = 0
    imag = 0
    for n in range(maxiter):
        nreal = real*real - imag*imag + z.real
        imag = 2*real*imag + z.imag
        real = nreal;
        if real * real + imag * imag > 4.0:
            return n
    return 0

@guvectorize([(complex64[:], int32[:], int32[:])], '(n),()->(n)', target='parallel')
def mandelbrot_numpy(z, maxit, output):
    maxiter = maxit[0]
    for i in range(z.shape[0]):
        output[i] = mandelbrot_iter(z[i],maxiter)

@guvectorize([(complex64[:], int32[:], int32[:])], '(n),(n)->(n)', target='cuda')
def mandelbrot_numpy_cuda(z, maxit, output):
    maxiter = maxit[0]
    for i in range(z.shape[0]):
        zreal = z[i].real
        zimag = z[i].imag
        real = zreal
        imag = zimag
        output[i] = 0
        for n in range(maxiter):
            real2 = real*real
            imag2 = imag*imag
            if real2 + imag2 > 4.0:
                output[i] = n
                break
            imag = 2*real*imag + zimag
            real = real2 - imag2 + zreal

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter, method='parallel'):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    z = r1 + r2[:,None]*1j
    mandelbrot = np.empty(z.shape, int)
    maxit = np.ones(z.shape, int) * maxiter

    if method == 'parallel':
        mandelbrot = mandelbrot_numpy(z, maxiter)
    elif method == 'cuda':
        mandelbrot = mandelbrot_numpy_cuda(z, maxit)
    elif method == 'single':
        for i in range(width):
            for j in range(height):
                mandelbrot[j,i] = mandelbrot_iter(r1[i]+r2[j]*1j, maxiter)
    else:
        raise NameError('Unknown method: {}'.format(method))

    return mandelbrot
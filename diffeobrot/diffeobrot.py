#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import jit, guvectorize, complex128, float64, int32

# diffeobrot set
# Continuous time dynamical system :
# dz(t)/dt + z(t) = z(t-t0)^2 + c
# z(0) = 0
# (continuous analogue of the Mandelbrot equation).
# The diffeobrot set is the set of point c in the complex 
# plane such that this system is stable.

@jit(int32(complex128, float64, float64, int32))
def diffeobrot_iter(c, t0, dt, maxiter):
    nreal = 0
    real = 0
    imag = 0
    creal = c.real
    cimag = c.imag
    
    # clearly the right container for that is a queue, but numba wouldn't like it...
    memory = []
    
    # assume z(t)=0 for t<0
    # the below is needed to initialize the lagged values z(t-t0).
    t = 0.0
    while t<t0:
        t += dt
        z_t_real = creal*(1-np.exp(-t))
        z_t_imag = cimag*(1-np.exp(-t))
        memory.append([z_t_real, z_t_imag])
        
    for n in range(maxiter):
        real2 = z_t_real*z_t_real
        imag2 = z_t_imag*z_t_imag
                
        if real2 + imag2 > 4.0:
            return n
        
        z_lag = memory[0]
        z_lag_real = z_lag[0]
        z_lag_imag = z_lag[1]
        z_t_real += (-z_t_real+creal+z_lag_real**2-z_lag_imag**2)
        z_t_imag += (-z_t_imag+cimag+2*z_lag_real*z_lag_imag)
        memory.append([z_t_real, z_t_imag])
        memory = memory[1:]
    return 0

@guvectorize([(complex128[:], float64[:], float64[:], int32[:], int32[:])], '(n),(),(),()->(n)', target='parallel')
def diffeobrot_numpy(z, t0, dt, maxit, output):
    maxiter = maxit[0]
    t0 = t0[0]
    dt = dt[0]
    for i in range(z.shape[0]):
        output[i] = diffeobrot_iter(z[i], t0, dt, maxiter)

def diffeobrot_set(xmin, xmax, ymin, ymax, width, height, t0, dt, maxiter, method='parallel'):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    z = r1 + r2[:,None]*1j
    diffeobrot = np.empty(z.shape, int)
    maxit = np.ones(z.shape, int) * maxiter

    if method == 'parallel':
        diffeobrot = diffeobrot_numpy(z, t0, dt, maxiter)
    elif method == 'single':
        for i in range(width):
            for j in range(height):
                diffeobrot[j,i] = diffeobrot_iter(r1[i]+r2[j]*1j, t0, dt, maxiter)
    else:
        raise NameError('Unknown method: {}'.format(method))

    return diffeobrot

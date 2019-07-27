#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import jit, float64, int32

@jit(int32(float64, float64, float64, float64,
           float64, float64, float64, float64,
           int32), nopython=True, nogil=True)
def julia_iter(a, b, c,d,
               ca, cb, cc, cd,
               maxiter):
    a2 = a*a
    b2 = b*b
    c2 = c*c
    d2 = d*d

    output = 0
    while a2 + b2 + c2 + d2 <= 4 and output < maxiter:
        tempa = a*a-b*b-c*c-d*d + ca
        tempb = 2*a*b + cb
        tempc = 2*a*c + cc
        tempd = 2*a*d + cd

        a = tempa
        b = tempb
        c = tempc
        d = tempd

        a2 = a*a
        b2 = b*b
        c2 = c*c
        d2 = d*d

        output += 1

    return output

def julia_set(xmins, xmaxs, dims, maxiter, c=[0, 0, 0, 0], slice_dim=-1, slice_idx=None):
    """ Julia set for the transformation z -> z^2 + c on the quaternions.
    This is a 4d transformation.
    To speed up the calculation, given we will only render a 3d slice of the 4d set,
    one can use slice_dim to specify the dimension along which the set will be sliced.
    Slice_idx is the coordinate of the slicing.
    """
    za = np.linspace(xmins[0], xmaxs[0], dims[0], dtype=np.float64)
    zb = np.linspace(xmins[1], xmaxs[1], dims[1], dtype=np.float64)
    zc = np.linspace(xmins[2], xmaxs[2], dims[2], dtype=np.float64)
    zd = np.linspace(xmins[3], xmaxs[3], dims[3], dtype=np.float64)

    sliced_dims=[dims[i] for i in range(len(dims)) if i != slice_dim]
    julia = np.empty(sliced_dims, int)

    # not exactly elegant but will do the job for now
    if slice_dim == 0:
        for j in range(dims[1]):
            for k in range(dims[2]):
                for l in range(dims[3]):
                    julia[l, k, j] = julia_iter(za[slice_idx], zb[j], zc[k], zd[l],
                                                c[0], c[1], c[2], c[3],
                                                maxiter)
    elif slice_dim == 1:
        for i in range(dims[0]):
            for k in range(dims[2]):
                for l in range(dims[3]):
                    julia[l, k, i] = julia_iter(za[i], zb[slice_idx], zc[k], zd[l],
                                                c[0], c[1], c[2], c[3],
                                                maxiter)
    elif slice_dim == 2:
        for i in range(dims[0]):
            for j in range(dims[1]):
                for l in range(dims[3]):
                    julia[l, j, i] = julia_iter(za[i], zb[j], zc[slice_idx], zd[l],
                                                c[0], c[1], c[2], c[3],
                                                maxiter)
    elif slice_dim == 3:
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    julia[k, j, i] = julia_iter(za[i], zb[j], zc[k], zd[slice_idx],
                                                c[0], c[1], c[2], c[3],
                                                maxiter)
    else:
        raise NameError('Must slice dimension 0, 1, 2 or 3, not {}'.format(slice_dim))

    return julia

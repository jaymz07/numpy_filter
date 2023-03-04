# distutils: language=c++

# ^ Compiler directive. Tells cython to compile to C++
from libcpp.vector cimport vector

import numpy as np

cimport numpy as np
np.import_array()

DTYPE = np.float64

ctypedef np.int_t DTYPE_t

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def lpf1(np.ndarray[double, ndim=1] y, float alpha):
    assert y.dtype == DTYPE
    cdef int N = y.shape[0]
    cdef np.ndarray yout = np.zeros(N,dtype=DTYPE)

    yout[0] = y[0]
    for i in range(1,N):
        yout[i] = alpha*y[i] + (1-alpha)*yout[i-1]
    return yout

def lpf2(np.ndarray[double, ndim=1] y, float alpha):
    assert y.dtype == DTYPE
    cdef int N = y.shape[0]
    cdef float a1 = 1.0-alpha
    cdef np.ndarray yout = np.zeros(N,dtype=DTYPE)

    yout[0] = y[0]
    yout[1] = alpha*y[1] + a1*y[0]
    for i in range(2,N):
        yout[i] = alpha*alpha*y[i] + 2*a1*yout[i-1] - a1**2*yout[i-2]
    return yout


def lpf(np.ndarray[double, ndim=1] y, float alpha, int order):
    assert y.dtype == DTYPE
    cdef int N = y.shape[0]
    cdef np.ndarray[double, ndim=2] yout = np.zeros((N,order+1),dtype=DTYPE)
    cdef np.ndarray[double, ndim=1] retval = np.zeros(N,dtype=DTYPE)

    for j in range(order+1):
        yout[0,j] = y[0]
    for i in range(N):
        yout[i,0] = y[i]

    for i in range(1,N):
        for j in range(1,order+1):
            yout[i,j] = alpha*yout[i,j-1] + (1-alpha)*yout[i-1,j]
        retval[i] = yout[i,order]
    return retval


def hpf(np.ndarray[double, ndim=1] y, float alpha):
    assert y.dtype == DTYPE
    cdef int N = y.shape[0]
    cdef np.ndarray yout = np.zeros(N,dtype=DTYPE)

    yout[0] = y[0]
    for i in range(1,N):
        yout[i] = (1-alpha)*yout[i-1] + (1-alpha)*(y[i] - y[i-1])
    return yout

def hpf2(np.ndarray[double, ndim=1] y, float alpha):
    assert y.dtype == DTYPE
    cdef int N = y.shape[0]
    cdef float a1 = 1.0-alpha
    cdef np.ndarray yout = np.zeros(N,dtype=DTYPE)

    yout[0] = y[0]
    yout[1] = a1*y[0] + a1*(y[1]-y[0])
    for i in range(2,N):
        yout[i] = a1*a1*(y[i] - 2*y[i-1] + y[i-2]) + 2*a1*yout[i-1] - a1*a1*yout[i-2]
    return yout

def alpha(freq, delta_t):
    pi2 = np.pi*2
    return pi2*freq*delta_t/(pi2*freq*delta_t + 1)

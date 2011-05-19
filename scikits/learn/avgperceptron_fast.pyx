# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# filename: avgperceptron.pyx
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
from __future__ import division

import numpy as np
import sys

cimport numpy as np
cimport cython

from time import time


# ----------------------------------------
# C functions for fast sparse-dense vector operations
# ----------------------------------------

cdef double dot(double *w_data_ptr, double *X_data_ptr, int *X_indices_ptr,
                int offset, int xnnz):
    cdef double sum = 0.0
    cdef int j
    for j from 0 <= j < xnnz:
        sum += w_data_ptr[X_indices_ptr[offset + j]] * X_data_ptr[offset + j]
    return sum


cdef void add(double *w_data_ptr, int w_offset, double *X_data_ptr,
                int *X_indices_ptr, int X_offset, int xnnz, double c):
    """Adds c*x to w."""
    cdef int i
    cdef int idx
    cdef double val
    #cdef int w_offset = (y * stride)
    for i from 0 <= i < xnnz:
        idx = X_indices_ptr[X_offset + i]
        val = X_data_ptr[X_offset + i]        
        w_data_ptr[w_offset + idx] += c * val


cdef int argmax(double *w_data_ptr, int wstride, double *X_data_ptr,
                int *X_indices_ptr, int X_offset, int xnnz, int y,
                int n_classes):
    cdef int j
    cdef double *w_k_data_ptr
    cdef double max_score = -1.0
    cdef double p = 0.0
    cdef int max_j = 0
    for j from 0 <= j < n_classes:
        w_j_data_ptr = w_data_ptr + (wstride * j)
        p = dot(w_j_data_ptr, X_data_ptr, X_indices_ptr, X_offset, xnnz)
        if p >= max_score:
            max_j = j
            max_score = p
    return max_j


def fit_sparse(np.ndarray[double, ndim=1, mode="c"] X_data,
        np.ndarray[int, ndim=1, mode="c"] X_indices,
        np.ndarray[int, ndim=1, mode="c"] X_indptr,
        np.ndarray[double, ndim=1, mode="c"] Y,
        np.ndarray[double, ndim=2, mode="c"] w,
        np.ndarray[double, ndim=2, mode="c"] w_bar,
        int epochs, int averaged,
        int shuffle, int verbose,
        int seed):

    cdef unsigned int n_samples = Y.shape[0]
    cdef unsigned int n_classes = w.shape[0]
    cdef unsigned int n_features = w.shape[1]

    # input X and y
    cdef double *X_data_ptr = <double *>X_data.data
    cdef int *X_indptr_ptr = <int *>X_indptr.data
    cdef int *X_indices_ptr = <int *>X_indices.data
    cdef double *Y_data_ptr = <double *>Y.data

    cdef double *w_data_ptr = <double *>w.data
    cdef double *w_bar_data_ptr = <double *>w_bar.data
    cdef int wstride0 = w.strides[0]
    cdef int wstride1 = w.strides[1]
    cdef int wstride = <int> (wstride0 / wstride1)

    # index array
    cdef np.ndarray[int, ndim=1, mode="c"] index = np.arange(n_samples,
                                                             dtype=np.int32)
    cdef int *index_data_ptr = <int *>index.data

    cdef unsigned int count = 0
    cdef unsigned int epoch = 0
    cdef unsigned int i = 0
    cdef int sample_idx = 0
    cdef int y = -1
    cdef int z = -1
    cdef int xnnz = 0
    cdef int nadds = 0
    cdef double u = 0.0
    cdef int w_offset = 0
    cdef int offset = 0

    t1=time()
    for epoch from 0 <= epoch < epochs:
        if verbose > 0:
            print("-- Epoch %d" % (epoch + 1))
        if shuffle:
            np.random.RandomState(seed).shuffle(index)
        nadds = 0
        i = 0
        for i from 0 <= i < n_samples:
            sample_idx = index_data_ptr[i]
            offset = X_indptr_ptr[sample_idx]
            xnnz = X_indptr_ptr[sample_idx + 1] - offset
            y = <int>Y_data_ptr[sample_idx]
            z = argmax(w_data_ptr, wstride, X_data_ptr, X_indices_ptr,
                       offset, xnnz, y, n_classes)
            if z != y:
                u = <double>(epochs * n_samples - (epoch * n_samples + i + 1))
                
                add(w_data_ptr, y * wstride, X_data_ptr, X_indices_ptr, offset, xnnz,
                    1.0)
                add(w_bar_data_ptr, y * wstride, X_data_ptr, X_indices_ptr, offset, xnnz,
                    u)
                
                add(w_data_ptr, z * wstride, X_data_ptr, X_indices_ptr, offset, xnnz,
                    -1.0)
                add(w_bar_data_ptr, z * wstride, X_data_ptr, X_indices_ptr, offset, xnnz,
                    -1.0 * u)
                nadds += 1

            i += 1
        # report epoche information
        if verbose > 0:
            print("NADDs: %d; NNZs: %d. " % (nadds, w.nonzero()[0].shape[0]))
            print("Total training time: %.2f seconds." % (time()-t1))

    # floating-point under-/overflow check.
    if np.any(np.isinf(w)) or np.any(np.isnan(w)):
        raise ValueError("floating-point under-/overflow occured.")

    w_bar *= (1.0 / (n_samples * epochs))
    return w, w_bar

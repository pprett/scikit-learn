"""Efficient (dense) parameter vector implementation for linear models. """

cimport numpy as np


from dtype cimport DTYPE


cdef extern from "math.h":
    cdef extern double sqrt(double x)


cdef class WeightVector(object):
    cdef np.ndarray w
    cdef double *w_data_ptr
    cdef double wscale
    cdef int n_features
    cdef double sq_norm

    cdef void add(self, DTYPE *x_data_ptr, int *x_ind_ptr,
                  int xnnz, double c) nogil
    cdef double dot(self, DTYPE *x_data_ptr, int *x_ind_ptr,
                    int xnnz) nogil
    cdef void scale(self, double c) nogil
    cdef void reset_wscale(self) nogil
    cdef double norm(self) nogil

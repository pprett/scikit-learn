# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.


import numpy as np


cimport numpy as np
cimport cython


cdef class WeightVector(object):
    """Dense vector represented by a scalar and a numpy array.

    The class provides methods to ``add`` a sparse vector
    and scale the vector.
    Representing a vector explicitly as a scalar times a
    vector allows for efficient scaling operations.

    Attributes
    ----------
    w : ndarray, dtype=np.float64, order='C'
        The numpy array which backs the weight vector.
    w_data_ptr : np.float64*
        A pointer to the data of the numpy array.
    wscale : double
        The scale of the vector.
    n_features : int
        The number of features (= dimensionality of ``w``).
    sq_norm : double
        The squared norm of ``w``.
    """

    def __cinit__(self, np.ndarray[DOUBLE, ndim=1, mode='c'] w):
        self.w = w
        self.w_data_ptr = <DOUBLE *>w.data
        self.wscale = 1.0
        self.n_features = w.shape[0]
        self.sq_norm = np.dot(w, w)

    cdef void add(self, FeatureVector f_vec, double c):
        """Scales example x by constant c and adds it to the weight vector.

        This operation updates ``sq_norm``.

        Parameters
        ----------
        f_vec : FeatureVector
            The feature vector of example x.
        c : double
            The scaling constant for the example.
        """
        cdef FVElem *fv_elem
        cdef int idx
        cdef double val
        cdef double innerprod = 0.0
        cdef double xsqnorm = 0.0

        # the next two lines save a factor of 2!
        cdef double wscale = self.wscale
        cdef DOUBLE* w_data_ptr = self.w_data_ptr

        f_vec.reset_iter()
        while f_vec.has_next() == 1:
            fv_elem = f_vec.next()
            idx = fv_elem.index
            val = fv_elem.value
            innerprod += (w_data_ptr[idx] * val)
            xsqnorm += (val * val)
            w_data_ptr[idx] += val * (c / wscale)

        self.sq_norm += (xsqnorm * c * c) + (2.0 * innerprod * wscale * c)

    cdef double dot(self, FeatureVector f_vec):
        """Computes the dot product of a sample x and the weight vector.

        Parameters
        ----------
        f_vec : FeatureVector
            The feature vector of example x.

        Returns
        -------
        innerprod : double
            The inner product of ``x`` and ``w``.
        """
        cdef FVElem *fv_elem
        cdef int idx
        cdef double innerprod = 0.0
        cdef DOUBLE* w_data_ptr = self.w_data_ptr

        f_vec.reset_iter()
        while f_vec.has_next() == 1:
            fv_elem = f_vec.next()
            idx = fv_elem.index
            innerprod += w_data_ptr[idx] * fv_elem.value

        innerprod *= self.wscale
        return innerprod

    ## cdef double dot_on_difference(self, DOUBLE *a_data_ptr,
    ##                               DOUBLE *b_data_ptr, INTEGER *x_ind_ptr,
    ##                               int xnnz_a, int xnnz_b):
    ##     """Computes the dot product of the weight vector and the difference
    ##        between samples a and b with disagreeing labels.

    ##     Parameters
    ##     ----------
    ##     a_data_ptr : double*
    ##         The array which holds the feature values of the first example
    ##         in the pair.
    ##     b_data_ptr : double*
    ##         The array which holds the feature values of the second example
    ##         in the pair.

    ##     Returns
    ##     -------
    ##     innerprod_on_difference : double
    ##         The inner product of ``w`` and the difference between
    ##         ``a`` and ``b``.
    ##     """
    ##     # <(a - b), w> = <a, w> + <-1.0 * b, w>
    ##     cdef double innerprod_on_difference = 0.0
    ##     innerprod_on_difference += self.dot(a_data_ptr, x_ind_ptr,
    ##                                         xnnz_a)
    ##     innerprod_on_difference += self.dot(b_data_ptr, x_ind_ptr,
    ##                                         xnnz_b) * -1.0
    ##     return innerprod_on_difference

    cdef void scale(self, double c):
        """Scales the weight vector by a constant ``c``.

        It updates ``wscale`` and ``sq_norm``. If ``wscale`` gets too
        small we call ``reset_swcale``."""
        self.wscale *= c
        self.sq_norm *= (c * c)
        if self.wscale < 1e-9:
            self.reset_wscale()

    cdef void reset_wscale(self):
        """Scales each coef of ``w`` by ``wscale`` and resets it to 1. """
        self.w *= self.wscale
        self.wscale = 1.0

    cdef double norm(self):
        """The L2 norm of the weight vector. """
        return sqrt(self.sq_norm)

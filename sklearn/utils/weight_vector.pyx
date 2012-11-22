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
        cdef FVElem fv_elem
        cdef INTEGER idx
        cdef DOUBLE val
        cdef double innerprod = 0.0
        cdef double xsqnorm = 0.0

        # the next two lines save a factor of 2!
        cdef double wscale = self.wscale
        cdef DOUBLE* w_data_ptr = self.w_data_ptr

        f_vec.reset_iter()
        while f_vec.next(&fv_elem) == 1:
            idx = (fv_elem.index)[0]
            val = (fv_elem.value)[0]
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
        cdef FVElem fv_elem
        cdef double innerprod = 0.0
        cdef DOUBLE* w_data_ptr = self.w_data_ptr

        f_vec.reset_iter()
        while f_vec.next(&fv_elem) == 1:
            innerprod += w_data_ptr[(fv_elem.index)[0]] * (fv_elem.value)[0]

        innerprod *= self.wscale
        return innerprod

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

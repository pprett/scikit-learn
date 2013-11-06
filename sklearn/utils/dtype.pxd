"""Fused dtype for feature matrix. """

cimport numpy as np

ctypedef fused DTYPE:
    np.float64_t
    np.float32_t
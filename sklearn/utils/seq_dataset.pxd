"""Dataset abstractions for sequential data access. """

cimport numpy as np

from cpython cimport bool

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INTEGER

cdef struct s_FVElem:
    INTEGER    *index
    DOUBLE     *value


ctypedef s_FVElem FVElem


cdef class FeatureVector:
    cdef public DOUBLE y
    cdef public DOUBLE sample_weight

    cdef Py_ssize_t n_features
    cdef INTEGER iter_pos
    cdef FVElem out

    cdef int next(self, FVElem *fv_elem)
    cdef void reset_iter(self)


cdef class ArrayFeatureVector(FeatureVector):
    cdef DOUBLE *x_data_ptr

    cdef void set_row(self, DOUBLE *x_data_ptr, DOUBLE y, DOUBLE sample_weight)
    cdef int next(self, FVElem *fv_elem)


cdef class CSRFeatureVector(FeatureVector):
    cdef DOUBLE *x_data_ptr
    cdef INTEGER *x_ind_ptr
    cdef int nnz

    cdef void set_row(self, DOUBLE *x_data_ptr, INTEGER *x_ind_ptr, int nnz,
                      DOUBLE y, DOUBLE sample_weight)
    cdef int next(self, FVElem *fv_elem)


cdef class PairwiseFeatureVector(FeatureVector):
    cdef FeatureVector f_vec_a
    cdef FeatureVector f_vec_b
    cdef DOUBLE inverted_value

    cdef void set_pair(self, FeatureVector f_vec_a, FeatureVector f_vec_b)
    cdef int next(self, FVElem *fv_elem)


cdef class SequentialDataset:
    cdef Py_ssize_t n_samples

    cdef void next(self)
    cdef void shuffle(self, seed)
    cdef FeatureVector get_feature_vector(self)


cdef class ArrayDataset(SequentialDataset):
    cdef Py_ssize_t n_features
    cdef int current_index
    cdef int stride
    cdef DOUBLE *X_data_ptr
    cdef DOUBLE *Y_data_ptr
    cdef np.ndarray index
    cdef INTEGER *index_data_ptr
    cdef DOUBLE *sample_weight_data
    cdef ArrayFeatureVector feature_vector

    cdef void next(self)
    cdef void shuffle(self, seed)
    cdef FeatureVector get_feature_vector(self)


cdef class CSRDataset(SequentialDataset):
    cdef int current_index
    cdef int stride
    cdef DOUBLE *X_data_ptr
    cdef INTEGER *X_indptr_ptr
    cdef INTEGER *X_indices_ptr
    cdef DOUBLE *Y_data_ptr
    cdef np.ndarray feature_indices
    cdef INTEGER *feature_indices_ptr
    cdef np.ndarray index
    cdef INTEGER *index_data_ptr
    cdef DOUBLE *sample_weight_data
    cdef CSRFeatureVector feature_vector

    cdef void next(self)
    cdef void shuffle(self, seed)
    cdef FeatureVector get_feature_vector(self)


cdef class PairwiseDataset(SequentialDataset):

    cdef np.ndarray pos_index
    cdef np.ndarray neg_index
    cdef INTEGER *pos_index_data_ptr
    cdef INTEGER *neg_index_data_ptr
    cdef int n_pos_samples
    cdef int n_neg_samples

    cdef PairwiseFeatureVector feature_vector

    cdef void next(self)
    cdef void shuffle(self, seed)
    cdef FeatureVector get_feature_vector(self)


cdef class PairwiseArrayDataset(PairwiseDataset):
    cdef Py_ssize_t n_features

    cdef int stride
    cdef DOUBLE *X_data_ptr
    cdef DOUBLE *Y_data_ptr

    cdef ArrayFeatureVector f_vec_a
    cdef ArrayFeatureVector f_vec_b

    cdef void next(self)
    cdef void shuffle(self, seed)
    cdef FeatureVector get_feature_vector(self)

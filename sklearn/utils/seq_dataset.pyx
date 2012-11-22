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


cdef class FeatureVector:
    """A vector that supports iteration over non-zero elements. """

    cdef int next(self, FVElem *fv_elem):
        raise NotImplementedError()

    cdef void reset_iter(self):
        self.iter_pos = -1


cdef class ArrayFeatureVector(FeatureVector):
    """Feature vector backed by a row in a numpy array. """

    def __cinit__(self, Py_ssize_t n_features):
        self.n_features = n_features
        self.iter_pos = 2**31 - 1

    cdef void set_row(self, DOUBLE *x_data_ptr, DOUBLE y, DOUBLE sample_weight):
        self.x_data_ptr = x_data_ptr
        self.y = y
        self.sample_weight = sample_weight
        self.reset_iter()

    cdef int next(self, FVElem *fv_elem):
        cdef INTEGER iter_pos = self.iter_pos + 1
        if iter_pos < self.n_features:
            fv_elem.index = &(self.iter_pos)
            fv_elem.value = &(self.x_data_ptr[iter_pos])
            self.iter_pos = iter_pos
            return 1
        else:
            return 0


cdef class CSRFeatureVector(FeatureVector):
    """Feature vector backed by a row in a CSR matrix. """

    def __cinit__(self):
        self.iter_pos = 2**31 - 1

    cdef void set_row(self, DOUBLE *x_data_ptr, INTEGER *x_ind_ptr, int nnz,
                      DOUBLE y, DOUBLE sample_weight):
        self.x_data_ptr = x_data_ptr
        self.x_ind_ptr = x_ind_ptr
        self.nnz = nnz
        self.reset_iter()

    cdef int next(self, FVElem *fv_elem):
        cdef INTEGER iter_pos = self.iter_pos + 1
        if iter_pos < self.nnz:
            fv_elem.index = self.x_ind_ptr + iter_pos
            fv_elem.value = self.x_data_ptr + iter_pos
            self.iter_pos = iter_pos
            return 1
        else:
            return 0


cdef class PairwiseFeatureVector(FeatureVector):
    """A vector that is backed by two ``FeatureVector`` objects. """

    cdef void set_pair(self, FeatureVector f_vec_a, FeatureVector f_vec_b):
        self.f_vec_a = f_vec_a
        self.f_vec_b = f_vec_b

    cdef int next(self, FVElem *fv_elem):
        cdef int has_a_next = 0
        cdef int has_b_next = 0
        print('before a.next')
        has_a_next = self.f_vec_a.next(fv_elem)
        print('a.next=%d' % has_a_next)
        if not has_a_next:
            # f_vec_a is exhausted
            has_b_next = self.f_vec_b.next(fv_elem)
            if not has_b_next:
                return 0
            else:
                # invert b's value
                self.inverted_value = (fv_elem.value)[0] * -1.0
                fv_elem.value = &(self.inverted_value)
                print('[b] %d:%.2f' % ((fv_elem.index)[0]),
                      (fv_elem.value)[0])
                return 1
        else:
            print('[a] %d:%.2f' % ((fv_elem.index)[0]),
                  (fv_elem.value)[0])
            return 1

    cdef void reset_iter(self):
        print('resetting iter')
        self.f_vec_a.reset_iter()
        self.f_vec_b.reset_iter()


cdef class SequentialDataset:
    """Base class for datasets with sequential data access. """

    cdef void next(self):
        """Get the next example ``x`` from the dataset.

        Parameters
        ----------
        x_data_ptr : np.float64**
            A pointer to the double array which holds the feature
            values of the next example.
        x_ind_ptr : np.int32**
            A pointer to the int32 array which holds the feature
            indices of the next example.
        nnz : int*
            A pointer to an int holding the number of non-zero
            values of the next example.
        y : np.float64*
            The target value of the next example.
        sample_weight : np.float64*
            The weight of the next example.
        """
        raise NotImplementedError()

    cdef void shuffle(self, seed):
        """Permutes the ordering of examples.  """
        raise NotImplementedError()

    cdef FeatureVector get_feature_vector(self):
        raise NotImplementedError()


cdef class ArrayDataset(SequentialDataset):
    """Dataset backed by a two-dimensional numpy array.

    The dtype of the numpy array is expected to be ``np.float64``
    and C-style memory layout.
    """

    def __cinit__(self, np.ndarray[DOUBLE, ndim=2, mode='c'] X,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] Y,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] sample_weights):
        """A ``SequentialDataset`` backed by a two-dimensional numpy array.

        Paramters
        ---------
        X : ndarray, dtype=np.float64, ndim=2, mode='c'
            The samples; a two-dimensional c-continuous numpy array of
            dtype np.float64.
        Y : ndarray, dtype=np.float64, ndim=1, mode='c'
            The target values; a one-dimensional c-continuous numpy array of
            dtype np.float64.
        sample_weights : ndarray, dtype=np.float64, ndim=1, mode='c'
            The weight of each sample; a one-dimensional c-continuous numpy
            array of dtype np.float64.
        """
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.current_index = -1
        self.stride = X.strides[0] / X.strides[1]
        self.X_data_ptr = <DOUBLE *>X.data
        self.Y_data_ptr = <DOUBLE *>Y.data
        self.sample_weight_data = <DOUBLE *>sample_weights.data

        # Use index array for fast shuffling
        cdef np.ndarray[INTEGER, ndim=1,
                        mode='c'] index = np.arange(0, self.n_samples,
                                                    dtype=np.int32)
        self.index = index
        self.index_data_ptr = <INTEGER *> index.data

        # the feature vector for this dataset
        self.feature_vector = ArrayFeatureVector(self.n_features)

    cdef void next(self):
        cdef int current_index = self.current_index
        if current_index >= (self.n_samples - 1):
            current_index = -1

        current_index += 1
        cdef int sample_idx = self.index_data_ptr[current_index]
        cdef int offset = sample_idx * self.stride

        self.feature_vector.set_row(self.X_data_ptr + offset,
                                    self.Y_data_ptr[sample_idx],
                                    self.sample_weight_data[sample_idx])

        self.current_index = current_index

    cdef void shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.index)

    cdef FeatureVector get_feature_vector(self):
        return self.feature_vector


cdef class CSRDataset(SequentialDataset):
    """A ``SequentialDataset`` backed by a scipy sparse CSR matrix. """

    def __cinit__(self, np.ndarray[DOUBLE, ndim=1, mode='c'] X_data,
                  np.ndarray[INTEGER, ndim=1, mode='c'] X_indptr,
                  np.ndarray[INTEGER, ndim=1, mode='c'] X_indices,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] Y,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] sample_weight):
        """Dataset backed by a scipy sparse CSR matrix.

        The feature indices of ``x`` are given by x_ind_ptr[0:nnz].
        The corresponding feature values are given by
        x_data_ptr[0:nnz].

        Parameters
        ----------
        X_data : ndarray, dtype=np.float64, ndim=1, mode='c'
            The data array of the CSR matrix; a one-dimensional c-continuous
            numpy array of dtype np.float64.
        X_indptr : ndarray, dtype=np.int32, ndim=1, mode='c'
            The index pointer array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.int32.
        X_indices : ndarray, dtype=np.int32, ndim=1, mode='c'
            The column indices array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.int32.
        Y : ndarray, dtype=np.float64, ndim=1, mode='c'
            The target values; a one-dimensional c-continuous numpy array of
            dtype np.float64.
        sample_weights : ndarray, dtype=np.float64, ndim=1, mode='c'
            The weight of each sample; a one-dimensional c-continuous numpy
            array of dtype np.float64.
        """
        self.n_samples = Y.shape[0]
        self.current_index = -1
        self.X_data_ptr = <DOUBLE *>X_data.data
        self.X_indptr_ptr = <INTEGER *>X_indptr.data
        self.X_indices_ptr = <INTEGER *>X_indices.data
        self.Y_data_ptr = <DOUBLE *>Y.data
        self.sample_weight_data = <DOUBLE *> sample_weight.data
        # Use index array for fast shuffling
        cdef np.ndarray[INTEGER, ndim=1,
                        mode='c'] index = np.arange(0, self.n_samples,
                                                    dtype=np.int32)
        self.index = index
        self.index_data_ptr = <INTEGER *> index.data

        self.feature_vector = CSRFeatureVector()

    cdef void next(self):
        cdef int current_index = self.current_index
        if current_index >= (self.n_samples - 1):
            current_index = -1

        current_index += 1
        cdef int sample_idx = self.index_data_ptr[current_index]
        cdef int offset = self.X_indptr_ptr[sample_idx]

        self.feature_vector.set_row(self.X_data_ptr + offset,
                                    self.X_indices_ptr + offset,
                                    self.X_indptr_ptr[sample_idx + 1] - offset,
                                    self.Y_data_ptr[sample_idx],
                                    self.sample_weight_data[sample_idx])

        self.current_index = current_index

    cdef void shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.index)

    cdef FeatureVector get_feature_vector(self):
        return self.feature_vector


cdef class PairwiseDataset(SequentialDataset):
    pass


cdef class PairwiseArrayDataset(PairwiseDataset):
    """Dataset backed by a two-dimensional numpy array. Calling next() returns a random pair of examples with disagreeing labels.

    The dtype of the numpy array is expected to be ``np.float64``
    and C-style memory layout.
    """
    def __cinit__(self, np.ndarray[DOUBLE, ndim=2, mode='c'] X,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] Y):
        """A ``PairwiseArrayDataset`` backed by a two-dimensional numpy array.

        Parameters
        ---------
        X : ndarray, dtype=np.float64, ndim=2, mode='c'
            The samples; a two-dimensional c-continuous numpy array of
            dtype np.float64.
        Y : ndarray, dtype=np.float64, ndim=1, mode='c'
            The target values; a one-dimensional c-continuous numpy array of
            dtype np.float64.
        sample_weights : ndarray, dtype=np.float64, ndim=1, mode='c'
            The weight of each sample; a one-dimensional c-continuous numpy
            array of dtype np.float64.
        """
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.stride = X.strides[0] / X.strides[1]
        self.X_data_ptr = <DOUBLE *>X.data
        self.Y_data_ptr = <DOUBLE *>Y.data

        # Create an index of positives and negatives for fast sampling
        # of disagreeing pairs
        positives = []
        negatives = []
        cdef Py_ssize_t i
        for i in range(self.n_samples):
            if Y[i] > 0:
                positives.append(i)
            else:
                negatives.append(i)
        cdef np.ndarray[INTEGER, ndim=1,
                        mode='c'] pos_index = np.array(positives, dtype=np.int32)
        cdef np.ndarray[INTEGER, ndim=1,
                        mode='c'] neg_index = np.array(negatives, dtype=np.int32)
        self.pos_index = pos_index
        self.neg_index = neg_index
        self.pos_index_data_ptr = <INTEGER *> pos_index.data
        self.neg_index_data_ptr = <INTEGER *> neg_index.data
        self.n_pos_samples = len(pos_index)
        self.n_neg_samples = len(neg_index)

        # the feature vectors for this dataset
        self.f_vec_a = ArrayFeatureVector(self.n_features)
        self.f_vec_b = ArrayFeatureVector(self.n_features)
        self.feature_vector = PairwiseFeatureVector(self.f_vec_a,
                                                    self.f_vec_b)

    cdef void next(self):
        cdef int current_pos_index = np.random.randint(self.n_pos_samples)
        cdef int current_neg_index = np.random.randint(self.n_neg_samples)

        # For each step, randomly sample one positive and one negative
        cdef int sample_pos_idx = self.pos_index_data_ptr[current_pos_index]
        cdef int sample_neg_idx = self.neg_index_data_ptr[current_neg_index]
        cdef int pos_offset = sample_pos_idx * self.stride
        cdef int neg_offset = sample_neg_idx * self.stride

        print("%d %d" % (current_pos_index, current_neg_index))
        print("%d %d" % (sample_pos_idx, sample_neg_idx))

        self.f_vec_a.set_row(self.X_data_ptr + pos_offset, 0.0, 1.0)
        self.f_vec_b.set_row(self.X_data_ptr + neg_offset, 0.0, 1.0)
        print("%.2f %.2f" % (self.X_data_ptr[pos_offset],
                             self.X_data_ptr[neg_offset]))

        cdef DOUBLE y_a = self.Y_data_ptr[sample_pos_idx]
        cdef DOUBLE y_b = self.Y_data_ptr[sample_neg_idx]
        cdef DOUBLE y = 0.0

        if y_a > y_b:
            y = 1.0
        elif y_a < y_b:
            y = -1.0

        self.feature_vector.y = y

    cdef void shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.index)

    cdef FeatureVector get_feature_vector(self):
        return self.feature_vector

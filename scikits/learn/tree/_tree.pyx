# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
#
# Adapted for CART by Brian Holt <bdholt1@gmail.com>
#

import numpy as np
cimport numpy as np

cimport cython

cdef extern from "math.h":
    cdef extern double log(double x)

cdef extern from "float.h":
    cdef extern double DBL_MAX


cdef class Criterion:
    """Interface for splitting criteria, both regression and classification.
    """
    
    cdef void init(self, np.float64_t *labels, np.int8_t *sample_mask,
                   int n_total_samples):
        """Init the criteria for each feature `i`. """
        pass

    cdef int update(self, int a, int b, np.float64_t *labels,
                     np.int8_t *sample_mask, np.float64_t *features_i,
                     int *sorted_features_i):
        """Update the criteria for each interval [a,b) where a and b
        are indices in `sorted_features_i`. """
        pass

    cdef double eval(self):
        """Evaluate the criteria (aka the split error). """
        pass

cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification.

    Attributes
    ----------
    pm_left_ptr : int*
        label counts for samples left of splitting point.
    pm_right_ptr : int*
        label counts for samples right of splitting point.
    nml : int
        number of samples left of splitting point.
    nmr : int
        number of samples right of splitting point.
    """

    cdef int *pm_left_ptr
    cdef int *pm_right_ptr

    cdef int nml, nmr, K, n_total_samples, n_samples
    
    def __init__(self, int K, np.ndarray[np.int32_t, ndim=1] pm_left,
                 np.ndarray[np.int32_t, ndim=1] pm_right):
        self.K = K
        self.nml = 0
        self.nmr = 0
        self.pm_left_ptr = <int *>pm_left.data
        self.pm_right_ptr = <int *>pm_right.data

    cdef void init(self, np.float64_t *labels, np.int8_t *sample_mask,
                   int n_total_samples):
        """Initializes the criterion for a new feature (col of `features`)."""
        cdef int c, j
        self.nml = 0
        self.nmr = 0
        for c from 0 <= c < self.K:
            self.pm_left_ptr[c] = 0
            self.pm_right_ptr[c] = 0

        for j from 0 <= j < n_total_samples:
            if sample_mask[j] == 0:
                continue
            c = <int>(labels[j])
            self.pm_right_ptr[c] += 1
            self.nmr += 1
            
        self.n_samples = self.nmr
        self.n_total_samples = n_total_samples

    cdef int update(self, int a, int b, np.float64_t *labels,
                    np.int8_t *sample_mask, np.float64_t *features_i,
                    int *sorted_features_i):
        """Update the counts for docs between `sorted_features_i[a]`
        and `sorted_features_i[b]`. """
        cdef int c, idx, j
        ## all samples from a to b-1 are on the left side
        for idx from a <= idx < b:
            j = sorted_features_i[idx]
            if sample_mask[j] == 0:
                continue
            ## we will distribute c from right to left
            #assert features_i[j] < t
            c = <int>(labels[j])
            self.pm_right_ptr[c] -= 1
            self.pm_left_ptr[c] += 1
            self.nmr -= 1
            self.nml += 1
            
        return self.nml

    cdef double eval(self):
        pass


cdef class Gini(ClassificationCriterion):
    """Gini Index splitting criteria.
    
    Gini index = \sum_{k=0}^{K-1} pmk (1 - pmk)
               = 1 - \sum_{k=0}^{K-1} pmk ** 2
    """

    cdef double eval(self):
        """Returns Gini index of left branch + Gini index of right branch. """
        cdef double gini_left = 1.0
        cdef double gini_right = 1.0
        cdef int k
        cdef double e1, e2
        cdef double nml = <double> self.nml
        cdef double nmr = <double> self.nmr
        #assert (self.nml + self.nmr) == self.n_samples
        
        for k from 0 <= k < self.K:
            gini_left -= (self.pm_left_ptr[k] / nml) * (self.pm_left_ptr[k] / nml)
            gini_right -= (self.pm_right_ptr[k] / nmr) * (self.pm_right_ptr[k] / nmr)

        e1 = (nml / self.n_samples) * gini_left
        e2 = (nmr / self.n_samples) * gini_right
        return e1 + e2


cdef class Entropy(ClassificationCriterion):
    """Entropy splitting criteria.

    Cross Entropy = - \sum_{k=0}^{K-1} pmk log(pmk)
    
    """

    cdef double eval(self):
        """Returns Entropy of left branch + Entropy index of right branch. """
        cdef double H_left = 0.0
        cdef double H_right = 0.0
        cdef int k
        cdef double e1, e2
        cdef double nml = <double> self.nml
        cdef double nmr = <double> self.nmr
        
        for k from 0 <= k < self.K:
            if self.pm_left_ptr[k] > 0:
                H_left += -(self.pm_left_ptr[k] / nml) * log(self.pm_left_ptr[k] / nml)
            if self.pm_right_ptr[k] > 0:
                H_right += -(self.pm_right_ptr[k] / nmr) * log(self.pm_right_ptr[k] / nmr)

        e1 = (nml / self.n_samples) * H_left
        e2 = (nmr / self.n_samples) * H_right
        return e1 + e2


cdef class RegressionCriterion(Criterion):
    """Abstract criterion for regression.

    Attributes
    ----------
    sum_left : double
        The sum of the samples left of the SP.
    sum_right : double
        The sum of the samples right of the SP.
    nml : int
        number of samples left of splitting point.
    nmr : int
        number of samples right of splitting point.
    """

    cdef double sum_left
    cdef double sum_right
    cdef int b

    cdef np.float64_t *labels
    cdef np.int8_t *sample_mask
    cdef int *sorted_features_i

    cdef int nml, nmr, K, n_total_samples, n_samples
    
    def __init__(self):
        self.sum_left = 0.0
        self.sum_right = 0.0
        self.nml = 0
        self.nmr = 0
        self.b = -1
        self.sorted_features_i = NULL

    cdef void init(self, np.float64_t *labels, np.int8_t *sample_mask,
                   int n_total_samples):
        """Initializes the criterion for a new feature (col of `features`)."""
        cdef int j = 0
        self.nml = 0
        self.nmr = 0
        self.sum_left = 0.0
        self.sum_right = 0.0
        self.labels = labels
        self.sample_mask = sample_mask
        self.b = -1
        self.sorted_features_i = NULL

        for j from 0 <= j < n_total_samples:
            if sample_mask[j] == 0:
                continue
            self.sum_right += labels[j]
            self.nmr += 1
            
        self.n_samples = self.nmr
        self.n_total_samples = n_total_samples

    cdef int update(self, int a, int b, np.float64_t *labels,
                    np.int8_t *sample_mask, np.float64_t *features_i,
                    int *sorted_features_i):
        """Update the mean of left and right branch for docs between
        `sorted_features_i[a]` and `sorted_features_i[b]`. """
        cdef int idx, j
        cdef double val
        ## all samples from a to b-1 are on the left side
        for idx from a <= idx < b:
            j = sorted_features_i[idx]
            if sample_mask[j] == 0:
                continue
            val = labels[j]
            self.sum_right -= val
            self.sum_left += val
            self.nmr -= 1
            self.nml += 1

        # set first idx of right branch
        self.b = b

        # store sorted features ptr for eval.
        self.sorted_features_i = sorted_features_i
            
        return self.nml

    cdef double eval(self):
        pass


cdef class MSE(RegressionCriterion):
    """Mean squared error splitting criterion for regression trees. """

    cdef double eval(self):
        """             
        MSE =  \sum_i (y_i - c0)^2  / N
        
        """
        cdef double mean_left = 0.0
        cdef double mean_right = 0.0
        cdef double var_left = 0.0
        cdef double var_right = 0.0
        cdef int j, idx
        cdef double e1, e2

        mean_left = self.sum_left / self.nml
        mean_right = self.sum_right / self.nmr

        for idx from 0 <= idx < self.n_total_samples:
            j = self.sorted_features_i[idx]
            if self.sample_mask[j] == 0:
                continue
            if idx < self.b:
                var_left += (self.labels[j] - mean_left) * \
                            (self.labels[j] - mean_left)
            else:
                var_right += (self.labels[j] - mean_right) * \
                             (self.labels[j] - mean_right)

        var_left /= self.nml
        var_right /= self.nmr

        e1 = (<double> self.nml) / self.n_samples * var_left
        e2 = (<double> self.nmr) / self.n_samples * var_right

        return e1 + e2


## cpdef np.float64_t eval_miss(np.ndarray[np.float64_t, ndim=1] labels,
##                        np.ndarray[np.float64_t, ndim=1] pm):
##     """
        
##         Misclassification error = (1 - pmk)
            
##     """
##     cdef int n_labels = labels.shape[0]
##     cdef int K = pm.shape[0]    
    
##     cdef int i = 0
##     for i in range(K):
##         pm[i] = 0.
    
##     cdef int value
##     for i in range(n_labels):       
##         value = (int)(labels[i])
##         pm[value] += 1. / n_labels
        
##     cdef np.float64_t H = 1. - pm.max()

##     ## FIXME this is broken!


cdef int smallest_sample_larger_than(int sample_idx, np.float64_t *features_i,
                                     int *sorted_features_i, np.int8_t *sample_mask,
                                     int n_total_samples):
    """Find index in the `sorted_features` matrix for sample
    who's feature `i` value is just about
    greater than those of the sample `sorted_features_i[sample_idx]`.
    Ignore samples where `sample_mask` is false.

    Returns
    -------
    next_sample_idx : int
        The index of the next smallest sample in `sorted_features`
        with different feature value than `sample_idx` .
        I.e. `sorted_features_i[sample_idx] < sorted_features_i[next_sample_idx]`
        -1 if no such element exists.
    """
    cdef int idx = 0, j
    cdef double threshold = -DBL_MAX
    if sample_idx > -1:
        threshold = features_i[sorted_features_i[sample_idx]]
    for idx from sample_idx < idx < n_total_samples:
        j = sorted_features_i[idx]
        if sample_mask[j] == 0:
            continue
        if features_i[j] > threshold:
            return idx
    return -1


def fill_counts(np.ndarray[np.float64_t, ndim=1, mode="c"] counts,
                np.ndarray[np.float64_t, ndim=1, mode="c"] labels,
                np.ndarray sample_mask):
    """The same as np.bincount but casts elements in `labels` to integers.

    Parameters
    ----------
    counts : ndarray, shape = K
        The bin counts to be filled.
    labels : ndarray, shape = n_total_samples
        The labels.
    sample_mask : ndarray, shape=n_total_samples, dtype=bool
        The samples to be considered.
    """
    cdef int j, c
    cdef int n_total_samples = labels.shape[0]
    cdef np.int8_t *sample_mask_ptr = <np.int8_t *>sample_mask.data
    for j from 0 <= j < n_total_samples:
        if sample_mask_ptr[j] == 0:
            continue
        c = <int>(labels[j])
        counts[c] += 1


def _find_best_split(np.ndarray sample_mask,
                     np.float64_t parent_split_error,
                     np.ndarray[np.float64_t, ndim=2, mode="fortran"] features,
                     np.ndarray[np.int32_t, ndim=2, mode="fortran"] sorted_features,
                     np.ndarray[np.float64_t, ndim=1, mode="c"] labels,
                     Criterion criterion, int K, int n_samples):
    """
    Parameters
    ----------
    sample_mask : ndarray, shape (n_samples,), dtype=bool
        A mask for the samples to be considered. Only samples `j` for which
        sample_mask[j] != 0 are considered.
    parent_split_error : np.float64
        The split error (aka criterion) of the parent node.
    features : ndarray, shape (n_samples, n_features), dtype=np.float64
        The feature values (aka `X`).
    sorted_features : ndarray, shape (n_samples, n_features)
        Argsort of cols of `features`. `sorted_features[0,j]` gives the example
        index of the smallest value of feature j.
    labels : ndarray, shape (n_samples,), dtype=float64
        The labels.
    criterion : Criterion
        The criterion function to be minimized.
    K : int
        The number of classes - for regression use 0.
    n_samples : int
        The number of samples in the current sample_mask (i.e. `sample_mask.sum()`).

    Returns
    -------
    best_i : int
        The split feature or -1 if criterion not smaller than `parent_split_error`.
    best_t : np.float64_t
        The split threshold
    best_error : np.float64_t
        The error (criterion) of the split.
    """
    cdef int n_total_samples = features.shape[0]
    cdef int n_features = features.shape[1]
    cdef int i, best_i = -1, best_nml, nml = 0, a, b
    cdef double t, error, best_error, best_t

    # pointer access to ndarray data
    cdef np.float64_t *labels_ptr = <np.float64_t *>labels.data
    cdef np.float64_t *features_i = NULL
    cdef int *sorted_features_i = NULL
    cdef np.int8_t *sample_mask_ptr = <np.int8_t *>sample_mask.data

    # Compute the column strides (inc in pointer elem to get from col i to i+1)
    # for `features` and `sorted_features`
    cdef int features_elem_stride = features.strides[0]
    cdef int features_col_stride = features.strides[1]
    cdef int features_stride = features_col_stride / features_elem_stride
    cdef int sorted_features_elem_stride = sorted_features.strides[0]
    cdef int sorted_features_col_stride = sorted_features.strides[1]
    cdef int sorted_features_stride = sorted_features_col_stride / sorted_features_elem_stride

    for i from 0 <= i < n_features:
        # get i-th col of features and features_sorted
        features_i = (<np.float64_t *>features.data) + features_stride * i
        sorted_features_i = (<int *>sorted_features.data) + sorted_features_stride * i

        # init the criterion for this feature
        criterion.init(labels_ptr, sample_mask_ptr, n_total_samples)

        # get sample in mask with smallest value for i-th feature
        a = smallest_sample_larger_than(-1, features_i, sorted_features_i,
                                        sample_mask_ptr, n_total_samples)
        while 1:
            # get sample in mask with val for i-th feature just larger than `a`
            b = smallest_sample_larger_than(a, features_i, sorted_features_i,
                                            sample_mask_ptr, n_total_samples)
            # if -1 there's none and we are fin
            if b == -1:
                break
            
            # update criterion for interval [a, b)
            nml = criterion.update(a, b, labels_ptr, sample_mask_ptr, features_i,
                                   sorted_features_i)

            # get criterion value
            error = criterion.eval()
            
            # check if current criterion smaller than parent criterion
            # if this is never true best_i is -1.
            if error < parent_split_error:
                # compute split point
                t = (features_i[sorted_features_i[a]] +
                     features_i[sorted_features_i[b]]) / 2.0
                parent_split_error = error
                best_i = i
                best_t = t
                best_error = error
                best_nml = nml
                
            a = b
    return best_i, best_t, best_error, best_nml

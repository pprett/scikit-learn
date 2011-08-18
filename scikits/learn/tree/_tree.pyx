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
    cdef extern double pow(double base, double exponent)

 
"""
 Classification entropy measures
 
    From Hastie et al. Elements of Statistical Learning, 2009.
         
    If a target is a classification outcome taking on values 0,1,...,K-1
    In node m, representing a region Rm with Nm observations, let
            
       pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)
          
    be the proportion of class k observations in node m   
"""  

cpdef double eval_gini(np.ndarray[np.float64_t, ndim=1] labels, 
                       np.ndarray[np.float64_t, ndim=1] pm):
    """
        
        Gini index = \sum_{k=0}^{K-1} pmk (1 - pmk)
                   = 1 - \sum_{k=0}^{K-1} pmk ** 2
            
    """       
    cdef int n_labels = labels.shape[0]
    cdef int K = pm.shape[0]
    
    cdef int i = 0
    cdef int c
    for i from 0 <= i < K:
        pm[i] = 0.0
    
    for i from 0 <= i < n_labels:
        c = (int)(labels[i])
        pm[c] += 1. / n_labels
    
    cdef double H = 1.
    cdef Py_ssize_t k
    for k in range(K): 
        H -=  pow(pm[k],2) 
         
    return H

cdef double gini(np.ndarray[np.float64_t, ndim=1] pm):
    """
        
        Gini index = \sum_{k=0}^{K-1} pmk (1 - pmk)
                   = 1 - \sum_{k=0}^{K-1} pmk ** 2
            
    """       
    cdef int K = pm.shape[0]
    cdef double H = 1.0
    cdef Py_ssize_t k
    for k from 0 <= k < K:
        H -= pm[k] * pm[k] 
    return H


cdef void fill_pm(np.ndarray[np.float64_t, ndim=1] pm_left,
                  np.ndarray[np.float64_t, ndim=1] pm_right,
                  double *nll, double *nlr, 
                  np.ndarray[np.float64_t, ndim=1] labels,
                  np.ndarray[np.float64_t, ndim=1] features_at_i,
                  double t, int n_samples, int K):
    cdef double *labels_data = <double *> labels.data
    cdef double *pm_left_data = <double *> pm_left.data
    cdef double *pm_right_data = <double *> pm_right.data
    # assert features_at_i.shape == labels.shape
    cdef int c = 0, j = 0, count
    
    for c from 0 <= c < K:
        pm_left_data[c] = 0.0
        pm_right_data[c] = 0.0

    for j from 0 <= j < n_samples:
        c = <int>(labels_data[j])
        if features_at_i[j] < t:
            pm_left_data[c] += 1.0
            count += 1
        else:
            pm_right_data[c] += 1.0
            
    nll[0] = count
    nlr[0] = n_samples - count
    for c from 0 <= c < K:
        pm_left_data[c] /= nll[0]
        pm_right_data[c] /= nlr[0]
    

cpdef double eval_entropy(np.ndarray[np.float64_t, ndim=1] labels,
                          np.ndarray[np.float64_t, ndim=1] pm):
    """
        
        Cross Entropy = - \sum_{k=0}^{K-1} pmk log(pmk)
            
    """
    cdef int n_labels = labels.shape[0]
    cdef int K = pm.shape[0]    
    
    cdef int i = 0
    cdef int c
    cdef double H = 0.0
    
    for i from 0 <= i < K:
        pm[i] = 0.0
    
    for i from 0 <= i < n_labels:
        c = (int)(labels[i])
        pm[c] += 1.0 / n_labels

    for i from 0 <= i < K:
        if pm[i] > 0 :    
            H +=  -pm[i] * log(pm[i])
         
    return H   

cpdef double eval_miss(np.ndarray[np.float64_t, ndim=1] labels,
                       np.ndarray[np.float64_t, ndim=1] pm):
    """
        
        Misclassification error = (1 - pmk)
            
    """
    cdef int n_labels = labels.shape[0]
    cdef int K = pm.shape[0]    
    
    cdef int i = 0
    for i in range(K):
        pm[i] = 0.
    
    cdef int value
    for i in range(n_labels):       
        value = (int)(labels[i])
        pm[value] += 1. / n_labels
        
    cdef double H = 1. - pm.max()

    ## FIXME this is broken!
    
"""
 Regression entropy measures
 
"""      
cpdef double eval_mse(np.ndarray[np.float64_t, ndim=1] labels,
                      np.ndarray[np.float64_t, ndim=1] pm):
    """             
        MSE =  \sum_i (y_i - c0)^2  / N
        
        pm is a redundant argument (intentional).
    """     
    cdef int n_labels = labels.shape[0]

    cdef float c0
    cdef Py_ssize_t i = 0
    for i in range(n_labels):
        c0 += labels[i]
    c0 /= n_labels

    cdef double H = 0.
    for i in range(n_labels):
        H += pow(labels[i] - c0, 2) 
    H /= n_labels
        
    return H


def _find_best_split(np.ndarray[np.float64_t, ndim=2, mode="c"] features,
                     np.ndarray[np.float64_t, ndim=1, mode="c"] labels,
                     criterion, int K):
    """
    Parameters
    ----------
    K : int
        The number of classes - for regression use 0.
    """
    # K = int(np.abs(labels.max())) + 1
    cdef int n_samples = features.shape[0]
    cdef int n_features = features.shape[1]

    cdef np.ndarray[np.float64_t, ndim=1] pm_left = np.zeros((K,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] pm_right = np.zeros((K,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] features_at_i = None
    cdef np.ndarray[np.float64_t, ndim=1] domain_i = None
    cdef double split_error = eval_gini(labels, pm_left)
    cdef int i, j, domain_size, best_i = -1
    cdef np.float64_t d1, d2, e1, e2, t, error, best_error, best_t, nll = 0.0, nlr = 0.0
    for i from 0 <= i < n_features:
        features_at_i = features[:, i]
        domain_i = np.unique(features_at_i)
        domain_size = domain_i.shape[0]
        for j from 0 <= j < domain_size - 1:
            d1 = domain_i[j]
            d2 = domain_i[j+1]
            t = (d1 + d2) / 2.0
            fill_pm(pm_left, pm_right, &nll, &nlr,
                    labels, features_at_i, t, n_samples, K)
            e1 = nll / n_samples * gini(pm_left)
            e2 = nlr / n_samples * gini(pm_right)
            error = e1 + e2
            if error < split_error:
                split_error = error
                best_i = i
                best_t = t
                best_error = error
                
    return best_i, best_t, best_error

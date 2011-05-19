from .base import BaseEstimator
from .utils import safe_asanyarray
import numpy as np
from scipy import sparse

from .avgperceptron_fast import fit_sparse


class AveragedPerceptron(BaseEstimator):
    '''

    Parameters
    ----------
    averaged : bool, default True
        Train as averaged perceptron.
    n_iter : int, default 1
        Number of iterations to perform per (partial) training set.
    shuffle : bool, default False
        Randomize input sequence between iterations.
    '''

    def __init__(self, averaged=False, n_iter=1, shuffle=False, seed=13,
                 verbose=0):
        self.averaged = averaged
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y, **params):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_params(**params)

        # check only y because X might be dense or sparse
        X = safe_asanyarray(X, order='C')
        y = np.asanyarray(y, dtype=np.float64, order='C')

        n_samples, n_features = X.shape

        if n_samples != y.shape[0]:
            raise ValueError("Shapes of X and y do not match.")

        # sort in asc order; largest class id is positive class
        self.classes = np.unique(y)
        n_classes = self.classes.shape[0]

        # Allocate datastructures from input arguments
        self._allocate_parameter_mem(n_classes, n_features)

        # delegate to concrete training procedure
        if n_classes > 1:
            self._fit(X, y)
        else:
            raise ValueError("The number of class labels must be "
                             "greater than one.")
        # return self for chaining fit and predict calls
        return self

    def _allocate_parameter_mem(self, n_classes, n_features):
        """Allocate mem for parameters; initialize if provided."""
        self.w = np.zeros((n_classes, n_features),
                              dtype=np.float64, order="C")

        self.wbar = np.zeros((n_classes, n_features),
                              dtype=np.float64, order="C")

    def _set_coef(self, coef):
        self.coef_ = coef
        if coef is None:
            self.sparse_coef_ = None
        else:
            # sparse representation of the fitted coef for the predict method
            self.sparse_coef_ = sparse.csr_matrix(coef)

    def _fit(self, X, y):
        if sparse.issparse(X):
            X = sparse.csr_matrix(X)

            # get sparse matrix datastructures
            X_data = np.array(X.data, dtype=np.float64, order="C")
            X_indices = np.array(X.indices, dtype=np.int32, order="C")
            X_indptr = np.array(X.indptr, dtype=np.int32, order="C")
            w, wbar = fit_sparse(X_data, X_indices, X_indptr, y, self.w,
                                 self.wbar, self.n_iter, self.averaged,
                                 self.shuffle, self.verbose, self.seed)
            if self.averaged:
                coef = wbar
            else:
                coef = w
            self._set_coef(coef)
        else:
            raise NotImplementedError("Dense case not implemented yet.")

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : array or scipy.sparse matrix of shape [n_samples, n_features]
           Whether the numpy.array or scipy.sparse matrix is accepted dependes
           on the actual implementation

        Returns
        -------
        array, shape = [n_samples]
           Array containing the predicted class labels.
        """
        X = safe_asanyarray(X, order='C')
        scores = self.decision_function(X)
        indices = scores.argmax(axis=1)
        return self.classes[np.ravel(indices)]

    def decision_function(self, X):
        """Predict signed 'distance' to the hyperplane (aka confidence score).

        Parameters
        ----------
        X : scipy.sparse matrix of shape [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples] if n_classes == 2 else [n_samples,n_classes]
          The signed 'distances' to the hyperplane(s).
        """
        # np.dot only works correctly if both arguments are sparse matrices
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        scores = np.asarray(np.dot(X, self.sparse_coef_.T).todense())
        return scores

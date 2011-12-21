"""
Algorithms for Boosting:
- Functional Gradient Descent
"""

# Authors: James Bergstra
# License: BSD3

import numpy as np

from ..utils import safe_asarray
from ..ensemble import BaseEnsemble
from ..base import RegressorMixin


class LossFunction(object):
    """Abstract base class for various loss functions."""

    def init_estimator(self, X, y):
        pass

    def __call__(self, y, pred):
        pass

    def negative_gradient(self, y, pred):
        """Compute the negative gradient."""
        pass


class LeastSquaresError(LossFunction):
    """Loss function for least squares (LS) estimation.
    Terminal regions need not to be updated for least squares. """

    def init_estimator(self):
        return MeanPredictor()

    def __call__(self, y, pred):
        return np.mean((y - pred) ** 2.0)

    def negative_gradient(self, y, pred):
        return y - pred


class LeastAbsoluteError(LossFunction):
    """Loss function for least absolute deviation (LAD) regression. """

    def init_estimator(self):
        return MedianPredictor()

    def __call__(self, y, pred):
        return np.abs(y - pred).mean()

    def negative_gradient(self, y, pred):
        return np.sign(y - pred)


class BinomialDeviance(LossFunction):
    """Binomial deviance loss function for binary classification."""

    def init_estimator(self):
        return ClassPriorPredictor()

    def __call__(self, y, pred):
        """Compute the deviance (= negative log-likelihood). """
        ## return -2.0 * np.sum(y * pred -
        ##                      np.log(1.0 + np.exp(pred))) / y.shape[0]

        # logaddexp(0, v) == log(1.0 + exp(v))
        return -2.0 * np.sum(y * pred -
                             np.logaddexp(0.0, pred)) / y.shape[0]

    def negative_gradient(self, y, pred):
        return y - 1.0 / (1.0 + np.exp(-pred))


class FunctionalGradient(object):
    def __init__(self, loss, X, y):
        self.loss = loss
        self.X = X
        self.y = y
        self.residual = np.array(y)   # copies, ensures array

    def current_Xy(self):
        return self.X, self.residual

    def update(self, prediction):
        self.residual = self.loss.negative_gradient(self.residual, prediction)


class FitNIter(object):
    """
    Iterations (self.next()) implement one round of functional gradient
    boosting.


    """
    def __init__(self, ensemble, fg, n_iters):
        self.ensemble = ensemble
        self.fg = fg
        self.n_iters = n_iters

    def __iter__(self):
        return self

    def next(self):
        if self.n_iters == len(self.ensemble.estimators_):
            raise StopIteration
        base = self.ensemble._make_estimator()
        X, y = self.fg.current_Xy()
        base.fit(X, y)
        self.fg.update(base.predict(X))
        return self


class GradientBoostedRegressor(BaseEnsemble, RegressorMixin):
    """
    Regression Boosting via functional gradient descent.

    The algorithm is to construct a regression ensemble by using a "base
    estimator" to repeatedly fit residual training error. So for example, the
    first iteration fits some function f() to the original (X, y) training
    data, and the second iteration fits some g() to (X, y - f(X)), the third
    iterations fits some h() to (X y - f(X) - g(X)), and so on.  The final
    ensemble is f() + g() + h() + ...

    This procedure is equivalent to functional gradient descent when the the
    training objective is to minimize mean squared error (MSE).

    For more information see e.g.:
    J. H. Friedman (2002). "Stochastic Gradient Boosting",
    Computational Statistics & Data Analysis.

    TODO: Mason has a good paper on the subject as well.
    """

    def __init__(self, base_estimator, n_estimators,
            loss=LeastSquaresError):
        super(GradientBoostedRegressor, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)
        self.loss = loss

    def fit_iter(self, X, y):
        """Create a fitting iterator for training set X, y.

        See class FitIter().
        """
        if 'int' in str(y.dtype):
            raise TypeError('Regression of int-valued targets is ambiguous'
                    '. Please cast to float if you want to train using a '
                    'regression criterion.')
        if issubclass(self.loss, LossFunction):
            loss = self.loss()
        else:
            loss = self.loss
        return FitNIter(
                ensemble=self,
                fg=FunctionalGradient(loss, X, y),
                n_iters=self.n_estimators)

    def fit(self, X, y):
        """Build a regression ensemble by funtional gradient boosting.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        Return
        ------
        self : object
            Returns self.
        """
        for _ in self.fit_iter(X, y):
            pass
        return self

    def predict(self, X):
        """Return the prediction for array-like X.
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            Test samples.

        Return
        ------
        prediction : numpy array of shape = [n_samples]
            Test predictions.

        """
        rval = self.estimators_[0].predict(X)
        for estimator in self.estimators_[1:]:
            pred_i = estimator.predict(X) 
            rval += pred_i
        return rval

"""Weight Boosting

This module contains weight boosting estimators for both classification and
regression.

The module structure is the following:

- The ``BaseAdaBoost`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ the the concrete ``LossFunction`` used.

- ``AdaBoostClassifier`` implements adaptive boosting (AdaBoost-SAMME) for
  classification problems.

- ``AdaBoostRegressor`` implements adaptive boosting (AdaBoost.R2) for
  classification problems.
"""

# Authors: Noel Dawe
# License: BSD Style

import math

import numpy as np

from .base import BaseEnsemble
from ..base import ClassifierMixin, RegressorMixin
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor


__all__ = [
    'AdaBoostClassifier',
    'AdaBoostRegressor',
]


class BaseWeightBoosting(BaseEnsemble):
    """Abstract base class for weight boosting.

    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the ensemble is built.

    n_estimators : integer, optional (default=10)
        The maximum number of estimators at which boosting is terminated.

    learn_rate : float, optional (default=0.5)
        learning rate shrinks the contribution of each classifier by
        `learn_rate`. There is a trade-off between learn_rate and n_estimators.
    """
    def __init__(self, base_estimator=None,
                       n_estimators=10,
                       learn_rate=.5,
                       compute_importances=False):

        if base_estimator is None:
            if isinstance(self, ClassifierMixin):
                base_estimator = DecisionTreeClassifier()
            else:
                base_estimator = DecisionTreeRegressor()
        elif (isinstance(self, ClassifierMixin)
              and not isinstance(base_estimator, ClassifierMixin)):
            raise TypeError("base_estimator must be a "
                            "subclass of ClassifierMixin")
        elif (isinstance(self, RegressorMixin)
              and not isinstance(base_estimator, RegressorMixin)):
            raise TypeError("base_estimator must be a "
                            "subclass of ClassifierMixin")

        if learn_rate <= 0:
            raise ValueError("learn_rate must be greater than 0")

        self.boost_weights_ = list()
        self.errs_ = list()
        self.learn_rate = learn_rate
        self.compute_importances = compute_importances
        self.feature_importances_ = None
        if compute_importances:
            try:
                base_estimator.compute_importances = True
            except AttributeError:
                raise AttributeError("Unable to compute feature importances "
                                     "since base_estimator does not have a "
                                     "compute_importances attribute")
        super(BaseWeightBoosting, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like of shape = [n_samples]
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights

        Returns
        -------
        self : object
            Returns self.
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        if isinstance(self, ClassifierMixin):
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            y = np.searchsorted(self.classes_, y)

        if sample_weight is None:
            # initialize weights to 1/N
            sample_weight = np.ones(X.shape[0], dtype=np.float32) \
                / X.shape[0]
        else:
            sample_weight = np.copy(sample_weight)
            # normalize
            sample_weight /= sample_weight.sum()

        # clear any previous fit results
        self.estimators_ = list()
        self.boost_weights_ = list()
        self.errs_ = list()

        for boost in xrange(self.n_estimators):
            estimator = self._make_estimator()
            if hasattr(estimator, 'fit_predict'):
                # optim for estimators that are able to save redundant
                # computations when calling fit + predict
                # on the same input X
                p = estimator.fit_predict(X, y,
                        sample_weight=sample_weight)
            else:
                p = estimator.fit(X, y,
                        sample_weight=sample_weight).predict(X)

            sample_weight = self.boost(sample_weight, p, y,
                    boost == self.n_estimators - 1)
            # early termination
            if sample_weight is None:
                break
            if boost < self.n_estimators - 1:
                # normalize
                sample_weight /= sample_weight.sum()

        # sum the importances
        try:
            if self.compute_importances:
                norm = sum(self.boost_weights_)
                self.feature_importances_ = \
                    sum(weight * clf.feature_importances_ for
                      weight, clf in zip(self.boost_weights_, self.estimators_)) \
                    / norm
        except AttributeError:
            raise AttributeError("Unable to compute feature importances "
                                 "since base_estimator does not have a "
                                 "feature_importances_ attribute")

        return self


class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):
    """An AdaBoosted classifier.

    An AdaBoosted classifier is a meta estimator that begins by fitting a
    classifier on a dataset and then fits additional copies of the classifer
    on the same dataset where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers
    focus more on difficult cases.

    See also
    --------
    DecisionTreeClassifier

    References
    ----------
    .. [1] Yoav Freund, Robert E. Schapire. "A Decision-Theoretic
           Generalization of on-Line Learning and an Application
           to Boosting", 1995
    .. [2] Ji Zhu, Hui Zou, Saharon Rosset, Trevor Hastie.
           "Multi-class AdaBoost" 2009
    """

    def boost(self, sample_weight, y_predict, y_true, is_last):
        """Implement a single boost

        Perform a single boost according to the multi-class SAMME algorithm and
        return the updated sample weights.

        Parameters
        ----------

        sample_weight : array-like of shape = [n_samples]
            The current sample weights.

        y_predict : array-like of shape = [n_samples]
            The predicted class labels.

        y_true : array-like of shape = [n_samples]
            The true class labels.

        is_last : Boolean
            True if this is the last boost.
        """
        # instances incorrectly classified
        incorrect = (y_predict != y_true).astype(np.int32)
        # error fraction
        err = (sample_weight * incorrect).sum() / sample_weight.sum()

        # stop if classification is perfect
        if err <= 0:
            self.boost_weights_.append(1.)
            self.errs_.append(err)
            return None

        # stop if the error is at least as bad as random guessing
        if err >= 1. - (1. / self.n_classes_):
            self.estimators_.pop(-1)
            return None

        # boost weight using multi-class AdaBoost SAMME alg
        alpha = self.learn_rate * (math.log((1. - err) / err) +
                             math.log(self.n_classes_ - 1.))

        self.boost_weights_.append(alpha)
        self.errs_.append(err)
        if not is_last:
            sample_weight *= np.exp(alpha * incorrect)
        return sample_weight

    def predict(self, X, limit=-1):
        """Predict class for X.

        The predicted class of an input sample is computed as the weighted
        mean prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        limit : int, optional (default=-1)
            Use only the first N=limit classifiers for the prediction. This is
            useful for grid searching the n_estimators parameter since it is not
            necessary to fit separately for all choices of n_estimators, but
            only the highest n_estimators.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        return self.classes_.take(
            np.argmax(self.predict_proba(X, limit=limit), axis=1),  axis=0)

    def staged_predict(self, X, limit=-1):
        """Predict class for X.

        The predicted class of an input sample is computed as the weighted
        mean prediction of the classifiers in the ensemble.
        This method allows monitoring (i.e. determine error on testing set)
        after each boost. See examples/ensemble/plot_boost_error.py

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        limit : int, optional (default=-1)
            See docs above for the predict method

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        for proba in self.staged_predict_proba(X, limit=limit):
            yield self.classes_.take(np.argmax(proba, axis=1),  axis=0)

    def predict_proba(self, X, limit=-1):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities
        of the classifiers in the ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        limit : int, optional (default=-1)
            See docs above for the predict method

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. Classes are
            ordered by arithmetical order.
        """
        X = np.atleast_2d(X)
        p = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)

        norm = sum(self.boost_weights_)
        for i, (alpha, estimator) in enumerate(
                zip(self.boost_weights_, self.estimators_)):
            if i == limit:
                break
            if self.n_classes_ == estimator.n_classes_:
                p += alpha * estimator.predict_proba(X)
            else:
                proba = alpha * estimator.predict_proba(X)
                for j, c in enumerate(estimator.classes_):
                    p[:, c] += proba[:, j]
        if norm > 0:
            p /= norm
        return p

    def staged_predict_proba(self, X, limit=-1):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities
        of the classifiers in the ensemble.
        This method allows monitoring (i.e. determine error on testing set)
        after each boost. See examples/ensemble/plot_boost_error.py

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        limit : int, optional (default=-1)
            See docs above for the predict method

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. Classes are
            ordered by arithmetical order.
        """
        X = np.atleast_2d(X)
        p = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)

        norm = 0.
        for i, (alpha, estimator) in enumerate(
                zip(self.boost_weights_, self.estimators_)):
            if i == limit:
                break
            if self.n_classes_ == estimator.n_classes_:
                p += alpha * estimator.predict_proba(X)
            else:
                proba = alpha * estimator.predict_proba(X)
                for j, c in enumerate(estimator.classes_):
                    p[:, c] += proba[:, j]
            norm += alpha
            yield p / norm if norm > 0 else p

    def predict_log_proba(self, X, limit=-1):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities
        of the classifiers in the ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        limit : int, optional (default=-1)
            See docs above for the predict method

        Returns
        -------
        p : array of shape = [n_samples]
            The class log-probabilities of the input samples. Classes are
            ordered by arithmetical order.
        """
        return np.log(self.predict_proba(X, limit=limit))


class AdaBoostRegressor(BaseWeightBoosting, RegressorMixin):
    """An AdaBoosted regressor.

    An AdaBoosted regressor is a meta estimator that begins by fitting a
    regressor on a dataset and then fits additional copies of the regressor
    on the same dataset where the weights of instances are adjusted
    according to the error of the prediction such that subsequent regressors
    focus more on difficult cases.

    See also
    --------
    DecisionTreeRegressor

    References
    ----------
    .. [1] Yoav Freund, Robert E. Schapire. "A Decision-Theoretic
           Generalization of on-Line Learning and an Application
           to Boosting", 1995
    .. [2] Drucker. AdaBoost.R2, 1997
    """
    def boost(self, sample_weight, y_predict, y_true, is_last):
        """Implement a single boost

        Perform a single boost according to the AdaBoost.R2 algorithm and
        return the updated sample weights.

        Parameters
        ----------

        sample_weight : array-like of shape = [n_samples]
            The current sample weights.

        y_predict : array-like of shape = [n_samples]
            The predicted class labels.

        y_true : array-like of shape = [n_samples]
            The true class labels.

        is_last : Boolean
            True if this is the last boost.
        """
        err_vect = np.abs(y_predict - y_true)
        err_vect /= err_vect.max()

        err = (sample_weight * err_vect).sum()

        # stop if fit is perfect
        if err == 0:
            self.boost_weights_.append(1.)
            self.errs_.append(err)
            return None

        # stop if the error is at least as bad as random guessing
        if err >= .5:
            self.estimators_.pop(-1)
            return None

        beta = err / (1. - err)

        # boost weight using AdaBoost.R2 alg
        alpha = self.learn_rate * math.log(1. / beta)

        self.boost_weights_.append(alpha)
        self.errs_.append(err)
        if not is_last:
            sample_weight *= np.power(beta, (1. - err_vect) * self.learn_rate)
        return sample_weight

    def predict(self, X, limit=-1):
        """Predict class for X.

        The predicted class of an input sample is computed as the weighted
        mean prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        limit : int, optional (default=-1)
            Use only the first N=limit classifiers for the prediction. This is
            useful for grid searching the n_estimators parameter since it is not
            necessary to fit separately for all choices of n_estimators, but
            only the highest n_estimators.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        X = np.atleast_2d(X)
        p = np.zeros(X.shape[0], dtype=np.float64)

        norm = sum(self.boost_weights_)
        for i, (alpha, estimator) in enumerate(
                zip(self.boost_weights_, self.estimators_)):
            if i == limit:
                break
            p += alpha * estimator.predict(X)
        if norm > 0:
            p /= norm
        return p

    def staged_predict(self, X, limit=-1):
        """Predict class for X.

        The predicted class of an input sample is computed as the weighted
        mean prediction of the classifiers in the ensemble.
        This method allows monitoring (i.e. determine error on testing set)
        after each boost. See examples/ensemble/plot_boost_error.py

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        limit : int, optional (default=-1)
            See docs above for the predict method

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        X = np.atleast_2d(X)
        p = np.zeros(X.shape[0], dtype=np.float64)

        norm = 0.
        for i, (alpha, estimator) in enumerate(
                zip(self.boost_weights_, self.estimators_)):
            if i == limit:
                break
            p += alpha * estimator.predict(X)
            norm += alpha
            yield p / norm if norm > 0 else p

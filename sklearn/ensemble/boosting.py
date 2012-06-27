"""Boosted classifiers

The module structure is the following:

- The ``BoostedClassifier`` class implements a ``fit`` method
  capable of boosting any classifier
"""

# Authors: Noel Dawe
# License: BSD 3

import numpy as np
from .base import BaseEnsemble
from ..base import ClassifierMixin
from ..tree import DecisionTreeClassifier
import math


__all__ = ['AdaBoostClassifier']


class AdaBoostClassifier(BaseEnsemble, ClassifierMixin):
    """An AdaBoosted classifier.

    An AdaBoosted classifier is a meta estimator that begins by fitting a
    classifier on a dataset and then fits additional copies of the classifer
    on the same dataset where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers
    focus more on difficult cases.

    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the ensemble is built.

    n_estimators : integer, optional (default=10)
        The maximum number of estimators at which boosting is terminated.

    beta : float, optional (default=.5)
        Scale boost weights. A low/high value corresponds to
        a slow/fast learning rate.

    fit_params : dict, optional
        parameters to pass to the fit method of the base estimator

    Notes
    -----
    .. [1] Yoav Freund, Robert E. Schapire. "A Decision-Theoretic
           Generalization of on-Line Learning and an Application
           to Boosting", 1995
    .. [2] Ji Zhu, Hui Zou, Saharon Rosset, Trevor Hastie.
           "Multi-class AdaBoost" 2009

    See also
    --------
    DecisionTreeClassifier
    """
    def __init__(self, base_estimator=None,
                       n_estimators=10,
                       beta=.5,
                       compute_importances=False):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier()
        elif not isinstance(base_estimator, ClassifierMixin):
            raise TypeError("base_estimator must be a "
                            "subclass of ClassifierMixin")

        super(AdaBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)

        if beta <= 0:
            raise ValueError("beta must be positive and non-zero")

        self.boost_weights_ = list()
        self.errs_ = list()
        self.beta = beta
        self.compute_importances = compute_importances
        self.feature_importances_ = None
        if compute_importances:
            try:
                self.base_estimator.compute_importances = True
            except AttributeError:
                raise AttributeError("Unable to compute feature importances "
                                     "since base_estimator does not have a "
                                     "compute_importances attribute")

    def fit(self, X, y, sample_weight=None, **params):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights

        params : dict
            extra keyword arguments passed to the fit method of
            base_estimator.

        Returns
        -------
        self : object
            Returns self.
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        y = np.searchsorted(self.classes_, y)

        if sample_weight is None:
            # initialize weights to 1/N
            sample_weight = np.ones(X.shape[0], dtype=np.float32) \
                / X.shape[0]
        else:
            sample_weight = np.copy(sample_weight)

        # clear any previous fit results
        self.estimators_ = list()
        self.boost_weights_ = list()
        self.errs_ = list()

        # boost the estimator using AdaBoost with the SAMME modification
        # for multi-class problems
        for boost in xrange(self.n_estimators):
            estimator = self._make_estimator()
            if hasattr(estimator, 'fit_predict'):
                # optim for estimators that are able to save redundant
                # computations when calling fit + predict
                # on the same input X
                p = estimator.fit_predict(X, y,
                                          sample_weight=sample_weight,
                                          **params)
            else:
                p = estimator.fit(X, y, sample_weight=sample_weight,
                                  **params).predict(X)
            # instances incorrectly classified
            incorrect = (p != y).astype(np.int32)
            # error fraction
            err = (sample_weight * incorrect).sum() / sample_weight.sum()
            # stop if classification is perfect
            if err <= 0:
                self.boost_weights_.append(1.)
                self.errs_.append(err)
                break
            # stop if the error is at least as bad as random guessing
            if err >= 1. - (1. / self.n_classes_):
                self.estimators_.pop(-1)
                break
            # boost weight using multi-class AdaBoost SAMME alg
            alpha = self.beta * (math.log((1. - err) / err) +
                                 math.log(self.n_classes_ - 1.))
            self.boost_weights_.append(alpha)
            self.errs_.append(err)
            if boost < self.n_estimators - 1:
                sample_weight *= np.exp(alpha * incorrect)
                # normalize
                sample_weight *= X.shape[0] / sample_weight.sum()

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

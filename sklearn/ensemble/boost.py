import numpy as np
from .base import BaseEnsemble
from ..base import ClassifierMixin, RegressorMixin
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
import math


class AdaBoostClassifier(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None,
                       n_estimators=10,
                       beta=.5,
                       two_class_cont=False,
                       two_class_threshold=0.,
                       **params):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(**params)

        BaseEnsemble.__init__(self,
            base_estimator=base_estimator,
            n_estimators=n_estimators)

        if beta <= 0:
            raise ValueError("Beta must be positive and non-zero")

        self.boost_weights = []
        self.beta = beta
        self.two_class_cont = two_class_cont
        self.two_class_threshold = two_class_threshold

    def fit(self, X, y, sample_weight=None, **kwargs):
        """
        X: list of instance vectors

        y: target values/classes

        sample_weight: sample weights
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        if isinstance(self.base_estimator, ClassifierMixin):
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            y = np.searchsorted(self.classes_, y)

        if not sample_weight:
            # initialize weights to 1/N
            sample_weight = np.ones(X.shape[0], dtype=np.float64)\
                / X.shape[0]
        else:
            sample_weight = np.copy(sample_weight)

        # boost the estimator
        for i in xrange(self.n_estimators):
            estimator = self._make_estimator()
            estimator.fit(X, y, sample_weight, **kwargs)
            # TODO request that classifiers return classification
            # of training sets when fitting
            # which would make the following line unnecessary
            T = estimator.predict(X)
            # instances incorrectly classified
            if self.two_class_cont:
                incorrect = (((T - self.two_class_threshold) * \
                              (y - self.two_class_threshold)) < 0).astype(np.int32)
            else:
                incorrect = (T != y).astype(np.int32)
            # error fraction
            err = np.sum(sample_weight * incorrect) / np.sum(sample_weight)
            # sanity check
            if err == 0:
                self.boost_weights.append(1.)
                break
            elif err >= 0.5:
                if i == 0:
                    self.boost_weights.append(1.)
                break
            # boost weight using multi-class SAMME alg
            alpha = self.beta * (math.log((1 - err) / err) + \
                            math.log(self.n_classes_ - 1))
            self.boost_weights.append(alpha)
            if i < self.n_estimators - 1:
                correct = incorrect ^ 1
                sample_weight *= np.exp(alpha * (incorrect - correct))
        return self

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the majority
        prediction of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        return self.classes_.take(
            np.argmax(self.predict_proba(X), axis=1),  axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. Classes are
            ordered by arithmetical order.
        """
        X = np.atleast_2d(X)
        p = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        norm = 0.
        for alpha, estimator in zip(self.boost_weights, self.estimators_):
            norm += alpha
            if self.n_classes_ == estimator.n_classes_:
                p += alpha * estimator.predict_proba(X)
            else:
                proba = alpha * estimator.predict_proba(X)
                for j, c in enumerate(estimator.classes_):
                    p[:, c] += proba[:, j]
        if norm > 0:
            p /= norm
        return p

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the mean predicted class log-probabilities of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples]
            The class log-probabilities of the input samples. Classes are
            ordered by arithmetical order.
        """
        return np.log(self.predict_proba(X))

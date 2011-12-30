import numpy as np
from .base import BaseEnsemble
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor, \
                   ExtraTreeClassifier, ExtraTreeRegressor
import math


class AdaBoost(BaseEnsemble):

    def __init__(self, base_estimator=None,
                       n_estimators=10,
                       estimator_params=[],
                       beta=.5,
                       discrete_class=False,
                       two_class_threshold=0.):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier()
        super(AdaBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        if boosts < 1:
            raise ValueError(
                "You must specify a number of boosts greater than 0")
        if beta <= 0:
            raise ValueError("Beta must be positive and non-zero")

        self.beta = beta
        self.discrete_class = discrete_class
        self.two_class_threshold = two_class_threshold

    def fit(self, X, y, sample_weight=None):
        """
        X: list of instance vectors
        y: target values/classes
        sample_weight: sample weights

        Notes: currently only binary classification is supported
        I am making the assumption that one class label is
        positive and the other is negative
        """

        if sample_weight is None:
            # initialize weights to 1/N
            sample_weight = np.ones(X.shape[0], dtype=np.float64)\
                / X.shape[0]
        else:
            sample_weight = np.copy(sample_weight)
        # remove any previous ensemble
        self[:] = []
        for i, boost in enumerate(xrange(boosts + 1)):
            estimator = self._make_estimator()
            estimator.fit(X, y, sample_weight=sample_weight)
            # TODO request that classifiers return classification
            # of training sets when fitting
            # which would make the following line unnecessary
            T = estimator.predict(X)
            # instances incorrectly classified
            if discrete_class:
                incorrect = (T != y).astype(np.int32)
            else:
                incorrect = ((T * y) < 0).astype(np.int32)
            print T, y
            print incorrect
            # error fraction
            err = np.sum(sample_weight * incorrect) / np.sum(sample_weight)
            # sanity check
            print i, err
            if err == 0:
                self.append((1., estimator))
                break
            elif err >= 0.5:
                if i == 0:
                    self.append((1., estimator))
                break
            # boost weight
            alpha = beta * math.log((1 - err) / err)
            self.append((alpha, estimator))
            if i < boosts:
                correct = incorrect ^ 1
                sample_weight *= np.exp(alpha * (incorrect - correct))
        return self

    def predict(self, X):

        prediction = np.zeros(X.shape[0], dtype=np.float64)
        norm = 0.
        for alpha, estimator in self:
            prediction += alpha * estimator.predict(X)
            norm += alpha
        if norm > 0:
            prediction /= norm
        return prediction

"""
YET TO BE IMPLEMENTED

class GradientBoost(BaseEnsemble): pass

class StochasticGradientBoost(BaseEnsemble): pass
"""

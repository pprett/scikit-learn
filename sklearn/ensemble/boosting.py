import numpy as np
from .base import BaseEnsemble
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
import math


class AdaBoost(BaseEnsemble):

    def __init__(self, base_estimator=None,
                       n_estimators=10,
                       estimator_params=[],
                       beta=.5,
                       two_class_cont=False,
                       two_class_threshold=0.):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier()
        super(AdaBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        if beta <= 0:
            raise ValueError("Beta must be positive and non-zero")

        self.beta = beta
        self.two_class_cont = two_class_cont
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
        if len(sample_weight) == 0:
            # initialize weights to 1/N
            sample_weight = np.ones(X.shape[0], dtype=np.float64)\
                / X.shape[0]
        else:
            sample_weight = np.copy(sample_weight)
        # determine number of classes
        # remove any previous ensemble
        self[:] = []
        for i, boost in enumerate(xrange(boosts + 1)):
            estimator = self.estimator(**self.params)
            estimator.fit(X, Y, sample_weight, **params)
            # TODO request that classifiers return classification
            # of training sets when fitting
            # which would make the following line unnecessary
            T = estimator.predict(X)
            # instances incorrectly classified
            if self.two_class_cont:
                incorrect = ((T * y) < 0).astype(np.int32)
            else:
                incorrect = (T != y).astype(np.int32)
            # error fraction
            err = np.sum(sample_weight * incorrect) / np.sum(sample_weight)
            # sanity check
            if err == 0:
                self.append((1., estimator))
                break
            elif err >= 0.5:
                if i == 0:
                    self.append((1., estimator))
                break
            # boost weight using multi-class SAMME alg
            alpha = beta * (math.log((1 - err) / err) + \
                            math.log(n_classes - 1))
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

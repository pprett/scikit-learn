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
            #y = np.searchsorted(self.classes_, y)

        if len(sample_weight) == 0:
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
                incorrect = ((T * y) < 0).astype(np.int32)
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
            alpha = beta * (math.log((1 - err) / err) + \
                            math.log(self.n_classes_ - 1))
            self.boost_weights.append(alpha)
            if i < self.n_estimators - 1:
                correct = incorrect ^ 1
                sample_weight *= np.exp(alpha * (incorrect - correct))
        return self

    def predict(self, X):

        prediction = np.zeros(X.shape[0], dtype=np.float64)
        norm = 0.
        for alpha, estimator in zip(self.boost_weights, self):
            prediction += alpha * estimator.predict(X)
            norm += alpha
        if norm > 0:
            prediction /= norm
        return prediction

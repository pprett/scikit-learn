from .base import BaseEnsemble
import numpy as np


class Bagged(BaseEnsemble):

    def __init__(self, base_estimator,
                       n_estimators=10,
                       estimator_params=[],
                       sample_fraction=.5):
        super(Bagged, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        if not 0 < sample_fraction < 1:
            raise ValueError(
                "You must specify sample_fraction between 0 and 1 (exclusive)")
        self.sample_fraction = sample_fraction

    def fit(self, X, y, sample_weight=None):
        """
        X: list of instance vectors
        y: target values/classes
        sample_fraction: fraction of X and y randomly sampled
        baggs: number of sampling/training iterations
        """
        # remove any previous ensemble
        self[:] = []
        for bagg in xrange(baggs):
            estimator = self._make_estimator()
            subsample = np.random.random_sample(sample_weight.shape[0]) \
                < sample_fraction
            estimator.fit(X[subsample], y[subsample],
                sample_weight=sample_weight[subsample], **params)
            self.append(estimator)
        return self

    def predict(self, X):

        if len(self) == 0:
            return None
        prediction = np.zeros(X.shape[0], dtype=np.float64)
        for estimator in self:
            prediction += estimator.predict(X)
        prediction /= len(self)
        return prediction

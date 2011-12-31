"""
Testing for the forest module (sklearn.ensemble.forest).
"""

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import BoostedClassifier
from sklearn import datasets

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
np.random.seed([1])
perm = np.random.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the boston dataset
# and randomly permute it
boston = datasets.load_boston()
perm = np.random.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_classification_toy():
    """Check classification on a toy dataset."""
    # Adaboost Classification
    clf = BoostedClassifier(n_estimators=10)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    assert_equal(10, len(clf))


def test_iris():
    """Check consistency on dataset iris."""
    for c in ("gini", "entropy"):
        # AdaBoost classification
        clf = BoostedClassifier(n_estimators=1, criterion=c)
        clf.fit(iris.data, iris.target)
        score = clf.score(iris.data, iris.target)
        assert score > 0.9, "Failed with criterion %s and score = %f" % (c,
                                                                         score)

'''
def test_boston():
    """Check consistency on dataset boston house prices."""
    for c in ("mse",):
        # Random forest
        """
        clf = RandomForestRegressor(n_estimators=10, criterion=c,
                                    random_state=1)
        clf.fit(boston.data, boston.target)
        score = clf.score(boston.data, boston.target)
        assert score < 3, ("Failed with max_features=None, "
                           "criterion %s and score = %f" % (c, score))

        clf = RandomForestRegressor(n_estimators=10, criterion=c,
                                    max_features=6, random_state=1)
        clf.fit(boston.data, boston.target)
        score = clf.score(boston.data, boston.target)
        assert score < 3, ("Failed with max_features=None, "
                           "criterion %s and score = %f" % (c, score))
        """
'''

def test_probability():
    """Predict probabilities."""
    # AdaBoost classification
    clf = BoostedClassifier(n_estimators=10)
    clf.fit(iris.data, iris.target)
    assert_array_almost_equal(np.sum(clf.predict_proba(iris.data), axis=1),
                              np.ones(iris.data.shape[0]))
    assert_array_almost_equal(clf.predict_proba(iris.data),
                              np.exp(clf.predict_log_proba(iris.data)))


def test_gridsearch():
    """Check that base trees can be grid-searched."""
    # AdaBoost classification
    boost = BoostedClassifier()
    parameters = {'n_estimators': (1, 2),
                  'base_estimator__max_depth': (1, 2)}
    clf = GridSearchCV(boost, parameters)
    clf.fit(iris.data, iris.target)


def test_pickle():
    """Check pickability."""
    import pickle

    # Adaboost
    obj = BoostedClassifier()
    obj.fit(iris.data, iris.target)
    score = obj.score(iris.data, iris.target)
    s = pickle.dumps(obj)

    obj2 = pickle.loads(s)
    assert_equal(type(obj2), obj.__class__)
    score2 = obj2.score(iris.data, iris.target)
    assert score == score2

    """
    obj = RandomForestRegressor()
    obj.fit(boston.data, boston.target)
    score = obj.score(boston.data, boston.target)
    s = pickle.dumps(obj)

    obj2 = pickle.loads(s)
    assert_equal(type(obj2), obj.__class__)
    score2 = obj2.score(boston.data, boston.target)
    assert score == score2
    """


if __name__ == "__main__":
    import nose
    nose.runmodule()

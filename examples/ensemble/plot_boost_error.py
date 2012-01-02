"""
========================================
Testing and Training Error with Boosting
========================================

This examples shows the use of forests of trees to evaluate the importance of
features on an artifical classification task. The red plots are the feature
importances of each individual tree, and the blue plot is the feature importance
of the whole forest.
"""
print __doc__

import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import BoostedClassifier
from sklearn.tree import DecisionTreeClassifier

# Build a classification task
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=8,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=5,
                           random_state=0,
                           shuffle=False)

X_test, X_train = X[:500], X[500:]
y_test, y_train = y[:500], y[500:]

test_errors = []
train_errors = []

for n_estimators in xrange(1, 10):
    # Build a boosted decision tree
    boost = BoostedClassifier(DecisionTreeClassifier(),
                              n_estimators=n_estimators)

    boost.fit(X_train, y_train)
    print boost.n_estimators, len(boost.estimators_)

    y_test_predict = boost.predict(X_test)
    y_train_predict = boost.predict(X_train)

    test_errors.append(sum(y_test_predict != y_test) / float(len(y_test)))
    train_errors.append(sum(y_train_predict != y_train) / float(len(y_train)))


# Plot the feature importances of the trees and of the forest
import pylab as pl
pl.figure()
pl.plot(xrange(1, 10), test_errors, "b", label='test error')
pl.plot(xrange(1, 10), train_errors, "r", label='train error')
pl.legend()
pl.show()

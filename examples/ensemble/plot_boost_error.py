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

from sklearn.datasets import make_blobs
from sklearn.ensemble import BoostedClassifier
from sklearn.tree import DecisionTreeClassifier

import scipy.stats

# Build multivariate normal distribution
n_features = 10
n_samples = 13000
n_split = 3000

cov = np.diag(np.ones(n_features))
mean = np.zeros(n_features)
X = np.random.multivariate_normal(mean, cov, n_samples)

# sort by distance from origin
X = np.array(sorted(list(X), key=lambda x: sum([x_i**2 for x_i in x])))


# label by quantile
y = []
for i, x in enumerate(X):
    if i < n_samples / 3.:
        y.append(1)
    elif i < 2 * n_samples / 3.:
        y.append(2)
    else:
        y.append(3)

y = np.array(y)

# random permutation
perm = np.random.permutation(n_samples)
y = y[perm]
X = X[perm]

X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]

test_errors = []
train_errors = []

bdt = BoostedClassifier(DecisionTreeClassifier(min_split=10),
                        n_estimators=100)

for boost in bdt.fit_generator(X_train, y_train):

    y_test_predict = boost.predict(X_test)
    y_train_predict = boost.predict(X_train)

    test_errors.append(sum(y_test_predict != y_test) / float(len(y_test)))
    train_errors.append(sum(y_train_predict != y_train) / float(len(y_train)))

# Plot the feature importances of the trees and of the forest
import pylab as pl
pl.figure()
pl.plot(xrange(1, len(test_errors)+1), test_errors, "b", label='test error')
pl.plot(xrange(1, len(train_errors)+1), train_errors, "r", label='train error')
pl.legend()
pl.show()

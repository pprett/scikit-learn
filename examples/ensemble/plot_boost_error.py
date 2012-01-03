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
                        n_estimators=10)

for boost in bdt.fit_generator(X_train, y_train):

    y_test_predict = boost.predict(X_test)
    y_train_predict = boost.predict(X_train)

    test_errors.append(sum(y_test_predict != y_test) / float(len(y_test)))
    train_errors.append(sum(y_train_predict != y_train) / float(len(y_train)))


n_trees = xrange(1, len(test_errors)+1)
# Plot the feature importances of the trees and of the forest
import pylab as pl
pl.figure()

pl.subplot(1, 3, 1)
pl.plot(n_trees, test_errors, "b", label='testing sample')
pl.plot(n_trees, train_errors, "r", label='training sample')
pl.legend()
pl.ylabel('Error')
pl.xlabel('Number of Trees')

pl.subplot(1, 3, 2)
pl.plot(n_trees, bdt.boost_weights_, "b")
pl.legend()
pl.ylabel('Boost Weight')
pl.xlabel('Number of Trees')

pl.show()

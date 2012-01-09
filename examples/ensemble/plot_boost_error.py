"""
========================================
Testing and Training Error with Boosting
========================================

This examples shows the use of boosting to improve prediction accuracy.
The error on the test and training sets after each boost is plotted on
the left. The boost weights and error of each tree are also shown.
"""
print __doc__

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.ensemble import BoostedClassifier
from sklearn.tree import DecisionTreeClassifier

import scipy.stats


n_features = 10
n_samples = 13000
n_split = 3000

# Build multivariate normal distribution
cov = np.diag(np.ones(n_features))
mean = np.zeros(n_features)
X = list(np.random.multivariate_normal(mean, cov, n_samples))

# Sort by distance from origin
X.sort(key=lambda x: sum([x_i**2 for x_i in x]))
X = np.array(X)

# Label by quantile.
# The decision boundaries separating successive classes
# are nested concentric ten-dimensional spheres [1].
#
# [1] Ji Zhu, Hui Zou, Saharon Rosset, Trevor Hastie.
#     "Multi-class AdaBoost" 2009
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
                        n_estimators=50)

for i, boost in enumerate(bdt.fit_generator(X_train, y_train)):

    y_test_predict = boost.predict(X_test)
    y_train_predict = boost.predict(X_train)
    print "boost %d: weight: %.3f error: %.3f" % \
          (i, boost.boost_weights_[-1], boost.errs_[-1])

    test_errors.append(sum(y_test_predict != y_test) / float(len(y_test)))
    train_errors.append(sum(y_train_predict != y_train) / float(len(y_train)))

n_trees = xrange(1, bdt.n_estimators + 1)

# Plot the feature importances of the trees and of the forest
import pylab as pl
pl.figure(figsize=(15, 5))

pl.subplot(1, 3, 1)
pl.plot(n_trees, test_errors, "b", label='test')
pl.plot(n_trees, train_errors, "r", label='train')
pl.legend()
pl.ylabel('Error')
pl.xlabel('Number of Trees')

pl.subplot(1, 3, 2)
pl.plot(n_trees, bdt.boost_weights_, "b")
pl.ylabel('Boost Weight')
pl.xlabel('Number of Trees')

pl.subplot(1, 3, 3)
pl.plot(n_trees, bdt.errs_, "b")
pl.ylabel('Error')
pl.xlabel('Tree')

pl.show()

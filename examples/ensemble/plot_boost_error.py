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
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

# Build a boosted decision tree
boost = BoostedClassifier(DecisionTreeClassifier(),
                          compute_importances=True,
                          n_estimators=100)

boost.fit(X, y)
importances = boost.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print "Feature ranking:"

for f in xrange(10):
    print "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])

# Plot the feature importances of the trees and of the forest
import pylab as pl
pl.figure()
pl.title("Feature importances")

for tree in boost.estimators_:
    pl.plot(xrange(10), tree.feature_importances_[indices], "r")

pl.plot(xrange(10), importances[indices], "b")
pl.show()

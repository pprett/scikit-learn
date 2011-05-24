#!/usr/bin/python

import sys
import numpy as np
from pprint import pprint

from scikits.learn.cross_val import StratifiedKFold
from scikits.learn.grid_search import GridSearchCV
from scikits.learn import svm
from scikits.learn.metrics import zero_one_score, f1_score, classification_report
from scikits.learn.preprocessing import Scaler


n_jobs = int(sys.argv[1])

A = np.loadtxt("featset1.csv", delimiter=",")
X = A[:, :-1]
Y = A[:, -1]
C_start, C_end, C_step = -3, 15, 2

train, test = iter(StratifiedKFold(Y, 2, indices=True)).next()

mean, std = X[train].mean(axis=0), X[train].std(axis=0)
std[std == 0.0] = 1.0
X[train] = (X[train] - mean) / std
X[test] =  (X[test] - mean) / std

#print np.unique(Y)
#print X[train].mean(axis=0), X[train].std(axis=0)

# Generate grid search values for C, gamma
C_val = 2. ** np.arange(C_start, C_end + C_step, C_step)


grid_clf = svm.sparse.LinearSVC()
print grid_clf

linear_SVC_params = {'C': C_val}

grid_search = GridSearchCV(grid_clf, linear_SVC_params, n_jobs=n_jobs,
                           score_func=f1_score)
grid_search.fit(X[train], Y[train], cv=StratifiedKFold(Y[train],
                                                       10, indices=True))
y_true, y_pred = Y[test], grid_search.predict(X[test])

print "Classification report for the best estimator: "
print grid_search.best_estimator

#print "Tuned for  with optimal value: %0.3f" % f1_score(y_true, y_pred)
#print classification_report(y_true, y_pred)

print "Grid scores:"
pprint(grid_search.grid_scores_)

print "Best score: %0.3f" % grid_search.best_score


best_parameters = grid_search.best_estimator._get_params()
print "Best C: %0.3f " % best_parameters['C']

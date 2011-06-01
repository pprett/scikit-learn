"""High difference in classifier accuracies with LinearSVC and SVC.


"""
print __doc__

import numpy as np
from functools import partial
from itertools import izip

from scipy import sparse
from scikits.learn import svm
from scikits.learn.grid_search import GridSearchCV
from scikits.learn.metrics.metrics import f1_score, zero_one_score
from scikits.learn.cross_val import StratifiedKFold
from scikits.learn.preprocessing.sparse import Normalizer
from scikits.learn.preprocessing import Scaler

# Initialize default C and gamma values
C_start, C_end, C_step = -3, 4, 2


def tolibsvm(X, Y, fname):
    fd = open(fname, "w+")
    for x, y in izip(X, Y):
        fd.write("%d " % int(y))
        nnz = x.nonzero()[0]
        fd.write(" ".join(["%d:%d" % (idx, v) for v, idx in zip(x[nnz], nnz)]))
        fd.write("\n")
    fd.close()


if __name__ == "__main__":
    cross_fold = 10

    A = np.loadtxt("featset1.csv", delimiter=",")
    X = A[:, :-1]
    Y = A[:, -1]

    ## get rid of empty examples!
    nnz = X.sum(axis=1) > 0.0
    print "%d examples are empty." % (X.shape[0] - nnz.sum())
    X = X[nnz]
    Y = Y[nnz]

    Y = np.array(Y, dtype=np.float64)

    print "X.shape=", X.shape
    print "Y.shape=", Y.shape

    ## get some held-out data (80%)
    offset = int(0.8 * X.shape[0])
    indices = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(indices)
    train = indices[:offset]
    test = indices[offset:]
    
    X_train = X[train]
    Y_train = Y[train]

    X_test = X[test]
    Y_test = Y[test]

    # standardize data - try to comment this out to see the effect!
    #scaler = Scaler()
    #scaler.fit(X[train])
    #X[train] = scaler.transform(X[train], copy=False)
    #X[test] = scaler.transform(X[test], copy=False)

    # Length normalizes rows of X to L1 == 1
    #X = Normalizer().transform(X, copy=True)

    # make X sparse
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)

    # Generate grid search values for C, gamma
    C_val = 2. ** np.arange(C_start, C_end + C_step, C_step)
    tol_val = [0.1, 0.01, 0.001, 0.0001]
    params = {'C': C_val, 'tol': tol_val}

    for clf_class in [svm.sparse.LinearSVC, partial(svm.sparse.SVC,
                                            kernel="linear")]:
        grid_clf = clf_class()
        print "_" * 80
        print grid_clf
        print

        grid_search = GridSearchCV(grid_clf, params, score_func=zero_one_score)
        grid_search.fit(X_train, Y_train,
                        cv=StratifiedKFold(Y_train,
                                           10, indices=True))
        y_true, y_pred = Y_test, grid_search.predict(X_test)

        print "Classification report for the best estimator: "
        print grid_search.best_estimator

        print "Tuned for  with optimal f1-score: %0.3f" % f1_score(y_true,
                                                                   y_pred)

        print "Best score: %0.3f" % grid_search.best_score

        best_parameters = grid_search.best_estimator._get_params()
        print "Best C: %0.6f " % best_parameters['C']
        print "Best tolerance: %0.16f " % best_parameters['tol']

        clf = clf_class(C=best_parameters['C'], tol=best_parameters['tol'])
        print clf
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        print "Accuracy:\t%.4f" % (y_true == y_pred).mean()
        print "F-Score:\t%.4f" % f1_score(y_true, y_pred)

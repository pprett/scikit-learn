import numpy as np

from time import time
from functools import wraps

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.utils import check_random_state

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

import pylab as pl


def repeat(n_repetitions=10):
    def wrap(f):
        def wrapper(*args, **kargs):
            scores = []
            for i in range(n_repetitions):
                scores.append(f(*args, random_state=i, **kargs))
            scores = np.array(scores)
            return scores.mean(axis=0), scores.std(axis=0)
        return wraps(f)(wrapper)
    return wrap


# ignore overflows due to exp
np.seterr(invalid='raise', under='raise', divide='raise', over='ignore')


classification_params = {'loss': 'deviance', 'n_iter': 250,
                         'min_split': 4, 'max_depth': 1,
                         'learn_rate': .6, 'subsample': 1.0}


@repeat()
def bench_random_gaussian(random_state=None):
    rs = check_random_state(random_state)
    shape = (12000, 10)
    X = rs.normal(size=shape).reshape(shape)
    y = ((X ** 2.0).sum(axis=1) > 9.34).astype(np.float64)
    y[y == 0.0] = -1.0

    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    ## stump = DecisionTreeClassifier(max_depth=1)
##     stump.fit(X_train, y_train)
##     error_rate = (1.0 - stump.score(X_test, y_test)) * 100.0

##     print "Stump: %.2f" % error_rate

##     tree = DecisionTreeClassifier(max_depth=20, min_split=5)
##     tree.fit(X_train, y_train)
##     error_rate = (1.0 - tree.score(X_test, y_test)) * 100.0

##     print "Tree: %.2f" % error_rate

    t0 = time()
    gbrt = GradientBoostingClassifier(**classification_params)
    gbrt.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    error_rate = (1.0 - gbrt.score(X_test, y_test))
    test_time = time() - t0
    ## for i in range(10) + [99, 199]:
    ##    print i+1, gbrt.train_deviance[i]
    #avg_splits = np.mean([np.sum(t.children[:,0] != -1) for t in gbrt.trees])
    #print "Avg. splits: %.4f" % avg_splits
    return error_rate, train_time, test_time


def random_gaussian_learning_curve(random_state=None):
    import pylab as pl
    rs = check_random_state(random_state)
    shape = (12000, 10)
    X = rs.normal(size=shape).reshape(shape).astype(np.float32)
    y = ((X ** 2.0).sum(axis=1) > 9.34).astype(np.float64)
    #y[y == 0] = -1

    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    n_iter = 2000
    max_depth = 2
    
    deviance = np.zeros((n_iter,), dtype=np.float64)
    error_rate = np.zeros((n_iter,), dtype=np.float64)
    y_pred = np.empty((X_test.shape[0],), dtype=np.float64)

    def monitor(clf, i):
        if i == 0:
            y_pred[:] = clf.init.predict(X_test)
        else:
            y_pred[:] = clf._predict(X_test, old_pred=y_pred)
        deviance[i] = clf.loss(y_test, y_pred)
        tmp = ((1.0 / (1.0 + np.exp(-y_pred))) >= 0.5).astype(np.float64)
        error_rate[i] = np.mean(tmp != y_test)

    print "Train first model..."
    gbrt = GradientBoostingClassifier(loss='deviance', n_iter=n_iter, min_split=1,
                                      max_depth=max_depth, learn_rate=1.0,
                                      subsample=1.0)
    gbrt.fit(X_train, y_train, monitor=monitor)
    n = deviance.shape[0]

    pl.subplot(122)
    pl.plot(np.arange(n), error_rate, "r-", label="No shrinkage")
    pl.subplot(121)
    pl.plot(np.arange(n), deviance, "r-", label="No shrinkage")

    print "Train second model..."
    gbrt = GradientBoostingClassifier(loss='deviance', n_iter=n_iter, min_split=1,
                                      max_depth=max_depth, learn_rate=0.2,
                                      subsample=1.0)
    gbrt.fit(X_train, y_train, monitor=monitor)
    pl.subplot(122)
    pl.plot(np.arange(n), error_rate, "b-", label="Shrinkage=0.2")
    pl.subplot(121)
    pl.plot(np.arange(n), deviance, "b-", label="Shrinkage=0.2")

    pl.subplot(122)
    pl.title("Stumps Misclassification Error")
    pl.xticks(np.linspace(0, n_iter, 5))
    pl.yticks(np.linspace(0.0, .5, 6))
    pl.xlabel("Boosting Iterations")
    pl.ylabel("Test Set Misclassification Error")
    pl.legend()

    pl.subplot(121)
    pl.title("Stumps Deviance")
    pl.xticks(np.linspace(0, n_iter, 5))
    pl.yticks(np.linspace(0.0, 2.0, 5))
    pl.xlabel("Boosting Iterations")
    pl.ylabel("Test Set Deviance")
    pl.legend()
    
    pl.show()


@repeat()
def bench_spam(random_state=None):
    X = np.loadtxt("/home/pprett/corpora/spam/spambase.data", delimiter=",")
    y = X[:, -1].ravel()
    X = X[:, :-1]
    f = open("/home/pprett/corpora/spam/spambase.names")
    feature_names = np.array([l.split(":")[0] for l in f])

    X, y = shuffle(X, y, random_state=random_state)
    X_test, y_test = X[:1536], y[:1536]
    X_train, y_train = X[1536:], y[1536:]

    clf = GradientBoostingClassifier(**classification_params)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    error_rate = (1.0 - clf.score(X_test, y_test))
    test_time = time() - t0
    ## for i in range(10) + [99, 199]:
    ##     print i+1, clf.train_deviance[i]
    ## feature_importances = clf.feature_importances_
    ## sorted_idx = np.argsort(feature_importances)[-20:]
    ## pos = np.arange(sorted_idx.shape[0]) + .5

    ## pl.barh(pos, feature_importances[sorted_idx], align='center')
    ## pl.yticks(pos, feature_names[sorted_idx])
    ## pl.ylabel('Relative influence')
    ## pl.show()
    return error_rate, train_time, test_time


@repeat(1)
def bench_madelon(random_state=None):
    X_train = np.loadtxt("/home/pprett/corpora/madelon/madelon_train.data")
    y_train = np.loadtxt("/home/pprett/corpora/madelon/madelon_train.labels")
    X_test = np.loadtxt("/home/pprett/corpora/madelon/madelon_valid.data")
    y_test = np.loadtxt("/home/pprett/corpora/madelon/madelon_valid.labels")
    clf = GradientBoostingClassifier(**classification_params)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    error_rate = (1.0 - clf.score(X_test, y_test))
    test_time = time() - t0
    return error_rate, train_time, test_time


@repeat(1)
def bench_arcene(random_state=None):
    X_train = np.loadtxt("/home/pprett/corpora/arcene/arcene_train.data")
    y_train = np.loadtxt("/home/pprett/corpora/arcene/arcene_train.labels")
    X_test = np.loadtxt("/home/pprett/corpora/arcene/arcene_valid.data")
    y_test = np.loadtxt("/home/pprett/corpora/arcene/arcene_valid.labels")
    clf = GradientBoostingClassifier(**classification_params)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    error_rate = (1.0 - clf.score(X_test, y_test))
    test_time = time() - t0
    return error_rate, train_time, test_time


regression_params = {'n_iter': 100, 'max_depth': 1,
                     'min_split': 1, 'learn_rate': 0.1,
                     'loss': 'ls'}


@repeat()
def bench_boston(random_state=None):
    boston = datasets.load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=random_state)
    X = np.asfortranarray(X, dtype=np.float32)
    y = y.astype(np.float32)
    
    offset = int(X.shape[0] * 0.9)
    X_train = X[:offset]
    y_train = y[:offset]
    X_test = X[offset:]
    y_test = y[offset:]
    clf = GradientBoostingRegressor(**regression_params)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    mse = np.mean((clf.predict(X_test) - y_test) ** 2.0)
    test_time = time() - t0
    return mse, train_time, test_time


@repeat()
def bench_friedman1(random_state=None):
    X, y = datasets.make_friedman1(n_samples=1200,
                                   random_state=random_state, noise=1.0)
    X = np.asfortranarray(X, dtype=np.float32)
    y = y.astype(np.float32)
    
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]
    clf = GradientBoostingRegressor(**regression_params)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    mse = np.mean((clf.predict(X_test) - y_test) ** 2.0)
    test_time = time() - t0
    return mse, train_time, test_time


@repeat()
def bench_friedman2(random_state=None):
    X, y = datasets.make_friedman2(n_samples=1200, random_state=random_state)
    X = np.asfortranarray(X, dtype=np.float32)
    y = y.astype(np.float32)
    
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]
    clf = GradientBoostingRegressor(**regression_params)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    mse = np.mean((clf.predict(X_test) - y_test) ** 2.0)
    test_time = time() - t0
    return mse, train_time, test_time


@repeat()
def bench_friedman3(random_state=None):
    X, y = datasets.make_friedman3(n_samples=1200, random_state=random_state)
    X = np.asfortranarray(X, dtype=np.float32)
    y = y.astype(np.float32)
    
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]
    clf = GradientBoostingRegressor(**regression_params)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    mse = np.mean((clf.predict(X_test) - y_test) ** 2.0)
    test_time = time() - t0
    return mse, train_time, test_time


gbrt_results = {
    ## "Example 10.2": bench_random_gaussian(),
    ## "Spam": bench_spam(),
    ## "Madelon": bench_madelon(),
    ## "Arcene": bench_arcene(),
   "Boston": bench_boston(), 
   "Friedman#1": bench_friedman1(),
   "Friedman#2": bench_friedman2(),
   "Friedman#3": bench_friedman3(),
    }

print gbrt_results

"""
To run this, you'll need to have installed.

  * scikit-learn 

Does two benchmarks

First, we fix a training set, increase the number of
samples to classify and plot number of classified samples as a
function of time.

In the second benchmark, we increase the number of dimensions of the
training set, classify a sample and plot the time taken as a function
of the number of dimensions.
"""
import numpy as np
import pylab as pl
import gc
from datetime import datetime
from time import time

# to store the results
scikit_classifier_results = []
scikit_regressor_results = []

mu_second = 0.0 + 10**6 # number of microseconds in a second


def bench_scikit_tree_classifier(X, Y):
    """
    bench with scikit-learn decision tree classifier
    """
    import scikits.learn
    from scikits.learn.tree import DecisionTreeClassifier

    #gc.collect()

    # start time
    #tstart = datetime.now()
    clf = DecisionTreeClassifier()
    clf.fit(X, Y).predict(X)
    #delta = (datetime.now() - tstart)
    # stop time

    #scikit_classifier_results.append(delta.seconds + delta.microseconds/mu_second)

def bench_scikit_tree_regressor(X, Y):
    """
    bench with scikit-learn decision tree regressor
    """
    import scikits.learn
    from scikits.learn.tree import DecisionTreeRegressor

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeRegressor()
    clf.fit(X, Y).predict(X)
    delta = (datetime.now() - tstart)
    # stop time

    scikit_regressor_results.append(delta.seconds + delta.microseconds/mu_second)

def profile(n_samples=1000, dim=50, K=2):
    np.random.seed(13)
    X = np.random.randn(n_samples, dim)
    Y = np.random.randint(0, K, (n_samples,))
    bench_scikit_tree_classifier(X, Y)

def bench_madelon():
    X_train = np.loadtxt("/home/pprett/corpora/madelon/madelon_train.data")
    y_train = np.loadtxt("/home/pprett/corpora/madelon/madelon_train.labels")
    X_test = np.loadtxt("/home/pprett/corpora/madelon/madelon_valid.data")
    y_test = np.loadtxt("/home/pprett/corpora/madelon/madelon_valid.labels")
    from scikits.learn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=100, min_split=5)
    t0 = datetime.now()
    clf.fit(X_train, y_train)
    delta = (datetime.now() - t0)
    score = np.mean(clf.predict(X_test) == y_test)
    clf.export_to_graphviz()
    print score, delta

def bench_arcene():
    X_train = np.loadtxt("/home/pprett/corpora/arcene/arcene_train.data")
    y_train = np.loadtxt("/home/pprett/corpora/arcene/arcene_train.labels")
    X_test = np.loadtxt("/home/pprett/corpora/arcene/arcene_valid.data")
    y_test = np.loadtxt("/home/pprett/corpora/arcene/arcene_valid.labels")
    from scikits.learn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=100, min_split=5)
    t0 = datetime.now()
    clf.fit(X_train, y_train)
    delta = (datetime.now() - t0)
    score = np.mean(clf.predict(X_test) == y_test)
    print score, delta

def bench_boston():
    from scikits.learn import datasets
    from scikits.learn.utils import shuffle
    boston = datasets.load_boston()
    np.random.seed(13)
    X, y = shuffle(boston.data, boston.target, random_state=13)
    offset = int(0.9 * X.shape[0])
    X_train = X[:offset]
    y_train = y[:offset]
    X_test = X[offset:]
    y_test = y[offset:]
    from scikits.learn.tree import DecisionTreeRegressor
    clf = DecisionTreeRegressor(max_depth=100, min_split=5)
    t0 = datetime.now()
    clf.fit(X_train, y_train)
    delta = (datetime.now() - t0)
    score = np.mean(clf.predict(X_test) - y_test)
    clf.export_to_graphviz()

    from scikits.learn.neighbors import NeighborsRegressor
    clf = NeighborsRegressor(n_neighbors=5, mode='mean')
    clf.fit(X_train, y_train)
    score2 = np.mean(clf.predict(X_test) - y_test)
    print score2, score, delta

if __name__ == '__main__':
    bench_madelon()
    #bench_arcene()
    bench_boston()

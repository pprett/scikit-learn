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
    

if __name__ == '__main__':

    X = np.loadtxt("/home/pprett/corpora/madelon/madelon_train.data")
    y = np.loadtxt("/home/pprett/corpora/madelon/madelon_train.labels")
    from scikits.learn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X, y)


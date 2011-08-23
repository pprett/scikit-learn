# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution
"""
================
Tree Classifier
================

A decision tree classifier

Implements Classification and Regression Trees (Breiman et al. 1984)

"""

from __future__ import division
from ..utils import check_random_state
import numpy as np
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
import _tree

__all__ = [
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    ]

lookup_c = \
      {'gini': _tree.Gini,
       'entropy': _tree.eval_entropy,
       'miss': _tree.eval_miss,
       }
lookup_r = \
      {'mse': _tree.eval_mse,
      }


class Leaf(object):
    '''
        v : target value
            Classification: array-like, shape = [n_features]
                Histogram of target values
            Regression:  real number
                Mean for the region
    '''

    def __init__(self, v):
        self.v = v

    def _graphviz(self):
        return 'Leaf(%s)' % (self.v)


class Node(object):
    '''
        value : target value
            Classification: array-like, shape = [n_features]
                Histogram of target values
            Regression:  real number
                Mean for the region
    '''

    def __init__(self, dimension, value, error, left, right):
        self.dimension = dimension
        self.value = value
        self.error = error
        self.left = left
        self.right = right

    def _graphviz(self):
        return "x[%s] < %s \\n error = %s" \
               % (self.dimension, self.value, self.error)


def _find_best_split(features, labels, criterion, K):
    n_samples, n_features = features.shape
    K = int(np.abs(labels.max())) + 1
    pm = np.zeros((K,), dtype=np.float64)

    best = None
    split_error = criterion(labels, pm)
    for i in xrange(n_features):
        features_at_i = features[:, i]
        domain_i = sorted(set(features_at_i))
        for d1, d2 in zip(domain_i[:-1], domain_i[1:]):
            t = (d1 + d2) / 2.
            cur_split = (features_at_i < t)
            left_labels = labels[cur_split]
            right_labels = labels[~cur_split]
            e1 = len(left_labels) / n_samples * \
                criterion(left_labels, pm)
            e2 = len(right_labels) / n_samples * \
                criterion(right_labels, pm)
            error = e1 + e2
            if error < split_error:
                split_error = error
                best = i, t, error
    return best


def _build_tree(is_classification, features, labels, criterion,
                max_depth, min_split, F, K, random_state):
    """
    Parameters
    ----------

    K : int
        Number of classes (for regression us 0).
    criterion : _tree.Criterion
        Split criterion extension type.
    """
    n_total_samples, n_dims = features.shape
    if labels.shape[0] != n_total_samples:
        raise ValueError("Number of labels does not match "
                         "number of features\n")
    labels = np.array(labels, dtype=np.float64, order="c")
    

    sample_dims = np.arange(n_dims)
    if F is not None:
        if F <= 0:
            raise ValueError("F must be > 0.\n"
                             "Did you mean to use None to signal no F?")
        if F > n_dims:
            raise ValueError("F must be < num dimensions of features.\n"
                             "F is %s, n_dims = %s "
                             % (F, n_dims))
        
        sample_dims = random_state.shuffle(np.arange(n_dims))[:F]
        features = features[:, sample_dims]
        n_total_samples, n_dims = features.shape

    # make data fortran layout
    if not features.flags["F_CONTIGUOUS"]:
        features = np.array(features, order="F")

    sorted_features = np.argsort(features, axis=0)
    sorted_features = sorted_features.astype(np.int32)
    if not sorted_features.flags["F_CONTIGUOUS"]:
        sorted_features = np.array(sorted_features, order="F")

    if min_split <= 0:
        raise ValueError("min_split must be greater than zero.\n"
                         "min_split is %s." % min_split)
    if max_depth <= 0:
        raise ValueError("max_depth must be greater than zero.\n"
                         "max_depth is %s." % max_depth)

    n_rec_part_called = np.zeros((1,), dtype=np.int)
    def recursive_partition(sample_mask, parent_split_error, depth, n_samples):
        n_rec_part_called[0] += 1
        is_leaf = False
        # If current depth larger than max return leaf.
        if depth >= max_depth:
            is_leaf = True
        # else try to find a split point
        else:
            dim, thresh, error, nll = _tree._find_best_split(sample_mask,
                                                             parent_split_error,
                                                             features, sorted_features,
                                                             labels, criterion, K,
                                                             n_samples)
            if dim != -1:
                # we found a split point
                # check if num samples to the left and right of split point
                # is larger than min_split
                if nll <= min_split or n_samples - nll <= min_split:
                    # splitting point does not suffice min_split
                    is_leaf = True
                else:
                    is_leaf = False
            else:
                # could not find splitting point
                is_leaf = True

        new_node = None
        if is_leaf:
            if is_classification:
                a = np.zeros((K, ))
                _tree.fill_counts(a, labels, sample_mask)
                new_node = Leaf(a)
            else:
                new_node = Leaf(np.mean(labels[sample_mask]))
        else:
            split = features[:, dim] < thresh
            left_sample_mask = split & sample_mask
            right_sample_mask = ~split & sample_mask
            new_node = Node(dimension=sample_dims[dim],
                            value=thresh, error=error,
                            left=recursive_partition(left_sample_mask, error,
                                                     depth + 1, nll),
                            right=recursive_partition(right_sample_mask,
                                                      error, depth + 1,
                                                      n_samples - nll))
    
        # assert new_node != None
        return new_node

    root = recursive_partition(np.ones((n_total_samples,), dtype=np.bool),
                               np.inf, 0, n_total_samples)
    print "recursive_partition called %d times" % n_rec_part_called[0]
    return root


def _apply_tree(tree, features):
    '''
    conf = apply_tree(tree, features)

    Applies the decision tree to a set of features.
    '''
    if type(tree) is Leaf:
        return tree.v
    if features[tree.dimension] < tree.value:
        return _apply_tree(tree.left, features)
    return _apply_tree(tree.right, features)


def _graphviz(tree):
    '''Print decision tree in .dot format
    '''
    if type(tree) is Leaf:
        return ""
    s = str(tree) + \
        " [label=" + "\"" + tree._graphviz() + "\"" + "] ;\n"
    s += str(tree.left) + \
        " [label=" + "\"" + tree.left._graphviz() + "\"" + "] ;\n"
    s += str(tree.right) + \
        " [label=" + "\"" + tree.right._graphviz() + "\"" + "] ;\n"

    s += str(tree) + " -> " + str(tree.left) + " ;\n"
    s += str(tree) + " -> " + str(tree.right) + " ;\n"

    return s + _graphviz(tree.left) + _graphviz(tree.right)


class BaseDecisionTree(BaseEstimator):
    '''
    Should not be used directly, use derived classes instead
    '''

    _dtree_types = ['classification', 'regression']

    def __init__(self, impl, criterion, max_depth,
                 min_split, F, random_state):

        if not impl in self._dtree_types:
            raise ValueError("impl should be one of %s, %s was given"
                             % (self._dtree_types, impl))

        self.type = impl
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_split = min_split
        self.F = F
        self.random_state = check_random_state(random_state)

        self.n_features = None
        self.tree = None

    def export_to_graphviz(self, filename="tree.dot"):
        """
        Export the tree in .dot format.  Render to PostScript using e.g.
        $ dot -Tps tree.dot -o tree.ps

        Parameters
        ----------
        filename : str
            The name of the file to write to.

        """
        if self.tree is None:
            raise Exception('Tree not initialized. Perform a fit first')

        with open(filename, 'w') as f:
            f.write("digraph Tree {\n")
            f.write(_graphviz(self.tree))
            f.write("\n}\n")

    def fit(self, X, y):
        """
        Fit the tree model according to the given training data and
        parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes 0,1,...,K-1

        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asanyarray(X, dtype=np.float64, order='C')
        _, self.n_features = X.shape
        
        if self.type == 'classification':
            y = np.asanyarray(y, dtype=np.int64)
            self.classes = np.unique(y)
            y = np.searchsorted(self.classes, y)
            self.K = self.classes.shape[0]
            if y.min() < 0:
                raise ValueError("Labels must be in the range [0 to %s)", self.K)
            
            # create new Criterion extension type
            criterion_clazz = lookup_c[self.criterion]
            pm_left = np.zeros((self.K,), dtype=np.float64)
            pm_right = np.zeros((self.K,), dtype=np.float64)
            criterion = criterion_clazz(self.K, pm_left, pm_right)
    
            self.tree = _build_tree(True, X, y, criterion,
                                    self.max_depth, self.min_split, self.F,
                                    self.K, self.random_state)
        else: # regression
            y = np.asanyarray(y, dtype=np.float64, order='C')
            self.tree = _build_tree(False, X, y, lookup_r[self.criterion],
                                    self.max_depth, self.min_split, self.F,
                                    0, self.random_state)
        return self

    def predict(self, X):
        """
        This function does classification or regression on an array of
        test vectors X.

        For a classification model, the predicted class for each
        sample in X is returned.  For a regression model, the function
        value of X calculated is returned.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
        """

        X = np.atleast_2d(X).astype(np.float64)
        n_samples, n_features = X.shape

        if self.tree is None:
            raise Exception('Tree not initialized. Perform a fit first')

        if self.n_features != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input.\n"
                             "Model n_features is %s and "
                             " input n_features is %s "
                             % (self.n_features, n_features))

        if self.type == 'classification':
            C = np.zeros(n_samples, dtype=int)
            for idx, sample in enumerate(X):
                c = np.argmax(_apply_tree(self.tree, sample))
                C[idx] = self.classes[c]
        else:
            C = np.zeros(n_samples, dtype=float)
            for idx, sample in enumerate(X):            
                C[idx] = _apply_tree(self.tree, sample)

        return C


class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    """Classify a multi-labeled dataset with a decision tree.

    Parameters
    ----------
    criterion : string
        function to measure goodness of split

    max_depth : integer
        maximum depth of the tree

    min_split : integer
        minimum size to split on

    F : integer, optional
        if given, then, choose F features

    random_state : integer or array_like, optional
        seed the random number generator


    Example
    -------
    >>> import numpy as np
    >>> from scikits.learn.datasets import load_iris
    >>> from scikits.learn.cross_val import StratifiedKFold
    >>> from scikits.learn.tree import DecisionTreeClassifier
    >>> data = load_iris()
    >>> skf = StratifiedKFold(data.target, 10)
    >>> for train_index, test_index in skf:
    ...     clf = DecisionTreeClassifier()
    ...     clf = clf.fit(data.data[train_index], data.target[train_index])
    ...     print np.mean(clf.predict(data.data[test_index]) == data.target[test_index])
    ...
    1.0
    0.933333333333
    0.866666666667
    0.933333333333
    0.933333333333
    0.933333333333
    0.933333333333
    1.0
    0.933333333333
    1.0

    """

    def __init__(self, criterion='gini', max_depth=10,
                  min_split=1, F=None, random_state=None):
        BaseDecisionTree.__init__(self, 'classification', criterion,
                                  max_depth, min_split, F, random_state)

    def predict_proba(self, X):
        """
        This function does classification on a test vector X
        given a model with probability information.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        P : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model, where classes are ordered by arithmetical
            order.

        """
        X = np.atleast_2d(X).astype(np.float64)
        n_samples, n_features = X.shape

        if self.tree is None:
            raise Exception('Tree not initialized. Perform a fit first')

        if self.n_features != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input.\n"
                             "Model n_features is %s and "
                             " input n_features is %s "
                             % (self.n_features, n_features))

        P = np.zeros((n_samples, self.K))
        for idx, sample in enumerate(X):
            P[idx, :] = _apply_tree(self.tree, sample)
            P[idx, :] /= np.sum(P[idx, :])
        return P

    def predict_log_proba(self, X):
        """
        This function does classification on a test vector X

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        P : array-like, shape = [n_samples, n_classes]
            Returns the log-probabilities of the sample for each class in
            the model, where classes are ordered by arithmetical
            order.

        """

        return np.log(self.predict_proba(X))


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    """Perform regression on dataset with a decision tree.

    Parameters
    ----------
    criterion : string
        function to measure goodness of split

    max_depth : integer
        maximum depth of the tree

    min_split : integer
        minimum size to split on

    F : integer, optional
        if given, then, choose F features

    random_state : integer or array_like, optional
        seed the random number generator

    Example
    -------
    >>> import numpy as np
    >>> from scikits.learn.datasets import load_boston
    >>> from scikits.learn.cross_val import KFold
    >>> from scikits.learn.tree import DecisionTreeRegressor
    >>> data = load_boston()
    >>> kf = KFold(len(data.target), 2)
    >>> for train_index, test_index in kf:
    ...     clf = DecisionTreeRegressor()
    ...     clf = clf.fit(data.data[train_index], data.target[train_index])
    ...     print np.mean(np.power(clf.predict(data.data[test_index]) - data.target[test_index], 2))
    ...
    19.2264679543
    41.2959435867

    """

    def __init__(self, criterion='mse', max_depth=10,
                  min_split=1, F=None, random_state=None):
        BaseDecisionTree.__init__(self, 'regression', criterion,
                                  max_depth, min_split, F, random_state)

"""
The :mod:`sklearn.grid_search` includes utilities to fine-tune the parameters
of an estimator.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD Style.

import time
from copy import copy
import operator

import numpy as np

from ..base import BaseEstimator, is_classifier, clone
from ..cross_validation import check_cv
from ..externals.joblib import Parallel, delayed, logger
from ..utils import check_arrays, safe_mask
from ..grid_search import GridSearchCV, IterGrid, _has_one_grid_point

__all__ = ['BoostGridSearchCV',]


def fit_grid_point(X, y, sample_weight, base_clf,
                   clf_params, train, test, verbose,
                   **fit_params):
    """Run fit on one set of parameters

    Returns the score and the instance of the classifier
    """
    if verbose > 1:
        start_time = time.time()
        msg = '%s' % (', '.join('%s=%s' % (k, v)
                                     for k, v in clf_params.iteritems()))
        print "[BoostGridSearchCV] %s %s" % (msg, (64 - len(msg)) * '.')

    X, y = check_arrays(X, y)
    # update parameters of the classifier after a copy of its base structure
    clf = clone(base_clf)
    clf.set_params(**clf_params)

    if hasattr(base_clf, 'kernel') and hasattr(base_clf.kernel, '__call__'):
        # cannot compute the kernel values with custom function
        raise ValueError(
            "Cannot use a custom kernel function. "
            "Precompute the kernel matrix instead.")

    if getattr(base_clf, "_pairwise", False):
        # X is a precomputed square kernel matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        X_train = X[np.ix_(train, train)]
        X_test = X[np.ix_(test, train)]
    else:
        X_train = X[safe_mask(X, train)]
        X_test = X[safe_mask(X, test)]

    if y is not None:
        y_test = y[safe_mask(y, test)]
        y_train = y[safe_mask(y, train)]
    else:
        y_test = None
        y_train = None

    if sample_weight is not None:
        sample_weight_test = sample_weight[safe_mask(sample_weight, test)]
        sample_weight_train = sample_weight[safe_mask(sample_weight, train)]
    else:
        sample_weight_test = None
        sample_weight_train = None

    if sample_weight is not None:
        clf.fit(X_train, y_train,
                sample_weight=sample_weight_train,
                **fit_params)
    else:
        clf.fit(X_train, y_train, **fit_params)

    if verbose > 1:
        end_msg = "%s -%s" % (msg,
                              logger.short_format_time(time.time() -
                                                       start_time))
        print "[BoostGridSearchCV] %s %s" % ((64 - len(end_msg)) * '.', end_msg)
    return clf, clf_params, train, test

def score_each_boost(X, y, sample_weight, clf, clf_params,
                     train, test, loss_func,
                     score_func, verbose):
    """Run fit on one set of parameters

    Returns the score and the instance of the classifier
    """
    if hasattr(clf, 'kernel') and hasattr(clf.kernel, '__call__'):
        # cannot compute the kernel values with custom function
        raise ValueError(
            "Cannot use a custom kernel function. "
            "Precompute the kernel matrix instead.")

    X, y = check_arrays(X, y)

    if getattr(clf, "_pairwise", False):
        # X is a precomputed square kernel matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        X_train = X[np.ix_(train, train)]
        X_test = X[np.ix_(test, train)]
    else:
        X_train = X[safe_mask(X, train)]
        X_test = X[safe_mask(X, test)]

    if y is not None:
        y_test = y[safe_mask(y, test)]
        y_train = y[safe_mask(y, train)]
    else:
        y_test = None
        y_train = None

    if sample_weight is not None:
        sample_weight_test = sample_weight[safe_mask(sample_weight, test)]
        sample_weight_train = sample_weight[safe_mask(sample_weight, train)]
    else:
        sample_weight_test = None
        sample_weight_train = None

    if verbose > 1:
        start_time = time.time()
        msg = '%s' % (', '.join('%s=%s' % (k, v)
                                     for k, v in clf_params.iteritems()))
        print "[BoostGridSearchCV] %s %s" % (msg, (64 - len(msg)) * '.')

    if y is not None:
        if hasattr(y, 'shape'):
            this_n_test_samples = y.shape[0]
        else:
            this_n_test_samples = len(y)
    else:
        if hasattr(X, 'shape'):
            this_n_test_samples = X.shape[0]
        else:
            this_n_test_samples = len(X)

    all_scores = []
    all_clf_params = []
    n_test_samples = []

    # TODO: include support for sample_weight in score functions
    if loss_func is not None or score_func is not None:
        for i, y_pred in enumerate(clf.staged_predict(X_test)):
            if loss_func is not None:
                score = -loss_func(y_test, y_pred)
            elif score_func is not None:
                score = score_func(y_test, y_pred)
            all_scores.append(score)
            clf_para = copy(clf_params)
            clf_para['n_estimators'] = i + 1
            all_clf_params.append(clf_para)
            n_test_samples.append(this_n_test_samples)

    else:
        if sample_weight_test is not None:
            for i, score in enumerate(clf.staged_score(X_test, y_test,
                sample_weight=sample_weight_test)):
                all_scores.append(score)
                clf_para = copy(clf_params)
                clf_para['n_estimators'] = i + 1
                all_clf_params.append(clf_para)
                n_test_samples.append(this_n_test_samples)

        else:
            for i, score in enumerate(clf.staged_score(X_test, y_test)):
                all_scores.append(score)
                clf_para = copy(clf_params)
                clf_para['n_estimators'] = i + 1
                all_clf_params.append(clf_para)
                n_test_samples.append(this_n_test_samples)

    if verbose > 1:
        end_msg = "%s -%s" % (msg,
                              logger.short_format_time(time.time() -
                                                       start_time))
        print "[BoostGridSearchCV] %s %s" % ((64 - len(end_msg)) * '.', end_msg)
    return all_scores, all_clf_params, n_test_samples


class BoostGridSearchCV(GridSearchCV):

    def __init__(self, estimator, param_grid, max_n_estimators,
            **kwargs):

        assert 'n_estimators' not in param_grid
        self.max_n_estimators = max_n_estimators
        super(BoostGridSearchCV, self).__init__(
                estimator=estimator,
                param_grid=param_grid,
                **kwargs)

    def fit(self, X, y=None, sample_weight=None):
        """Run fit with all sets of parameters

        Returns the best classifier

        Parameters
        ----------

        X: array, [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y: array-like, shape = [n_samples], optional
            Target vector relative to X for classification;
            None for unsupervised learning.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights
        """
        estimator = self.estimator
        cv = self.cv

        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            # support list of unstructured objects on which feature
            # extraction will be applied later in the tranformer chain
            n_samples = len(X)
        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
            y = np.asarray(y)
        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
        base_clf = clone(self.estimator)

        # first fit at each grid point using the maximum n_estimators
        param_grid = self.param_grid
        param_grid['n_estimators'] = [self.max_n_estimators]
        grid = IterGrid(param_grid)

        pre_dispatch = self.pre_dispatch
        clfs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=pre_dispatch)(
            delayed(fit_grid_point)(
                X, y, sample_weight, base_clf, clf_params, train, test,
                self.verbose, **self.fit_params)
                    for clf_params in grid
                    for train, test in cv)
        # now use the already fitted ensembles but trancate to N estimators for
        # N from 1 to n_estimators_max - 1 (inclusive)
        out = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=pre_dispatch)(
            delayed(score_each_boost)(
                X, y, sample_weight, clf, clf_params, train, test,
                self.loss_func, self.score_func, self.verbose)
                    for clf, clf_params, train, test in clfs)

        out = reduce(operator.add, [zip(*stage) for stage in out])
        # out is now a list of triplet: score, estimator_params, n_test_samples

        n_grid_points = len(list(grid)) * self.max_n_estimators
        n_fits = len(out)
        n_folds = n_fits // n_grid_points

        scores = list()
        cv_scores = list()
        for block in range(0, n_fits, n_folds * self.max_n_estimators):
            for grid_start in range(block, block + self.max_n_estimators):
                n_test_samples = 0
                score = 0
                these_points = list()
                for this_score, clf_params, this_n_test_samples in \
                        out[grid_start:
                            grid_start + n_folds * self.max_n_estimators:
                            self.max_n_estimators]:
                    these_points.append(this_score)
                    if self.iid:
                        this_score *= this_n_test_samples
                    score += this_score
                    n_test_samples += this_n_test_samples
                if self.iid:
                    score /= float(n_test_samples)
                scores.append((score, clf_params))
                cv_scores.append(these_points)

        cv_scores = np.asarray(cv_scores)

        # Note: we do not use max(out) to make ties deterministic even if
        # comparison on estimator instances is not deterministic
        best_score = -np.inf
        for score, params in scores:
            if score > best_score:
                best_score = score
                best_params = params

        if best_score is None:
            raise ValueError('Best score could not be found')
        self.best_score_ = best_score
        self.best_params_ = best_params

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_clf).set_params(**best_params)
            if sample_weight is not None:
                best_estimator.fit(X, y, sample_weight, **self.fit_params)
            else:
                best_estimator.fit(X, y, **self.fit_params)
            self._best_estimator_ = best_estimator
            self._set_methods()

        # Store the computed scores
        # XXX: the name is too specific, it shouldn't have
        # 'grid' in it. Also, we should be retrieving/storing variance
        self.grid_scores_ = [
            (clf_params, score, all_scores)
                    for (score, clf_params), all_scores
                    in zip(scores, cv_scores)]
        return self

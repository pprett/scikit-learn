"""
================================================================
Compare the speed of the default and boost-optimized grid search
================================================================

This example performs a grid search of the minimum leaf size and number of
estimators in an Adaboosted decision tree using
`sklearn.grid_search.GridSearchCV`
and a grid search optimized for boosted classifiers:
`sklearn.ensemble.grid_search.BoostGridSearchCV`.

The optimized grid search BoostGridSearchCV does not fit a classifier for all
values of n_estimators, but only the maximum. This classifier can then be
truncated to determine the scores for all number of estimators less than
n_estimators.
"""
print __doc__

from time import time

import numpy as np

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble.grid_search import BoostGridSearchCV
from sklearn.grid_search import GridSearchCV

def plot_grid_scores(
        grid_scores, best_point, params,
        label_all_bins=False,
        label_all_ticks=False,
        n_ticks=11,
        title=None):

    param_names = sorted(grid_scores[0][0].keys())
    param_values = dict([(pname, []) for pname in param_names])
    for pvalues, score, cv_scores in grid_scores:
        for pname in param_names:
            param_values[pname].append(pvalues[pname])

    # remove duplicates
    for pname in param_names:
        param_values[pname] = np.unique(param_values[pname]).tolist()

    scores = np.empty(shape=[len(param_values[pname]) for pname in param_names])

    for pvalues, score, cv_scores in grid_scores:
        index = []
        for pname in param_names:
            index.append(param_values[pname].index(pvalues[pname]))
        scores.itemset(tuple(index), score)

    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = plt.axes([.12, .15, .8, .75])
    cmap = cm.get_cmap('jet', 100)
    img = ax.imshow(scores, interpolation="nearest", cmap=cmap,
            aspect='auto',
            origin='lower')

    if label_all_ticks:
        plt.xticks(range(len(param_values[param_names[1]])),
                param_values[param_names[1]])
        plt.yticks(range(len(param_values[param_names[0]])),
                param_values[param_names[0]])
    else:
        trees = param_values[param_names[1]]
        def tree_formatter(x, pos):
            if x >= len(trees) or x < 0:
                return ''
            return str(trees[int(x)])

        leaves = param_values[param_names[0]]
        def leaf_formatter(x, pos):
            if x >= len(leaves) or x < 0:
                return ''
            return str(leaves[int(x)])

        ax.xaxis.set_major_formatter(FuncFormatter(tree_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(leaf_formatter))
        ax.xaxis.set_major_locator(MaxNLocator(n_ticks, integer=True,
            prune='lower', steps=[1, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(n_ticks, integer=True,
            steps=[1, 5, 10]))
        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_rotation(45)

    ax.set_xlabel(params[param_names[1]], fontsize=12,
            position=(1., 0.), ha='right')
    ax.set_ylabel(params[param_names[0]], fontsize=12,
            position=(0., 1.), va='top')

    ax.set_frame_on(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    for row in range(scores.shape[0]):
        for col in range(scores.shape[1]):
            decor={}
            if ((param_values[param_names[0]].index(best_point[param_names[0]])
                 == row) and
                (param_values[param_names[1]].index(best_point[param_names[1]])
                 == col)):
                decor = dict(weight='bold',
                             bbox=dict(boxstyle="round,pad=0.5",
                                       ec='black',
                                       fill=False))
            if label_all_bins or decor:
                plt.text(col, row, "%.3f" % (scores[row][col]), ha='center',
                         va='center', **decor)
    if title:
        plt.suptitle(title)

    plt.colorbar(img, fraction=.06, pad=0.03)
    plt.axis("tight")

# Load data
X, y = make_classification(n_samples=2000, n_features=5, n_classes=2,
        n_informative=3,
        n_redundant=2)

clf = AdaBoostClassifier(DecisionTreeClassifier(), learn_rate=0.5)

grid_params_slow = {
    'base_estimator__min_samples_leaf': range(1, 600, 50),
    'n_estimators': range(1, 1001, 100)
}

grid_params_fast = {
    'base_estimator__min_samples_leaf': range(1, 600, 20),
}

grid_clf_slow = GridSearchCV(
        clf, grid_params_slow,
        cv=StratifiedKFold(y, 3),
        n_jobs=-1,
        verbose=10)

grid_clf_fast = BoostGridSearchCV(
        clf, grid_params_fast,
        max_n_estimators=1000,
        cv=StratifiedKFold(y, 3),
        n_jobs=-1,
        verbose=10)

print "=" * 30
print "slow grid search ..."
print "=" * 30

tstart = time()
grid_clf_slow.fit(X, y)
slow_time = time() - tstart

print "=" * 30
print "fast grid search ..."
print "=" * 30

tstart = time()
grid_clf_fast.fit(X, y)
fast_time = time() - tstart

print "slow grid search: %f [sec]" % slow_time
print "fast grid search: %f [sec]" % fast_time

clf_slow = grid_clf_slow.best_estimator_
grid_scores_slow = grid_clf_slow.grid_scores_

clf_fast = grid_clf_fast.best_estimator_
grid_scores_fast = grid_clf_fast.grid_scores_

plot_grid_scores(
    grid_scores_slow,
    best_point={
        'base_estimator__min_samples_leaf':
        clf_slow.base_estimator.min_samples_leaf,
        'n_estimators':
        clf_slow.n_estimators},
    params={
        'base_estimator__min_samples_leaf':
        'minimum leaf size',
        'n_estimators':
        'number of trees'},
    title="Classification score over slow parameter grid search.")

plot_grid_scores(
    grid_scores_fast,
    best_point={
        'base_estimator__min_samples_leaf':
        clf_fast.base_estimator.min_samples_leaf,
        'n_estimators':
        clf_fast.n_estimators},
    params={
        'base_estimator__min_samples_leaf':
        'minimum leaf size',
        'n_estimators':
        'number of trees'},
    title="Classification score over fast parameter grid search.")

plt.show()

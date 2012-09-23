"""
============================================================
Plot the decision surfaces on the Gaussian quantiles dataset
============================================================

This plot shows the decision surfaces learned by an AdaBoosted decision tree
classifier on the Gaussian quantiles dataset.
"""
print __doc__

import numpy as np

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import IndexLocator, FuncFormatter

from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble.grid_search import BoostGridSearchCV
from sklearn.metrics import classification_report

# Load data
X, y = make_classification(n_samples=2000, n_features=5, n_classes=2,
        n_informative=3,
        n_redundant=2)

clf = AdaBoostClassifier(DecisionTreeClassifier(), learn_rate=0.5)

grid_params = {
    'base_estimator__min_samples_leaf': range(20, 600, 20),
}

grid_clf = BoostGridSearchCV(
        clf, grid_params,
        max_n_estimators=200,
        cv=StratifiedKFold(y, 3),
        n_jobs=-1,
        verbose=10)

grid_clf.fit(X, y)


def plot_grid_scores(
        grid_scores, best_point, params,
        label_all_bins=False,
        label_all_ticks=False,
        n_ticks=10):

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
    ax = plt.axes([.1, .15, .8, .75])
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
            return str(trees[int(x)])

        leaves = param_values[param_names[0]]
        def leaf_formatter(x, pos):
            return str(leaves[int(x)])

        x_base = scores.shape[1] / n_ticks
        y_base = scores.shape[0] / n_ticks

        ax.xaxis.set_major_formatter(FuncFormatter(tree_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(leaf_formatter))
        ax.xaxis.set_major_locator(IndexLocator(max(1, x_base), 0))
        ax.yaxis.set_major_locator(IndexLocator(max(1, y_base), 0))
        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_rotation(45)

    ax.set_xlabel(params[param_names[1]], fontsize=12, position=(1., 0.), ha='right')
    ax.set_ylabel(params[param_names[0]], fontsize=12, position=(0., 1.), va='top')

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

    plt.suptitle("Classification score over parameter grid search.")
    plt.colorbar(img, fraction=.06, pad=0.03)
    plt.axis("tight")
    plt.show()

clf = grid_clf.best_estimator_
grid_scores = grid_clf.grid_scores_

plot_grid_scores(
    grid_scores,
    best_point={
        'base_estimator__min_samples_leaf':
        clf.base_estimator.min_samples_leaf,
        'n_estimators':
        clf.n_estimators},
    params={
        'base_estimator__min_samples_leaf':
        'minimum leaf size',
        'n_estimators':
        'number of trees'})

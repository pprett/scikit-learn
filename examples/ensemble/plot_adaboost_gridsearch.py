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

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble.grid_search import BoostGridSearchCV
from sklearn.metrics import classification_report

# Load data
X, y = make_gaussian_quantiles(n_samples=5000, n_features=3,
                               n_classes=3)

clf = AdaBoostClassifier(DecisionTreeClassifier(), learn_rate=0.5)

grid_params = {
    'base_estimator__min_samples_leaf': range(10, 500, 50),
    #'n_estimators': range(10, 800, 50)
}

grid_clf = BoostGridSearchCV(
        clf, grid_params,
        n_estimators_range=range(10, 800, 50),
        cv=StratifiedKFold(y, 3),
        n_jobs=-1,
        verbose=10)

grid_clf.fit(X, y)


def plot_grid_scores(
        grid_scores, best_point, params,
        label_all_bins=False,
        label_all_ticks=False):

    param_names = sorted(grid_scores[0][0].keys())
    param_values = dict([(pname, []) for pname in param_names])
    for pvalues, score, cv_scores in grid_scores:
        for pname in param_names:
            param_values[pname].append(pvalues[pname])

    for pname in param_names:
        param_values[pname] = np.unique(param_values[pname]).tolist()

    scores = np.empty(shape=[len(param_values[pname]) for pname in param_names])

    for pvalues, score, cv_scores in grid_scores:
        index = []
        for pname in param_names:
            index.append(param_values[pname].index(pvalues[pname]))
        scores.itemset(tuple(index), score)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 100) # jet doesn't have white color
    #cmap.set_bad('w') # default value is 'k'
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

        ax.xaxis.set_major_formatter(FuncFormatter(tree_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(leaf_formatter))
        ax.xaxis.set_major_locator(IndexLocator(2, 0))
        ax.yaxis.set_major_locator(IndexLocator(2, 0))
        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_rotation(45)

    ax.set_xlabel(params[param_names[1]], fontsize=20, position=(1., 0.), ha='right')
    ax.set_ylabel(params[param_names[0]], fontsize=20, position=(0., 1.), va='top')

    ax.set_frame_on(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    if label_all_bins:
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
                plt.text(col, row, "%.3f" % (scores[row][col]), ha='center',
                         va='center', **decor)
    else:
        # circle the best bin and label the parameters
        pass

    plt.suptitle("Classification accuracy over parameter grid search.")
    plt.colorbar(img)
    plt.axis("tight")
    plt.show()

clf = grid_clf.best_estimator_
grid_scores = grid_clf.grid_scores_

y_pred = clf.predict(X)
print classification_report(y, y_pred)

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

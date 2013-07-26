import numpy as np

from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split

boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,
                                                    random_state=1)

est = DecisionTreeRegressor(random_state=1)

print np.diff(y_train).sum()
est.fit(X_train, y_train)
print np.diff(y_train).sum()
print est.score(X_test, y_test)

export_graphviz(est, max_depth=None)

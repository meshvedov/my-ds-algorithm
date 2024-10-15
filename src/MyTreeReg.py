import numpy as np
import pandas as pd
from pprint import pprint

from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
X, y = data['data'], data['target']


class MyTreeReg:
    def __init__(self,
                 max_depth=5,
                 min_samples_split=2,
                 max_leafs=20) -> None:
        self.max_leafs = max_leafs
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.leafs_cnt = 0
        self.tree = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__} class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, "
            f"max_leafs={self.max_leafs}"
        )

    # ['node', 'col_name', split_val, [...], [...]] | ['leaf', avg, ]
    def fit(self, X: pd.DataFrame, y: pd.Series):
        def create_tree(X: pd.DataFrame, y: pd.Series, name: str) -> (int, list):
            if True:
                avg = np.mean(y)
                return 1, ['leaf_'+name, avg]

            col_name, split_val, _ = get_best_split(X, y)
            left = X[col_name] <= split_val
            leaf_l, node = create_tree(X.loc[left], y.loc[left], name='left')
            sub_l = ['node', col_name, split_val, node]

        _, self.tree = create_tree(X, y)


def get_best_split(X: pd.DataFrame, y: pd.Series):
    def _mse(y):
        mse = 1/len(y)*np.sum((y - np.mean(y))**2)
        return mse

    def best_split(X: pd.Series, y: pd.Series):
        thresh = np.sort(np.unique(X))
        rules = (thresh[1:] + thresh[:-1]) * 0.5

        mp = _mse(y)
        result = []
        for rule in rules:
            left, right = np.where(X <= rule)[0], np.where(X > rule)[0]
            ml, mr = _mse(y[left]), _mse(y[right])
            gain = mp - (len(left) * ml + len(right) * mr) / len(y)
            result.append((X.name, rule, gain))
        return max(result, key=lambda x: x[2])

    splits = []
    for col in X.columns:
        res = best_split(X[col], y)
        splits.append(res)
    col_name, split_value, gain = max(splits, key=lambda x: x[2])
    return col_name, split_value, gain


mtr = MyTreeReg(max_depth=1, min_samples_split=1, max_leafs=2)
print(mtr)
print(get_best_split(X, y))

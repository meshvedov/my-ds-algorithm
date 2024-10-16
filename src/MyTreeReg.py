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
                 max_leafs=20,
                 bins=None) -> None:
        self.y_len = 0
        self.bins = bins
        self.max_leafs = 2 if max_leafs <= 1 else max_leafs
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.leafs_cnt = 0
        self.tree = None
        self.leafs_sum = 0
        self.dict_bins = {}
        self.fi = {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__} class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, "
            f"max_leafs={self.max_leafs}"
        )

    # ['node', 'col_name', split_val, [...], [...]] | ['leaf', avg, ]
    def fit(self, X: pd.DataFrame, y: pd.Series):
        def create_tree(X: pd.DataFrame, y: pd.Series, name: str, leafs: int, depth: int) -> (int, list):
            depth -= 1
            splits = self.get_best_split(X, y)
            if leafs - 1 <= 0 or depth < 0 or len(y) < self.min_samples_split or splits is None:
                avg = np.mean(y)
                self.leafs_sum += avg
                return 1, ['leaf_' + name, avg]

            col_name, split_val, _ = splits  # self.get_best_split(X, y)
            left = X[col_name] <= split_val
            leaf_l, node_l = create_tree(X[left].reset_index(drop=1),
                                         y[left].reset_index(drop=1),
                                         name='left',
                                         leafs=leafs - 1,
                                         depth=depth)
            sub_l = ['node', col_name, split_val, node_l]
            right = X[col_name] > split_val
            leaf_r, node_r = create_tree(X[right].reset_index(drop=1),
                                         y[right].reset_index(drop=1),
                                         name='right',
                                         leafs=leafs - leaf_l, depth=depth)
            self.leafs_cnt = leaf_l + leaf_r
            sub_l.append(node_r)
            self._fi_create(self.mse(y),
                            self.mse(y[left]),
                            self.mse(y[right]),
                            len(y), len(X[left]), len(X[right]), col_name)
            return leaf_l + leaf_r, sub_l

        self.fi = {col: 0 for col in X.columns}
        self.y_len = len(y)
        _, self.tree = create_tree(X, y, name='', leafs=self.max_leafs, depth=self.max_depth)

    def mse(self, y):
        if len(y) == 0:
            return 0
        mse = 1 / len(y) * np.sum((y - np.mean(y)) ** 2)
        return mse

    def _fi_create(self, mp, ml, mr, len_y, len_l, len_r, name):
        fi = len_y / self.y_len * (mp - (len_l * ml + len_r * mr) / len_y)
        self.fi[name] += fi

    def print_tree(self):
        pprint(self.tree, indent=4)

    def _walk_tree(self, X: pd.Series, tree: list):
        if tree[0] in ['leaf_left', 'leaf_right']:
            return tree[1]
        if tree[0] == 'node' and X[tree[1]] <= tree[2]:
            return self._walk_tree(X, tree=tree[3])
        elif tree[0] == 'node' and X[tree[1]] > tree[2]:
            return self._walk_tree(X, tree=tree[4])

    def predict(self, X: pd.DataFrame):
        return X.apply(lambda x: self._walk_tree(x, self.tree), axis=1).to_numpy()

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        def hist_edges(thresh, col):
            if self.dict_bins.get(col) is None:
                _, bins = np.histogram(thresh, bins=self.bins)
                self.dict_bins[col] = bins[1:-1]
            return self.dict_bins[col]

        def best_split(X: pd.Series, y: pd.Series):
            thresh = np.sort(np.unique(X))
            if len(thresh) == 1:
                rules = thresh
            elif self.bins is not None:
                rules = hist_edges(thresh, X.name)
            else:
                rules = (thresh[1:] + thresh[:-1]) * 0.5

            mp = self.mse(y)
            result = []
            for rule in rules:
                if X.min() <= rule < X.max():
                    left, right = np.where(X <= rule)[0], np.where(X > rule)[0]
                    ml, mr = self.mse(y[left]), self.mse(y[right])
                    gain = mp - (len(left) * ml + len(right) * mr) / len(y)
                    result.append((X.name, rule, gain))
            if len(result) == 0:
                return None
            return max(result, key=lambda x: x[2])

        splits = []
        for col in X.columns:
            res = best_split(X[col], y)
            if res is None:
                continue
            splits.append(res)
        if len(splits) == 0:
            return None
        col_name, split_value, gain = max(splits, key=lambda x: x[2])
        return col_name, split_value, gain


# mtr = MyTreeReg(max_depth=3, min_samples_split=2, max_leafs=1)
# mtr = MyTreeReg(max_depth=1, min_samples_split=1, max_leafs=2, bins=8)
# mtr = MyTreeReg(max_depth=3, min_samples_split=2, max_leafs=5)
# mtr = MyTreeReg(max_depth=5, min_samples_split=100, max_leafs=10, bins=4)
# mtr = MyTreeReg(max_depth=4, min_samples_split=50, max_leafs=17, bins=16)
# mtr = MyTreeReg(max_depth=10, min_samples_split=40, max_leafs=21, bins=10)
mtr = MyTreeReg(max_depth=15, min_samples_split=35, max_leafs=30, bins=6)

mtr.fit(X, y)
mtr.print_tree()
print(mtr.leafs_cnt, round(mtr.leafs_sum, 6))
pprint(mtr.fi)
# print(mtr.predict(X.sample(10, random_state=42)))
# print(get_best_split(X, y))

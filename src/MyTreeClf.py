import numpy as np
import pandas as pd
from pprint import pprint

df = pd.read_csv('https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:, :4], df['target']

# X, y = df.iloc[:, :4].sample(50, random_state=42).reset_index(drop=1) , df['target'].sample(50, random_state=42).reset_index(drop=1)

X_test = X.sample(10, random_state=42)


class MyTreeClf:
    def __init__(self,
                 max_depth=5,
                 min_samples_split=2,
                 max_leafs=20,
                 bins=None,
                 criterion='entropy') -> None:
        self.criterion = criterion
        self.dict_bins = {}
        self.bins = bins
        self.tree = None
        self.max_leafs = max_leafs
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.leafs_cnt = 0
        self.leafs_sum = 0

    def __str__(self) -> str:
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    # ['node', 'col_name', split_val, [...], [...]] | ['leaf', prob1, ]
    def fit(self, X: pd.DataFrame, y: pd.Series):
        def is_leaf(sub: pd.Series, depth):
            return any([depth < 0,
                        sub.size == 1,
                        sub.value_counts().size == 1,
                        sub.size < self.min_samples_split,
                        self.leafs_cnt + 1 >= self.max_leafs]
                       )

        def probability_one(labels: pd.Series):
            return np.mean(labels)

        def tree_create_(X, y):
            if self.max_leafs in [1, 2]:
                col_name, split, ig = self.get_best_split2(X, y)
                left = X[X[col_name] <= split]
                p1 = probability_one(y[left.index])
                self.leafs_sum += p1
                leaf_left = ['leaf_left', p1]
                right = X[X[col_name] > split]
                p1 = probability_one(y[right.index])
                self.leafs_sum += p1
                leaf_right = ['leaf_right', p1]
                self.leafs_cnt = 2
                return ['node', col_name, split, leaf_left, leaf_right]
            else:
                return tree_create(X, y, depth=self.max_depth)

        def tree_create(X, y, depth=1):
            depth -= 1
            if is_leaf(y, depth) or (self.get_best_split2(X, y) is None):
                self.leafs_cnt += 1
                p1 = probability_one(y)
                self.leafs_sum += p1
                return ['leaf_left', p1]

            res = self.get_best_split2(X, y)
            col_name, split, ig = res
            left = X[X[col_name] <= split]
            sub_left = ['node', col_name, split,
                        tree_create(left.reset_index(drop=1), y[left.index].reset_index(drop=1), depth)]

            right = X[X[col_name] > split]
            if is_leaf(y[right.index], depth - 1) or (self.get_best_split2(right.reset_index(drop=1), y[right.index].reset_index(drop=1)) is None):
                self.leafs_cnt += 1
                p1 = probability_one(y[right.index])
                self.leafs_sum += p1
                sub_right = ['leaf_right', p1]
            else:
                sub_right = tree_create(right.reset_index(drop=1), y[right.index].reset_index(drop=1), depth)
            sub_left.append(sub_right)
            return sub_left

        self.tree = tree_create_(X, y)

    def print_tree(self):
        pprint(self.tree, indent=4)

    def predict_proba(self, X: pd.DataFrame):
        def walk_tree(obj, lst: list):
            if lst[0] in ['leaf_left', 'leaf_right']:
                return lst[1]
            if lst[0] == 'node':
                col_name = lst[1]
                if obj[col_name] <= lst[2]:
                    return walk_tree(obj, lst[3])
                else:
                    return walk_tree(obj, lst[4])

        res = X.apply(lambda x: walk_tree(x, self.tree), axis=1)
        return res

    def predict(self, X: pd.DataFrame):
        return np.where(self.predict_proba(X) > .5, 1, 0)

    def _hist_split(self, vals: np.ndarray, col: str):
        # if vals.size <= self.bins - 1:
        #     return vals
        if self.dict_bins.get(col) is None:
            _, bins = np.histogram(vals, bins=self.bins)
            self.dict_bins[col] = bins[1:-1]
        return self.dict_bins[col]

    def get_best_split2(self, X: pd.DataFrame, y: pd.Series):
        def best_split(X: pd.Series, y: pd.Series, col_name: str):
            # s0 = -(np.mean(y) * np.log2(np.mean(y)) + (1 - np.mean(y)) * np.log2(1 - np.mean(y)))
            unique_vals = np.sort(X.unique())
            if self.bins is not None:
                rules = self._hist_split(unique_vals, col_name)
            else:
                rules = (unique_vals[:-1] + unique_vals[1:]) * 0.5
            output = []

            def get_log(x):
                if x == 0:
                    return 0
                return np.log2(x)

            def entropy(left, right):
                s0 = -(np.mean(y) * np.log2(np.mean(y)) + (1 - np.mean(y)) * np.log2(1 - np.mean(y)))
                s1 = -(np.mean(y[left]) * get_log(np.mean(y[left])) + (1 - np.mean(y[left])) * get_log(
                    1 - np.mean(y[left])))
                s2 = -(np.mean(y[right]) * get_log(np.mean(y[right])) + (1 - np.mean(y[right])) * get_log(
                    1 - np.mean(y[right])))
                ig = s0 - (s1 * len(left) + s2 * len(right)) / len(y)
                return ig

            def gini(left: np.ndarray, right: np.ndarray):
                gp = 1 - np.mean(y)**2 - (1 - np.mean(y))**2
                gl = 1 - np.mean(y[left])**2 - (1 - np.mean(y[left]))**2
                gr = 1 - np.mean(y[right])**2 - (1 - np.mean(y[right]))**2
                return gp - (len(left) * gl + len(right) * gr) / len(y)

            for rule in rules:
                if X.min() <= rule < X.max():
                    left, right = np.where(X <= rule)[0], np.where(X > rule)[0]
                    # s1 = -(np.mean(y[left]) * get_log(np.mean(y[left])) + (1 - np.mean(y[left])) * get_log(
                    #     1 - np.mean(y[left])))
                    # s2 = -(np.mean(y[right]) * get_log(np.mean(y[right])) + (1 - np.mean(y[right])) * get_log(
                    #     1 - np.mean(y[right])))
                    # ig = s0 - (s1 * len(left) + s2 * len(right)) / len(y)
                    ig = 0
                    if self.criterion == 'entropy':
                        ig = entropy(left, right)
                    elif self.criterion == 'gini':
                        ig = gini(left, right)
                    output.append((X.name, rule, ig))

            if len(output) == 0:
                return None
            return max(output, key=lambda x: x[2])

        result = []
        for col in X.columns:
            res = best_split(X[col], y, col)
            if res is None:
                continue
            result.append(res)
        if result is None or len(result) == 0:
            return None
        col_name, split_value, ig = max(result, key=lambda x: x[2])

        return col_name, split_value, ig


def get_best_split(X: pd.DataFrame, y: pd.Series):
    col_name, split_value, ig = X.columns[0], 0, 0
    for col in range(X.shape[1]):
        feature = X.iloc[:, col].sort_values()
        label = y.iloc[feature.index]
        P0, P1 = label.value_counts(normalize=True).ravel()
        S0 = -(P0 * np.log2(P0) + P1 * np.log2(P1))
        IG, thresh = 0, 0
        thresh_feature = feature.drop_duplicates()
        for i in range(0, len(thresh_feature) - 1):
            thresh_temp = (thresh_feature.iloc[i] + thresh_feature.iloc[i + 1]) / 2
            left = feature[feature <= thresh_temp]
            right = feature[feature > thresh_temp]
            lab_left = label[left.index].value_counts(normalize=True)
            lab_right = label[right.index].value_counts(normalize=True)
            s_left, s_right = 0, 0
            if lab_left.size == 1:
                p0_r, p1_r = lab_right.ravel()
                s_right = -(p0_r * np.log2(p0_r) + p1_r * np.log2(p1_r))
            elif lab_right.size == 1:
                p0_l, p1_l = lab_left.ravel()
                s_left = -(p0_l * np.log2(p0_l) + p1_l * np.log2(p1_l))
            else:
                p0_l, p1_l = lab_left.ravel()
                p0_r, p1_r = lab_right.ravel()
                s_left = -(p0_l * np.log2(p0_l) + p1_l * np.log2(p1_l))
                s_right = -(p0_r * np.log2(p0_r) + p1_r * np.log2(p1_r))

            IG_temp = S0 - (left.size / feature.shape[0]) * s_left - (right.size / feature.shape[0]) * s_right
            if IG_temp > IG:
                IG, thresh = IG_temp, thresh_temp

        if IG > ig:
            col_name, split_value, ig = X.columns[col], thresh, IG
    return col_name, split_value, ig


# tr = MyTreeClf(max_depth=1, min_samples_split=1, max_leafs=2)
# tr = MyTreeClf(max_depth=3, min_samples_split=2, max_leafs=5)
# tr = MyTreeClf(max_depth=5, min_samples_split=200, max_leafs=10)
# tr = MyTreeClf(max_depth=4, min_samples_split=100, max_leafs=17)
# tr = MyTreeClf(max_depth=10, min_samples_split=40, max_leafs=21)
# tr = MyTreeClf(max_depth=15, min_samples_split=20, max_leafs=30)
# tr = MyTreeClf(max_depth=3, min_samples_split=2, max_leafs=2)

# tr = MyTreeClf(max_depth=1, min_samples_split=1, max_leafs=2, bins=8)
# tr = MyTreeClf(max_depth=3, min_samples_split=3, max_leafs=5, bins=None)
# tr = MyTreeClf(max_depth=5, min_samples_split=200, max_leafs=10, bins=4)
# tr = MyTreeClf(max_depth=4, min_samples_split=100, max_leafs=17, bins=16)
# tr = MyTreeClf(max_depth=10, min_samples_split=40, max_leafs=21, bins=10)
# tr = MyTreeClf(max_depth=15, min_samples_split=20, max_leafs=30, bins=6)

# tr = MyTreeClf(max_depth=1, min_samples_split=1, max_leafs=2, bins=8, criterion='gini')
# tr = MyTreeClf(max_depth=3, min_samples_split=2, max_leafs=5, bins=None, criterion='gini')
# tr = MyTreeClf(max_depth=5, min_samples_split=200, max_leafs=10, bins=4, criterion='entropy')
# tr = MyTreeClf(max_depth=4, min_samples_split=100, max_leafs=17, bins=16, criterion='gini')
tr = MyTreeClf(max_depth=10, min_samples_split=40, max_leafs=21, bins=10, criterion='gini')
# tr = MyTreeClf(max_depth=15, min_samples_split=20, max_leafs=30, bins=6, criterion='gini')

tr.fit(X, y)
pprint(tr.tree, indent=4)
print(tr.leafs_cnt, round(tr.leafs_sum, 6))
# print(tr.predict_proba(X_test))
# print(tr.predict(X_test))

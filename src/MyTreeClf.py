import numpy as np
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:,:4], df['target']

class MyTreeClf:
    def __init__(self,
                 max_depth=5,
                 min_samples_split=2,
                 max_leafs=20) -> None:
        self.max_leafs = max_leafs
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.leafs_cnt = 0

    def __str__(self) -> str:
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    def fit(self, X: pd.DataFrame, y: pd.Series):

        def tree_create(X, y, depth=0):
            depth += 1
            col_name, split, ig = get_best_split(X, y)
            left = X[col_name <= split]
            right = X[col_name > split]




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
                s_left = -(p0_l*np.log2(p0_l) + p1_l*np.log2(p1_l))
            else:
                p0_l, p1_l = lab_left.ravel()
                p0_r, p1_r = lab_right.ravel()
                s_left = -(p0_l*np.log2(p0_l) + p1_l*np.log2(p1_l))
                s_right = -(p0_r * np.log2(p0_r) + p1_r * np.log2(p1_r))

            IG_temp = S0 - (left.size / feature.shape[0]) * s_left - (right.size / feature.shape[0]) * s_right
            if IG_temp > IG:
                IG, thresh = IG_temp, thresh_temp

        if IG > ig:
            col_name, split_value, ig = X.columns[col], thresh, IG
    return col_name, split_value, ig

def get_best_split2(X: pd.DataFrame, y: pd.Series):
    def best_split(X: pd.Series, y: pd.Series):
        s0 = -(np.mean(y) * np.log2(np.mean(y)) + (1 - np.mean(y)) * np.log2(1 - np.mean(y)))
        unique_vals = np.sort(X.unique())
        rules = (unique_vals[:-1] + unique_vals[1:]) * 0.5
        output = []

        def get_log(x):
            if x == 0:
                return 0
            return np.log2(x)

        for rule in rules:
            left, right = np.where(X <= rule)[0], np.where(X > rule)[0]
            s1 = -(np.mean(y[left]) * get_log(np.mean(y[left])) + (1 - np.mean(y[left])) * get_log(1 - np.mean(y[left])))
            s2 = -(np.mean(y[right]) * get_log(np.mean(y[right])) + (1 - np.mean(y[right])) * get_log(1 - np.mean(y[right])))
            ig = s0 - (s1 * len(left) + s2 * len(right)) / len(y)
            output.append((X.name, rule, ig))
        return max(output, key=lambda x: x[2])

    result = []
    for col in X.columns:
        result.append(best_split(X[col], y))
    col_name, split_value, ig = max(result, key=lambda x: x[2])

    return col_name, split_value, ig

tree = MyTreeClf()
print(get_best_split(X, y))
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=100, n_features=3, n_informative=2, random_state=42)
# X = pd.DataFrame(X)
# y = pd.Series(y)
# X.columns = [f'col_{col}' for col in X.columns]

class MyKNNClf:
    def __init__(self,
                 k=3,
                 metric='euclidean',
                 weight='uniform') -> None:
        self.weight = weight
        self.metric = metric
        self.k = k
        self.train_size = 0

    def __str__(self) -> str:
        return f"MyKNNClf class: k={self.k}"


    def _distance_euclidean(self, row1, row2):
        return np.sqrt(np.sum((row1 - row2) ** 2))

    def _distance_chebyshev(self, row1, row2):
        return np.max(np.abs(row1 - row2))

    def _distance_manhattan(self, row1, row2):
        return np.sum(np.abs(row1 - row2))

    def _distance_cosine(self, row1, row2):
        return 1 - (np.sum(row1 * row2) / (np.sqrt(np.sum(row1 ** 2)) * np.sqrt(np.sum(row2 ** 2))))

    def _get_prediction_by_weights(self, lst, distance=None, prob=False):
        lst = np.array(lst)
        if distance is not None:
            distance = np.array(distance)
        dcl = defaultdict(list)
        for i in range(len(lst)):
            denominator = (i + 1) if self.weight == 'rank' else distance[i]
            if lst[i] == 0:
                dcl['0'].append(1 / denominator)
            elif lst[i] == 1:
                dcl['1'].append(1 / denominator)
        if len(dcl) == 1:
            return lst[0]
        q0 = np.sum(dcl['0']) / np.sum(dcl['0'] + dcl['1'])
        q1 = np.sum(dcl['1']) / np.sum(dcl['0'] + dcl['1'])
        if prob:
            return q1
        return 0 if q0 > q1 else 1

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.train_size = X_train.shape

    def predict(self, X_test: pd.DataFrame):
        predictions=[1 if p >= .5 else 0 for p in self.predict_proba(X_test)]
        return np.array(predictions)

    def predict_proba(self, X_test: pd.DataFrame):
        # train = np.expand_dims(self.X, axis=0)
        # test = np.expand_dims(X.to_numpy(), axis=1)
        # distances = np.sqrt(np.sum((test - train) ** 2, axis=-1))
        # indx = np.argsort(distances)[:, :self.k]
        #
        # return np.mean(self.y[indx], axis=1)
        predictions = []
        for _, row1 in X_test.iterrows():
            distances = []
            for _, row2 in self.X_train.iterrows():
                distances.append(getattr(self, '_distance_' + self.metric)(row1, row2))
            k_indices = np.argsort(distances)[:self.k]
            label = self.y_train[k_indices]
            if self.weight == 'uniform':
                prob = label.value_counts(normalize=True).to_dict()
                predictions.append(prob.get(1, 0))
            elif self.weight == 'rank':
                predictions.append(self._get_prediction_by_weights(label, prob=True))
            elif self.weight == 'distance':
                dist = np.array(distances)[k_indices]
                predictions.append(self._get_prediction_by_weights(label, dist, prob=True))

        return np.array(predictions)


train_data = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'feature2': [1, 2, 3, 4],
    'feature3': [1, 2, 3, 4]
})

train_labels = pd.Series([0, 1, 1, 0])  # Метки классов

# Тестовая выборка
test_data = pd.DataFrame({
    'feature1': [1.5, 2.5],
    'feature2': [1.5, 2.5],
    'feature3': [1.5, 2.5]
})

knn = MyKNNClf(k=3, weight='uniform')
knn.fit(train_data, train_labels)
print(knn.predict(test_data))
print(knn.predict_proba(test_data).sum())

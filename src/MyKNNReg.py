import numpy as np
import pandas as pd


class MyKNNReg:
    def __init__(self,
                 k: int = 3,
                 metric='euclidean',
                 weight='uniform') -> None:
        self.weight = weight
        self.metric = metric
        self.train_size = None
        self.k = k

    def __str__(self) -> str:
        return f"MyKNNReg class: k={self.k}"

    def _get_distances(self, train: np.ndarray, test: np.ndarray):
        met = {
            'euclidean': np.sqrt(np.sum((test - train) ** 2, axis=-1)),
            'chebyshev': np.max(np.abs(test - train), axis=-1),
            'manhattan': np.sum(np.abs(test - train), axis=-1),
            'cosine': 1 - np.sum(test * train, axis=-1) / (np.sqrt(np.sum(test ** 2, axis=-1)) * np.sqrt(np.sum(train ** 2, axis=-1)))
        }

        if self.metric:
            return met[self.metric]
        return None

    def _get_ws(self, idx, dist):
        pass

    def _get_y_weight(self, idx, dist):
        ws = (1 / np.arange(1, self.k + 1)) / (np.sum(1 / np.arange(1, self.k + 1)))
        dict_w = {
            'uniform': np.mean(self.y[idx], axis=1),
            'rank': np.dot(self.y[idx], ws),
            'distance': np.sum((1/dist) * self.y[idx], axis=1) / np.sum(1/dist, axis=1)
        }

        return dict_w[self.weight]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X = X_train.to_numpy()
        self.y = y_train.to_numpy()
        self.train_size = X_train.shape

    def predict(self, X_test: pd.DataFrame):
        train = np.expand_dims(self.X, axis=0)
        test = np.expand_dims(X_test.to_numpy(), axis=1)
        distances = self._get_distances(train, test)
        idx = np.argsort(distances)[:, :self.k]
        # dist_idx =  np.take_along_axis(distances, idx, axis=1)
        dist_idx = np.sort(distances)[:, :self.k]
        pred = self._get_y_weight(idx, dist_idx)
        # pred = np.mean(self.y[idx], axis=1)
        return pred


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

knn_reg = MyKNNReg(weight='distance')
knn_reg.fit(train_data, train_labels)
print(knn_reg.predict(test_data))

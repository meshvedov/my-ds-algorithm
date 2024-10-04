import numpy as np
import pandas as pd
class MyKNNClf:

    def __init__(self, k: int=3, metric="euclidean", weight="uniform"):
        self.k = k
        self.train_size = None
        self.metric=metric
        self.weight=weight

    def __str__(self):
        return f"MyKNNClf class: k={self.k}"

    def __repr__(self):
        return f"MyKNNClf class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.train_size = self.X.shape

    def predict_proba(self, X: pd.DataFrame):
        train = np.expand_dims(self.X, axis=0)
        test = np.expand_dims(X.to_numpy(), axis=1)
        distances = self.__get_distance(train, test)
        indx = np.argsort(distances)[:, :self.k]
        dist = np.sort(distances)[:, :self.k]

        return self.__get_proba(indx, dist)

    def predict(self, X: pd.DataFrame):
        return np.array([1 if prob >= 0.5 else 0 for prob in self.predict_proba(X)])

    def __get_distance(self, train, test):
        cosine = 1 - np.sum(train * test, axis=-1) / (np.sqrt(np.sum(train**2, axis=-1)) * np.sqrt(np.sum(test**2, axis=-1)))
        dist_formulas ={
            "euclidean": np.sqrt(np.sum((test - train) ** 2, axis=-1)),
            "manhattan": np.sum(np.abs(test - train), axis=-1),
            "chebyshev": np.max(np.abs(test - train), axis=-1),
            "cosine": cosine
        }
        if self.metric in dist_formulas:
            return dist_formulas[self.metric]

    def __get_proba(self, indx, dist):
        proba_formulas = {
            "uniform": np.mean(self.y[indx], axis=1),
            "rank": np.sum(self.y[indx] / np.arange(1, self.k + 1), axis=1) / np.sum(1 / np.arange(1, self.k + 1)),
            "distance": np.sum(1 / dist * self.y[indx], axis=1) / np.sum(1 / dist, axis=1)
        }
        if self.weight in proba_formulas:
            return proba_formulas[self.weight]


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

knn = MyKNNClf(weight='rank')
knn.fit(train_data, train_labels)
knn.predict(test_data)
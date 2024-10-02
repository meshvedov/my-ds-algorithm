import numpy as np
import pandas as pd

from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=100, n_features=3, n_informative=2, random_state=42)
# X = pd.DataFrame(X)
# y = pd.Series(y)
# X.columns = [f'col_{col}' for col in X.columns]

class MyKNNClf:
    def __init__(self, k=3) -> None:
        self.k = k
        self.train_size = 0

    def __str__(self) -> str:
        return f"MyKNNClf class: k={self.k}"

    def __distance_evclid(self, row1, row2):
        return np.sqrt(np.sum((row1 - row2) ** 2))


    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.train_size = X_train.shape

    def predict(self, X_test: pd.DataFrame):
        predictions = []
        for _, row1 in X_test.iterrows():
            distances = []
            for _, row2 in self.X_train.iterrows():
                distances.append(self.__distance_evclid(row1, row2))
            k_indices = np.argsort(distances)[:self.k]
            # k_nearest_labels = self.X_train.iloc[k_indices].index.values
            most_common_label = self.y_train[k_indices].mode()[0]
            predictions.append(most_common_label)

        return np.array(predictions)

    def predict_proba(self, X_test: pd.DataFrame):
        if self.k == 1:
            return np.ones(shape=len(X_test))
        predictions = []
        for _, row1 in X_test.iterrows():
            distances = []
            for _, row2 in self.X_train.iterrows():
                distances.append(self.__distance_evclid(row1, row2))
            k_indices = np.argsort(distances)[:self.k]
            prob = self.y_train[k_indices].value_counts(normalize=True)[1]
            predictions.append(prob)

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

knn = MyKNNClf(k=1)
knn.fit(train_data, train_labels)
print(knn.predict(test_data))
print(knn.predict_proba(test_data).sum())

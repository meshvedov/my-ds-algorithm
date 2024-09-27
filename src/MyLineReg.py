import pandas as pd
import numpy as np

from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=5, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learn_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learn_rate}"

    def fit(self, X, y, verbose=False):
        x = X.copy()
        x.insert(0, 'inter', 1)
        self.weights = np.ones(x.shape[1])

        for i in range(self.n_iter):
            y_pred = np.dot(x, self.weights)
            mse = np.mean((y - y_pred) ** 2)
            grad = 2 / len(x) * np.dot((y_pred - y), x)
            self.weights -= self.learn_rate * grad
            if verbose:
                if i == 0:
                    print(f"start | loss: {mse}")
                elif i % verbose == 0:
                    print(f"{i}    | loss: {mse}")

    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def predict(self, X: pd.DataFrame):
        temp = X.copy()
        temp.insert(0, 'inter', 1)
        return np.dot(temp, self.weights)


myreg = MyLineReg()
myreg.fit(X, y, verbose=10)

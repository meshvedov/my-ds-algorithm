from tabnanny import verbose

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Optional, Union, Callable

from sklearn.metrics import log_loss

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1, weights=None):
        self.weights = weights
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        eps = 1e-15
        x = X[:]
        x.insert(0, 'inter', 1)
        self.weights = np.ones(x.shape[1])
        y_sigmoid = 1 / (1 + np.exp(-np.dot(x, self.weights)))

        for i in range(self.n_iter):
            log_loss = -np.mean(y*np.log(y_sigmoid + eps) + (1 - y) * np.log(1 - y_sigmoid + eps))
            grad = 1 / len(x) * ((y_sigmoid - y) @ x)
            self.weights -= self.learning_rate * grad
            y_sigmoid = 1 / (1 + np.exp(-np.dot(x, self.weights)))

            if verbose and i % verbose == 0:
                print(f"{i} | loss: {log_loss:.4f}")

    def get_coef(self):
        return self.weights.to_numpy()[1:]





mylog = MyLogReg(n_iter=50)
mylog.fit(X, y, verbose=10)
print(np.mean(mylog.get_coef()))
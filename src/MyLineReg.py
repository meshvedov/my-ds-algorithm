import random

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# нампай: 1.26.4
# пандас: 2.1.2

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]


class MyLineReg:
    def __init__(self,
                 n_iter=100,
                 learning_rate=0.1,
                 weights=None,
                 metric=None,
                 reg: str = None,
                 l1_coef=0,
                 l2_coef=0,
                 sgd_sample=None,
                 random_state=42):
        self.random_state = random_state
        self.sgd_sample = sgd_sample
        self.l2_coef = l2_coef
        self.l1_coef = l1_coef
        self.reg = reg
        self.n_iter = n_iter
        self.learn_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.metric_score = None

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learn_rate}"

    def _eval_metric_(self, y, y_pred):
        # mae mse rmmse mape r2
        residual = np.subtract(y, y_pred)
        metric_name = self.metric
        metric = 0
        if metric_name == 'mae':
            metric = np.mean(np.abs(residual))
        elif metric_name == 'mse':
            metric = np.mean(residual ** 2)
        elif metric_name == 'rmse':
            metric = np.sqrt(np.mean(residual ** 2))
        elif metric_name == 'mape':
            metric = 100 * np.mean(np.abs(np.divide(residual, y)))
        elif metric_name == 'r2':
            y_avg = np.mean(y)
            metric = 1 - np.divide(np.sum(residual ** 2), np.sum((y - y_avg) ** 2))
        else:
            metric = 0
        return metric

    def _loss_grad(self, y: pd.Series, y_pred: pd.Series, x: pd.DataFrame, idx=None):
        if idx:
            err = y_pred[idx] - y[idx]
            x = x.iloc[idx, :]
        else:
            err = y_pred - y
        loss = np.mean((y - y_pred) ** 2)
        grad = 2 / len(x) * np.dot(err, x)
        if self.reg and (self.l1_coef or self.l2_coef):
            if self.reg == 'l1' and self.l1_coef:
                loss += self.l1_coef * np.sum(np.abs(self.weights))
                grad += self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2' and self.l2_coef:
                loss += self.l2_coef * np.sum(self.weights ** 2)
                grad += self.l2_coef * 2 * self.weights
            elif self.reg == 'elasticnet' and self.l1_coef and self.l2_coef:
                loss += self.l1_coef * np.sum(np.abs(self.weights))
                grad += self.l1_coef * np.sign(self.weights)
                loss += self.l2_coef * np.sum(self.weights ** 2)
                grad += self.l2_coef * 2 * self.weights
        return loss, grad

    def _get_learning_rate(self, step: int):
        if callable(self.learn_rate):
            return self.learn_rate(step)
        else:
            return self.learn_rate

    def _get_batch_idx(self, size: int):
        idxes = None
        if self.sgd_sample:
            len_=size
            if type(self.sgd_sample) == int and self.sgd_sample > 1:
                len_ = self.sgd_sample
            elif type(self.sgd_sample) == float and self.sgd_sample <= 1:
                len_ = round(size * self.sgd_sample)
            idxes = random.sample(range(size), k=len_ )
        return idxes

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)
        x = X.reset_index(drop=True)
        x.insert(0, 'inter', 1)
        batch_idx= self._get_batch_idx(len(x))
        self.weights = np.ones(x.shape[1])
        y_pred = np.dot(x, self.weights)
        loss_metric, grad = self._loss_grad(y, y_pred, x, batch_idx)
        if verbose:
            print(f"start | loss: {loss_metric} | "
                  f"{self.metric + ': ' + str(self._eval_metric_(y, y_pred)) if self.metric else ''}")

        for i in range(1, self.n_iter + 1):
            lr =  self._get_learning_rate(i)
            self.weights -= lr * grad
            batch_idx = self._get_batch_idx(len(x))
            y_pred = np.dot(x, self.weights)
            loss_metric, grad = self._loss_grad(y, y_pred, x, batch_idx)
            if self.metric:
                self.metric_score = self._eval_metric_(y, y_pred)

            if verbose:
                if i % verbose == 0:
                    print(f"{i}    | loss: {loss_metric} | "
                          f"{self.metric + ': ' + str(self.metric_score) if self.metric else ''}"
                          f" | lr: {lr}")

    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def predict(self, X: pd.DataFrame):
        temp = X.copy()
        temp.insert(0, 'inter', 1)
        return np.dot(temp, self.weights)

    def get_best_score(self):
        return self.metric_score



myreg = MyLineReg(n_iter=100, metric='mae', learning_rate = lambda iter: 0.5 * (0.85 ** iter), sgd_sample=.2)
myreg.fit(X, y, verbose=10)
print(myreg.get_best_score())
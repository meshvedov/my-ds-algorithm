import pandas as pd
import numpy as np

from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg: str = None, l1_coef=0, l2_coef=0):
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

    def _loss_grad(self, y_pred, x):
        loss = np.mean((y - y_pred) ** 2)
        grad = 2 / len(x) * np.dot((y_pred - y), x)
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

    def _func_loss(self, y_pred, x):
        loss = np.mean((y - y_pred) ** 2)

        return loss
    def fit(self, X, y, verbose=False):
        x = X.copy()
        x.insert(0, 'inter', 1)
        self.weights = np.ones(x.shape[1])
        y_pred = np.dot(x, self.weights)
        loss_metric, grad = self._loss_grad(y_pred, x)
        if verbose:
            print(f"start | loss: {loss_metric} | "
                  f"{self.metric + ': ' + str(self._eval_metric_(y, y_pred)) if self.metric else ''}")

        for i in range(self.n_iter):
            # grad = self._grad(y_pred, x)
            self.weights -= self.learn_rate * grad
            y_pred = np.dot(x, self.weights)
            loss_metric, grad = self._loss_grad(y_pred, x)
            if self.metric:
                self.metric_score = self._eval_metric_(y, y_pred)

            if verbose:
                if i % verbose == 0:
                    print(f"{i}    | loss: {loss_metric} | "
                          f"{self.metric + ': ' + str(self.metric_score) if self.metric else ''}")

    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def predict(self, X: pd.DataFrame):
        temp = X.copy()
        temp.insert(0, 'inter', 1)
        return np.dot(temp, self.weights)

    def get_best_score(self):
        return self.metric_score


myreg = MyLineReg(n_iter=100, metric='mae')
myreg.fit(X, y, verbose=10)
print(myreg.get_best_score())

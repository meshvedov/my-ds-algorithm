from tabnanny import verbose

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Optional, Union, Callable
import random

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyLogReg:
    def __init__(self,
                 n_iter=10,
                 learning_rate=0.1,
                 weights=None,
                 metric: Optional[Union['accuracy', 'precision', 'recall', 'f1', 'roc_auc']] = None,
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
        self.metric = metric
        self.weights = weights
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metrics = []

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def __metric_classes(self, y_true, y_pred) -> (int, int, int, int):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp, tn, fp, fn

    def __metric_accuracy(self, tp, tn, fp, fn):
        return (tp + tn) / (tp + tn + fn + fp)

    def __metric_precision(self, tp, fp):
        return tp / (tp + fp)

    def __metric_recall(self, tp, fn):
        return tp / (tp + fn)

    def __calculate_auc(self, probabilities, labels):
        probabilities = np.array(probabilities)
        labels = np.array(labels)

        # Сортируем вероятности по убыванию, сохраняя индексы
        sorted_indices = np.argsort(-probabilities)
        sorted_labels = labels[sorted_indices]
        sorted_probs = probabilities[sorted_indices]

        # Определяем количество положительных и отрицательных классов
        P = np.sum(labels == 1)  # Количество положительных классов (1)
        N = np.sum(labels == 0)  # Количество отрицательных классов (0)

        # Переменные для расчета
        auc_sum = 0
        positive_count = 0  # Счётчик положительных классов выше текущего элемента

        # Итерируем по каждому объекту в отсортированном списке
        i = 0
        while i < len(sorted_labels):
            current_prob = sorted_probs[i]
            same_prob_positives = 0
            same_prob_negatives = 0

            # Считаем количество элементов с тем же score
            while i < len(sorted_labels) and sorted_probs[i] == current_prob:
                if sorted_labels[i] == 1:
                    same_prob_positives += 1
                else:
                    same_prob_negatives += 1
                i += 1

            # Для каждого отрицательного класса с тем же score
            auc_sum += same_prob_negatives * positive_count + same_prob_negatives * same_prob_positives / 2

            # Увеличиваем количество положительных классов выше текущего уровня
            positive_count += same_prob_positives

        # Рассчитываем финальный AUC
        auc = auc_sum / (P * N)
        return auc

    def __metric_score(self, X: pd.DataFrame, y: pd.Series):
        score = None
        x = X[:]
        y_pred = self.predict(x)
        tp, tn, fp, fn = self.__metric_classes(y, y_pred)
        if self.metric == 'accuracy':
            score = self.__metric_accuracy(tp, tn, fp, fn)
        elif self.metric == "precision":
            score = self.__metric_precision(tp, fp)
        elif self.metric == "recall":
            score = self.__metric_recall(tp, fn)
        elif self.metric == 'f1':
            prec = self.__metric_precision(tp, fp)
            recall = self.__metric_recall(tp, fn)
            score = 2 * (prec * recall) / (prec + recall)
        elif self.metric == "roc_auc":
            score = self.__calculate_auc(np.round(self.predict_proba(x), 10), y)
        return score

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)
        eps = 1e-15
        x = X[:].reset_index(drop=True)
        x.insert(0, 'inter', 1)
        self.weights = np.ones(x.shape[1])
        y_sigmoid = 1 / (1 + np.exp(-np.dot(x, self.weights)))

        for i in range(1, self.n_iter + 1):
            log_loss = -np.mean(y*np.log(y_sigmoid + eps) + (1 - y) * np.log(1 - y_sigmoid + eps))

            k=0
            if self.sgd_sample:
                if 0 < self.sgd_sample <= 1:
                    k=round(len(x) * self.sgd_sample)
                elif self.sgd_sample > 1:
                    k = self.sgd_sample
                idx = random.sample(range(len(x)), k=k)
                grad = 1 / len(x.iloc[idx, :]) * ((y_sigmoid[idx] - y.to_numpy()[idx]) @ x.iloc[idx, :])
            else:
                grad = 1 / len(x) * np.dot((y_sigmoid - y), x)

            if self.reg:
                if self.reg in ['l1', 'elasticnet']:
                    log_loss += self.l1_coef * np.sum(np.abs(self.weights))
                    grad += self.l1_coef * np.sign(self.weights)
                elif self.reg in ['l2', 'elasticnet']:
                    log_loss += self.l2_coef * np.sum(self.weights**2)
                    grad += self.l2_coef * 2 * self.weights


            lr = self.learning_rate
            if callable(self.learning_rate):
                lr = self.learning_rate(i)

            self.weights -= lr * grad
            y_sigmoid = 1 / (1 + np.exp(-np.dot(x, self.weights)))

            metric_score = self.__metric_score(x, y)
            self.metrics.append(metric_score)
            if verbose and i % verbose == 0:
                res_print = f"{i} | loss: {log_loss:.4f}"
                if self.metric:
                    res_print += f" | {self.metric}: {metric_score}"
                res_print += f" | lr: {lr}"
                print(res_print)

    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self, X: pd.DataFrame):
        x = X[:]
        if 'inter' not in x.columns:
            x.insert(0, 'inter', 1)
        return 1 / (1 + np.exp(-np.dot(x, self.weights)))

    def predict(self, X: pd.DataFrame):
        y_sigmoid = self.predict_proba(X)
        return np.where(y_sigmoid > .5, 1, 0)

    def get_best_score(self):
        return self.metrics[-1]


mylog = MyLogReg(n_iter=50, metric='recall', learning_rate= .425 #lambda iter: 0.5 * (0.85 ** iter)
, sgd_sample=.1)
mylog.fit(X, y, verbose=10)
print(mylog.get_best_score())
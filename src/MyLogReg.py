from tabnanny import verbose

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Optional, Union, Callable

from sklearn.metrics import log_loss

X, y = make_classification(n_samples=100, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyLogReg:
    def __init__(self,
                 n_iter=10,
                 learning_rate=0.1,
                 weights=None,
                 metric: Optional[Union['accuracy', 'precision', 'recall', 'f1', 'roc_auc']] = None):
        self.metric = metric
        self.weights = weights
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def __metric_classes(self, y_true, y_pred) -> (int, int, int, int):
        assert len(y_true) == len(y_pred), "len(y_true) must equal len(y_pred)"
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(y_true)):
            if y_true[i] == 0:
                if y_pred[i] == 0:
                    tn += 1
                elif y_pred[i] == 1:
                    fn += 1
            elif y_true[i] == 1:
                if y_pred[i] == 1:
                    tp += 1
                elif y_pred[i] == 0:
                    fp += 1
        return tp, tn, fp, fn

    def __metric_accuracy(self, tp, tn, fp, fn):
        return (tp + tn) / tp+tn+fn+fp

    def __metric_precision(self, tp, fp):
        return tp / (tp + fp)

    def __metric_recall(self, tp, fn):
        return tp / (tp + fn)

        import numpy as np

    def __calculate_auc(self, probabilities, labels):
        # Преобразуем данные в numpy массивы
        probabilities = np.array(probabilities)
        labels = np.array(labels)

        # Сортируем вероятности по убыванию, сохраняя индексы
        sorted_indices = np.argsort(-probabilities)
        sorted_labels = labels[sorted_indices]

        # Определяем количество положительных и отрицательных классов
        P = np.sum(labels == 1)  # Количество положительных классов (1)
        N = np.sum(labels == 0)  # Количество отрицательных классов (0)

        # Переменные для расчета
        auc_sum = 0

        # Итерируем по каждому объекту с классом 0
        for i, label in enumerate(sorted_labels):
            if label == 0:  # Если это отрицательный класс (0)
                # Считаем количество положительных классов выше (с большей вероятностью)
                positives_above = np.sum(sorted_labels[:i] == 1)

                # Считаем количество положительных классов с той же вероятностью
                positives_same = np.sum(sorted_labels[i:] == 1)

                # Добавляем количество положительных классов выше плюс половину положительных с таким же скором
                auc_sum += positives_above + 0.5 * positives_same

        # Рассчитываем финальный AUC
        auc = auc_sum / (P * N)
        return auc

    # Пример использования:
    # probabilities = [0.9, 0.8, 0.4, 0.7, 0.6, 0.5]
    # labels = [1, 0, 1, 0, 1, 0]  # Данные классов 0 и 1
    #
    # auc = calculate_auc(probabilities, labels)
    # print(f"AUC: {auc:.3f}")

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
            pass

        return score

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

            metric_score = 1
            if verbose and i % verbose == 0:
                res_print = f"{i} | loss: {log_loss:.4f} |"
                if self.metric:
                    res_print += f" {self.metric}: {metric_score}"
                print(res_print)

    def get_coef(self):
        return self.weights.to_numpy()[1:]

    def predict_proba(self, X: pd.DataFrame):
        x = X[:]
        x.insert(0, 'inter', 1)
        return 1 / (1 + np.exp(-np.dot(x, self.weights)))

    def predict(self, X: pd.DataFrame):
        y_sigmoid = self.predict_proba(X)
        return np.where(y_sigmoid > .5, 1, 0)





mylog = MyLogReg(n_iter=50, metric='f1')
mylog.fit(X, y, verbose=10)
print(np.mean(mylog.get_coef()))
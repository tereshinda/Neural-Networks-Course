import numpy as np


class CrossEntropy:
    def __init__(self, softmax: bool = True) -> None:
        self.softmax = softmax

    def fn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.softmax:
            return -np.sum(y_true * np.log(y_pred))

    def dfn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.softmax:
            return y_pred - y_true

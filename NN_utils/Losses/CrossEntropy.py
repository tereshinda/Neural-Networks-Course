import numpy as np


class CrossEntropy:
    def __init__(self, softmax: bool = True) -> None:
        self.softmax = softmax

    def fn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.softmax:
            #  Adding a small value for stability
            return -np.sum(y_true * np.log(y_pred + 1e-9))

    def dfn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.softmax:
            return y_pred - y_true

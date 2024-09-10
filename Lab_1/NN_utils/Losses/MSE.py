import numpy as np


class MSE:
    def fn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.square(y_true - y_pred))

    def dfn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return (2 / n) * (y_pred - y_true)

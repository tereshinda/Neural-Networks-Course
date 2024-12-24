import numpy as np


class BinaryCrossEntropy:
    def __init__(self):
        self.epsilon = 1e-15

    def fn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def dfn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

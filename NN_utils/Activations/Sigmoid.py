from NN_utils.Layers.Layer import Layer
import numpy as np


class Sigmoid(Layer):
    @staticmethod
    def fn(x: float | np.ndarray) -> float | np.ndarray:
        return 1 / (1 + np.exp(-x))

    # derivative
    @staticmethod
    def dfn(x: float | np.ndarray) -> float | np.ndarray:
        return Sigmoid.fn(x) * (1 - Sigmoid.fn(x))

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.a = x
        return self.fn(self.a)

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        return delta * self.dfn(self.a)

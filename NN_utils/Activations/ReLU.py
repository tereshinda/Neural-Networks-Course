from NN_utils.Layers.Layer import Layer
import numpy as np


class ReLU(Layer):
    @staticmethod
    def fn(x: float | np.ndarray) -> float | np.ndarray:
        return (x > 0) * x

    # Derivative
    @staticmethod
    def dfn(x: float | np.ndarray) -> float | np.ndarray:
        return x > 0

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.a = x
        return self.fn(self.a)

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        return delta * self.dfn(self.a)

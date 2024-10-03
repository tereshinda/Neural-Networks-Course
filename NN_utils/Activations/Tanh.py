from NN_utils.Layers.Layer import Layer
import numpy as np


class Tanh(Layer):
    @staticmethod
    def fn(x: float | np.ndarray) -> float | np.ndarray:
        return np.tanh(x)

    # Derivative
    @staticmethod
    def dfn(x: float | np.ndarray) -> float | np.ndarray:
        return 1 - np.square(Tanh.fn(x))

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.a = x
        return self.fn(self.a)

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        return delta * self.dfn(self.a)

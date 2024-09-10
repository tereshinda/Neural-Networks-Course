import numpy as np


class Layer:
    """Abstract class for any NN layer
    """

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        pass

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        pass

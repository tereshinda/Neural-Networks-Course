from NN_utils.Layers.Layer import Layer
import numpy as np


class Flatten(Layer):
    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.original_shape = x.shape

        return x.flatten().reshape(-1, 1)

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        return delta.reshape(self.original_shape)

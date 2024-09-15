from NN_utils.Layers.Layer import Layer
import numpy as np


class Softmax(Layer):
    def __init__(self, cross_entropy_loss: bool = True):
        self.y = None
        self.cross_entropy_loss = cross_entropy_loss

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x)
        self.y = exps / np.sum(exps)
        return self.y

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        # if we use CE, we immediately pass (y_pred - y_true) as a gradient
        if self.cross_entropy_loss:
            return delta

        delta_input = np.zeros_like(self.y)

        n_classes = self.y.shape[0]
        for i in range(n_classes):
            for j in range(n_classes):
                if i == j:
                    delta_input[i] = delta[i] * (self.y[i] * (1 - self.y[j]))
                else:
                    delta_input[i] -= delta[j] * self.y[i] * self.y[j]

        return delta_input

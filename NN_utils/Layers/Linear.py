from NN_utils.Layers.Layer import Layer
import numpy as np


class Linear(Layer):
    def __init__(self, in_neurons: int, out_neurons: int, random_seed: int = 42) -> None:
        # random generator
        self.rng = np.random.default_rng(random_seed)

        # parameter k for initialize weights
        self.k = np.sqrt(6. / (in_neurons + out_neurons))
        # initialize layer weights (Xavier)
        self.W = self.rng.uniform(-self.k, self.k + 1e-5, size=(out_neurons, in_neurons))
        # initialize bias
        self.b = self.rng.random((out_neurons, 1)) * 2 - 1

        # weight and bias changes
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # result from previous layer (current input)
        self.a = None

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.a = x
        return self.W @ x + self.b

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        self.dW = delta @ self.a.T
        self.db = np.sum(delta, axis=1, keepdims=True)
        new_delta = self.W.T @ delta

        # update params
        # assert self.W.shape == self.dW.shape

        self.W -= lr * self.dW
        self.b -= lr * self.db

        return new_delta

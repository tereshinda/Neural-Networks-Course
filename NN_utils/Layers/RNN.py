from NN_utils.Layers.Layer import Layer
from NN_utils.Activations.Tanh import Tanh
import numpy as np


class RNN(Layer):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, random_seed: int = 42) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # random generator
        self.rng = np.random.default_rng(random_seed)

        # Initialize weights
        self.k = 1. / self.hidden_size

        self.Whi = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.input_size))
        self.Whh = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.hidden_size))
        self.Why = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.output_size, self.hidden_size))
        # Initialize biases
        self.bh = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                   size=(self.hidden_size, 1))
        self.by = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                   size=(self.output_size, 1))

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        # we're taking a data batch at once
        self.batch_size = x.shape[0]
        # history of all vectors h_i for backprop (first is zeros)
        self.h_raw = np.zeros((self.batch_size + 1, self.hidden_size, 1))
        self.h = np.zeros((self.batch_size + 1, self.hidden_size, 1))

        for i in range(self.batch_size):
            self.h_raw[i + 1] = self.Whi @ x[i] + self.Whh @ self.h[i] + self.bh
            self.h[i + 1] = Tanh.fn(self.h_raw[i + 1])

        return self.Why @ self.h[-1] + self.by

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        self.dWhi = np.zeros_like(self.Whi)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)

        self.dWhy = delta @ self.h[-1].T
        self.dby = delta
        dh = self.Why.T @ delta

        for i in range(self.batch_size, 0, -1):
            dh_raw = dh * Tanh.dfn(self.h_raw[i])

            self.dbh += dh_raw
            self.dWhi += dh_raw @ self.x[i - 1].T
            self.dWhh += dh_raw @ self.h[i - 1].T

            dh = self.Whh.T @ dh_raw

        # update params
        self.Whi -= lr * self.dWhi
        self.Whh -= lr * self.dWhh
        self.Why -= lr * self.dWhy
        self.bh -= lr * self.dbh
        self.by -= lr * self.dby

        return np.array([])

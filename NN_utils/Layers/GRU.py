from NN_utils.Layers.Layer import Layer
from NN_utils.Activations.Tanh import Tanh
from NN_utils.Activations.Sigmoid import Sigmoid
import numpy as np


class GRU(Layer):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, random_seed: int = 42) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # random generator
        self.rng = np.random.default_rng(random_seed)

        # Initialize weights
        self.k = 1. / self.hidden_size
        # update gate
        self.Wiz = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.input_size))
        self.Whz = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.hidden_size))
        # reset gate
        self.Wir = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.input_size))
        self.Whr = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.hidden_size))
        # h_temp
        self.Wih = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.input_size))
        self.Whh = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.hidden_size))

        self.Why = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.output_size, self.hidden_size))
        # Initialize biases
        self.bz = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                   size=(self.hidden_size, 1))
        self.br = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                   size=(self.hidden_size, 1))
        self.bh = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                   size=(self.hidden_size, 1))
        self.by = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                   size=(self.output_size, 1))

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.batch_size = x.shape[0]

        self.h = np.zeros((self.batch_size + 1, self.hidden_size, 1))
        self.z_raw = np.zeros((self.batch_size, self.hidden_size, 1))
        self.z = np.zeros((self.batch_size, self.hidden_size, 1))
        self.r_raw = np.zeros((self.batch_size, self.hidden_size, 1))
        self.r = np.zeros((self.batch_size, self.hidden_size, 1))
        self.h_temp_raw = np.zeros((self.batch_size, self.hidden_size, 1))
        self.h_temp = np.zeros((self.batch_size, self.hidden_size, 1))

        for i in range(self.batch_size):
            # update gate
            self.z_raw[i] = self.Wiz @ self.x[i] + self.Whz @ self.h[i] + self.bz
            self.z[i] = Sigmoid.fn(self.z_raw[i])
            # reset gate
            self.r_raw[i] = self.Wir @ self.x[i] + self.Whr @ self.h[i] + self.br
            self.r[i] = Sigmoid.fn(self.r_raw[i])
            # h_temp
            self.h_temp_raw[i] = self.Wih @ self.x[i] + self.Whh @ (self.h[i] * self.r[i]) + self.bh
            self.h_temp[i] = Tanh.fn(self.h_temp_raw[i])

            self.h[i + 1] = (1 - self.z[i]) * self.h[i] + self.z[i] * self.h_temp[i]

        return self.Why @ self.h[-1] + self.by

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        self.dWiz = np.zeros_like(self.Wiz)
        self.dWhz = np.zeros_like(self.Whz)
        self.dWir = np.zeros_like(self.Wir)
        self.dWhr = np.zeros_like(self.Whr)
        self.dWih = np.zeros_like(self.Wih)
        self.dWhh = np.zeros_like(self.Whh)

        self.dbz = np.zeros_like(self.bz)
        self.dbr = np.zeros_like(self.br)
        self.dbh = np.zeros_like(self.bh)

        self.dWhy = delta @ self.h[-1].T
        self.dby = delta
        dh = self.Why.T @ delta

        for i in range(self.batch_size, 0, -1):
            # update gate
            dz_raw = (dh * self.h_temp[i - 1] - dh * self.h[i - 1]) * Sigmoid.dfn(self.z_raw[i - 1])
            self.dbz += dz_raw
            self.dWiz += dz_raw @ self.x[i - 1].T
            self.dWhz += dz_raw @ self.h[i - 1].T
            # h_temp
            dh_temp_raw = dh * self.z[i - 1] * Tanh.dfn(self.h_temp_raw[i - 1])
            self.dbh += dh_temp_raw
            self.dWih += dh_temp_raw @ self.x[i - 1].T
            self.dWhh += dh_temp_raw * self.r[i - 1] @ self.h[i - 1].T
            # reset gate
            dr_raw = dh_temp_raw * self.h[i - 1] * Sigmoid.dfn(self.r_raw[i - 1])
            self.dbr += dr_raw
            self.dWir += dr_raw @ self.x[i - 1].T
            self.dWhr += dr_raw @ self.h[i - 1].T

            dh = self.Whz.T @ dz_raw + self.Whr.T @ dr_raw + self.Whh.T @ (dh_temp_raw * self.r[i - 1])

        # update params
        self.Wiz -= lr * self.dWiz
        self.Whz -= lr * self.dWhz
        self.Wir -= lr * self.dWir
        self.Whr -= lr * self.dWhr
        self.Wih -= lr * self.dWih
        self.Whh -= lr * self.dWhh
        self.Why -= lr * self.dWhy

        self.bz -= lr * self.dbz
        self.br -= lr * self.dbr
        self.bh -= lr * self.dbh
        self.by -= lr * self.dby

        return np.array([])
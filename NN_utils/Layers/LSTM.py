from NN_utils.Layers.Layer import Layer
from NN_utils.Activations.Tanh import Tanh
from NN_utils.Activations.Sigmoid import Sigmoid
import numpy as np


class LSTM(Layer):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, random_seed: int = 42) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # random generator
        self.rng = np.random.default_rng(random_seed)

        # Initialize weights
        self.k = 1. / self.hidden_size
        # forget gate
        self.Wif = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.input_size))
        self.Whf = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.hidden_size))
        # input gate
        self.Wii = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.input_size))
        self.Whi = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.hidden_size))
        # cell state
        self.Wic = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.input_size))
        self.Whc = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.hidden_size))
        # output gate
        self.Wio = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.input_size))
        self.Who = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.hidden_size, self.hidden_size))

        self.Why = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                    size=(self.output_size, self.hidden_size))
        # Initialize biases
        self.bf = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                   size=(self.hidden_size, 1))
        self.bi = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                   size=(self.hidden_size, 1))
        self.bc = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                   size=(self.hidden_size, 1))
        self.bo = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                   size=(self.hidden_size, 1))
        self.by = self.rng.uniform(-self.k ** 0.5, self.k ** 0.5 + 1e-5,
                                   size=(self.output_size, 1))

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.batch_size = x.shape[0]

        # history of all hidden states for backprop
        # + 1 to store first vectors of zeros
        self.h = np.zeros((self.batch_size + 1, self.hidden_size, 1))
        # self.C_raw = np.zeros((self.batch_size + 1, self.hidden_size, 1))
        self.C = np.zeros((self.batch_size + 1, self.hidden_size, 1))
        self.C_temp_raw = np.zeros((self.batch_size, self.hidden_size, 1))
        self.C_temp = np.zeros((self.batch_size, self.hidden_size, 1))
        self.f_raw = np.zeros((self.batch_size, self.hidden_size, 1))
        self.f = np.zeros((self.batch_size, self.hidden_size, 1))
        self.i_raw = np.zeros((self.batch_size, self.hidden_size, 1))
        self.i = np.zeros((self.batch_size, self.hidden_size, 1))
        self.o_raw = np.zeros((self.batch_size, self.hidden_size, 1))
        self.o = np.zeros((self.batch_size, self.hidden_size, 1))

        for j in range(self.batch_size):
            # forget gate
            self.f_raw[j] = self.Wif @ x[j] + self.Whf @ self.h[j] + self.bf
            self.f[j] = Sigmoid.fn(self.f_raw[j])
            # cell state
            self.C_temp_raw[j] = self.Wic @ x[j] + self.Whc @ self.h[j] + self.bc
            self.C_temp[j] = Tanh.fn(self.C_temp_raw[j])
            # input gate
            self.i_raw[j] = self.Wii @ x[j] + self.Whi @ self.h[j] + self.bi
            self.i[j] = Sigmoid.fn(self.i_raw[j])
            # update cell
            self.C[j + 1] = self.C[j] * self.f[j] + self.i[j] * self.C_temp[j]
            # output gate
            self.o_raw[j] = self.Wio @ x[j] + self.Who @ self.h[j] + self.bo
            self.o[j] = Sigmoid.fn(self.o_raw[j])

            self.h[j + 1] = self.o[j] * Tanh.fn(self.C[j + 1])

        return self.Why @ self.h[-1] + self.by

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        self.dWif = np.zeros_like(self.Wif)
        self.dWhf = np.zeros_like(self.Whf)
        self.dWii = np.zeros_like(self.Wii)
        self.dWhi = np.zeros_like(self.Whi)
        self.dWic = np.zeros_like(self.Wic)
        self.dWhc = np.zeros_like(self.Whc)
        self.dWio = np.zeros_like(self.Wio)
        self.dWho = np.zeros_like(self.Who)

        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)

        self.dWhy = delta @ self.h[-1].T
        self.dby = delta
        dh = self.Why.T @ delta

        for j in range(self.batch_size, 0, -1):
            # output gate
            do_raw = dh * Tanh.fn(self.C[j]) * Sigmoid.dfn(self.o_raw[j - 1])
            self.dbo += do_raw
            self.dWio += do_raw @ self.x[j - 1].T
            self.dWho += do_raw @ self.h[j - 1].T

            dC_raw = dh * self.o[j - 1] * Tanh.dfn(self.C[j])
            # input gate
            di_raw = dC_raw * self.C_temp[j - 1] * Sigmoid.dfn(self.i_raw[j - 1])
            self.dbi += di_raw
            self.dWii += di_raw @ self.x[j - 1].T
            self.dWhi += di_raw @ self.h[j - 1].T
            # cell state
            dC_temp_raw = dC_raw * self.i[j - 1] * Tanh.dfn(self.C_temp_raw[j - 1])
            self.dbc += dC_temp_raw
            self.dWic += dC_temp_raw @ self.x[j - 1].T
            self.dWhc += dC_temp_raw @ self.h[j - 1].T
            # forget gate
            df_raw = dC_raw * self.C[j - 1] * Sigmoid.dfn(self.f_raw[j - 1])
            self.dbf += df_raw
            self.dWif += df_raw @ self.x[j - 1].T
            self.dWhf += df_raw @ self.h[j - 1].T

            dh = self.Who.T @ do_raw + self.Whi.T @ di_raw + self.Whc.T @ dC_temp_raw + self.Whf.T @ df_raw

        # update params
        self.Wif -= lr * self.dWif
        self.Whf -= lr * self.dWhf
        self.Wii -= lr * self.dWii
        self.Whi -= lr * self.dWhi
        self.Wic -= lr * self.dWic
        self.Whc -= lr * self.dWhc
        self.Wio -= lr * self.dWio
        self.Who -= lr * self.dWho
        self.Why -= lr * self.dWhy

        self.bf -= lr * self.dbf
        self.bi -= lr * self.dbi
        self.bc -= lr * self.dbc
        self.bo -= lr * self.dbo
        self.by -= lr * self.dby

        return np.array([])

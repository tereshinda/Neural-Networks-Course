from NN_utils.Layers.Layer import Layer
import numpy as np


class FeedForward(Layer):
    """
    FeedForward layer for transformer block
    """

    def __init__(self, d_model: int, d_ff: int, random_seed: int = 21) -> None:
        self.rng = np.random.default_rng(random_seed)

        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = self.rng.standard_normal((d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = self.rng.standard_normal((d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def d_relu(self, x: np.ndarray) -> np.ndarray:
        return x >= 0

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.x = x

        self.z1 = self.x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        return self.a1 @ self.W2 + self.b2

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        db2 = np.sum(delta, axis=0)
        dW2 = self.a1.T @ delta

        da1 = delta @ self.W2.T
        dz1 = da1 * self.d_relu(self.z1)

        db1 = np.sum(dz1, axis=0)
        dW1 = self.x.T @ dz1

        d_x = dz1 @ self.W1.T

        # update params
        self.W2 -= lr * dW2
        self.W1 -= lr * dW1
        self.b2 -= lr * db2
        self.b1 -= lr * db1

        return d_x

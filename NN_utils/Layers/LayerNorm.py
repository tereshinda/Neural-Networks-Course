from NN_utils.Layers.Layer import Layer
import numpy as np


class LayerNorm(Layer):
    def __init__(self, embed_size: int, eps: float = 1e-5):
        self.embed_size = embed_size

        self.gamma = np.ones(embed_size)
        self.beta = np.zeros(embed_size)
        self.eps = eps

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.x_norm = (x - self.mean) / self.std

        return self.gamma * self.x_norm + self.beta

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        d_beta = np.sum(delta, axis=0)
        d_gamma = np.sum(delta * self.x_norm, axis=0)

        d_x_norm = self.gamma * delta
        d_std = -np.sum(d_x_norm * (self.x - self.mean), axis=1, keepdims=True) / (self.std ** 2)
        d_var = 0.5 * d_std / self.std
        d_mean = -np.sum(d_x_norm, axis=1, keepdims=True) / self.std - \
                 2 * d_var * np.sum(self.x - self.mean, axis=1, keepdims=True) / self.x.shape[1]
        d_x = d_x_norm / self.std + 2 * d_var * (self.x - self.mean) / self.x.shape[1] \
              + d_mean / self.x.shape[1]

        # update params
        self.gamma -= lr * d_gamma
        self.beta -= lr * d_beta

        return d_x

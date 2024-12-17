from NN_utils.Layers.Layer import Layer
import numpy as np


class PositionalEncoding(Layer):
    """
    A class for constructing immutable position vectors of embeddings
    """

    def __init__(self, embed_size: int, max_len: int = 64) -> None:
        # embed_size = d
        self.embed_size = embed_size
        self.max_len = max_len

        # Initialization
        self.embeddings = np.zeros((self.max_len, self.embed_size))

        position = np.arange(0, self.max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_size, 2) * -(np.log(10000.0) / self.embed_size))

        self.embeddings[:, 0::2] = np.sin(position * div_term)
        self.embeddings[:, 1::2] = np.cos(position * div_term)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        return x + self.embeddings

    def backprop(self, delta: np.ndarray, lr: float = 1e-3) -> None:
        return delta

from NN_utils.Layers.Layer import Layer
import numpy as np


class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int, random_seed: int = 42) -> None:
        self.rng = np.random.default_rng(random_seed)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Dictionary of embeddings
        self.embeddings = self.rng.standard_normal((self.num_embeddings, self.embedding_dim))

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.indexes = x
        return self.embeddings[x]

    def backprop(self, delta: np.ndarray, lr: float) -> None:
        self.d_embeddings = np.zeros_like(self.embeddings)
        np.add.at(self.d_embeddings, self.indexes, delta)

        # Update params
        self.embeddings -= lr * self.d_embeddings

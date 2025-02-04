from NN_utils.Layers.Layer import Layer
import numpy as np


class SelfAttention(Layer):
    """
    Single-head Self Attention layer without mask
    """

    def __init__(self, input_dim: int, random_seed: int = 21) -> None:
        self.rng = np.random.default_rng(random_seed)
        # input_dim = embedding_size
        self.input_dim = input_dim
        # self.internal_dim = input_dim // 2

        self.Wq = self.rng.standard_normal((self.input_dim, self.input_dim))
        self.Wk = self.rng.standard_normal((self.input_dim, self.input_dim))
        self.Wv = self.rng.standard_normal((self.input_dim, self.input_dim))

    def softmax(self, x: np.ndarray) -> np.ndarray:
        # Stabilize
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def softmax_grad(self, softmax_output: np.ndarray, d_output: np.ndarray) -> np.ndarray:
        # np.diagflat(s)
        s_diag = np.einsum('ij,jk->ijk', softmax_output, np.eye(softmax_output.shape[1]))
        # s @ s.T
        s_dot = np.einsum('ij,ik->ijk', softmax_output, softmax_output)
        jacobian_matrix = s_diag - s_dot
        # jacobian_matrix @ d_output
        d_input = np.einsum('ijk,ik->ij', jacobian_matrix, d_output)
        return d_input

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.x = x

        self.Q = x @ self.Wq
        self.K = x @ self.Wk
        self.V = x @ self.Wv

        self.attention_scores = (self.Q @ self.K.T) / np.sqrt(self.input_dim)
        self.attention_weights = self.softmax(self.attention_scores)
        return self.attention_weights @ self.V

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        d_attention_weights = delta @ self.V.T
        d_V = self.attention_weights.T @ delta

        d_attention_scores = self.softmax_grad(softmax_output=self.attention_weights,
                                               d_output=d_attention_weights)

        d_attention_scores /= np.sqrt(self.input_dim)

        d_Q = d_attention_scores @ self.K
        d_K = d_attention_scores.T @ self.Q

        d_Wq = self.x.T @ d_Q
        d_Wk = self.x.T @ d_K
        d_Wv = self.x.T @ d_V

        # update params
        self.Wq -= lr * d_Wq
        self.Wk -= lr * d_Wk
        self.Wv -= lr * d_Wv

        return d_Q @ self.Wq.T + d_K @ self.Wk.T + d_V @ self.Wv.T

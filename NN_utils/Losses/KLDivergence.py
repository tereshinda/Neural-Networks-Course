import numpy as np


class KLDivergence:
    """
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """

    def fn(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        return -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var))

    def dfn(self, mu: np.ndarray, log_var: np.ndarray) -> (np.ndarray, np.ndarray):
        return mu, 0.5 * (np.exp(log_var) - 1)

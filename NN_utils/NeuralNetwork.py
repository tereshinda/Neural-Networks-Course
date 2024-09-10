import numpy as np
from NN_utils.Layers.Layer import Layer


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.feedforward(x)

        return x

    def backprop(self, delta: np.ndarray, lr: float):
        for layer in reversed(self.layers):
            delta = layer.backprop(delta, lr)

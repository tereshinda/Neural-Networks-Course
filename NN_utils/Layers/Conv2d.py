from NN_utils.Layers.Layer import Layer
import numpy as np


class Conv2d(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple, stride: int | tuple = 1,
                 padding: int | tuple = 0, random_seed: int = 42) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding if type(padding) == tuple else (padding, padding)

        # random generator
        self.rng = np.random.default_rng(random_seed)

        # Init filters
        self.filters = self.rng.standard_normal(
            (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))

        self.input = None
        self.input_padded = None
        self.d_input = None

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.input = x

        # Add padding to the input data
        self.input_padded = np.pad(self.input,
                                   pad_width=((0, 0),
                                              (self.padding[0], self.padding[0]),
                                              (self.padding[1], self.padding[1])),
                                   mode='constant', constant_values=0)

        # Calculate the matrix output dimension
        h_out = (self.input.shape[1] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (self.input.shape[2] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        self.output_result = np.zeros((self.out_channels, h_out, w_out))

        # Convolve through channels
        for i in range(h_out):
            for j in range(w_out):
                i_start = i * self.stride[0]
                j_start = j * self.stride[1]

                self.output_result[:, i, j] += np.sum(self.input_padded[:, i_start:i_start + self.kernel_size[0],
                                                      j_start:j_start + self.kernel_size[1]] * self.filters,
                                                      axis=(1, 2, 3))

        return self.output_result

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        self.d_filters = np.zeros_like(self.filters)
        self.d_input = np.zeros_like(self.input_padded)

        for f in range(self.out_channels):

            for c in range(self.in_channels):

                for i in range(delta.shape[1]):
                    for j in range(delta.shape[2]):
                        i_start = i * self.stride[0]
                        j_start = j * self.stride[1]

                        # weight gradients
                        self.d_filters[f][c] += self.input_padded[c][i_start:i_start + self.kernel_size[0],
                                                j_start:j_start + self.kernel_size[1]] * delta[f][i, j]
                        # input gradients
                        self.d_input[c][i_start:i_start + self.kernel_size[0],
                        j_start:j_start + self.kernel_size[1]] += self.filters[f][c] * delta[f][i, j]

        # Remove the padding
        self.d_input = self.d_input[:, self.padding[0]:-self.padding[0], self.padding[1]: -self.padding[1]]

        # update params
        self.filters -= lr * self.d_filters

        return self.d_input

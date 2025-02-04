from NN_utils.Layers.Layer import Layer
import numpy as np


class AvgPool2d(Layer):
    def __init__(self, kernel_size: int | tuple, stride: int | tuple = 2, padding: int | tuple = 0) -> None:
        self.kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding if type(padding) == tuple else (padding, padding)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.n_channels = self.input.shape[0]

        if self.padding != (0, 0):
            self.input_padded = np.pad(self.input,
                                       pad_width=((0, 0),
                                                  (self.padding[0], self.padding[0]),
                                                  (self.padding[1], self.padding[1])),
                                       mode='constant', constant_values=0)
        else:
            self.input_padded = self.input

        # Сalculate the matrix output dimension
        self.h_out = (self.input.shape[1] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        self.w_out = (self.input.shape[2] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        self.output_result = np.zeros((self.n_channels, self.h_out, self.w_out))

        # pooling on all channels in parallel
        for i in range(self.h_out):
            for j in range(self.w_out):
                i_start = i * self.stride[0]
                j_start = j * self.stride[1]

                self.output_result[:, i, j] = np.mean(self.input_padded[:, i_start:i_start + self.kernel_size[0],
                                                      j_start:j_start + self.kernel_size[1]], axis=(1, 2)
                                                      )

        return self.output_result

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        # value for a uniform gradient distribution
        avg_value = self.kernel_size[0] * self.kernel_size[1]

        # gradients to return
        delta_input = np.zeros(self.input_padded.shape)

        for i in range(self.h_out):
            for j in range(self.w_out):
                i_start = i * self.stride[0]
                j_start = j * self.stride[1]

                delta_input[:, i_start:i_start + self.kernel_size[0],
                j_start:j_start + self.kernel_size[1]] += delta[:, i:i + 1, j:j + 1] / avg_value

        # get rid of padding (even for asymmetrical windows)
        if self.padding[0] != 0:
            delta_input = delta_input[:, self.padding[0]:-self.padding[0], :]
        if self.padding[1] != 0:
            delta_input = delta_input[:, :, self.padding[1]: -self.padding[1]]

        return delta_input

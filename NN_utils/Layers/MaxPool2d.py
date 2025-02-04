from NN_utils.Layers.Layer import Layer
import numpy as np


class MaxPool2d(Layer):
    def __init__(self, kernel_size: int | tuple, stride: int | tuple = 2, padding: int | tuple = 0) -> None:
        self.kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding if type(padding) == tuple else (padding, padding)

        self.input = None

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
        h_out = (self.input.shape[1] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (self.input.shape[2] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        # value to return
        self.output_result = np.zeros((self.n_channels, h_out, w_out))

        # mask of max elements for backprop (array of tuples with coordinates)
        # self.mask = np.empty((self.n_channels, h_out, w_out), dtype=object)
        self.mask = np.zeros((self.n_channels, h_out, w_out, 2), dtype=np.int)

        for i in range(h_out):
            for j in range(w_out):
                i_start = i * self.stride[0]
                j_start = j * self.stride[1]

                current_window = self.input_padded[:, i_start:i_start + self.kernel_size[0],
                                 j_start:j_start + self.kernel_size[1]]
                # write max_values
                self.output_result[:, i, j] = np.max(current_window, axis=(1, 2))
                # store the indices of the maximum values
                idx, idy = np.unravel_index(np.argmax(current_window.reshape(self.n_channels, -1), axis=1),
                                            self.kernel_size)

                self.mask[:, i, j, 0] = i_start + idx
                self.mask[:, i, j, 1] = j_start + idy

        return self.output_result

    def backprop(self, delta: np.ndarray, lr: float) -> np.ndarray:
        # gradients to return
        delta_input = np.zeros(self.input_padded.shape)
        _, h_out, w_out = delta.shape

        for c in range(self.n_channels):

            for i in range(delta.shape[1]):
                for j in range(delta.shape[2]):
                    delta_input[c, self.mask[c, i, j, 0], self.mask[c, i, j, 1]] += delta[c][i, j]

        # get rid of padding (even for asymmetrical windows)
        if self.padding[0] != 0:
            delta_input = delta_input[:, self.padding[0]:-self.padding[0], :]
        if self.padding[1] != 0:
            delta_input = delta_input[:, :, self.padding[1]:-self.padding[1]]

        return delta_input

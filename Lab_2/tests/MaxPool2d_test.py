from NN_utils.Layers.MaxPool2d import MaxPool2d
import numpy as np

kernel_size = 2
stride = 1
padding = 1

input_data = np.array([[[7, 1, 3],
                        [4, 5, 6],
                        [3, 8, 7]],

                       [[1, 6, 8],
                        [5, 7, 2],
                        [4, 2, 4]],

                       [[1, 5, 7],
                        [7, 3, 3],
                        [8, 8, 5]]])

layer = MaxPool2d(kernel_size=2, stride=stride, padding=padding)
output_data = layer.feedforward(input_data)

expected_output = np.array([[[7, 7, 3, 3],
                             [7, 7, 6, 6],
                             [4, 8, 8, 7],
                             [3, 8, 8, 7]],

                            [[1, 6, 8, 8],
                             [5, 7, 8, 8],
                             [5, 7, 7, 4],
                             [4, 4, 4, 4]],

                            [[1, 5, 7, 7],
                             [7, 7, 7, 7],
                             [8, 8, 8, 5],
                             [8, 8, 8, 5]]])

np.testing.assert_array_equal(output_data, expected_output)

delta = np.array([[[0, 0.25, 1.25, 0],
                   [1, 0.75, 2, 0],
                   [0.25, 0, 1.75, 1],
                   [1, 2, 0.25, 0.75]],

                  [[0.25, 0, 2, 0],
                   [0.75, 1, 0.25, 0.25],
                   [1, 1, 1, 1],
                   [0, 0.75, 2, 0]],

                  [[0, 1, 1, 0],
                   [1, 2, 2, 1],
                   [1, 2, 2, 1],
                   [0, 1, 1, 0]]])

delta_input = layer.backprop(delta, 1e-3)
expected_delta_input = np.array([[[2, 0, 1.25],
                                  [0.25, 0, 2],
                                  [1, 4, 1.75]],

                                 [[0.25, 0, 2.5],
                                  [1.75, 3, 0],
                                  [0.75, 0, 3]],

                                 [[0, 1, 4],
                                  [3, 0, 0],
                                  [4, 3, 1]]])

# backprop testing
np.testing.assert_array_equal(delta_input, expected_delta_input)

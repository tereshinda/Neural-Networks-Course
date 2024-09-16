from NN_utils.Layers.AvgPool2d import AvgPool2d
import numpy as np

kernel_size = 2
stride = 2
padding = 1

input_data = np.array([[[1, 2, 3],
                        [0, 0, 0],
                        [4, 5, 6],
                        [1, 1, 1]],

                       [[2, 2, 2],
                        [0, 0, 0],
                        [0, 1, 0],
                        [1, 1, 1]]])

layer = AvgPool2d(kernel_size=2, stride=stride, padding=padding)
output_data = layer.feedforward(input_data)

expected_output = np.array([[[0.25, 1.25],
                             [1, 2.75],
                             [0.25, 0.5]],

                            [[0.5, 1],
                             [0, 0.25],
                             [0.25, 0.5]]])

# feedforward test
np.testing.assert_array_equal(output_data, expected_output)

delta = np.array([[[1, 2],
                   [3, 4],
                   [5, 6]],

                  [[7, 8],
                   [9, 10],
                   [11, 12]]])

delta_input = layer.backprop(delta, 1e-3)

expected_delta_input = np.array([[[0.25, 0.5, 0.5],
                                  [0.75, 1, 1],
                                  [0.75, 1, 1],
                                  [1.25, 1.5, 1.5]],

                                 [[1.75, 2, 2],
                                  [2.25, 2.5, 2.5],
                                  [2.25, 2.5, 2.5],
                                  [2.75, 3, 3]]])

# backprop testing
np.testing.assert_array_equal(delta_input, expected_delta_input)
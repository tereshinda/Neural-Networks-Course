from NN_utils.Layers.AvgPool2d import AvgPool2d
import numpy as np

# FEEDFORWARD
#######################################################
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

np.testing.assert_array_equal(output_data, expected_output)

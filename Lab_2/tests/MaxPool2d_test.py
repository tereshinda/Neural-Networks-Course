from NN_utils.Layers.MaxPool2d import MaxPool2d
import numpy as np

# FEEDFORWARD
#######################################################
kernel_size = 2
stride = 2
padding = 0

input_data = np.array([[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]],

                       [[17, 18, 19, 20],
                        [21, 22, 23, 24],
                        [25, 26, 27, 28],
                        [29, 30, 31, 32]],

                       [[33, 34, 35, 36],
                        [37, 38, 39, 40],
                        [41, 42, 43, 44],
                        [45, 46, 47, 48]],

                       [[49, 50, 51, 52],
                        [53, 54, 55, 56],
                        [57, 58, 59, 60],
                        [61, 62, 63, 64]]])

layer = MaxPool2d(kernel_size=2, stride=stride, padding=padding)
output_data = layer.feedforward(input_data)

expected_output = np.array([[[6, 8],
                             [14, 16]],

                            [[22, 24],
                             [30, 32]],

                            [[38, 40],
                             [46, 48]],

                            [[54, 56],
                             [62, 64]]])

np.testing.assert_array_equal(output_data, expected_output)

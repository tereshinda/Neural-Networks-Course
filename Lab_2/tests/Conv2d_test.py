from NN_utils.Layers.Conv2d import Conv2d
import numpy as np

input_channels = 2
output_channels = 3
filter_size = 3
stride = 2
padding = 1

# 2 x 4 x 4
input_data = np.array([[[0, 2, 0, 2],
                        [-2, 3, 1, -1],
                        [-2, 1, -1, 2],
                        [3, 2, 1, -2]],

                       [[2, 3, 0, 2],
                        [3, 3, 3, 3],
                        [0, -2, 0, 2],
                        [0, 1, 3, 2]],
                       ])

# 3 x 2 x 3 x 3
filters = np.array([[[[2, 0, 0],
                      [-1, 2, -1],
                      [1, 1, 2]],

                     [[0, -1, 1],
                      [0, -2, 3],
                      [1, 3, 3]]
                     ],

                    [[[1, 3, -1],
                      [-1, 2, -2],
                      [-1, 1, 0]],

                     [[3, 3, 0],
                      [0, -2, 3],
                      [-2, 3, 1]]
                     ],

                    [[[2, 2, 3],
                      [-2, 0, 0],
                      [0, -1, 3]],

                     [[-2, -1, 1],
                      [-2, 3, -2],
                      [-2, -2, 3]]
                     ]
                    ])

conv_layer = Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=filter_size, stride=stride,
                    padding=padding)
conv_layer.filters = filters

output_data = conv_layer.feedforward(input_data)
expected_output = np.array([[[25, 25],
                             [-1, 22]],

                            [[11, 4],
                             [-8, 32]],

                            [[14, -21],
                             [15, -12]]])

np.testing.assert_array_equal(output_data, expected_output)

from NN_utils.Layers.Conv2d import Conv2d
import numpy as np

input_channels = 2
output_channels = 3
filter_size = 3
stride = 2
padding = 1

# 2 x 4 x 4
input_data = np.array([[[0., 2., 0., 2.],
                        [-2., 3., 1., -1.],
                        [-2., 1., -1., 2.],
                        [3., 2., 1., -2.]],

                       [[2., 3., 0., 2.],
                        [3., 3., 3., 3.],
                        [0., -2., 0., 2.],
                        [0., 1., 3., 2.]],
                       ])

# 3 x 2 x 3 x 3
filters = np.array([[[[2., 0., 0.],
                      [-1., 2., -1.],
                      [1., 1., 2.]],

                     [[0., -1., 1.],
                      [0., -2., 3.],
                      [1., 3., 3.]]
                     ],

                    [[[1., 3., -1.],
                      [-1., 2., -2.],
                      [-1., 1., 0.]],

                     [[3., 3., 0.],
                      [0., -2., 3.],
                      [-2., 3., 1.]]
                     ],

                    [[[2., 2., 3.],
                      [-2., 0., 0.],
                      [0., -1., 3.]],

                     [[-2., -1., 1.],
                      [-2., 3., -2.],
                      [-2., -2., 3.]]
                     ]
                    ])

conv_layer = Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=filter_size, stride=stride,
                    padding=padding)
conv_layer.filters = filters

output_data = conv_layer.feedforward(input_data)
expected_output = np.array([[[25., 25.],
                             [-1., 22.]],

                            [[11., 4.],
                             [-8., 32.]],

                            [[14., -21.],
                             [15., -12.]]])

# forward
np.testing.assert_array_equal(output_data, expected_output)

delta = np.array([[[1., 2.],
                   [3., 4.]],

                  [[5., 6.],
                   [7., 8.]],

                  [[9., 10.],
                   [11., 12.]]])

expected_delta_input = np.array([[[12., -39., 16., -14.],
                                  [40., 91., 46., 62.],
                                  [20., -53., 24., -20.],
                                  [-1., 35., 0., 44.]],

                                 [[15., -20., 14., 4.],
                                  [7., 19., 12., 58.],
                                  [13., -16., 12., 12.],
                                  [8., 13., 12., 56.]]])

expected_delta_filters = np.array([[[[12., -2., 5.],
                                     [8., -10., 17.],
                                     [14., 13., -1.]],

                                    [[12., 21., 21.],
                                     [-2., 2., 9.],
                                     [10., 21., 20.]]],

                                   [[[24., -6., 13.],
                                     [20., -22., 45.],
                                     [34., 25., 7.]],

                                    [[24., 45., 45.],
                                     [2., 10., 29.],
                                     [26., 57., 56.]]],

                                   [[[36., -10., 21.],
                                     [32., -34., 73.],
                                     [54., 37., 15.]],

                                    [[36., 69., 69.],
                                     [6., 18., 49.],
                                     [42., 93., 92.]]]])

delta_input = conv_layer.backprop(delta, lr=1e-2)

# backprop
np.testing.assert_array_equal(delta_input, expected_delta_input)
np.testing.assert_array_equal(conv_layer.d_filters, expected_delta_filters)

# Test with PyTorch tensors
#############################
import torch
import torch.nn as nn

conv_layer_torch = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=2,
                             padding=1)

weights = torch.tensor([[[[2., 0., 0.],
                          [-1., 2., -1.],
                          [1., 1., 2.]],

                         [[0., -1., 1.],
                          [0., -2., 3.],
                          [1., 3., 3.]]
                         ],

                        [[[1., 3., -1.],
                          [-1., 2., -2.],
                          [-1., 1., 0.]],

                         [[3., 3., 0.],
                          [0., -2., 3.],
                          [-2., 3., 1.]]
                         ],

                        [[[2., 2., 3.],
                          [-2., 0., 0.],
                          [0., -1., 3.]],

                         [[-2., -1., 1.],
                          [-2., 3., -2.],
                          [-2., -2., 3.]]
                         ]
                        ], dtype=torch.float32)

bias = torch.tensor([0, 0, 0], dtype=torch.float32)

input_data_torch = torch.tensor([[[0., 2., 0., 2.],
                                  [-2., 3., 1., -1.],
                                  [-2., 1., -1., 2.],
                                  [3., 2., 1., -2.]],

                                 [[2., 3., 0., 2.],
                                  [3., 3., 3., 3.],
                                  [0., -2., 0., 2.],
                                  [0., 1., 3., 2.]],
                                 ], dtype=torch.float32, requires_grad=True)

conv_layer_torch.weight.data = weights
conv_layer_torch.bias.data = bias

output_torch = conv_layer_torch(input_data_torch)

delta_torch = torch.tensor([[[1., 2.],
                             [3., 4.]],

                            [[5., 6.],
                             [7., 8.]],

                            [[9., 10.],
                             [11., 12.]]], dtype=torch.float32)

output_torch.backward(delta_torch)

grad_input_torch = input_data_torch.grad.numpy()
grad_weight_torch = conv_layer_torch.weight.grad.numpy()

np.testing.assert_array_equal(delta_input, grad_input_torch)
np.testing.assert_array_equal(conv_layer.d_filters, grad_weight_torch)

#%%

'''A simple, flexible CNN for multi-class problems using Torch.

This CNN has two convolutional blocks. More blocks can be easily added.

version=0.0.1
date=2024-11-28
'''
from torch.nn import Module, Sequential, Conv2d, ReLU, BatchNorm2d, MaxPool2d, Linear

class SimpleCNN(Module):
    def __init__(self, num_classes, input_shape, out_initial = 32):
        super(SimpleCNN, self).__init__()
        self.input_shape = input_shape
        self.input_channels = self.input_shape[0]
        self.conv1 = self.conv_block(self.input_channels, out_initial)
        self.conv2 = self.conv_block(out_initial, out_initial * 2)
        self.flat_size = self.calculate_flattened_size(input_shape)
        self.linear1 = Linear(self.flat_size, out_initial * 4)
        self.relu = ReLU()
        self.linear2 = Linear(out_initial * 4, num_classes)
    
    def calculate_flattened_size(self, input_shape):
        c, h, w = input_shape
        x = torch.zeros(1, c, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        return x.numel()
    
    def conv_block(
        self, in_channels, out_channels,
        kernel = 3, stride = 1, padding = 1, pool_kernel = 2, pool_stride = 2
    ):
        block = Sequential(
            Conv2d(
                in_channels, out_channels,
                kernel_size = kernel, stride = stride, padding = padding
            ),
            ReLU(),
            BatchNorm2d(out_channels),
            MaxPool2d(kernel_size = pool_kernel, stride = pool_stride)
        )
        return block
    
    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

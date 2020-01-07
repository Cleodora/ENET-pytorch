import torch
import torch.nn as nn

kernel_stride = {
    True: 2,
    False: 1
}

def conv2x2(in_channels: int, out_channels: int):
    """
        A convolution with kernel size 2 x 2 and stride 2
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        2,
        2
    )


def batch_norm_activation(no_channels: int, activation_function):
    """
        Batch normalization and activation after each convolution
        as stated in the paper
    """
    batch_norm = nn.BatchNorm2d(no_channels, 1e3)
    activation = activation_function()
    return batch_norm, activation


def initial_block(in_channels: int, out_channels: int, conv_type: bool = False):
    """
        A convolution with kernel size 1 x 1 and stride 1 and bias true
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_stride[conv_type],
        kernel_stride[conv_type]
    )

def middle_block(in_channels: int, out_channels: int, dilation_rate: int = 1):
    """
        A convolution with kernel size 3 x 3 and stride 1 and bias true
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = 1,
        dilation = dilation_rate
    )

def middle_block_assymetric(in_channels: int):
    """
        An assymetric convolution with kernel size 5 x 1 and 1 x 5 with padding (2, 0) and (0, 2)
    """
    first_symmetry = nn.Conv2d(in_channels = in_channels, out_channels =  in_channels, kernel_size = (5, 1), padding=(2, 0), bias=False)
    second_symmetry = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = (1, 5), padding=(0, 2), bias=False)
    return first_symmetry, second_symmetry

def maxpool_2x2(return_indices: bool = True):
    """
        A Maxpool with kernel size 2 and stride 2
    """
    return nn.MaxPool2d(
        kernel_size = 2,
        stride = 2,
        return_indices = return_indices
    )


def dropout(dropout_prob: float = 0.1):
    """
        A Spatial dropout with either probability 0.1 or 0.01
    """
    return nn.Dropout2d(p = dropout_prob)

def maxunpool_2x2():
    """
        A Max unpooling with kernel size 2 x 2 
    """
    return nn.MaxUnpool2d(
        kernel_size = 2
    )


def ConvTranspose2_2x2(in_channels: int, out_channels: int):
    """
        A convolutional transpose used in the upsampling bottelneck
    """
    return nn.ConvTranspose2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 2,
        stride = 2
    )

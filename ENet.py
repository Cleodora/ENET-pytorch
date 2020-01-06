import torch
import torch.nn as nn

from Helper import *

class EnetInitialBlock(nn.Module):
    """ Initial Block for the ENET architecture
    There are two paths in the initial black.
    1. The first path takes the input, performs a 
    3 * 3 convolution with stride 2 and output 13 filters.
    2. The second path takes the input and performs 2 * 2 
    non-overlapping maxpooling abd gives out 3 filters corresponding
    to the RGB values.
    Both the filters are concatenated which gives 16 filters.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        use_PRelu: bool = True,
    ):
        super(EnetInitialBlock, self).__init__()
        if use_PRelu:
            activation_function = nn.PReLU(out_channels - 3)
        else:
            activation_function = nn.ReLU()
        
        self.conv3_3 = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size = (3,3),
            stride=2,
            padding=1
        )
        self.BN = nn.BatchNorm2d(out_channels - 3)
        self.activation = activation_function


        self.max2_2 = nn.MaxPool2d(
            kernel_size = (2, 2),
            stride= 2,
            padding= 1
        )
    

    def forward(self, input):
        conv = self.conv3_3(input)
        conv = self.BN(conv)
        conv = self.activation

        maxpool = self.max2_2
        concat = torch.cat((conv, maxpool),1)
        return concat


class DownSamplingBottleNeck(nn.Module):

    def init(self, in_channels, out_channels, internal_channels, activation_function, dropout_prob):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxpool = maxpool_2x2()
        self.initial_block = initial_block(in_channels, internal_channels, conv_type = 'downsampling')
        self.batch_norm1, self.activation1 = batch_norm_activation(internal_channels, activation_function)
        self.middle_block = middle_block(internal_channels, internal_channels)
        self.batch_norm2, self.activation2 = batch_norm_activation(internal_channels, activation_function)
        self.final_block = initial_block(internal_channels, out_channels)
        self.batch_norm3, self.activation3 = batch_norm_activation(out_channels, activation_function)
        self.dropout = dropout(dropout_prob)
        self.activation = activation_function()
    
    def forward(self, input):
        main, max_indices = self.maxpool(input)
        input_size = input.size()

        secondary = self.initial_block(input)
        secondary = self.batch_norm1(secondary)
        secondary = self.activation1(secondary)

        secondary = self.middle_block(secondary)
        secondary = self.batch_norm2(secondary)
        secondary = self.activation2(secondary)

        secondary = self.final_block(secondary)
        secondary = self.batch_norm3(secondary)
        secondary = self.activation3(secondary)



        if self.in_channels != self.out_channels:
            padding = torch.zeros(
                input_size[0],
                self.out_channels - self.in_channels,
                input_size[2] // 2,
                input_size[3] // 2
            )
            if main.is_cuda:
                padding = padding.cuda()

            concatenate = torch.cat((main, padding), 1)
            final_merge = concatenate + secondary
        else:    
            final_merge = main + secondary
        
        return final_merge, max_indices


class BottleNeck(nn.Module):

    def init(self, in_channels, out_channels, internal_channels, activation_function, dropout_prob, downsampling = False,upsampling = False, dilation = False, dilation_rate = 1, assymetric  = False  ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.initial_block = initial_block(in_channels, internal_channels, downsampling )
        self.batch_norm1, self.activation1 = batch_norm_activation(internal_channels, activation_function)

        if downsampling:
            self.maxpool = maxpool_2x2()
        elif upsampling:
            self.extension_block = initial_block(in_channels, out_channels)
            self.batch_norm_ext, _ = batch_norm_activation(out_channels, activation_function)
            self.max_unpool_ext = maxunpool_2x2()
            self.middle_block = ConvTranspose2_2x2(internal_channels, internal_channels)
        elif dilation:
            self.middle_block = middle_block(internal_channels, internal_channels, dilation_rate = dilation_rate)
        elif assymetric:
            self.middle_block = middle_block_assymetric(internal_channels)
        else:
            self.middle_block = middle_block(internal_channels, internal_channels)

        self.batch_norm2, self.activation2 = batch_norm_activation(internal_channels, activation_function)
        self.final_block = initial_block(internal_channels, out_channels)
        self.batch_norm3, self.activation3 = batch_norm_activation(out_channels, activation_function)
        self.dropout = dropout(dropout_prob)
        self.activation = activation_function()
    

    def forward(self, input, pool_indices):

        main = None
        secondary = self.initial_block(input)
        secondary = self.batch_norm1(secondary)
        secondary = self.activation1(secondary)

        secondary = self.middle_block(secondary)
        secondary = self.batch_norm2(secondary)
        secondary = self.activation2(secondary)

        secondary = self.final_block(secondary)
        secondary = self.batch_norm3(secondary)
        secondary = self.activation3(secondary)

        max_indices = None
        if self.downsampling:
            main, max_indices = self.maxpool(input)
            input_size = input.size()
            if self.in_channels != self.out_channels:
                padding = torch.zeros(
                    input_size[0],
                    self.out_channels - self.in_channels,
                    input_size[2] // 2,
                    input_size[3] // 2
                )
                if main.is_cuda:
                    padding = padding.cuda()

                extension_block = torch.cat((main, padding), 1)
            secondary = extension_block + secondary
        elif self.upsampling:
            extension_block = self.extension_block(input)
            extension_block = self.batch_norm_ext(extension_block)
            extension_block = self.max_unpool_ext(extension_block, pool_indices)
            secondary = extension_block + secondary
        else:
            secondary = input + secondary
        
        Final = F.PReLU(secondary)
        if self.downsampling:
            return Final, max_indices
        return Final
        

        


        
            
            
            
            














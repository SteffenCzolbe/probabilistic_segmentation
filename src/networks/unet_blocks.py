import torch
import torch.nn as nn
import numpy as np


class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """

    def __init__(self, input_dim, output_dim, padding, pool=True, p=0):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(nn.Dropout2d(p=p))
            layers.append(nn.AvgPool2d(
                kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Dropout2d(p=p))
        layers.append(nn.Conv2d(input_dim, output_dim,
                                kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout2d(p=p))
        layers.append(nn.Conv2d(output_dim, output_dim,
                                kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout2d(p=p))
        layers.append(nn.Conv2d(output_dim, output_dim,
                                kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """

    def __init__(self, input_dim, output_dim, padding, bilinear=True, p=0):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            upconv_layer = []
            upconv_layer.append(nn.Dropout2d(p=p))
            upconv_layer.append(nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2))
            self.upconv_layer = upconv_layer

        self.conv_block = DownConvBlock(
            input_dim, output_dim, padding, pool=False, p=p)

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(
                x, mode='bilinear', scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)

        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out

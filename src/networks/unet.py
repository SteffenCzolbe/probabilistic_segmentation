import torch.nn.functional as F
from .unet_blocks import *


class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    p: dropout probability
    """

    def __init__(self, input_channels, num_classes, num_filters, apply_last_layer=True, padding=True, p=0, batch_norm=False):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.apply_last_layer = apply_last_layer

        self.contracting_path = nn.ModuleList()
        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True
            self.contracting_path.append(
                DownConvBlock(input, output, padding, pool=pool, p=p, batch_norm=batch_norm))

        self.upsampling_path = nn.ModuleList()
        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(
                input, output, padding, p=p, batch_norm=batch_norm))

        if self.apply_last_layer:
            last_layer = []
            last_layer.append(nn.Conv2d(output, num_classes, kernel_size=1))
            self.last_layer = nn.Sequential(*last_layer)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])

        del blocks

        if self.apply_last_layer:
            x = self.last_layer(x)

        return x

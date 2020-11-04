import torch.nn as nn
from src.networks.unet import Unet


class MCDropoutUnet(nn.Module):
    """
    U-Net with a few dropout and convolutional layers at the end.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    p: dropout probability
    """

    def __init__(self, input_channels, num_classes, num_filters, p=0, batch_norm=False):
        super().__init__()

        self.unet = Unet(
            input_channels=input_channels,
            num_classes=None,
            num_filters=num_filters,
            batch_norm=batch_norm,
            p=p,
            apply_last_layer=False
        )

        channels = num_filters[0]
        dropout_layers = []
        dropout_layers.append(nn.Dropout2d(p=p))
        dropout_layers.append(nn.Conv2d(channels, channels,
                                        kernel_size=3, stride=1, padding=1))
        dropout_layers.append(nn.ReLU())
        # dropout_layers.append(nn.Dropout2d(p=p))
        dropout_layers.append(nn.Conv2d(channels, channels,
                                        kernel_size=3, stride=1, padding=1))
        dropout_layers.append(nn.ReLU())
        # dropout_layers.append(nn.Dropout2d(p=p))
        dropout_layers.append(nn.Conv2d(channels, num_classes,
                                        kernel_size=3, stride=1, padding=1))

        self.dropout_net = nn.Sequential(*dropout_layers)

    def forward(self, x):
        h = self.unet(x)
        return self.dropout_net(h)

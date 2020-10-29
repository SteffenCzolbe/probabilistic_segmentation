from .unet_blocks import *
from .unet import Unet
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl


class Encoder(nn.Module):
    """
    Encoder. Reducing the input with convolution and fully connected layers into a 64dim output
    """

    def __init__(self, data_dims, input_channels, num_filters, padding=True, dropout=False):
        super().__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters

        layers = []
        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True
            layers.append(
                DownConvBlock(input, output, padding, pool=pool, p=0.5 if dropout else 0.))
        self.contracting_cnn = nn.Sequential(*layers)

        encoder_output_size = num_filters[-1] * \
            (data_dims[1] // 2**(len(num_filters)-1)) * \
            (data_dims[2] // 2**(len(num_filters)-1))
        self.fully_connected = nn.Sequential(nn.Flatten(),
                                             nn.Linear(
                                                 encoder_output_size, 1032),
                                             nn.LeakyReLU(),
                                             nn.Linear(1032, 256),
                                             nn.LeakyReLU(),
                                             nn.Linear(256, 64),
                                             nn.LeakyReLU(),)

    def forward(self, input):
        h = self.contracting_cnn(input)
        output = self.fully_connected(h)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """

    def __init__(self, data_dims, input_channels, num_filters, latent_dim, dropout=False):
        super().__init__()
        self.encoder = Encoder(data_dims, input_channels,
                               num_filters, dropout=dropout)
        self.mu = nn.Linear(64, latent_dim)
        self.log_std = nn.Linear(64, latent_dim)

        # We init mu and std with small weights
        # for layer in [self.mu, self.log_std]:
        #     torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        #     torch.nn.init.normal_(layer.bias, mean=0.0, std=0.1)

    def forward(self, input):
        h = self.encoder(input)
        mu = self.mu(h)
        log_std = self.log_std(h)

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_std)), 1)
        return dist


class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """

    def __init__(self, latent_dim, unet_output_channels, num_classes, no_fcomb_layers):
        super().__init__()
        self.latent_dim = latent_dim
        filters = latent_dim + unet_output_channels

        layers = []
        for _ in range(no_fcomb_layers-1):
            layers.append(nn.Conv2d(filters, filters,
                                    kernel_size=1, padding=0))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(filters, num_classes,
                                kernel_size=1, padding=0))
        self.model = nn.Sequential(*layers)

    def forward(self, feature_map, z):
        """
        Z is batch_size x latent_dim and feature_map is batch_size x no_channels x H x W.
        So broadcast Z to batch_size x latent_dim x H x W. Behavior is exactly the same as tf.tile (verified)
        """
        B, c_feat, H, W = feature_map.shape
        z = z.view(B, self.latent_dim, 1, 1).expand(B, self.latent_dim, H, W)
        concatinated = torch.cat([feature_map, z], dim=1)
        output = self.model(concatinated)
        return output


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    data_dims: shape of the data, eg. [1,128,128]
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_convs_fcomb: no of 1x1 convs in the conbination function
    """

    def __init__(self, data_dims, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=6, no_fcomb_layers=4, beta=10.0, dropout=False):
        super().__init__()
        self.data_dims = data_dims
        self.input_channels = data_dims[0]
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.beta = beta

        self.unet = Unet(self.input_channels, num_classes, num_filters,
                         apply_last_layer=False, padding=True, p=0.5 if dropout else 0.)
        self.prior = AxisAlignedConvGaussian(data_dims,
                                             self.input_channels, num_filters, latent_dim)
        self.posterior = AxisAlignedConvGaussian(data_dims,
                                                 self.input_channels + num_classes, num_filters, latent_dim)
        self.fcomb = Fcomb(
            latent_dim, num_filters[0], num_classes, no_fcomb_layers)

    def forward(self, x, y=None):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        # propagate networks
        self.prior_latent_distribution = self.prior(x)
        self.unet_features = self.unet.forward(x)
        if y is not None:
            y_onehot = F.one_hot(
                y[:, 0], num_classes=self.num_classes).permute(0, -1, 1, 2)
            xy = torch.cat([x, y_onehot], dim=1)
            self.posterior_latent_distribution = self.posterior(xy)

        # sample latent
        if y is not None:
            self.z = self.posterior_latent_distribution.rsample()
        else:
            self.z = self.prior_latent_distribution.sample()

        # reconstruct image
        self.y_hat_raw = self.fcomb(self.unet_features, self.z)
        y_hat = torch.sigmoid(self.y_hat_raw)

        return y_hat

    def resample(self):
        """
        resamples a prediction based on inputs from the previous forward pass.
        """
        # propagate networks
        self.z = self.prior_latent_distribution.sample()
        # reconstruct image
        self.y_hat_raw = self.fcomb(self.unet_features, self.z)
        y_hat = torch.sigmoid(self.y_hat_raw)

        return y_hat

    def elbo(self, x, y):
        """
        feeds the samples through the network and calculates the
        evidence lower bound of the log-likelihood of P(Y|X)
        """
        self.forward(x, y)

        # prior-posterior divergence
        kl_loss = kl.kl_divergence(
            self.prior_latent_distribution, self.posterior_latent_distribution).mean()

        # reconstruction loss
        if not self.training:
            # resample output based on prior, not posterior
            self.forward(x)

        reconstruction_loss = F.cross_entropy(self.y_hat_raw, y[:, 0])

        loss = reconstruction_loss + self.beta * kl_loss

        return loss, reconstruction_loss, kl_loss

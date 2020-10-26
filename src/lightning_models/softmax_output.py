import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch.nn.functional as F

from src.networks.unet import Unet


class SoftmaxOutput(pl.LightningModule):
    def __init__(self, hparms):
        super().__init__()
        self.save_hyperparameters(hparms)

        self.unet = Unet(input_channels=1,
                         num_classes=2,
                         num_filters=self.hparams.num_filters)

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y[:, 0])
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y[:, 0])
        self.log('val/loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y[:, 0])
        self.log('test/loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def model_name():
        return 'Softmax Output'

    @staticmethod
    def model_shortname():
        return 'softm'

    def pixel_wise_probabaility(self, x, sample_cnt=None):
        """return the pixel-wise probability map

        Args:
            x: the input
            sample_cnt (optional): Amount of samples to draw for internal approximation

        Returns:
            tensor: B x C x H x W, with probability values summing up to 1 across the channel dimension.
        """
        y = self.forward(x)
        return F.softmax(y, dim=1)

    def pixel_wise_uncertainty(self, x, sample_cnt=None):
        """return the pixel-wise entropy

        Args:
            x: the input
            sample_cnt (optional): Amount of samples to draw for internal approximation

        Returns:
            tensor: B x 1 x H x W
        """
        p = self.pixel_wise_probabaility(x, sample_cnt=sample_cnt)
        h = torch.sum(-p * torch.log(p), dim=1, keepdim=True)
        return h

    def sample_prediction(self, x):
        y = self.forward(x)
        _, pred = y.max(dim=1, keepdim=True)
        return pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--num_filters', type=int, nargs='+', default=[
                            32, 64, 128, 192], help='Number of Channels for the U-Net architecture. Decoder uses the reverse. Default: 32 64 128 192')
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

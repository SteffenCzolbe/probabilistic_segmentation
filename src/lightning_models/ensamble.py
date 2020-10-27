import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch.nn.functional as F

from src.networks.unet import Unet


class Ensamble(pl.LightningModule):
    def __init__(self, hparms):
        super().__init__()
        self.save_hyperparameters(hparms)

    def forward(self, x):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = None
        raise NotImplementedError()
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        raise NotImplementedError()
        loss = None
        self.log('val/loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = None
        raise NotImplementedError()
        self.log('test/loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def model_name():
        raise NotImplementedError()
        return ''

    @staticmethod
    def model_shortname():
        raise NotImplementedError()
        return ''

    @staticmethod
    def train_dataset_annotaters_separated():
        return False

    def pixel_wise_probabaility(self, x, sample_cnt=None):
        """return the pixel-wise probability map

        Args:
            x: the input
            sample_cnt (optional): Amount of samples to draw for internal approximation

        Returns:
            tensor: B x C x H x W, with probability values summing up to 1 across the channel dimension.
        """
        raise NotImplementedError()

    def pixel_wise_uncertainty(self, x, sample_cnt=None):
        """return the pixel-wise entropy

        Args:
            x: the input
            sample_cnt (optional): Amount of samples to draw for internal approximation

        Returns:
            tensor: B x 1 x H x W
        """
        raise NotImplementedError()

    def sample_prediction(self, x):
        raise NotImplementedError()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--num_filters', type=int, nargs='+', default=[
                            32, 64, 128, 192], help='Number of Channels for the U-Net architecture. Decoder uses the reverse. Default: 32 64 128 192')
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        raise NotImplementedError()
        return parser

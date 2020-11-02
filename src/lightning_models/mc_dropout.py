import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch.nn.functional as F

from src.networks.unet import Unet


class MCDropout(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.unet = Unet(
            input_channels=1,
            num_classes=2,
            num_filters=self.hparams.num_filters,
            batch_norm=self.hparams.batch_norm
        )

    def forward(self, x):
        """perfroms a probability-mask prediction

        Args:
            x: the input

        Returns:
            tensor: 1 x C x H x W of probabilities, summing up to 1 across the channel dimension.
        """
        self.train()
        return F.softmax(self.unet(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)
        loss = F.cross_entropy(y_hat, y[:, 0])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.train()
        y_hat = self.unet(x)
        loss = F.cross_entropy(y_hat, y[:, 0])
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.train()
        y_hat = self.unet(x)
        loss = F.cross_entropy(y_hat, y[:, 0])
        self.log("test/loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def model_name():
        return "MC Dropout"

    @staticmethod
    def model_shortname():
        return "mcdropout"

    @staticmethod
    def train_dataset_annotaters_separated():
        return True

    def pixel_wise_probabaility(self, x, sample_cnt=16):
        """return the pixel-wise probability map

        Args:
            x: the input
            sample_cnt (optional): Amount of samples to draw for internal approximation

        Returns:
            tensor: B x C x H x W, with probability values summing up to 1 across the channel dimension.
        """
        # we approximate the pixel wise probability by sampling  sample_cnt predictions, then avergaging
        ps = [self.forward(x) for _ in range(sample_cnt)]
        p = torch.stack(ps).mean(dim=0)
        return p

    def pixel_wise_uncertainty(self, x, sample_cnt=16):
        """return the pixel-wise entropy

        Args:
            x: the input
            sample_cnt (optional): Amount of samples to draw for internal approximation

        Returns:
            tensor: B x 1 x H x W
        """
        p = self.pixel_wise_probabaility(x, sample_cnt=sample_cnt)
        mask = p > 0
        h = torch.zeros_like(p)
        h[mask] = torch.log2(1 / p[mask])
        H = torch.sum(p * h, dim=1, keepdim=True)
        return H

    def sample_prediction(self, x):
        """samples a concrete (thresholded) prediction.

        Args:
            x: the input

        Returns:
            tensor: B x 1 x H x W, Long type (int) with class labels.
        """
        y = self.forward(x)
        _, pred = y.max(dim=1, keepdim=True)
        return pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument(
            "--num_filters",
            type=int,
            nargs="+",
            default=[32, 64, 128, 192],
            help="Number of Channels for the U-Net architecture. Decoder uses the reverse. Default: 32 64 128 192",
        )
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument(
            "--dropout_prob",
            type=float,
            default=0,
            help="The probability of setting output to zero.",
        )
        parser.add_argument('--batch_norm', action='store_true',
                            help='Set to use batch normalization during training.')

        return parser

import torch
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
import torch.nn.functional as F

from src.networks.probabilistic_unet import ProbabilisticUnet


class ProbUnet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.punet = ProbabilisticUnet(
            data_dims=self.hparams.data_dims,
            num_classes=2,
            num_filters=self.hparams.num_filters,
            latent_dim=self.hparams.latent_space_dim,
            no_fcomb_layers=4,
            beta=self.hparams.beta,
            dropout=self.hparams.dropout,
            batch_norm=self.hparams.batch_norm,
        )

    def forward(self, x):
        """perfroms a probability-mask prediction

        Args:
            x: the input

        Returns:
            tensor: 1 x C x H x W of probabilities, summing up to 1 across the channel dimension.
        """
        return F.softmax(self.punet(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        (
            loss,
            reconstruction_loss,
            kl_loss,
            mu_dist,
            std_prior,
            std_posterior,
        ) = self.punet.elbo(x, y)
        self.log("train/loss", loss)
        self.log("train/kl_div", kl_loss)
        self.log("train/recon_loss", reconstruction_loss)
        self.log("train/mu_dist", mu_dist)
        self.log("train/std_post_norm", std_posterior)
        self.log("train/std_prior_norm", std_prior)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        (
            loss,
            reconstruction_loss,
            kl_loss,
            mu_dist,
            std_prior,
            std_posterior,
        ) = self.punet.elbo(x, y)
        self.log("val/loss", loss)
        self.log("val/kl_div", kl_loss)
        self.log("val/recon_loss", reconstruction_loss)
        self.log("val/mu_dist", mu_dist)
        self.log("val/std_post_norm", std_posterior)
        self.log("val/std_prior_norm", std_prior)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        (
            loss,
            reconstruction_loss,
            kl_loss,
            mu_dist,
            std_prior,
            std_posterior,
        ) = self.punet.elbo(x, y)
        self.log("test/loss", loss)
        self.log("test/kl_div", kl_loss)
        self.log("test/recon_loss", reconstruction_loss)
        self.log("test/mu_dist", mu_dist)
        self.log("test/std_post_norm", std_posterior)
        self.log("test/std_prior_norm", std_prior)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def model_name():
        return "Prob. U-Net"

    @staticmethod
    def model_shortname():
        return "punet"

    @staticmethod
    def train_dataset_annotaters_separated():
        return True

    def max_unique_samples(self):
        return float('inf')

    def pixel_wise_probabaility(self, x, sample_cnt=16):
        """return the pixel-wise probability map

        Args:
            x: the input
            sample_cnt (optional): Amount of samples to draw for internal approximation

        Returns:
            tensor: B x C x H x W, with probability values summing up to 1 across the channel dimension.
        """
        # we approximate the pixel whise probability by sampling  sample_cnt predictions, then avergaging
        self.sample_prediction(x)
        ps = [self.resample_prediction_non_threshholded()
              for _ in range(sample_cnt)]
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

    def resample_prediction_non_threshholded(self):
        # resampling z for the last sample_prediction
        y = self.punet.resample()
        return F.softmax(y, dim=1)

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
        parser.add_argument("--learning_rate", type=float, default=0.0005)
        parser.add_argument(
            "--latent_space_dim",
            type=int,
            default=6,
            help="Probabalistic-Unet: Dimensionality of the latent space (Default 6)",
        )
        parser.add_argument(
            "--beta",
            type=float,
            default=0.001,
            help="Probabalistic-Unet: Weight factor for the KL-divergence loss (Default 0.001)",
        )
        parser.add_argument(
            "--dropout", action="store_true", help="Set to use dropout during training."
        )
        parser.add_argument(
            "--batch_norm",
            action="store_true",
            help="Set to use batch normalization during training.",
        )
        return parser

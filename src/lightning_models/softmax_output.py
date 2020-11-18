import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch.nn.functional as F

from src.networks.unet import Unet
from src.metrics.generalized_energy_distance import generalized_energy_distance
from src.metrics.soft_dice_loss import heatmap_dice_loss


class SoftmaxOutput(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # add default for backwards-compatebility
        if 'compute_comparison_metrics' not in self.hparams:
            self.hparams.compute_comparison_metrics = True

        self.unet = Unet(
            input_channels=self.hparams.data_dims[0],
            num_classes=self.hparams.data_classes,
            num_filters=self.hparams.num_filters,
        )

    def forward(self, x):
        """perfroms a probability-mask prediction

        Args:
            x: the input

        Returns:
            tensor: 1 x C x H x W of probabilities, summing up to 1 across the channel dimension.
        """
        y = self.unet(x)
        return F.softmax(y, dim=1)

    def training_step(self, batch, batch_idx):
        x, ys = batch
        y = ys[torch.randint(len(ys), ())]
        y_hat = self.unet(x)
        loss = F.cross_entropy(y_hat, y[:, 0])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, ys = batch
        y_hat = self.unet(x)
        for y in ys:
            loss = F.cross_entropy(y_hat, y[:, 0])
            self.log("val/loss", loss)
        if self.hparams.compute_comparison_metrics:
            # calculate aditional metrics every 5 epochs
            if self.current_epoch % 5 == 0:
                for sample_count in [1, 4, 8, 16]:
                    ged, sample_diversity = generalized_energy_distance(
                        self, x, ys, sample_count=sample_count
                    )
                    self.log(f"val/ged/{sample_count}", ged)
                    self.log(
                        f"val/sample_diversity/{sample_count}", sample_diversity)

                    dice = heatmap_dice_loss(
                        self, x, ys, sample_count=sample_count)
                    self.log(f"val/diceloss/{sample_count}", dice)

    def test_step(self, batch, batch_idx):
        x, ys = batch
        y_hat = self.unet(x)
        for y in ys:
            loss = F.cross_entropy(y_hat, y[:, 0])
            self.log("test/loss", loss)
        if self.hparams.compute_comparison_metrics:
            for sample_count in [1, 4, 8, 16]:
                ged, sample_diversity = generalized_energy_distance(
                    self, x, ys, sample_count=sample_count
                )
                self.log(f"test/ged/{sample_count}", ged)
                self.log(
                    f"test/sample_diversity/{sample_count}", sample_diversity)

                dice = heatmap_dice_loss(
                    self, x, ys, sample_count=sample_count)
                self.log(f"test/diceloss/{sample_count}", dice)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def model_name():
        return "Softmax Output"

    @staticmethod
    def model_shortname():
        return "softm"

    def max_unique_samples(self):
        return 1

    def pixel_wise_probabaility(self, x, sample_cnt=None):
        """return the pixel-wise probability map

        Args:
            x: the input
            sample_cnt (optional): Amount of samples to draw for internal approximation

        Returns:
            tensor: B x C x H x W, with probability values summing up to 1 across the channel dimension.
        """
        return self.forward(x)

    def pixel_wise_uncertainty(self, x, sample_cnt=None):
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
        return parser

import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch.nn.functional as F

from src.networks.mcdropout_unet import MCDropoutUnet
from src.metrics.generalized_energy_distance import generalized_energy_distance
from src.metrics.soft_dice_loss import heatmap_dice_loss


class MCDropout(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.net = MCDropoutUnet(
            input_channels=self.hparams.data_dims[0],
            num_classes=self.hparams.data_classes,
            num_filters=self.hparams.num_filters,
            batch_norm=self.hparams.batch_norm,
            p=self.hparams.dropout_prob,
        )

    def forward(self, x):
        """perfroms a probability-mask prediction
        Args:
            x: the input
        Returns:
            tensor: 1 x C x H x W of probabilities, summing up to 1 across the channel dimension.
        """
        self.train()
        return F.softmax(self.net(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, ys = batch
        y = ys[torch.randint(len(ys), ())]
        y_hat = self.net(x)
        loss = F.cross_entropy(y_hat, y[:, 0])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.train()
        x, ys = batch
        for y in ys:
            y_hat = self.net(x)
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
        self.train()
        x, ys = batch
        for y in ys:
            y_hat = self.net(x)
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
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def model_name():
        return "MC Dropout"

    @staticmethod
    def model_shortname():
        return "mcdropout"

    def max_unique_samples(self):
        return float("inf")

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
        eps = torch.tensor(10 ** -8).type_as(x)
        p = self.pixel_wise_probabaility(x, sample_cnt=sample_cnt)
        p = torch.max(p, eps)
        h = torch.sum(-p * torch.log2(p), dim=1, keepdim=True)
        return h

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
            default=0.5,
            help="The probability of setting output to zero.",
        )
        parser.add_argument(
            "--batch_norm",
            action="store_true",
            help="Set to use batch normalization during training.",
        )

        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0,
            help="L2 regularization on weights and biases called in Optimizer.",
        )

        return parser

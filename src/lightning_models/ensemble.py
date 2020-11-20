import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch.nn.functional as F
from itertools import cycle

from src.networks.unet import Unet
from src.metrics.generalized_energy_distance import generalized_energy_distance
from src.metrics.soft_dice_loss import heatmap_dice_loss
import src.util as util


class Ensemble(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # add default for backwards-compatebility
        if 'compute_comparison_metrics' not in self.hparams:
            self.hparams.compute_comparison_metrics = True
        if 'class_weights' not in self.hparams:
            self.hparams.class_weights = [1., 1.]

        self.models = torch.nn.ModuleList(
            [
                Unet(
                    input_channels=self.hparams.data_dims[0],
                    num_classes=self.hparams.data_classes,
                    num_filters=self.hparams.num_filters,
                )
                for _ in range(self.hparams.num_models)
            ]
        )
        weight = torch.tensor(self.hparams.class_weights)
        self.lossfun = torch.nn.CrossEntropyLoss(weight=weight)

    def forward(self, x):
        """perfroms a probability-mask prediction

        Args:
            x: the input

        Returns:
            tensor: 1 x C x H x W of probabilities, summing up to 1 across the channel dimension.
        """
        ps = [F.softmax(model.forward(x), dim=1) for model in self.models]
        p = torch.stack(ps).mean(dim=0)
        return p

    def training_step(self, batch, batch_idx):
        x, ys = batch
        y_hats = [model.forward(x) for model in self.models]
        ensemble_loss = []
        # We cycle through constituent models and ground truth annotations, repeating annotations if nessesary
        for i, (y_hat, y) in enumerate(zip(y_hats, cycle(ys))):
            loss = self.lossfun(y_hat, y[:, 0])
            self.log(f"train/loss_model_{i}", loss)
            ensemble_loss.append(loss)
        ensemble_loss = torch.stack(ensemble_loss).mean()

        self.log("train/loss", ensemble_loss)
        return ensemble_loss

    def validation_step(self, batch, batch_idx):
        x, ys = batch
        y_hats = [model.forward(x) for model in self.models]
        ensemble_loss = []
        # We cycle through constituent models and ground truth annotations, repeating annotations if nessesary
        for i, (y_hat, y) in enumerate(zip(y_hats, cycle(ys))):
            loss = self.lossfun(y_hat, y[:, 0])
            self.log(f"val/loss_model_{i}", loss)
            ensemble_loss.append(loss)
        ensemble_loss = torch.stack(ensemble_loss).mean()

        self.log("val/loss", ensemble_loss)

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

        return ensemble_loss

    def test_step(self, batch, batch_idx):
        x, ys = batch
        y_hats = [model.forward(x) for model in self.models]
        ensemble_loss = []
        # We cycle through constituent models and ground truth annotations, repeating annotations if nessesary
        for i, (y_hat, y) in enumerate(zip(y_hats, cycle(ys))):
            loss = self.lossfun(y_hat, y[:, 0])
            self.log(f"test/loss_model_{i}", loss)
            ensemble_loss.append(loss)
        ensemble_loss = torch.stack(ensemble_loss).mean()

        self.log("test/loss", ensemble_loss)

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
        return "Ensemble"

    @staticmethod
    def model_shortname():
        return "ensemble"

    def max_unique_samples(self):
        return self.hparams.num_models

    def pixel_wise_probabaility(self, x, sample_cnt=None):
        """return the pixel-wise probability map

        Args:
            x: the input
            sample_cnt (optional): Amount of samples to draw for internal approximation

        Returns:
            tensor: B x C x H x W, with probability values summing up to 1 across the channel dimension.
        """
        if sample_cnt is None:
            # draw all samples
            return self.forward(x)
        else:
            ps = [
                F.softmax(
                    self.models[torch.randint(
                        self.hparams.num_models, ())].forward(x),
                    dim=1,
                )
                for _ in range(sample_cnt)
            ]
            return torch.stack(ps).mean(dim=0)

    def pixel_wise_uncertainty(self, x, sample_cnt=None):
        """return the pixel-wise entropy

        Args:
            x: the input
            sample_cnt (optional): Amount of samples to draw for internal approximation

        Returns:
            tensor: B x 1 x H x W
        """
        p = self.pixel_wise_probabaility(x, sample_cnt=sample_cnt)
        return util.entropy(p)

    def sample_prediction(self, x):
        """samples a concrete (thresholded) prediction.

        Args:
            x: the input

        Returns:
            tensor: B x 1 x H x W, Long type (int) with class labels.
        """
        model = self.models[torch.randint(self.hparams.num_models, ())]

        y = model.forward(x)
        _, pred = y.max(dim=1, keepdim=True)
        return pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument(
            "--num_models",
            type=int,
            default=4,
            help="Number of ensemble models. Default: 4",
        )
        parser.add_argument(
            "--num_filters",
            type=int,
            nargs="+",
            default=[32, 64, 128, 192],
            help="Number of Channels for the U-Net architecture. Decoder uses the reverse. Default: 32 64 128 192",
        )
        parser.add_argument(
            "--class_weights",
            type=float,
            nargs="+",
            default=[1., 1.],
            help="Weight assigned to the classes in the loss computation. Default 1 1",
        )
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        return parser

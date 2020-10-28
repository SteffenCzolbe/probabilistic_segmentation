import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch.nn.functional as F

from src.networks.unet import Unet


class Ensemble(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.models = torch.nn.ModuleList([Unet(input_channels=1,
                                                num_classes=2,
                                                num_filters=self.hparams.num_filters) for _ in range(self.hparams.num_models)])
        self.next_model_to_sample = 0

    def forward(self, x):
        y_hats = [model.forward(x) for model in self.models]
        return y_hats

    def training_step(self, batch, batch_idx):
        x, ys = batch
        y_hats = self.forward(x)
        ensemble_loss = 0
        for i, (y_hat, y) in enumerate(zip(y_hats, ys)):
            loss = F.cross_entropy(y_hat, y[:, 0])
            self.log(f'train/loss_model_{i}', loss)
            ensemble_loss += loss

        self.log('train/loss', ensemble_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, ys = batch
        y_hats = self.forward(x)
        ensemble_loss = 0
        for i, (y_hat, y) in enumerate(zip(y_hats, ys)):
            loss = F.cross_entropy(y_hat, y[:, 0])
            self.log(f'val/loss_model_{i}', loss)
            ensemble_loss += loss

        self.log('val/loss', ensemble_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, ys = batch
        y_hats = self.forward(x)
        ensemble_loss = 0
        for i, (y_hat, y) in enumerate(zip(y_hats, ys)):
            loss = F.cross_entropy(y_hat, y[:, 0])
            self.log(f'test/loss_model_{i}', loss)
            ensemble_loss += loss

        self.log('test/loss', ensemble_loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def model_name():
        return 'Ensemble'

    @staticmethod
    def model_shortname():
        return 'ensemble'

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
        preds = [self.sample_prediction(x) for _ in range(len(self.models))]
        p_c1 = torch.cat(preds, dim=1).float().mean(dim=1, keepdim=True)
        p = torch.cat([1 - p_c1, p_c1], dim=1)
        return p

    def pixel_wise_uncertainty(self, x, sample_cnt=None):
        """return the pixel-wise entropy

        Args:
            x: the input
            sample_cnt (optional): Amount of samples to draw for internal approximation

        Returns:
            tensor: B x 1 x H x W
        """
        p = self.pixel_wise_probabaility(x, sample_cnt=sample_cnt)
        h = torch.sum(-p * torch.log(p + 10**-8), dim=1, keepdim=True)
        return h

    def sample_prediction(self, x):
        model = self.models[self.next_model_to_sample]

        self.next_model_to_sample = (
            self.next_model_to_sample + 1) % len(self.models)

        y = model.forward(x)
        _, pred = y.max(dim=1, keepdim=True)
        return pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--num_models', type=int, default=4,
                            help='Number of ensemble models. Default: 4')
        parser.add_argument('--num_filters', type=int, nargs='+', default=[
                            32, 64, 128, 192], help='Number of Channels for the U-Net architecture. Decoder uses the reverse. Default: 32 64 128 192')
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

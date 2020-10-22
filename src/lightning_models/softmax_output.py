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
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y[:, 0])
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y[:, 0])
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def model_name():
        return 'Softmax Output'

    @staticmethod
    def model_shortname():
        return 'softm'

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--num_filters', type=int, nargs='+', default=[
                            32, 64, 128, 192], help='Number of Channels for the U-Net architecture. Decoder uses the reverse. Default: 32 64 128 192')
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

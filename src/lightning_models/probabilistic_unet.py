import torch
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
import torch.nn.functional as F

from src.networks.probabilistic_unet import ProbabilisticUnet


class ProbUnet(pl.LightningModule):
    def __init__(self, hparms):
        super().__init__()
        self.save_hyperparameters(hparms)

        # make sure hparams is a dict from here on. Sadly pytrorch lightning is inconsistent, and passes us a namespace when training, but a dict when loading from save :/
        if isinstance(hparms, Namespace):
            hparms = vars(hparms)
        self.punet = ProbabilisticUnet(data_dims=hparms['data_dims'],
                                       num_classes=2,
                                       num_filters=hparms['num_filters'],
                                       latent_dim=hparms['latent_space_dim'],
                                       no_fcomb_layers=4,
                                       beta=hparms['beta'])

    def forward(self, x):
        return self.punet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, reconstruction_loss, kl_loss = self.punet.elbo(x, y)
        self.log('train/loss', loss)
        self.log('train/kl_div', kl_loss)
        self.log('train/recon_loss', reconstruction_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, reconstruction_loss, kl_loss = self.punet.elbo(x, y)
        self.log('val/loss', loss)
        self.log('val/kl_div', kl_loss)
        self.log('val/recon_loss', reconstruction_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, reconstruction_loss, kl_loss = self.punet.elbo(x, y)
        self.log('test/loss', loss)
        self.log('test/kl_div', kl_loss)
        self.log('test/recon_loss', reconstruction_loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def model_name():
        return 'Prob. U-Net'

    @staticmethod
    def model_shortname():
        return 'punet'

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
        preds = [self.resample_prediction() for _ in range(sample_cnt)]
        p_c1 = torch.cat(preds, dim=1).float().mean(dim=1, keepdim=True)
        p = torch.cat([1 - p_c1, p_c1], dim=1)
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
        h = torch.sum(-p * torch.log(p + 10**-8), dim=1, keepdim=True)
        return h

    def sample_prediction(self, x):
        y = self.forward(x)
        _, pred = y.max(dim=1, keepdim=True)
        return pred

    def resample_prediction(self):
        # resampling z for the last sample_prediction
        y = self.punet.resample()
        _, pred = y.max(dim=1, keepdim=True)
        return pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--num_filters', type=int, nargs='+', default=[
                            32, 64, 128, 192], help='Number of Channels for the U-Net architecture. Decoder uses the reverse. Default: 32 64 128 192')
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--latent_space_dim', type=int, default=6,
                            help='Probabalistic-Unet: Dimensionality of the latent space (Default 6)')
        parser.add_argument('--beta', type=float, default=10.0,
                            help='Probabalistic-Unet: Weight factor for the KL-divergence loss (Default 10.0)')
        return parser

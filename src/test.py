import pytorch_lightning as pl
from argparse import ArgumentParser

from src.lightning_models.softmax_output import SoftmaxOutput
from src.datamodels.lidc_datamodule import LIDCDataModule
import src.util as util
import os
import yaml
import glob
import re


def cli_main():
    pl.seed_everything(1234)
    supported_models = util.get_supported_models()

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        '--model_path', type=str, help=f'Path to the trained model.')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # model
    # ------------
    model = util.load_model_from_checkpoint(args.model_path)
    checkpoint_path = util.get_checkpoint_path(args.model_path)
    datamodule = util.load_datamodule_for_model(model)

    # ------------
    # testing
    # ------------
    print(model)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.test(model=model, ckpt_path=checkpoint_path, datamodule=datamodule)


if __name__ == '__main__':
    cli_main()

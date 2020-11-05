import pytorch_lightning as pl
from argparse import ArgumentParser

from src.lightning_models.softmax_output import SoftmaxOutput
from src.datamodels.lidc_datamodule import LIDCDataModule
import src.util as util
import os
import yaml
import glob
import re


def load_model(model_path):
    def find_checkpoints():
        files = []
        for fname in glob.glob(os.path.join(model_path, "checkpoints", "*")):
            if re.match(r".*epoch=[0-9]+.ckpt", fname):
                files.append(fname)
        return files

    # read config
    with open(os.path.join(model_path, "hparams.yaml")) as f:
        hparams = yaml.load(f, Loader=yaml.Loader)
    model_type = hparams["model"]

    # load model
    model_class = util.get_supported_models()[model_type]
    checkpoints = find_checkpoints()
    assert len(
        checkpoints) == 1, f"An unexpected amount of checkpoints detected! Checkpoints found: {checkpoints}"
    print(f'Loading checkpoint file {checkpoints[0]}')
    model = model_class.load_from_checkpoint(checkpoint_path=checkpoints[0])

    dataset_type = hparams["dataset"]
    dataset_mode = model.train_dataset_annotaters_separated()
    supported_datasets = util.get_supported_datasets()
    dataset = supported_datasets[dataset_type](
        separate_multiple_annotations=dataset_mode)
    return model, dataset, checkpoints[0]


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
    model, dataset, checkpoint_path = load_model(args.model_path)

    # ------------
    # testing
    # ------------
    print(model)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.test(model=model, ckpt_path=checkpoint_path, datamodule=dataset)


if __name__ == '__main__':
    cli_main()

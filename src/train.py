import pytorch_lightning as pl
from argparse import ArgumentParser

from src.lightning_models.softmax_output import SoftmaxOutput
from src.datamodels.lidc_datamodule import LIDCDataModule
import src.util as util


def cli_main():
    pl.seed_everything(1234)
    supported_models = util.get_supported_models()

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        '--model', type=str, help=f'Model architecute. Options: {list(supported_models.keys())}')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='Batchsize. Default 64.')
    parser.add_argument('--learning_rate', default=0.0001,
                        type=float, help='Learning rate. Default 0.0001')
    parser = pl.Trainer.add_argparse_args(parser)
    for model in supported_models.values():
        parser = model.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = LIDCDataModule()

    # ------------
    # model
    # ------------
    if args.model not in supported_models:
        raise Exception(f'Model {args.model} unknown.')
    model = supported_models[args.model](args)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dataset)

    # ------------
    # testing
    # ------------
    trainer.test(model, dataset)


if __name__ == '__main__':
    cli_main()

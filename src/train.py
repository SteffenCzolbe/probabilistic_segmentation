import pytorch_lightning as pl
from argparse import ArgumentParser

from src.lightning_models.softmax_output import SoftmaxOutput
from src.datamodels.lidc_datamodule import LIDCDataModule
import src.util as util


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help=f"Model architecute. Options: {list(util.get_supported_models().keys())}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help=f"Dataset. Options: {list(util.get_supported_datamodules().keys())}",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batchsize. Default 64."
    )
    parser.add_argument(
        "--learning_rate",
        default=0.0001,
        type=float,
        help="Learning rate. Default 0.0001",
    )
    parser.add_argument(
        "--notest", action="store_true", help="Set to not run test after training."
    )
    parser.add_argument(
        "--compute_comparison_metrics",
        type=bool,
        default=True,
        help="Compute GED etc. metrics in val and test steps.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Early stopping oatience, in Epcohs. Default: 10",
    )

    parser = pl.Trainer.add_argparse_args(parser)
    for model in util.get_supported_models().values():
        parser = model.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = util.load_damodule(args.dataset, batch_size=args.batch_size)
    args.data_dims = dataset.dims
    args.data_classes = dataset.classes

    # ------------
    # model
    # ------------
    model_cls = util.get_model_cls(args.model)
    model = model_cls(args)

    # ------------
    # training
    # ------------
    # save model with best validation loss
    checkpointing_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss", mode="min"
    )
    # early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val/loss", min_delta=0.00, patience=args.early_stop_patience, verbose=True, mode="min"
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpointing_callback,
        callbacks=[early_stop_callback],
    )
    trainer.fit(model, dataset)

    # ------------
    # testing
    # ------------
    if not args.notest:
        trainer.test()


if __name__ == "__main__":
    cli_main()

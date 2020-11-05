import pytorch_lightning as pl
from argparse import ArgumentParser

from src.lightning_models.softmax_output import SoftmaxOutput
from src.datamodels.lidc_datamodule import LIDCDataModule
import src.util as util


def cli_main():
    pl.seed_everything(1234)
    supported_models = util.get_supported_models()
    supported_datasets = util.get_supported_datasets()

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        '--model', type=str, help=f'Model architecute. Options: {list(supported_models.keys())}')
    parser.add_argument(
        '--dataset', type=str, help=f'Dataset. Options: {list(supported_datasets.keys())}')
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
    if args.dataset not in supported_datasets:
        raise Exception(f'Dataset {args.dataset} unknown.')
    dataset_mode = supported_models[args.model].train_dataset_annotaters_separated(
    )
    dataset = supported_datasets[args.dataset](
        separate_multiple_annotations=dataset_mode)
    args.data_dims = dataset.dims

    # ------------
    # model
    # ------------
    if args.model not in supported_models:
        raise Exception(f'Model {args.model} unknown.')
    model = supported_models[args.model](args)

    # ------------
    # training
    # ------------
    # save model with best validation loss
    checkpointing_callback = pl.callbacks.ModelCheckpoint(monitor='val/loss',
                                                          mode='min')
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val/loss',
                                                     min_delta=0.00,
                                                     patience=10,
                                                     verbose=True,
                                                     mode='min')

    # early stopping

    trainer = pl.Trainer.from_argparse_args(
        args, checkpoint_callback=checkpointing_callback, callbacks=[early_stop_callback])
    trainer.fit(model, dataset)

    # ------------
    # testing
    # ------------
    trainer.test()


if __name__ == '__main__':
    cli_main()

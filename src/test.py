import pytorch_lightning as pl
from argparse import ArgumentParser
import src.util as util
import os
import json


def cli_main():
    pl.seed_everything(1234)
    supported_models = util.get_supported_models()

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        '--model_path', type=str, help=f'Path to the trained model.')
    parser.add_argument(
        '--file', type=str, default='./plots/experiment_results.json', help=f'File to save the results in.')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # model
    # ------------
    model = util.load_model_from_checkpoint(args.model_path)
    checkpoint_path = util.get_checkpoint_path(args.model_path)
    datamodule = util.load_datamodule_for_model(model)
    dataset = model.hparams.dataset

    # ------------
    # file
    # ------------
    if os.path.isfile(args.file):
        with open(args.file) as f:
            test_results = json.load(f)
    else:
        test_results = {}

    if dataset not in test_results:
        test_results[dataset] = {}
    test_results[dataset][args.model_path] = {}
    test_results[dataset][args.model_path]['model_name'] = model.model_name()
    test_results[dataset][args.model_path]['model_shortname'] = model.model_shortname()

    # ------------
    # Run model test script
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    test_metrics = trainer.test(
        model=model, ckpt_path=checkpoint_path, datamodule=datamodule)
    for k, v in test_metrics[0].items():
        test_results[dataset][args.model_path][k] = v

    # ------------
    # save results
    # ------------
    print('Test results:')
    print(json.dumps(test_results[dataset]
                     [args.model_path], indent=4, sort_keys=True))
    with open(args.file, 'w') as json_file:
        json.dump(test_results, json_file, indent=4, sort_keys=True)


if __name__ == '__main__':
    cli_main()

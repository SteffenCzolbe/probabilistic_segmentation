import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from src.metrics.generalized_energy_distance import generalized_energy_distance
from src.metrics.soft_dice_loss import heatmap_dice_loss
from src.metrics.correlation import pearsonr
import src.util as util
import os
import pickle
from tqdm import tqdm
from collections import defaultdict


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
        '--file', type=str, default='./plots/experiment_results.pickl', help=f'File to save the results in.')
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
        with open(args.file, 'rb') as f:
            test_results = pickle.load(f)
    else:
        test_results = {}

    if dataset not in test_results:
        test_results[dataset] = {}
    test_results[dataset][model.model_shortname()] = {}
    test_results[dataset][model.model_shortname(
    )]['model_name'] = model.model_name()
    test_results[dataset][model.model_shortname(
    )]['model_shortname'] = model.model_shortname()

    # ------------
    # Run model test script
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    test_metrics = trainer.test(
        model=model, ckpt_path=checkpoint_path, datamodule=datamodule)
    for k, v in test_metrics[0].items():
        test_results[dataset][model.model_shortname()][k] = v

    # ------------
    # Record sample-specific metrics
    # ------------
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        metrics = defaultdict(list)
        for i, (x, ys) in enumerate(tqdm(datamodule.test_dataloader(), desc='Collecting sample-individual metrics...')):
            x, ys = util.to_device((x, ys), device)
            assert ys[0].max() <= 1
            y_mean = torch.stack(ys).float().mean(dim=0)

            for sample_count in [1, 4, 8, 16]:
                ged, sample_diversity = generalized_energy_distance(
                    model, x, ys, sample_count=sample_count)
                metrics[f"test/ged/{sample_count}"].append(ged)
                metrics[f"test/sample_diversity/{sample_count}"].append(
                    sample_diversity)

                dice = heatmap_dice_loss(
                    model, x, ys, sample_count=sample_count)
                metrics[f"test/diceloss/{sample_count}"].append(dice)

            sample_count = 16
            uncertainty = model.pixel_wise_uncertainty(
                x, sample_cnt=sample_count)
            correl = torch.stack([pearsonr(uncertainty, torch.nn.functional.binary_cross_entropy(
                model.sample_prediction(x).float(),
                ys[torch.randint(len(ys), ())].float(),
                reduction='none')) for _ in range(16)]).mean(dim=0)
            metrics["test/uncertainty_seg_error_correl"].append(correl)
            if i < 3:
                # record pixel-wise metrics for only a few batchs
                y_hat = model.sample_prediction(x).flatten()
                y_consens = torch.stack(ys).sum(dim=0).flatten()
                model_uncertainty = uncertainty.flatten()

                y_mean = torch.stack(ys).float().mean(dim=0)
                annot_uncertainty = util.binary_entropy(y_mean).flatten()

                annotators = len(ys)
                # record conditional uncetainty only if the is consensus
                metrics["test/tp_uncertainty"].append(
                    model_uncertainty[(y_hat == 1) & (y_consens == annotators)])
                metrics["test/fp_uncertainty"].append(
                    model_uncertainty[(y_hat == 1) & (y_consens == 0)])
                metrics["test/fn_uncertainty"].append(
                    model_uncertainty[(y_hat == 0) & (y_consens == annotators)])
                metrics["test/tn_uncertainty"].append(
                    model_uncertainty[(y_hat == 0) & (y_consens == 0)])
                metrics["test/model_uncertainty"].append(model_uncertainty)
                metrics["test/annotator_uncertainty"].append(annot_uncertainty)

    # map metrics into lists of floats
    test_results[dataset][model.model_shortname()]['per_sample'] = {}
    for k in metrics:
        test_results[dataset][model.model_shortname()]['per_sample'][k] = torch.cat(
            metrics[k]).cpu().numpy()

    # ------------
    # save results
    # ------------
    with open(args.file, 'wb') as f:
        pickle.dump(test_results, f)


if __name__ == '__main__':
    cli_main()

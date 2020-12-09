from .plot2d import Fig
from src.datamodels.lidc_datamodule import LIDCDataModule
from src import util
from tqdm import tqdm
import torch
import os
import glob
from argparse import ArgumentParser


def set_up_figure(model_cnt, sample):
    fig = Fig(
        rows=1 + model_cnt,
        cols=7,
        title="Model Predictions",
        figsize=None,
        background=True,
    )
    fig.plot_img(0, 0, sample[0], title="Image", vmin=0, vmax=1)

    for i in range(len(sample[1])):
        fig.plot_img(0, i+1, sample[1][i], title=f"l_{i}", vmin=0, vmax=1)
    return fig


def load_sample(datamodule, idx, device):
    dl = datamodule.test_dataloader()
    img = dl.dataset[idx]
    return util.to_device(img, device)


def predict(model, x):
    with torch.no_grad():
        predictions = []
        # draw 4 predictions
        for i in range(4):
            pred = model.sample_prediction(x.unsqueeze(0)).squeeze(0)
            predictions.append(pred)

        # pixel-wise probability
        pred = model.pixel_wise_probabaility(x.unsqueeze(0), sample_cnt=16).squeeze(0)[
            [1]
        ]
        predictions.append(pred)

        # uncertainty
        pred = model.pixel_wise_uncertainty(
            x.unsqueeze(0), sample_cnt=16).squeeze(0)
        predictions.append(pred)

    return predictions


def plot_predictions(fig, row, predictions, model_name):
    for col, pred in enumerate(predictions):
        fig.plot_img(
            row,
            col + 1,
            pred,
            vmin=0,
            vmax=1,
            title=model_name if col == 0 else None,
        )


def make_fig(model_checkpoints, output_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for row, model_checkpoint in enumerate(tqdm(model_checkpoints, desc='Plotting Model outputs...')):
        model = util.load_model_from_checkpoint(model_checkpoint).to(device)
        datamodule = util.load_datamodule_for_model(
            model, batch_size=1)
        sample = load_sample(datamodule, idx=0, device=device)
        if row == 0:
            fig = set_up_figure(len(model_checkpoints), sample)
        predictions = predict(model, sample[0])
        plot_predictions(fig, row + 1, predictions, model.model_shortname())

    os.makedirs("./plots/", exist_ok=True)
    fig.save(output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--checkpoint', type=str, help='The trained model.')
    parser.add_argument(
        '--output_file', type=str, help='File to save the results in.')
    args = parser.parse_args()

    make_fig(args.checkpoint, args.output_file)

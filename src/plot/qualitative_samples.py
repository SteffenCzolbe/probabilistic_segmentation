from .plot2d import Fig
from src.datamodels.lidc_datamodule import LIDCDataModule
from src import util
from tqdm import tqdm
import torch
import pytorch_lightning as pl
import os
import seaborn as sns
import numpy as np
import glob
from argparse import ArgumentParser
from matplotlib import pyplot as plt


def set_up_figure(model_cnt, x, ys):
    cols = 6
    fig = Fig(
        rows=5,
        cols=cols,
        figsize=(4, 4),
        background=True,
    )
    # image goes top left
    fig.plot_img(0, 0, x, vmin=0, vmax=1)

    # annotations go top right
    n = len(ys)
    for col, y in zip(range(cols-n, cols), ys):
        fig.plot_img(0, col, y, vmin=0, vmax=1)
    return fig


def load_sample(datamodule, idx, device):
    dl = datamodule.test_dataloader()
    img = dl.dataset[idx]
    return util.to_device(img, device)


def predict(model, x):
    with torch.no_grad():
        predictions = []
        # draw 8 predictions
        for i in range(6):
            pred = model.sample_prediction(x.unsqueeze(0)).squeeze(0)
            predictions.append(pred)

    return predictions


def plot_predictions(fig, row, predictions,):
    for col, pred in enumerate(predictions):
        fig.plot_img(
            row,
            col,
            pred,
            vmin=0,
            vmax=1,
        )


def make_fig(args):
    device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
    models = [util.load_model_from_checkpoint(ckpt).to(
        device) for ckpt in args.model_checkpoints]
    datamodule = util.load_datamodule_for_model(
        models[0], batch_size=1)
    x, ys = load_sample(datamodule, idx=args.sample_idx, device=device)
    fig = set_up_figure(4, x, ys)

    for row, model in enumerate(tqdm(models, desc='Plotting Model outputs...')):
        predictions = predict(model, x)
        plot_predictions(fig, row + 1, predictions)

    # adjust spacing
    plt.subplots_adjust(left=0, bottom=0, right=1,
                        top=0.9, wspace=0.0, hspace=0.06)

    os.makedirs("./plots/", exist_ok=True)
    for f in args.output_file:
        fig.save(f, close=False)


def order_model_checkpoints(cktps):
    def order(name):
        if 'softm' in name:
            return 0
        elif 'ensemble' in name:
            return 1
        elif 'mcdropout' in name:
            return 2
        elif 'punet' in name:
            return 3
        else:
            raise Exception(f'do not know how to order model {name}')
    return sorted(cktps, key=order)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--sample_idx', type=int, default=8, help='Idx of the sample to plot')
    parser.add_argument(
        '--model_dir', type=str, help='Directory of trained models.')
    parser.add_argument(
        '--output_file', type=str, nargs='+', help='File to save the results in.')
    args = parser.parse_args()

    model_checkpoints = glob.glob(os.path.join(args.model_dir, '*'))
    args.model_checkpoints = order_model_checkpoints(model_checkpoints)

    make_fig(args)

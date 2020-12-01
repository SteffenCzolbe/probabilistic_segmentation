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


def load_sample(datamodule, idx, device):
    dl = datamodule.test_dataloader()
    batch = dl.dataset[idx]
    batch = util.to_device(batch, device)
    x, ys = batch
    return x.unsqueeze(0), [y.unsqueeze(0) for y in ys]


def add_edge(img, width=2, c=0):
    # adds a black frame to the image

    # top
    img[:, :width, :] = c
    # bottom
    img[:, -width:, :] = c
    # left
    img[:, :, :width] = c
    # right
    img[:, :, -width:] = c

    return img


def plot_row(fig, row, img_idx, device, models, edge_width, color_gt, color_model):
    datamodule = util.load_datamodule_for_model(models[0], batch_size=1)
    x, ys = load_sample(datamodule, idx=img_idx, device=device)
    y_mean = torch.stack(ys).float().mean(dim=0).round().long()

    # plot image
    fig.plot_img(row, 0, add_edge(x[0], width=edge_width), vmin=0, vmax=1)
    # plot gt seg outline
    fig.plot_contour(row, 0, y_mean[0],
                     contour_class=1, width=2, rgba=color_gt)

    # plot model predictions
    for col, model in enumerate(models):
        pl.seed_everything(42)
        with torch.no_grad():
            p = model.pixel_wise_probabaility(x, sample_cnt=16)
            _, y_pred_mean = p.max(dim=1, keepdim=True)
            uncertainty = util.entropy(p)

        # plot uncertainty heatmap
        fig.plot_overlay(
            row, 1 + col, add_edge(uncertainty[0], c=1, width=edge_width), alpha=1, vmin=0, vmax=1, cmap='Greys')

        # plot gt seg outline
        fig.plot_contour(
            row, 1 + col, y_mean[0], contour_class=1, width=2, rgba=color_gt)

        # plot model prediction outline
        fig.plot_contour(row, 1 + col, y_pred_mean[0], contour_class=1, width=2, rgba=color_model
                         )


def make_fig(args):
    # set up
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    isic_model_checkpoints = ['./trained_models/isic18/softm/', './trained_models/isic18/ensemble/',
                              './trained_models/isic18/mcdropout/', './trained_models/isic18/punet/']
    isic_models = [util.load_model_from_checkpoint(ckpt).to(
        device) for ckpt in isic_model_checkpoints]
    lidc_model_checkpoints = ['./trained_models/lidc/softm/', './trained_models/lidc/ensemble/',
                              './trained_models/lidc/mcdropout/', './trained_models/lidc/punet/']
    lidc_models = [util.load_model_from_checkpoint(ckpt).to(
        device) for ckpt in lidc_model_checkpoints]

    fig = Fig(
        rows=args.images_each*2,
        cols=5,
        title=None,
        figsize=(6, 8),
        background=True,
    )
    plt.tight_layout()
    colors = np.array(sns.color_palette("Paired")) * 255
    color_gt = colors[1]
    color_model = colors[7]
    np.random.seed(7)
    img_indices = np.random.choice(range(500), args.images_each*2)

    # plot isic dataset
    for i in range(args.images_each):
        plot_row(fig, i, img_indices[i], device, isic_models,
                 4, color_gt, color_model)

    # plot lidc dataset
    for i in range(args.images_each, args.images_each*2):
        plot_row(fig, i, img_indices[i], device, lidc_models,
                 2, color_gt, color_model)

    # set labels
    fig.axs[0, 0].title.set_text('Image')
    fig.axs[0, 0].title.set_fontsize(10)
    for i, model in enumerate(isic_models):
        fig.axs[0, 1 + i].title.set_text(model.model_name())
        fig.axs[0, 1 + i].title.set_fontsize(10)

    # adjust spacing
    plt.subplots_adjust(left=0, bottom=0, right=1,
                        top=0.9, wspace=0.06, hspace=0.06)

    for f in args.output_file:
        fig.save(f, close=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--images_each', type=int, default=3, help='Amount of images per dataset')
    parser.add_argument(
        '--output_file', type=str, nargs='+', help='File to save the results in.')
    args = parser.parse_args()

    make_fig(args)

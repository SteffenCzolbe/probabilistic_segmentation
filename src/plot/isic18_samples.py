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
    return x.unsqueeze(0), ys[0].unsqueeze(0)


def add_edge(img, width=4, c=0):
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


def make_fig(args, img_cnt):
    # set up
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = [util.load_model_from_checkpoint(ckpt).to(
        device) for ckpt in args.model_checkpoints]
    datamodule = util.load_datamodule_for_model(models[0], batch_size=1)
    fig = Fig(
        rows=img_cnt,
        cols=1 + len(models),
        title=None,
        figsize=(6, 8),
        background=True,
    )
    plt.tight_layout()
    colors = np.array(sns.color_palette("Paired")) * 255
    color_gt = colors[1]
    color_model = colors[7]
    np.random.seed(42)
    img_indices = np.random.choice(
        range(len(datamodule.test_dataloader().dataset)), img_cnt)

    for row, img_idx in enumerate(tqdm(img_indices, desc='Generating Images')):
        x, y = load_sample(datamodule, idx=img_idx, device=device)

        # plot image
        fig.plot_img(row, 0, add_edge(x[0]), vmin=0, vmax=1)
        # plot gt seg outline
        fig.plot_contour(row, 0, y[0],
                         contour_class=1, width=2, rgba=color_gt)

        # plot model predictions
        for col, model in enumerate(models):
            pl.seed_everything(42)
            with torch.no_grad():
                p = model.pixel_wise_probabaility(x, sample_cnt=16)
                _, y_pred = p.max(dim=1, keepdim=True)
                uncertainty = util.entropy(p)

            # plot uncertainty heatmap
            fig.plot_overlay(
                row, 1 + col, add_edge(uncertainty[0], c=1), alpha=1, vmin=0, vmax=1, cmap='Greys')

            # plot gt seg outline
            fig.plot_contour(row, 1 + col, y[0], contour_class=1, width=2, rgba=color_gt
                             )

            # plot model prediction outline
            fig.plot_contour(row, 1 + col, y_pred[0], contour_class=1, width=2, rgba=color_model
                             )

    # set labels
    fig.axs[0, 0].title.set_text('Image')
    fig.axs[0, 0].title.set_fontsize(10)
    for i, model in enumerate(models):
        fig.axs[0, 1 + i].title.set_text(model.model_name())
        fig.axs[0, 1 + i].title.set_fontsize(10)

    # adjust spacing
    plt.subplots_adjust(left=0, bottom=0, right=1,
                        top=0.9, wspace=0.06, hspace=0.06)

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
        '--model_dir', type=str, help='Directory of trained models.')
    parser.add_argument(
        '--output_file', type=str, nargs='+', help='File to save the results in.')
    args = parser.parse_args()

    model_checkpoints = glob.glob(os.path.join(args.model_dir, '*'))
    args.model_checkpoints = order_model_checkpoints(model_checkpoints)

    make_fig(args, img_cnt=6)

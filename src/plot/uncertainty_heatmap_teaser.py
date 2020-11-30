from .plot2d import Fig
from src.datamodels.lidc_datamodule import LIDCDataModule
from src import util
from tqdm import tqdm
import torch
import pytorch_lightning as pl
import os
import seaborn as sns
import numpy as np
from argparse import ArgumentParser


def load_sample(datamodule, idx, device):
    dl = datamodule.test_dataloader()
    batch = dl.dataset[idx]
    batch = util.to_device(batch, device)
    x, ys = batch
    return x.unsqueeze(0), ys[0].unsqueeze(0)


def make_fig(args):
    # set up
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = util.load_model_from_checkpoint(args.model_path).to(device)
    datamodule = util.load_datamodule_for_model(model, batch_size=1)

    for idx in tqdm(range(100), desc='Generating Images'):
        x, y = load_sample(datamodule, idx=idx, device=device)
        fig = Fig(
            rows=1,
            cols=2,
            title=None,
            figsize=None,
            background=True,
        )
        colors = np.array(sns.color_palette("Paired")) * 255
        color_gt = colors[1]
        color_model = colors[7]

        # draw samples
        pl.seed_everything(42)
        with torch.no_grad():
            p = model.pixel_wise_probabaility(x, sample_cnt=args.samples)
            _, y_pred = p.max(dim=1, keepdim=True)
            uncertainty = util.entropy(p)

        # plot image
        fig.plot_img(0, 0, x[0], vmin=0, vmax=1)

        # plot uncertainty heatmap
        fig.plot_overlay(
            0, 1, uncertainty[0], alpha=1, vmin=None, vmax=None, cmap='Greys', colorbar=True, colorbar_label="Model Uncertainty")

        # plot model prediction outline
        fig.plot_contour(0, 0, y_pred[0], contour_class=1, width=2, rgba=color_model
                         )
        fig.plot_contour(0, 1, y_pred[0], contour_class=1, width=2, rgba=color_model
                         )

        # plot gt seg outline
        fig.plot_contour(0, 0, y[0], contour_class=1, width=2, rgba=color_gt
                         )
        fig.plot_contour(0, 1, y[0], contour_class=1, width=2, rgba=color_gt
                         )

        # add legend
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle
        legend_data = [
            [0, color_gt, "GT Annotation"],
            [1, color_model, "Model Prediction"], ]
        handles = [
            Rectangle((0, 0), 1, 1, color=[v/255 for v in c]) for k, c, n in legend_data
        ]
        labels = [n for k, c, n in legend_data]

        plt.legend(handles, labels, ncol=len(legend_data))

        os.makedirs("./plots/", exist_ok=True)
        # fig.save(args.output_file)
        os.makedirs(args.output_folder, exist_ok=True)
        fig.save(os.path.join(args.output_folder,
                              f'test_{idx}.png'), close=False)
        fig.save(os.path.join(args.output_folder, f'test_{idx}.pdf'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--model_path', type=str, help='Path to the trained model')
    parser.add_argument(
        '--output_folder', type=str, help='File to save the results in.')
    parser.add_argument(
        '--samples', type=int, default=16, help='Samples to draw. Default 16.')
    args = parser.parse_args()

    make_fig(args)

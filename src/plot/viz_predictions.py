from .plot2d import Fig
from src.datamodels.lidc_datamodule import LIDCDataModule
from src import util
import torch
import yaml
import os
import glob
import re


def set_up_figure(model_cnt, sample):
    fig = Fig(
        rows=1 + model_cnt,
        cols=7,
        title="Model Predictions",
        figsize=None,
        background=True,
    )
    fig.plot_img(0, 0, sample[0], title="Image", vmin=0, vmax=1)

    fig.plot_img(0, 1, sample[1][0], title="l_0", vmin=0, vmax=1)
    fig.plot_img(0, 2, sample[1][1], title="l_1", vmin=0, vmax=1)
    fig.plot_img(0, 3, sample[1][2], title="l_2", vmin=0, vmax=1)
    fig.plot_img(0, 4, sample[1][3], title="l_3", vmin=0, vmax=1)
    return fig


def load_sample(sample_indx):
    dm = LIDCDataModule(batch_size=1, separate_multiple_annotations=False)
    dl = dm.test_dataloader()
    img = dl.dataset[sample_indx]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return util.to_device(img, device)


def load_model(model_path):
    def find_checkpoints():
        files = []
        for fname in glob.glob(os.path.join(model_path, "checkpoints", "*")):
            if re.match(r".*epoch=[0-9]+.ckpt", fname):
                files.append(fname)
        return files

    # read config
    with open(os.path.join(model_path, "hparams.yaml")) as f:
        hparams = yaml.load(f, Loader=yaml.Loader)
    model_type = hparams["model"]

    # load model
    model_class = util.get_supported_models()[model_type]
    checkpoints = find_checkpoints()
    assert len(
        checkpoints) == 1, f"multiple checkpoints detected!: {checkpoints}"
    model = model_class.load_from_checkpoint(checkpoint_path=checkpoints[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model


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


def make_fig(model_checkpoints):
    sample = load_sample(0)
    fig = set_up_figure(len(model_checkpoints), sample)

    for row, model_checkpoint in enumerate(model_checkpoints):
        model = load_model(model_checkpoint)
        predictions = predict(model, sample[0])
        plot_predictions(fig, row + 1, predictions, model.model_shortname())

    os.makedirs("./plots/", exist_ok=True)
    fig.save("./plots/predictions.png")


if __name__ == "__main__":
    # './trained_models/softmax',
    model_checkpoints = glob.glob("./trained_models/*")
    make_fig(model_checkpoints)

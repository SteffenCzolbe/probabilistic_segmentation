import glob
import os
import re
import yaml
import torch


def to_device(obj, device):
    """Maps a torch tensor, or a collection containing torch tesnors recursively onto the gpu

    Args:
        obj ([type]): [description]
    """
    if hasattr(obj, 'to'):
        return obj.to(device)
    elif hasattr(obj, '__iter__'):
        return [to_device(o, device) for o in obj]
    else:
        raise Exception(f'Do not know how to map object {obj} to {device}')


def get_supported_datamodules():
    from src.datamodels.lidc_datamodule import LIDCDataModule
    from src.datamodels.isic18_datamodule import ISIC18DataModule

    supported_datamodels = {'lidc': LIDCDataModule,
                            'isic18': ISIC18DataModule}

    return supported_datamodels


def load_damodule(dataset_name, batch_size=32):
    """Loads a Datamodule

    Args:
        dataset_name (str): Name of dataset
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        Datamodule
    """
    supported_datamodels = get_supported_datamodules()
    if dataset_name not in supported_datamodels:
        raise Exception(
            f'Dataset {dataset_name} unknown. Supported datasets: {supported_datamodels.keys()}')

    datamodule = supported_datamodels[dataset_name](batch_size=batch_size)

    return datamodule


def load_datamodule_for_model(model, batch_size=None):
    """Loads the datamodule for the model. kwargs set will overwrite model defaults.

    Args:
        model: The model
        batchsize (bool, optional): Set to overwrite batch size.
    """
    batch_size = batch_size if batch_size is not None else model.hparams.batch_size
    datamodule_name = model.hparams.dataset
    return load_damodule(datamodule_name, batch_size=batch_size)


def get_checkpoint_path(model_path):
    epoch_to_checkpoint = {}
    regex = r".*epoch=([0-9]+).ckpt"
    for fname in glob.glob(os.path.join(model_path, "checkpoints", "*")):
        if re.match(regex, fname):
            epoch = re.search(regex, fname).group(1)
            epoch_to_checkpoint[int(epoch)] = fname
    return sorted(epoch_to_checkpoint.items(), key=lambda t: t[0])[-1][1]


def get_supported_models():
    from src.lightning_models.softmax_output import SoftmaxOutput
    from src.lightning_models.mc_dropout import MCDropout
    from src.lightning_models.probabilistic_unet import ProbUnet
    from src.lightning_models.ensemble import Ensemble

    supported_models = [SoftmaxOutput, MCDropout, ProbUnet, Ensemble]

    # remap supported models to dict
    supported_models = dict([(model.model_shortname(), model)
                             for model in supported_models])
    return supported_models


def get_model_cls(model_name):
    supported_models = get_supported_models()
    if model_name not in supported_models:
        raise Exception(
            f'Model {model_name} unknown. Models available: {supported_models.keys()}.')
    return supported_models[model_name]


def load_model_from_checkpoint(model_path):
    # read config
    with open(os.path.join(model_path, "hparams.yaml")) as f:
        hparams = yaml.load(f, Loader=yaml.Loader)
    model_type = hparams["model"]

    # load model
    model_class = get_supported_models()[model_type]
    checkpoint = get_checkpoint_path(model_path)
    print(f'Loading model {model_type} from checkpoint file {checkpoint}')
    try:
        model = model_class.load_from_checkpoint(
            checkpoint_path=checkpoint, strict=True)
    except RuntimeError, e:
        print('WARNING: ', e.message)
        print('reloading model with non-strict mapping...')
        model = model_class.load_from_checkpoint(
            checkpoint_path=checkpoint, strict=False)

    return model


def entropy(p):
    """
    Calculates the entropy (uncertainty) of p

    Args:
        p (Tensor BxCxHxW): probability per class

    Returns:
        Tensor Bx1xHxw
    """
    mask = p > 0
    h = torch.zeros_like(p)
    h[mask] = torch.log2(1 / p[mask])
    H = torch.sum(p * h, dim=1, keepdim=True)
    return H


def binary_entropy(p):
    """
    Calculates the entropy (uncertainty) of p

    Args:
        p (Tensor Bx1xHxW): probability per class

    Returns:
        Tensor Bx1xHxw
    """
    p = torch.cat([p, 1-p])
    return entropy(p)

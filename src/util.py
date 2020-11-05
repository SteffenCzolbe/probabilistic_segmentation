

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


def get_supported_datasets():
    from src.datamodels.lidc_datamodule import LIDCDataModule
    from src.datamodels.addi_datamodule import ADDIDataModule

    supported_datasets = {'lidc': LIDCDataModule,
                          'addi': ADDIDataModule}

    return supported_datasets

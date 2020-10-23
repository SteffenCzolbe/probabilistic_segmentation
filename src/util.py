

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

    supported_models = [SoftmaxOutput]

    # remap supported models to dict
    supported_models = dict([(model.model_shortname(), model)
                             for model in supported_models])
    return supported_models

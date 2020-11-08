import torch


def dice_loss(input, target, eps=1e-6):
    if target.max() > 1:
        raise Exception('Dice is only implemented for 2 classes')

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + eps) /
                (iflat.sum() + tflat.sum() + eps))


def heatmap_dice_loss(model, x, ys, sample_count=16):
    """Calculates the Soft Dice overlap between heatmaps.

    Args:
        model: the model, implementing a sample_prediction method
        x: The images, Float-Tensor of shape BxCxHxW
        ys: The annotations, List of Long-Tensors of shape Bx1xHxW of segmentation maps
        sample_count: the amount of samples to draw from the model

    Returns:
        List of distances, length B
    """
    input = model.pixel_wise_probabaility(x, sample_cnt=sample_count)[:, [1]]
    target = torch.stack(ys).float().mean(dim=0)
    return dice_loss(input, target)

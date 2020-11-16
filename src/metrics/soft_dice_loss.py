import torch


def dice_loss(input, target, eps=1e-6):
    if target.max() > 1:
        raise Exception('Dice is only implemented for 2 classes')

    B = input.shape[0]
    x = input.view(B, -1)
    y = target.view(B, -1)
    intersection = (x * y).sum(dim=1)

    return 1 - ((2. * intersection + eps) /
                ((x**2).sum(dim=1) + (y**2).sum(dim=1) + eps))


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

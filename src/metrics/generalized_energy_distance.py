import torch
from itertools import product, combinations


def iou(a, b, eps=1e-6):
    """Intersection over Union operation

    Args:
        a: Long-Tensors of shape Bx1xHxW of segmentation maps
        b: Long-Tensors of shape Bx1xHxW of segmentation maps

    Returns:
        List of distances, length B
    """
    if a.max() > 1:
        raise Exception('IoU is only implemented for 2 classes')

    intersection = (a & b).float().sum(
        dim=(1, 2, 3))
    union = (a | b).float().sum(
        dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)
    return iou


def ged(samples_a, samples_b):
    """Calculates the Generalized Energy Distance.

    Args:
        samples_a: List of Long-Tensors of shape Bx1xHxW of segmentation maps
        samples_b: List of Long-Tensors of shape Bx1xHxW of segmentation maps

    Returns:
        ged, sample_diversity: Lists of distances, length B
    """
    idx_a = range(len(samples_a))
    idx_b = range(len(samples_b))

    e_ab = torch.stack([1 - iou(samples_a[i], samples_b[j])
                        for i, j in product(idx_a, idx_b)]).mean(dim=0)
    if len(samples_a) > 1:
        e_aa = torch.stack([1 - iou(samples_a[i], samples_a[j])
                            for i, j in product(idx_a, idx_a)]).mean(dim=0)
    else:
        e_aa = 0

    if len(samples_b) > 1:
        e_bb = torch.stack([1 - iou(samples_b[i], samples_b[j])
                            for i, j in product(idx_b, idx_b)]).mean(dim=0)
    else:
        e_bb = 0

    ged = 2 * e_ab - e_aa - e_bb
    sample_diversity = e_aa

    return ged, sample_diversity


def generalized_energy_distance(model, x, ys, sample_count=16):
    """Calculates the Generalized Energy Distance.

    Args:
        model: the model, implementing a sample_prediction method
        x: The images, Float-Tensor of shape BxCxHxW
        ys: The annotations, List of Long-Tensors of shape Bx1xHxW of segmentation maps
        sample_count: the amount of samples to draw from the model

    Returns:
        ged, sample_diversity: Lists of distances, length B
    """
    y_hats = [model.sample_prediction(x) for _ in range(sample_count)]
    return ged(y_hats, ys)


if __name__ == '__main__':
    a = torch.tensor([[[[1, 1], [0, 0]]], [[[1, 1], [0, 0]]]])
    b = torch.tensor([[[[1, 0], [1, 0]]], [[[1, 1], [0, 0]]]])
    print(iou(a, b))

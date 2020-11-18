import torch


def pearsonr(x, y):
    """
    Pearson-correlation coefficient for batched operation
    WARNING: returns Nan if one of the inputs is a zero-matrix

    Arguments
    ---------
    x : 2+D torch.Tensor
    y : 2+D torch.Tensor

    Returns
    -------
    r_val : 1d torch.Tensor
        pearsonr correlation coefficient between x and y
    """
    B = x.shape[0]
    x = x.view(B, -1)
    y = y.view(B, -1)
    mean_x = torch.mean(x, dim=1, keepdim=True)
    mean_y = torch.mean(y, dim=1, keepdim=True)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.sum(xm * ym, dim=1)
    r_den = torch.norm(xm, p=2, dim=1) * torch.norm(ym, p=2, dim=1)
    r_val = r_num / r_den
    return r_val

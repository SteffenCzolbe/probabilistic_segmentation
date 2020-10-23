import torch.nn as nn
from torchvision import transforms
import numpy as np


class RandomAffine(nn.Module):

    def __init__(
        self,
        degrees=(-180, 180),
        translate=(-0.5, 0.5),
        scale=(0.9, 1.1),
        shear=(-0.03, 0.03),
        flip=True,
    ):
        """Random affine transformation of the image keeping center invariant

        Args:
            degrees (sequence or float or int): Range of degrees to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to None to deactivate rotations.
            translate (tuple, optional): tuple of maximum absolute fraction for horizontal
                and vertical translations. For example translate=(a, b), then horizontal shift
                is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
                randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
            scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
                randomly sampled from the range a <= scale <= b. Will keep original scale by default.
            shear (sequence of float or int, optional): Range of degrees to select from.
                If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
                will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
                range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
                a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
                Will not apply shear by default
            flip (boolean), random flips along axis
        """
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.flip = flip

    def get_transform_params(self):
        """
        randomizes the affine transformation
        """
        angle = np.random.uniform(
            *self.degrees, size=1) if self.degrees is not None else 0
        translate = tuple(np.random.uniform(
            *self.translate, size=2)) if self.translate is not None else 0
        scale = np.random.uniform(
            *self.scale, size=1) if self.scale is not None else 0
        shear = tuple(np.random.uniform(
            *self.scale, size=2)) if self.shear is not None else 0
        flip = np.random.choice(
            [True, False], size=2) if self.flip else (False, False)
        return angle, translate, scale, shear, flip

    def randomize(self):
        angle, translate, scale, shear, flip = self.get_transform_params()
        self.affine_params = [angle, translate, scale, shear]
        self.flip_params = flip

    def forward(self, img):
        if self.flip_params[0]:
            img = transforms.functional.hflip(img)
        if self.flip_params[1]:
            imgs = transforms.functional.vflip(img)

        img = transforms.functional.affine(
            img, *self.affine_params, resample=0, fillcolor=0)
        return img

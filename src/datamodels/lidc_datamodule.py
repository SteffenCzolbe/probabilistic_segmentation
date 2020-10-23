import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os
import re
import glob
from PIL import Image
from torchvision import transforms
from .augmentation import RandomAffine


class LIDCDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data/lidc/', batch_size: int = 32, separate_multiple_annotations=True):
        """Data module for the LIDC dataset

        Args:
            data_dir (str, optional): the lidc data directory. Defaults to './data/lidc/'.
            batch_size (int, optional): Batch size. Defaults to 32.
            separate_multiple_annotations (bool, optional): Defines the format of the dataset.
                The dataset comes with multiple annotations, eg X, Y1, Y2, Y3, ...
                if set to True, Samples will be retuned as tuples (X, Y1), (X, Y2), (X,Y3)
                if set to false, Samples will be returned as tuples (X, (Y1, Y2, Y3))
                Defaults to True.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.separate_multiple_annotations = separate_multiple_annotations

        self.dims = (1, 128, 128)
        self.augment = RandomAffine()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        # check if data available
        if not os.path.isdir(os.path.join(self.data_dir, "LIDC_crops")):
            print('Data not found. Downloading...')
            from data.lidc.download import main as download_lidc
            download_lidc()

    def train_dataloader(self):
        dataset = LIDCDataset(
            self.data_dir, "train", separate_multiple_annotations=self.separate_multiple_annotations, transform=self.transform, augment=self.augment)
        return DataLoader(dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        dataset = LIDCDataset(
            self.data_dir, "val", separate_multiple_annotations=self.separate_multiple_annotations, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        dataset = LIDCDataset(
            self.data_dir, "test", separate_multiple_annotations=self.separate_multiple_annotations, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size)


class LIDCDataset(Dataset):
    def __init__(self, data_dir, datasplit, separate_multiple_annotations, transform=None, augment=None):
        self.separate_multiple_annotations = separate_multiple_annotations
        self.transform = transform
        self.augment = augment
        self.annotation_dir = os.path.join(
            data_dir, "LIDC_crops", "LIDC_DLCV_version", datasplit, "lesions")
        self.annotations = os.listdir(self.annotation_dir)
        self.img_dir = os.path.join(
            data_dir, "LIDC_crops", "LIDC_DLCV_version", datasplit, "images")
        self.images = os.listdir(self.img_dir)

        # biold tuples of X, Y
        if self.separate_multiple_annotations:
            self.build_data_for_separate_annotations()
        else:
            self.build_data_for_collected_annotations()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # randomizer augmentation
        if self.augment:
            self.augment.randomize()

        # load, augment and transform images
        if self.separate_multiple_annotations:
            x, y = self.data[index]
            return self.load_and_transform_image(x), self.load_and_transform_image(y)
        else:
            x, ys = self.data[index]
            return self.load_and_transform_image(x), [self.load_and_transform_image(y) for y in ys]

    def load_and_transform_image(self, fname):
        # load image
        image = Image.open(fname)

        # app augmentation
        if self.augment:
            image = self.augment(image)

        # apply transforms
        tensor_image = self.transform(image)

        # labels should be integers
        is_image = "images" in fname
        if not is_image:
            tensor_image = tensor_image.type(torch.LongTensor)
        return tensor_image

    def build_data_for_separate_annotations(self):
        self.data = []
        for annotation in self.annotations:
            annotation_prefix = annotation[:-7]
            image = os.path.join(self.img_dir, annotation_prefix + ".png")
            annotation = os.path.join(self.annotation_dir, annotation)
            self.data.append((image, annotation))

    def build_data_for_collected_annotations(self):
        images = sorted(self.images)
        annotations = sorted(self.annotations)
        self.data = []
        for image in images:
            image_prefix = image[:-4]
            matches_for_this_image = []
            while len(annotations) > 0 and annotations[0].startswith(image_prefix):
                annotation = annotations.pop(0)
                matches_for_this_image.append(
                    os.path.join(self.annotation_dir, annotation))
            self.data.append((os.path.join(self.img_dir, image),
                              tuple(matches_for_this_image)))


if __name__ == '__main__':
    dm = LIDCDataModule(separate_multiple_annotations=False)
    ds = dm.test_dataloader()
    print(ds.dataset.data[122])
    sample = ds.dataset[122]

    print(sample[0].shape)
    print(sample[1][0].shape)

    from torchvision.utils import save_image
    save_image(sample[0], 'test.png')

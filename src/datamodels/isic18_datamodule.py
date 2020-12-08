import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os
import re
import glob
from PIL import Image
from torchvision import transforms
from .augmentation import RandomAffine


class ISIC18DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data/isic18/254/', batch_size: int = 32):
        """Data module for the ISIC18 dataset

        Args:
            data_dir (str, optional): the isic18 data directory. Defaults to './data/isic18/'.
            batch_size (int, optional): Batch size. Defaults to 32.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sampler = None
        self.num_workers = 0
        self.dims = (3, 256, 256)
        self.classes = 2
        self.augment = RandomAffine()
        self.transform = transforms.Compose([
            transforms.Pad(padding=1, padding_mode='edge'),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        # check if data available
        if not os.path.isdir(os.path.join(self.data_dir, "train")):
            print('Data not found. Downloading...')
            from data.isic18.download import main as download_isic18
            download_isic18()

    def train_dataloader(self):
        dataset = ISIC18Dataset(
            self.data_dir, "train", transform=self.transform, augment=self.augment)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        dataset = ISIC18Dataset(
            self.data_dir, "val", transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        dataset = ISIC18Dataset(
            self.data_dir, "test", transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class ISIC18Dataset(Dataset):
    def __init__(self, data_dir, datasplit, transform=None, augment=None):
        self.transform = transform
        self.augment = augment
        self.annotation_dir = os.path.join(
            data_dir, datasplit, "segmentations")
        self.annotations = os.listdir(self.annotation_dir)
        self.img_dir = os.path.join(
            data_dir, datasplit, "images")
        self.images = os.listdir(self.img_dir)

        # biold tuples of X, Y
        self.build_file_tuples()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # randomizer augmentation
        if self.augment:
            self.augment.randomize()

        # load, augment and transform images
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

    def build_file_tuples(self):
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
    dm = ISIC18DataModule()
    ds = dm.test_dataloader()
    print(ds.dataset.data[122])
    sample = ds.dataset[122]

    print(sample[0].shape)
    print(sample[1][0].shape)

    from torchvision.utils import save_image
    save_image(sample[0], 'test.png')

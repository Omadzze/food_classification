import json

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import  datasets
import torchvision.transforms.v2 as transforms
import src.config as config
class Dataset:

    def __init__(self):
        self.root = config.PATH
        self.batch_size = config.BATCH_SIZE
        self.num_workers = config.NUM_WORKERS
        self.validation_split = config.VALIDATION_SPLIT

    # https://rumn.medium.com/ultimate-guide-to-fine-tuning-in-pytorch-part-3-deep-dive-to-pytorch-data-transforms-53ed29d18dde
    def train_transform(self) -> transforms.Compose:
        """
        Normalizes image and transforms it. Applies data augmentation techniques to the images.
        Note apply this only to the training images and not the test images!

        :return:
        """
        transform = transforms.Compose([
            # resize image to H, W = 224
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),

            # Data augmentation
            # apply contrast and brightness to the image
            transforms.ColorJitter(brightness = config.BRIGHTNESS, contrast = config.CONTRAST,
                                   saturation = config.SATURATION, hue = config.HUE),

            # adjust image rotation
            transforms.RandomRotation(degrees=(config.ROTATION_RANGE_MIN, config.ROTATION_RANGE_MAX)),

            transforms.RandomHorizontalFlip(p=config.FLIPPING),
            transforms.RandomVerticalFlip(p=config.FLIPPING),
        ])

        return transform

    def eva_transform(self) -> transforms.Compose:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),
        ])

        return transform

    def data_preparation(self):

        """
        Applies data augmentation techniques to the training images. Loads the dataset
        :return: augmented training and test data
        """

        train_data = datasets.Food101(
            root=self.root,
            split="train",
            download=True,
            transform = self.train_transform()
        )

        test_data = datasets.Food101(
            root=self.root,
            split="test",
            download=True,
            transform=self.eva_transform(),
            target_transform = None
        )

        return train_data, test_data

    def data_loading(self):

        """
        Creates validation dataset out of training dataset.
        Then makes dataloader for the training, testing, validation sets.
        :param validation_set: ration of the validation set
        :return: dataloaders for the train, testing and validation sets
        """

        train_data, test_data = self.data_preparation()

        # Splitting data to the training and validation
        val_size = int(len(train_data) * self.validation_split)
        train_size = len(train_data) - val_size

        train_data, val_data = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_data, batch_size=self.batch_size, num_workers=self.batch_size, shuffle=True, pin_memory=True)
        validation_loader = DataLoader(val_data, batch_size=self.batch_size, num_workers=self.batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, num_workers=self.batch_size, shuffle=False, pin_memory=True)

        return train_loader, validation_loader, test_loader

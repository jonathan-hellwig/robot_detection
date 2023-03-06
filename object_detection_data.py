import os
import itertools
import shutil
from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from PIL import Image
import torchvision
import torchvision.transforms as T
from utils import Encoder


class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path: str, image_transforms=None, bounding_box_transforms=None
    ) -> None:
        self.image_transforms = image_transforms
        self.bounding_box_transforms = bounding_box_transforms
        self.data_path = data_path
        self.images = sorted(os.listdir(data_path + "/images/"))
        self.labels = sorted(os.listdir(data_path + "/labels/"))

    def __getitem__(self, idx):
        # Check if paths are equal
        image_path = self.data_path + "/images/" + self.images[idx]
        label_path = self.data_path + "/labels/" + self.labels[idx]
        image = Image.open(image_path)
        label_strings = open(label_path).read().splitlines()
        target_bounding_boxes = []
        target_classes = []
        for label_string in label_strings:
            target_bounding_boxes.append(
                np.array(np.fromstring(label_string[1:], sep=" ", dtype=np.float32))
            )
            target_classes.append(
                np.array(
                    np.fromstring(label_string[0], sep=" ", dtype=np.float32),
                    dtype=np.int64,
                )
            )
        target_bounding_boxes = torch.tensor(np.array(target_bounding_boxes))
        target_classes = torch.tensor(np.array(target_classes)) + 1
        if self.bounding_box_transforms:
            target_bounding_boxes = self.bounding_box_transforms(target_bounding_boxes)
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, target_bounding_boxes, target_classes

    def __len__(self):
        return len(self.images)


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, encoder: Encoder) -> None:
        self.encoder = encoder
        self.images = torch.load(data_path + "/transformed_images.pt")
        bounding_boxes = torch.load(data_path + "/target_bounding_boxes.pt")
        object_classes = torch.load(data_path + "/target_classes.pt")
        self.encoded_bounding_boxes = []
        self.encoded_target_classes = []
        self.target_masks = []
        for bounding_box, object_class in zip(bounding_boxes, object_classes):
            encoded_bounding_boxes, target_mask, target_classes = self.encoder.apply(
                bounding_box, object_class
            )
            self.encoded_bounding_boxes.append(encoded_bounding_boxes)
            self.encoded_target_classes.append(target_classes)
            self.target_masks.append(target_mask)

    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.encoded_bounding_boxes[idx],
            self.target_masks[idx],
            self.encoded_target_classes[idx],
        )

    def __len__(self):
        return len(self.images)


def calculate_mean_std(dataset: ObjectDetectionDataset):
    means = []
    for image, _, _ in tqdm.tqdm(dataset):
        means.append(torch.mean(image, dim=(1, 2)))
    stacked_means = torch.stack(means)
    mean = torch.mean(stacked_means, dim=0)
    std = torch.std(stacked_means, dim=0)
    return mean, std


def create_train_val_split():
    images = os.listdir("data/train/images/")
    index = torch.randperm(len(images))
    val_index = index[: int(0.2 * len(images))]
    val_files = [images[idx][:-3] for idx in val_index]
    for files in val_files:
        shutil.move(
            "data/train/images/" + files + "png", "data/val/images/" + files + "png"
        )
        shutil.move(
            "data/train/labels/" + files + "txt", "data/val/labels/" + files + "txt"
        )


def preprocess_data(
    data_path,
    image_augmentations=[lambda x: x],
    bounding_box_augmentations=[lambda x: x],
):
    image_transforms = T.Compose(
        [
            T.Grayscale(),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Resize((60, 80)),
        ]
    )
    bounding_box_transforms = T.Compose([])
    train_data = ObjectDetectionDataset(
        data_path,
        image_transforms=image_transforms,
        bounding_box_transforms=bounding_box_transforms,
    )
    images = []
    image_means = []
    image_stds = []
    target_bounding_boxes = []
    target_classes = []
    for i, (image, target_bounding_box, target_class) in enumerate(train_data):
        for image_augmentation, bounding_box_augmentation in zip(
            image_augmentations, bounding_box_augmentations
        ):
            images.append(image_augmentation(image))
            target_bounding_boxes.append(bounding_box_augmentation(target_bounding_box))
            target_classes.append(target_class)
            image_means.append(torch.mean(images[-1], dim=(1, 2)))
            image_stds.append(torch.std(images[-1], dim=(1, 2)))
        if i % 10 == 0:
            print(f"{i}/{len(train_data)}")
    image_means = torch.tensor(image_means)
    image_stds = torch.tensor(image_stds)
    image_mean = torch.mean(image_means)
    image_std = torch.mean(image_stds)

    stacked_images = (torch.stack(images) - image_mean) / image_std
    torch.save(stacked_images, data_path + "/transformed_images.pt")
    torch.save(target_bounding_boxes, data_path + "/target_bounding_boxes.pt")
    torch.save(target_classes, data_path + "/target_classes.pt")
    torch.save(torch.tensor([image_mean, image_std]), data_path + "/image_normalize.pt")


def flip_bounding_boxes(bounding_boxes):
    bounding_boxes = bounding_boxes.clone()
    bounding_boxes[:, 0] = 1 - bounding_boxes[:, 0]
    return bounding_boxes


if __name__ == "__main__":
    image_augmentations = [lambda x: x, torchvision.transforms.functional.hflip]
    bounding_box_augmentations = [lambda x: x, flip_bounding_boxes]
    preprocess_data("data/val")
    preprocess_data("data/train", image_augmentations, bounding_box_augmentations)

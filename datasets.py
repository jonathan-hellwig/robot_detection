import pytorch_lightning as pl
import os
import torch
import numpy as np
import tqdm
from PIL import Image, ImageDraw
import torchvision
import torchvision.transforms as T

import utils


class RoboEireanData(torch.utils.data.Dataset):
    """Custom PyTorch Dataset class for loading RoboEireann dataset.

    Args:
        data_path (str): Path to dataset directory.
        selected_classes (List[str]): List of classes to select from ["ball", "robot", "goal_post", "penalty_spot"].
        image_transforms (optional): Torchvision transforms to apply on the images. Defaults to None.
        bounding_box_transforms (optional): A function that takes in a tensor of bounding boxes and applies the transformation. Defaults to None.
    """

    CLASSES = ["ball", "robot", "goal_post", "penalty_spot"]

    def __init__(
        self,
        data_path: str,
        selected_classes: list[str],
        image_transforms=None,
        bounding_box_transforms=None,
    ) -> None:
        assert set(selected_classes) <= set(self.CLASSES)
        self.image_transforms = image_transforms
        self.bounding_box_transforms = bounding_box_transforms
        self.data_path = data_path
        self.selected_classes = selected_classes

        self.images = sorted(os.listdir(os.path.join(data_path, "images")))
        self.labels = sorted(os.listdir(os.path.join(data_path, "labels")))

    def __getitem__(self, idx):
        assert self.images[idx][:-3] == self.labels[idx][:-3]
        image_path = os.path.join(self.data_path, "images", self.images[idx])
        label_path = os.path.join(self.data_path, "labels", self.labels[idx])

        image = Image.open(image_path)
        with open(label_path) as f:
            label_strings = f.read().splitlines()

        target_bounding_boxes_list = []
        target_classes_list = []

        for label_string in label_strings:
            parsed_target_class = int(label_string[0])
            if self.CLASSES[parsed_target_class] in self.selected_classes:
                target_classes_list.append(
                    self.selected_classes.index(self.CLASSES[parsed_target_class])
                )
                target_bounding_boxes_list.append(
                    np.fromstring(label_string[1:], sep=" ", dtype=np.float32)
                )

        target_bounding_boxes = torch.tensor(np.array(target_bounding_boxes_list))
        target_classes = (
            torch.tensor(np.array(target_classes_list), dtype=torch.long) + 1
        )

        for transform in self.bounding_box_transforms:
            target_bounding_boxes, target_classes = transform(
                target_bounding_boxes, target_classes
            )
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, target_bounding_boxes, target_classes

    def __len__(self):
        return len(self.images)


class RoboEireanDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        selected_classes: list[str],
        encoder: utils.Encoder,
        batch_size: int = 32,
    ):
        super().__init__()
        self.data_path = data_path
        self.selected_classes = selected_classes
        self.encoder = encoder
        self.batch_size = batch_size
        self.image_transforms = T.Compose(
            [
                T.Grayscale(),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Resize((60, 80), antialias=True),
            ]
        )
        self.bounding_box_transforms = [self.encoder.encode]

    def setup(self, stage: str = None):
        # TODO: Consider joining the traing and validation data
        # TODO: Consider encoding the data once at the start of training
        self.train_dataset = RoboEireanData(
            os.path.join(self.data_path, "train"),
            self.selected_classes,
            image_transforms=self.image_transforms,
            bounding_box_transforms=self.bounding_box_transforms,
        )
        self.val_dataset = RoboEireanData(
            os.path.join(self.data_path, "val"),
            self.selected_classes,
            image_transforms=self.image_transforms,
            bounding_box_transforms=self.bounding_box_transforms,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )


class SyntheticDataModule(pl.LightningDataModule):
    IMAGE_WIDTH = 80
    IMAGE_HEIGHT = 60
    LENGTH = 16

    def __init__(
        self,
        encoder: utils.Encoder,
        batch_size: int = 32,
    ):
        super().__init__()

        self.encoder = encoder
        self.batch_size = batch_size

    def setup(self, stage: str = None):
        self.train_dataset = SyntheticData(
            self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.LENGTH, self.encoder
        )
        self.val_dataset = SyntheticData(
            self.IMAGE_WIDTH, self.IMAGE_HEIGHT, int(self.LENGTH * 0.1), self.encoder
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )


class SyntheticData(torch.utils.data.Dataset):
    """
    SyntheticData generates a synthetic dataset of images with bounding boxes and classes.

    Args:
        image_width (int): Width of the generated images.
        image_height (int): Height of the generated images.
        length (int): Number of images in the dataset.
        encoder (utils.Encoder): An encoder object to encode bounding boxes and classes.
    """

    def __init__(
        self, image_width: int, image_height: int, length: int, encoder: utils.Encoder
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.length = length
        self.encoder = encoder

        self.images = []
        self.encoded_bounding_boxes = []
        self.encoded_target_classes = []
        for _ in range(length):
            image, bounding_box = self._generate_image()
            encoded_bounding_boxes, target_classes = self.encoder.encode(
                bounding_box, torch.tensor([[1]])
            )
            self.images.append(image)
            self.encoded_bounding_boxes.append(encoded_bounding_boxes)
            self.encoded_target_classes.append(target_classes)

    def __getitem__(self, idx: int):
        return (
            self.images[idx],
            self.encoded_bounding_boxes[idx],
            self.encoded_target_classes[idx],
        )

    def __len__(self) -> int:
        return self.length

    def _generate_image(self):
        image = Image.new("1", (self.image_width, self.image_height))
        image_draw = ImageDraw.Draw(image)
        center = torch.tensor(
            [
                torch.randint(0, self.image_width - 1, (1,)).item(),
                torch.randint(0, self.image_height - 1, (1,)).item(),
            ]
        )
        size = torch.tensor([self.image_width * 0.25, self.image_height * 0.25])
        upper_left = center - size / 2
        lower_right = center + size / 2
        image_draw.rectangle(
            [upper_left[0], upper_left[1], lower_right[0], lower_right[1]], fill=255
        )
        bounding_box = torch.tensor(
            [
                [
                    center[0].item() / self.image_width,
                    center[1].item() / self.image_height,
                    size[0].item() / self.image_width,
                    size[1].item() / self.image_height,
                ]
            ]
        )
        return (
            torchvision.transforms.functional.pil_to_tensor(image).to(torch.float),
            bounding_box,
        )


def calculate_mean_std(dataset: RoboEireanData):
    means = []
    for image, _, _ in tqdm.tqdm(dataset):
        means.append(torch.mean(image, dim=(1, 2)))
    stacked_means = torch.stack(means)
    mean = torch.mean(stacked_means, dim=0)
    std = torch.std(stacked_means, dim=0)
    return mean, std


def preprocess_data(
    base_path: str,
    split_path: str,
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
    train_data = RoboEireanData(
        os.path.join(base_path, "raw", split_path),
        ["robot"],
        image_transforms=image_transforms,
        bounding_box_transforms=bounding_box_transforms,
    )
    images = []
    image_means = []
    image_stds = []
    target_bounding_boxes = []
    target_classes = []
    for image, target_bounding_box, target_class in tqdm.tqdm(train_data):
        for image_augmentation, bounding_box_augmentation in zip(
            image_augmentations, bounding_box_augmentations
        ):
            images.append(image_augmentation(image))
            target_bounding_boxes.append(bounding_box_augmentation(target_bounding_box))
            target_classes.append(target_class)
            image_means.append(torch.mean(images[-1], dim=(1, 2)))
            image_stds.append(torch.std(images[-1], dim=(1, 2)))
    image_means = torch.tensor(image_means)
    image_stds = torch.tensor(image_stds)
    image_mean = torch.mean(image_means)
    image_std = torch.mean(image_stds)

    stacked_images = (torch.stack(images) - image_mean) / image_std
    transformed_images_path = os.path.join(
        base_path, "transformed", split_path, "transformed_images.pt"
    )
    target_bounding_boxes_path = os.path.join(
        base_path, "transformed", split_path, "target_bounding_boxes.pt"
    )
    target_classes_path = os.path.join(
        base_path, "transformed", split_path, "target_classes.pt"
    )
    image_normalize_path = os.path.join(
        base_path, "transformed", split_path, "image_normalize.pt"
    )
    torch.save(stacked_images, transformed_images_path)
    torch.save(target_bounding_boxes, target_bounding_boxes_path)
    torch.save(target_classes, target_classes_path)
    torch.save(torch.tensor([image_mean, image_std]), image_normalize_path)


def flip_bounding_boxes(bounding_boxes: torch.Tensor) -> torch.Tensor:
    bounding_boxes = bounding_boxes.clone()
    if bounding_boxes.dim() == 2:
        bounding_boxes[:, 0] = 1 - bounding_boxes[:, 0]
    return bounding_boxes


if __name__ == "__main__":
    image_augmentations = [lambda x: x, torchvision.transforms.functional.hflip]
    bounding_box_augmentations = [lambda x: x, flip_bounding_boxes]
    preprocess_data("data", "val")
    preprocess_data(
        "data",
        "train",
        image_augmentations,
        bounding_box_augmentations,
    )

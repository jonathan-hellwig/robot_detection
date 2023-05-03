import os
import shutil
import torch
import numpy as np
import tqdm
from PIL import Image, ImageDraw
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
        assert self.images[idx][:-3] == self.labels[idx][:-3]
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


def index_to_class(index):
    return ["no_object", "ball", "robot", "penalty_spot", "goal_post"][index]


def index_to_updated_index(index, classes):
    updated_index = torch.zeros_like(index)
    for i in range(index.size(0)):
        updated_index[i] = classes.index(index_to_class(index[i])) + 1
    return updated_index


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        encoder: Encoder,
    ) -> None:
        self.encoder = encoder
        loaded_images = torch.load(data_path + "/transformed_images.pt")
        bounding_boxes = torch.load(data_path + "/target_bounding_boxes.pt")
        object_classes = torch.load(data_path + "/target_classes.pt")
        self.images = []
        self.encoded_bounding_boxes = []
        self.encoded_target_classes = []
        self.target_masks = []
        for bounding_box, object_class, image in zip(
            bounding_boxes, object_classes, loaded_images
        ):
            is_selected_object_class = [
                index_to_class(c) in encoder.classes for c in object_class
            ]
            if any(is_selected_object_class):
                is_selected_object_class = torch.tensor(is_selected_object_class)
                bounding_box = bounding_box[is_selected_object_class]
                object_class = index_to_updated_index(
                    object_class[is_selected_object_class], encoder.classes
                )
                (
                    encoded_bounding_boxes,
                    target_mask,
                    target_classes,
                ) = self.encoder.apply(bounding_box, object_class)
                self.encoded_bounding_boxes.append(encoded_bounding_boxes)
                self.encoded_target_classes.append(target_classes)
                self.target_masks.append(target_mask)
                self.images.append(image)

    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.encoded_bounding_boxes[idx],
            self.target_masks[idx],
            self.encoded_target_classes[idx],
        )

    def __len__(self):
        return len(self.encoded_bounding_boxes)


class SyntheticData(torch.utils.data.Dataset):
    def __init__(self, image_width, image_height, length, encoder):
        print('Generating synthetic data...')
        self.image_width = image_width
        self.image_height = image_height
        self.length = length
        self.encoder = encoder

        self.images = []
        self.encoded_bounding_boxes = []
        self.encoded_target_classes = []
        self.target_masks = []
        for _ in range(length):
            image, bounding_box = self._generate_image()
            encoded_bounding_boxes, target_mask, target_classes = self.encoder.apply(
                bounding_box, torch.tensor([[1]])
            )
            self.images.append(image)
            self.encoded_bounding_boxes.append(encoded_bounding_boxes)
            self.encoded_target_classes.append(target_classes)
            self.target_masks.append(target_mask)
        print('Done!')

    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.encoded_bounding_boxes[idx],
            self.target_masks[idx],
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

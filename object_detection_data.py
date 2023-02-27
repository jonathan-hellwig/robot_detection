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


class Encoder:
    def __init__(
        self,
        default_scalings: torch.Tensor,
        feature_map_size: Tuple[int, int],
        num_classes: int,
        threshold: float = 0.5,
    ) -> None:
        # width x height
        self.default_scalings = default_scalings
        self.feature_map_size = feature_map_size
        self.num_classes = num_classes
        self.threshold = threshold
        self.default_boxes_tr_bl = self._default_boxes_("xywh")
        self.default_boxes_xy_wh = self._default_boxes_("tlbr")

    def apply(self, target_boxes: torch.Tensor, target_classes: torch.Tensor):
        NUM_BOX_PARAMETERS = 4
        # Transform bounding boxes from (cx, cy, w, h) to (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
        target_boxes_tl_br = torch.zeros((target_boxes.size(0), 4))
        target_boxes_tl_br[:, 0:2] = target_boxes[:, 0:2] - target_boxes[:, 2:4] / 2
        target_boxes_tl_br[:, 2:4] = target_boxes[:, 0:2] + target_boxes[:, 2:4] / 2
        # Select the default box with the highest IoU and with IoU higher than the threshold value
        ious = calc_iou_tensor(target_boxes_tl_br, self.default_boxes_tr_bl)
        _, best_dbox_idx = ious.max(dim=1)
        masked_ious = (
            torch.logical_or(
                (
                    best_dbox_idx
                    == torch.arange(0, self.default_boxes_tr_bl.size(0)).reshape(-1, 1)
                ).T,
                ious > self.threshold,
            )
            * ious
        )
        # Select the target box with the highest IoU for each default box
        best_value, best_idx = masked_ious.max(dim=0)

        mask_idx = best_value > 0
        selected_target_boxes = target_boxes[best_idx[mask_idx]]
        selected_default_boxes = self.default_boxes_xy_wh[mask_idx]

        # Encode target boxes with relative offsets to default box
        encoded_target_boxes = torch.zeros(
            (self.default_boxes_xy_wh.size(0), NUM_BOX_PARAMETERS)
        )
        encoded_target_boxes[mask_idx, 0:2] = (
            selected_default_boxes[:, 0:2] - selected_target_boxes[:, 0:2]
        ) / selected_default_boxes[:, 2:4]
        encoded_target_boxes[mask_idx, 2:4] = torch.log(
            selected_target_boxes[:, 2:4] / selected_default_boxes[:, 2:4]
        )

        dbox_classes = torch.zeros(self.default_boxes_xy_wh.size(0), dtype=torch.long)

        dbox_classes[mask_idx] = target_classes[best_idx[mask_idx]].flatten()
        return encoded_target_boxes, mask_idx, dbox_classes


    def _default_boxes(self, type):
        assert type in ["xywh", "tlbr"]
        NUM_BOX_PARAMETERS = 4
        feature_map_height, feature_map_width = self.feature_map_size
        default_boxes = torch.zeros(
            (
                feature_map_height,
                feature_map_width,
                self.default_scalings.size(0),
                NUM_BOX_PARAMETERS,
            )
        )
        for i, j in itertools.product(
            range(feature_map_height), range(feature_map_width)
        ):
            center = torch.tensor(
                [
                    (i + 0.5) / feature_map_height,
                    (j + 0.5) / feature_map_width,
                ]
            )
            if type == "xywh":
                default_boxes[i, j, :, 0:2] = center
                default_boxes[i, j, :, 2:4] = self.default_scalings
            else:
                default_boxes[i, j, :, 0:2] = center - self.default_scalings / 2
                default_boxes[i, j, :, 2:4] = center + self.default_scalings / 2

        return default_boxes.reshape((-1, NUM_BOX_PARAMETERS))

    def decode_model_output(self, boxes, class_logits):
        """
        boxes: batch_size x f_h x f_w x num_default_scalings x 4
        classes: batch_size x f_h x f_w x num_default_scalings x num_classes + 1
        """
        class_probs = F.softmax(class_logits, dim=-1)
        decoded_boxes = torch.zeros(boxes.shape)
        decoded_boxes[:, :, :, :, 0:2] = self.default_boxes_xy_wh[:, :, :, :, 2:4] * (
            self.default_boxes_xy_wh[:, 0:2] - boxes
        )
        decoded_boxes[:, :, :, :, 2:4] = self.default_boxes_xy_wh[
            :, :, :, :, 2:4
        ] * torch.exp(boxes[:, :, :, :, 2:4])
        return decoded_boxes, class_probs


def calc_iou_tensor(box1, box2):
    """Calculation of IoU based on two boxes tensor,
    Reference to https://github.com/kuangliu/pytorch-src
    input:
        box1 (N, 4)
        box2 (M, 4)
    output:
        IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(be1[:, :, :2], be2[:, :, :2])
    rb = torch.min(be1[:, :, 2:], be2[:, :, 2:])

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:, :, 0] * delta[:, :, 1]

    delta1 = be1[:, :, 2:] - be1[:, :, :2]
    area1 = delta1[:, :, 0] * delta1[:, :, 1]
    delta2 = be2[:, :, 2:] - be2[:, :, :2]
    area2 = delta2[:, :, 0] * delta2[:, :, 1]

    iou = intersect / (area1 + area2 - intersect)
    return iou


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
        self.encoded_object_classes = []
        self.target_masks = []
        for i, (bounding_box, object_class) in enumerate(
            zip(bounding_boxes, object_classes)
        ):
            encoded_bounding_boxes, target_mask, target_classes = self.encoder.apply(
                bounding_box, object_class
            )
            self.encoded_bounding_boxes.append(encoded_bounding_boxes)
            self.encoded_object_classes.append(target_classes)
            self.target_masks.append(target_mask)

    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.encoded_bounding_boxes[idx],
            self.target_masks[idx],
            self.encoded_object_classes[idx],
            idx,
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

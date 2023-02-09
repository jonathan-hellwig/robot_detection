import os
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
        self.default_scalings = default_scalings
        self.feature_map_size = feature_map_size
        self.num_classes = num_classes
        self.threshold = threshold
        self.default_boxes_tr_bl = self._default_boxes_tr_bl()
        self.default_boxes_xy_hw = self._default_boxes_xy_hw()

    def apply(self, target_boxes: torch.Tensor, target_classes: torch.Tensor):
        NUM_BOX_PARAMETERS = 4
        # Select the default box with the highest IoU and with IoU higher than the threshold value
        target_boxes_tl_br = torch.zeros((target_boxes.size(0), 4))
        target_boxes_tl_br[:, 0:2] = target_boxes[:, 0:2] - target_boxes[:, 3:4] / 2
        target_boxes_tl_br[:, 2:4] = target_boxes[:, 0:2] + target_boxes[:, 3:4] / 2
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
        selected_default_boxes = self.default_boxes_xy_hw[mask_idx]

        # Encode target boxes with relative offsets to default box
        encoded_target_boxes = torch.zeros(
            (self.default_boxes_xy_hw.size(0), NUM_BOX_PARAMETERS)
        )
        encoded_target_boxes[mask_idx, 0:2] = (
            selected_default_boxes[:, 0:2] - selected_target_boxes[:, 0:2]
        ) / selected_default_boxes[:, 2:4]
        encoded_target_boxes[mask_idx, 2:4] = torch.log(
            selected_target_boxes[:, 2:4] / selected_default_boxes[:, 2:4]
        )

        dbox_classes = torch.zeros(self.default_boxes_xy_hw.size(0), dtype=torch.long)

        dbox_classes[mask_idx] = target_classes[best_idx[mask_idx]].flatten()
        return encoded_target_boxes, mask_idx, dbox_classes

    def _default_boxes_tr_bl(self):
        NUM_BOX_PARAMETERS = 4
        feature_map_height, feature_map_width = self.feature_map_size
        default_boxes = torch.zeros(
            (
                feature_map_height * feature_map_width * self.default_scalings.size(0),
                NUM_BOX_PARAMETERS,
            )
        )
        for i in range(default_boxes.size(0) // self.default_scalings.size(0)):
            selected_boxes_idx = range(
                i * self.default_scalings.size(0),
                (i + 1) * self.default_scalings.size(0),
            )
            top_left = torch.tensor(
                [
                    (i % feature_map_width) / feature_map_width,
                    (i // feature_map_width) / feature_map_height,
                ]
            )
            default_boxes[selected_boxes_idx, 0:2] = top_left
            default_boxes[selected_boxes_idx, 2:4] = top_left + self.default_scalings
        return default_boxes

    def _default_boxes_xy_hw(self):
        NUM_BOX_PARAMETERS = 4
        feature_map_height, feature_map_width = self.feature_map_size
        default_boxes = torch.zeros(
            (
                feature_map_height * feature_map_width * self.default_scalings.size(0),
                NUM_BOX_PARAMETERS,
            )
        )
        for i in range(default_boxes.size(0) // self.default_scalings.size(0)):
            selected_boxes_idx = range(
                i * self.default_scalings.size(0),
                (i + 1) * self.default_scalings.size(0),
            )
            top_left = torch.tensor(
                [
                    (i % feature_map_width) / feature_map_width,
                    (i // feature_map_width) / feature_map_height,
                ]
            )
            default_boxes[selected_boxes_idx, 0:2] = (
                top_left + self.default_scalings / 2
            )
            default_boxes[selected_boxes_idx, 2:4] = self.default_scalings
        return default_boxes

    def decode_model_output(self, boxes, class_logits):
        """
        boxes: batch_size x f_h x f_w x num_default_scalings x 4
        classes: batch_size x f_h x f_w x num_default_scalings x num_classes + 1
        """
        class_probs = F.softmax(class_logits, dim=-1)
        decoded_boxes = torch.zeros(boxes.shape)
        decoded_boxes[:, :, :, :, 0:2] = self.default_boxes_xy_hw[:, :, :, :, 2:4] * (
            self.default_boxes_xy_hw[:, 0:2] - boxes
        )
        decoded_boxes[:, :, :, :, 2:4] = self.default_boxes_xy_hw[
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
    def __init__(self, data_path: str, encoder: Encoder, transforms=None) -> None:
        self.encoder = encoder
        self.transforms = transforms
        self.data_path = data_path
        self.images = sorted(os.listdir(data_path + "/images/"))
        self.labels = sorted(os.listdir(data_path + "/labels/"))

    def __getitem__(self, idx):
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

        (
            encoded_bounding_boxes,
            target_mask,
            encoded_target_classes,
        ) = self.encoder.apply(target_bounding_boxes, target_classes)

        if self.transforms is not None:
            image = self.transforms(image)
        return image, encoded_bounding_boxes, target_mask, encoded_target_classes

    def __len__(self):
        return len(self.images)


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, encoder: Encoder) -> None:
        self.encoder = encoder
        self.images = torch.load(data_path + "/transformed_images.pt")
        bounding_boxes = torch.load(data_path + "/bounding_boxes.pt")
        object_classes = torch.load(data_path + "/object_classes.pt")
        self.encoded_bounding_boxes = []
        self.encoded_object_classes = []
        self.target_masks = []
        for bounding_box, object_class in zip(bounding_boxes, object_classes):
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


def save_bounding_boxes(data_path):
    labels = sorted(os.listdir(data_path + "/labels/"))
    bounding_boxes = []
    object_classes = []
    for label_path in labels:
        label_strings = open(data_path + "/labels/" + label_path).read().splitlines()
        bounding_boxes_image = []
        object_classes_image = []
        for label_string in label_strings:
            bounding_boxes_image.append(
                np.fromstring(label_string[1:], sep=" ", dtype=np.float32)
            )
            object_classes_image.append(
                np.fromstring(label_string[0], sep=" ", dtype=np.float32)
            )
        bounding_boxes.append(torch.tensor(np.array(bounding_boxes_image)))
        bounding_boxes.append(torch.tensor(np.array(bounding_boxes_image)))
        object_classes.append(
            torch.tensor(np.array(object_classes_image), dtype=torch.long)
        )
        object_classes.append(
            torch.tensor(np.array(object_classes_image), dtype=torch.long)
        )
    torch.save(bounding_boxes, data_path + "/bounding_boxes.pt")
    torch.save(object_classes, data_path + "/object_classes.pt")


def save_transformed_images(data_path):
    transforms = T.Compose(
        [
            T.Grayscale(),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Resize((60, 80)),
        ]
    )
    default_scalings = [
        torch.tensor([0.25, 0.25]),
        torch.tensor([0.125, 0.125]),
        torch.tensor([0.125 / 2, 0.125 / 2]),
        torch.tensor([0.125 / 4, 0.125 / 4]),
    ]
    feature_map_size = (8, 10)
    num_classes = 4
    encoder = Encoder(default_scalings, feature_map_size, num_classes)
    train_data = ObjectDetectionDataset(data_path, encoder, transforms=transforms)
    images = []
    image_means = []
    i = 0
    for image, _, _, _ in train_data:
        images.append(image)
        images.append(torchvision.transforms.functional.hflip(image))
        image_means.append(torch.mean(image, dim=(1, 2)))
        i += 1
        if i % 10 == 0:
            print(f"{i}/{len(train_data)}")
    image_means = torch.tensor(image_means)
    image_mean = torch.mean(image_means)
    image_std = torch.std(image_means)
    stacked_images = (torch.stack(images) - image_mean) / image_std
    torch.save(stacked_images, data_path + "/transformed_images.pt")


if __name__ == "__main__":
    save_bounding_boxes("data/train")
    save_bounding_boxes("data/val")

    # save_transformed_images('data/val')
    # save_transformed_images('data/train')

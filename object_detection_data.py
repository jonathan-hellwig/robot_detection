import shutil
from typing import List, Tuple
import torch
import itertools
import os
import numpy as np
import tqdm
from PIL import Image
import torchvision
import torchvision.transforms as T


class Encoder:
    def __init__(self, default_scalings: List[torch.Tensor], feature_map_size: Tuple[int, int], num_classes: int, threshold: float = 0.5) -> None:
        self.default_scalings = default_scalings
        self.feature_map_size = feature_map_size
        self.num_classes = num_classes
        self.threshold = threshold

    # TODO: Validate encoding
    def apply(self, object_classes, bounding_boxes):
        NUM_BOX_PARAMETERS = 4
        feature_map_height, feature_map_width = self.feature_map_size
        encoded_bounding_boxes = torch.zeros(
            (feature_map_height, feature_map_width, len(self.default_scalings), NUM_BOX_PARAMETERS))
        target_mask = torch.zeros(
            (feature_map_height, feature_map_width), dtype=torch.bool)
        target_classes = torch.zeros(
            (feature_map_height, feature_map_width, len(self.default_scalings)), dtype=torch.long)
        for target_box, ground_truth_class in zip(bounding_boxes, object_classes):
            ground_truth_center = target_box[0:2]
            ground_truth_size = target_box[2:4]
            jaccard_overlaps = []
            for i, j in itertools.product(range(feature_map_height), range(feature_map_width)):
                for k, default_scaling in enumerate(self.default_scalings):
                    default_box_center = torch.tensor(
                        [i / feature_map_width, j / feature_map_height])
                    offset_box = torch.concat(
                        ((ground_truth_center-default_box_center) / default_scaling, torch.log(ground_truth_size / default_scaling)))
                    # TODO: Think about the scaling
                    jaccard_overlaps.append((jaccard_overlap(default_box_center,
                                                             default_scaling, ground_truth_center, ground_truth_size), (i, j, k), offset_box))
            max_overlap = max(jaccard_overlaps, key=lambda item: item[0])
            threshold_overlap = [
                (overlap, index, offset_box) for overlap, index, offset_box in jaccard_overlaps if overlap > self.threshold]
            # TODO: Resolve conflicts between boxes
            for _, (i, j, k), offset_box in [max_overlap] + threshold_overlap:
                encoded_bounding_boxes[i, j, k, :] = offset_box
                target_mask[i, j] = True
                target_classes[i, j, :] = ground_truth_class + 1
        target_classes = target_classes.flatten()
        return encoded_bounding_boxes, target_mask, target_classes


def jaccard_overlap(center_first: torch.Tensor, size_first: torch.Tensor, center_second: torch.Tensor, size_second: torch.Tensor) -> float:
    upper_left_first = center_first - size_first / 2
    lower_right_first = center_first + size_first / 2
    upper_left_second = center_second - size_second / 2
    lower_right_second = center_second + size_second / 2

    intersection_area = max(0, min(lower_right_first[0], lower_right_second[0]) - max(upper_left_first[0], upper_left_second[0])) * max(
        0, min(lower_right_first[1], lower_right_second[1]) - max(upper_left_first[1], upper_left_second[1]))
    union_area = size_first[0] * size_first[1] + \
        size_second[0] * size_second[1] - intersection_area
    return intersection_area / union_area


class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, encoder: Encoder, transforms=None) -> None:
        self.encoder = encoder
        self.transforms = transforms
        self.data_path = data_path
        self.images = sorted(os.listdir(
            data_path + '/images/'))
        self.labels = sorted(os.listdir(
            data_path + '/labels/'))

    def __getitem__(self, idx):
        # TODO: This might break if one label file is missing!
        image_path = self.data_path + '/images/' + \
            self.images[idx]
        label_path = self.data_path + '/labels/' + \
            self.labels[idx]
        # print(image_path)
        # print(label_path)
        image = Image.open(image_path)
        label_strings = open(label_path).read().splitlines()
        bounding_boxes = []
        object_classes = []
        # TODO: Use the encoding here
        for label_string in label_strings:
            bounding_boxes.append(torch.tensor(np.fromstring(
                label_string[1:], sep=' ', dtype=np.float32)))
            object_classes.append(torch.tensor(np.fromstring(
                label_string[0], sep=' ', dtype=np.float32), dtype=torch.long))
        encoded_bounding_boxes, target_mask, target_classes = self.encoder.apply(
            object_classes, bounding_boxes)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, encoded_bounding_boxes, target_mask, target_classes

    def __len__(self):
        return len(self.images)


class TransformedObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, encoder: Encoder) -> None:
        self.encoder = encoder
        self.images = torch.load(data_path + '/transformed_images.pt')
        self.bounding_boxes = torch.load(data_path + '/bounding_boxes.pt')
        self.object_classes = torch.load(data_path + '/object_classes.pt')

    def __getitem__(self, idx):
        encoded_bounding_boxes, target_mask, target_classes = self.encoder.apply(
            self.object_classes[idx], self.bounding_boxes[idx])
        return self.images[idx], encoded_bounding_boxes, target_mask, target_classes

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
    images = os.listdir('SPLObjDetectDatasetV2/train/images/')
    index = torch.randperm(len(images))
    val_index = index[:int(0.2 * len(images))]
    val_files = [images[idx][:-3] for idx in val_index]
    for files in val_files:
        shutil.move('SPLObjDetectDatasetV2/train/images/' +
                    files + 'png', 'SPLObjDetectDatasetV2/val/images/' + files + 'png')
        shutil.move('SPLObjDetectDatasetV2/train/labels/' +
                    files + 'txt', 'SPLObjDetectDatasetV2/val/labels/' + files + 'txt')


def save_bounding_boxes(data_path):
    labels = sorted(os.listdir(data_path + '/labels/'))
    bounding_boxes = []
    object_classes = []
    for label_path in labels:
        label_strings = open(data_path + '/labels/' +
                             label_path).read().splitlines()
        bounding_boxes_image = []
        object_classes_image = []
        for label_string in label_strings:
            bounding_boxes_image.append(torch.tensor(np.fromstring(
                label_string[1:], sep=' ', dtype=np.float32)))
            object_classes_image.append(torch.tensor(np.fromstring(
                label_string[0], sep=' ', dtype=np.float32), dtype=torch.long))
        bounding_boxes.append(bounding_boxes_image)
        object_classes.append(object_classes_image)
    torch.save(bounding_boxes, data_path + '/bounding_boxes.pt')
    torch.save(object_classes, data_path + '/object_classes.pt')


def save_transformed_images(data_path):
    transforms = T.Compose([T.Grayscale(), T.PILToTensor(), T.ConvertImageDtype(
        torch.float32), T.Resize((60, 80))])
    default_scalings = [torch.tensor(
        [0.25, 0.25]), torch.tensor(
        [0.125, 0.125]), torch.tensor([0.125 / 2, 0.125 / 2]), torch.tensor([0.125 / 4, 0.125 / 4])]
    feature_map_size = (8, 10)
    num_classes = 4
    encoder = Encoder(default_scalings, feature_map_size, num_classes)
    train_data = ObjectDetectionDataset(
        data_path, encoder, transforms=transforms)
    images = []
    image_means = []
    i = 0
    for image, _, _, _ in train_data:
        # print(image.shape)
        images.append(image)
        images.append(torchvision.transforms.functional.hflip(image))
        image_means.append(torch.mean(image, dim=(1, 2)))
        i += 1
        if i % 10 == 0:
            print(f'{i}/{len(train_data)}')
    image_means = torch.tensor(image_means)
    image_mean = torch.mean(image_means)
    image_std = torch.std(image_means)
    stacked_images = (torch.stack(images) - image_mean) / image_std
    torch.save(stacked_images, data_path + '/transformed_images.pt')


if __name__ == "__main__":
    save_bounding_boxes('SPLObjDetectDatasetV2/train')
    save_bounding_boxes('SPLObjDetectDatasetV2/val')

    save_transformed_images('SPLObjDetectDatasetV2/val')
    save_transformed_images('SPLObjDetectDatasetV2/train')

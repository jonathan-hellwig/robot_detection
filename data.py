from typing import List, Tuple
import torch
import itertools
import os
import numpy as np
import tqdm
from PIL import Image


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


def calculate_mean_std(dataset: ObjectDetectionDataset):
    means = []
    for image, _, _ in tqdm.tqdm(dataset):
        means.append(torch.mean(image, dim=(1, 2)))
    stacked_means = torch.stack(means)
    mean = torch.mean(stacked_means, dim=0)
    std = torch.std(stacked_means, dim=0)
    return mean, std

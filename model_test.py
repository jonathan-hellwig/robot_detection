import torch
from models import (
    SingleShotDetectorLoss,
    ObjectDetectionTask,
    convert_to_absolute_coordinates,
    draw_image_grid,
)
import unittest
import matplotlib.pyplot as plt


class DrawImageGridTest(unittest.TestCase):
    def test_draw_grid(self):
        images = torch.randn(10, 1, 60, 80)
        target_bounding_boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]).repeat(
            10, 1, 1
        )
        target_classes = torch.tensor([[1]]).repeat(10, 1)
        predicted_bounding_boxes = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]]).repeat(
            10, 1, 1
        )
        predicted_classes = torch.tensor([[[0.0, 1.0]]]).repeat(10, 1, 1)
        image_grid = draw_image_grid(
            images,
            target_bounding_boxes,
            target_classes,
            predicted_bounding_boxes,
            predicted_classes,
        )
        assert image_grid.shape == (3, 188, 330)

    def test_convert_to_absolute(self):
        predicted_bounding_boxes = torch.tensor([[[0.5, 0.5, 1.0, 1.0]]]).repeat(
            10, 1, 1
        )
        image_size = (60, 80)
        absolute_bounding_boxes = convert_to_absolute_coordinates(
            predicted_bounding_boxes, image_size
        )
        assert absolute_bounding_boxes.shape == (10, 1, 4)
        assert torch.allclose(
            absolute_bounding_boxes,
            torch.tensor([[[0.0, 0.0, 80.0, 60.0]]]).repeat(10, 1, 1),
        )


class ObjectDetectionLossTest(unittest.TestCase):
    def test_object_detection_loss(self):
        target_bounding_boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]).repeat(
            10, 1, 1
        )
        target_classes = torch.tensor([[1]]).repeat(10, 1)
        predicted_bounding_boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]).repeat(
            10, 1, 1
        )
        predicted_class_logits = torch.tensor([[[0.0, 1000.0]]]).repeat(10, 1, 1)
        loss = SingleShotDetectorLoss()
        total_loss, location_loss, mined_classification_loss = loss(
            target_bounding_boxes,
            target_classes,
            predicted_bounding_boxes,
            predicted_class_logits,
        )
        assert total_loss == torch.tensor(0.0)
        assert location_loss == torch.tensor(0.0)
        assert mined_classification_loss == torch.tensor(0.0)


if __name__ == "__main__":
    unittest.main()

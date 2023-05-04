import utils
import unittest
import torchvision.transforms as T
import torch

from datasets import (
    RoboEireanData,
    RoboEireanDataWithEncoder,
    TransformedRoboEireanData,
)


class TestRoboEireanData(unittest.TestCase):
    def test_getitem(self):
        image_transforms = T.Compose(
            [
                T.Grayscale(),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Resize((60, 80)),
            ]
        )
        bounding_box_transforms = T.Compose([])
        dataset = RoboEireanData(
            "data/train",
            ["ball", "robot", "goal_post", "penalty_spot"],
            image_transforms,
            bounding_box_transforms,
        )
        image, target_bounding_boxes, target_classes = dataset[0]
        self.assertEqual(image.shape, (1, 60, 80))
        self.assertEqual(target_bounding_boxes.shape, (1, 4))
        self.assertEqual(target_classes.shape, (1, 1))


class TestTransformedDataset(unittest.TestCase):
    def test_getitem(self):
        default_scalings = torch.tensor(
            [
                [0.06549374, 0.12928654],
                [0.11965626, 0.26605093],
                [0.20708716, 0.38876095],
                [0.31018215, 0.47485098],
                [0.415882, 0.8048184],
                [0.7293086, 0.8216225],
            ]
        )
        encoder = utils.Encoder(default_scalings, ["robot"])
        dataset = TransformedRoboEireanData("data/train", encoder)
        image, encoded_bounding_boxes, target_masks, encoded_target_classes = dataset[0]
        self.assertEqual(image.shape, (1, 60, 80))
        self.assertEqual(encoded_bounding_boxes.shape, (480, 4))
        self.assertEqual(target_masks.shape, (480,))
        self.assertEqual(encoded_target_classes.shape, (480,))


class TestRoboEireanDataWithEncoder(unittest.TestCase):
    def test_getitem(self):
        image_transforms = T.Compose(
            [
                T.Grayscale(),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Resize((60, 80)),
            ]
        )
        bounding_box_transforms = T.Compose([])
        default_scalings = torch.tensor(
            [
                [0.06549374, 0.12928654],
                [0.11965626, 0.26605093],
                [0.20708716, 0.38876095],
                [0.31018215, 0.47485098],
                [0.415882, 0.8048184],
                [0.7293086, 0.8216225],
            ]
        )
        encoder = utils.Encoder(default_scalings, ["robot"])
        dataset = RoboEireanDataWithEncoder(
            "data/train",
            encoder,
            ["robot"],
            image_transforms,
            bounding_box_transforms,
        )
        image, encoded_bounding_boxes, target_masks, encoded_target_classes = dataset[0]
        self.assertEqual(image.shape, (1, 60, 80))
        self.assertEqual(encoded_bounding_boxes.shape, (480, 4))
        self.assertEqual(target_masks.shape, (480,))
        self.assertEqual(encoded_target_classes.shape, (480,))


if __name__ == "__main__":
    unittest.main()

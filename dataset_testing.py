from datasets import RoboEireanData
import unittest
from torchvision import transforms as T
import torch


class TestRoboEireanData(unittest.TestCase):
    def test_getitem(self):
        image_transforms = T.Compose(
            [
                T.Grayscale(),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Resize((60, 80), antialias=True),
            ]
        )
        bounding_box_transforms = T.Compose([])
        dataset = RoboEireanData(
            "data/raw/train",
            ["ball", "robot", "goal_post", "penalty_spot"],
            image_transforms,
            bounding_box_transforms,
        )
        image, target_bounding_boxes, target_classes = dataset[0]
        self.assertEqual(image.shape, (1, 60, 80))
        self.assertEqual(target_bounding_boxes.shape, (10, 4))
        self.assertEqual(target_classes.shape, (10, 1))


if __name__ == "__main__":
    unittest.main()

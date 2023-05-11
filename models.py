import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

import utils


class DepthWise(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, use_stride: bool = False
    ) -> None:
        super().__init__()
        self.batch_normalization = nn.BatchNorm2d(in_channels)
        if use_stride:
            self.conv2d_1 = nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=2,
                groups=in_channels,
                bias=False,
                padding=1,
            )
        else:
            self.conv2d_1 = nn.Conv2d(
                in_channels,
                in_channels,
                3,
                groups=in_channels,
                bias=False,
                padding="same",
            )
        self.conv2d_2 = nn.Conv2d(in_channels, out_channels, 1, padding="same")
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_normalization(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        return x


class NormConv2dReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.batch_normalization = nn.BatchNorm2d(in_channels)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, 3, bias=False, padding="same"
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_normalization(x)
        x = self.conv2d(x)
        x = self.relu(x)
        return x


# TODO: handle hyperparameters differently
class MultiClassJetNet(pl.LightningModule):
    NUM_BOX_PARAMETERS = 4

    def __init__(self, num_classes, num_boxes, learning_rate: float) -> None:
        super().__init__()
        self.mean_average_precisions = []
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.accuracy = MulticlassAccuracy(num_classes=num_classes + 1, average=None)
        self.block_channels = [[24, 16, 16, 20], [20, 20, 20, 20, 24]]

        self.input_layer = NormConv2dReLU(1, 16)

        self.depth_wise_backbone = [DepthWise(16, 24, use_stride=True)]
        for block_channel in self.block_channels:
            for in_channels, out_channels in zip(
                block_channel[:-2], block_channel[1:-1]
            ):
                self.depth_wise_backbone.append(DepthWise(in_channels, out_channels))
            self.depth_wise_backbone.append(
                DepthWise(block_channel[-2], block_channel[-1], use_stride=True)
            )
        self.depth_wise_backbone = nn.Sequential(*self.depth_wise_backbone)

        self.classifier_channels = [24, 24, 24, 24]
        self.classifier = []
        for classifier_channel in self.classifier_channels:
            self.classifier.append(
                NormConv2dReLU(classifier_channel, classifier_channel)
            )
        self.classifier = nn.Sequential(*self.classifier)

        self.output_layer = nn.Conv2d(
            24,
            (self.num_classes + 1 + self.NUM_BOX_PARAMETERS) * self.num_boxes,
            1,
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.depth_wise_backbone(x)
        x = self.classifier(x)
        x = self.output_layer(x)
        return self._format_model_output(x)

    def _format_model_output(
        self, output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        output: (batch_size, num_anchors * (num_classes + 1 + 4), feature_map_height, feature_map_height)
        """
        batch_size, _, feature_map_height, feature_map_width = output.shape
        output = output.permute((0, 2, 3, 1))
        output = output.reshape(
            (
                -1,
                feature_map_height,
                feature_map_width,
                self.num_boxes,
                self.NUM_BOX_PARAMETERS + self.num_classes + 1,
            )
        )
        predicted_boxes = output[:, :, :, :, 0 : self.NUM_BOX_PARAMETERS].reshape(
            (batch_size, -1, self.NUM_BOX_PARAMETERS)
        )
        predicted_class_logits = output[:, :, :, :, self.NUM_BOX_PARAMETERS :].reshape(
            (-1, self.num_classes + 1)
        )
        return predicted_boxes, predicted_class_logits

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], _
    ):
        image, encoded_target_boxes, target_is_object, encoded_target_classes = batch
        encoded_predicted_boxes, predicted_class_logits = self(image)
        mined_classification_loss, location_loss = self.bounding_box_loss(
            encoded_target_boxes,
            target_is_object,
            encoded_target_classes,
            encoded_predicted_boxes,
            predicted_class_logits,
        )
        with torch.no_grad():
            accuracy = self.accuracy(
                predicted_class_logits, encoded_target_classes.flatten()
            )
        self.log("train/accuracy/no_object", accuracy[0])
        self.log(
            f"train/accuracy/robot",
            accuracy[1],
        )
        self.log("train/loss/classification", mined_classification_loss)
        self.log("train/loss/location", location_loss)
        return mined_classification_loss + location_loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], _
    ):
        image, encoded_target_boxes, target_is_object, encoded_target_classes = batch
        encoded_predicted_boxes, predicted_class_logits = self(image)
        mined_classification_loss, location_loss = self.bounding_box_loss(
            encoded_target_boxes,
            target_is_object,
            encoded_target_classes,
            encoded_predicted_boxes,
            predicted_class_logits,
        )
        with torch.no_grad():
            accuracy = self.accuracy(
                predicted_class_logits, encoded_target_classes.flatten()
            )
        self.log("val/accuracy/no_object", accuracy[0])
        self.log(
            f"val/accuracy/robot",
            accuracy[1],
        )
        self.log("val/loss/classification", mined_classification_loss)
        self.log("val/loss/location", location_loss)

    def bounding_box_loss(
        self,
        target_boxes: torch.Tensor,
        target_is_object: torch.Tensor,
        target_classes: torch.Tensor,
        predicted_boxes: torch.Tensor,
        predicted_class_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_predicted_boxes = predicted_boxes[target_is_object]
        selected_target_boxes = target_boxes[target_is_object]
        number_of_positive = selected_predicted_boxes.size(0)
        if number_of_positive > 0:
            location_loss = (
                F.smooth_l1_loss(
                    selected_predicted_boxes, selected_target_boxes, reduction="sum"
                )
                / number_of_positive
            )
        else:
            location_loss = 0.0

        classfication_loss = F.cross_entropy(
            predicted_class_logits,
            target_classes.flatten(),
            reduction="none",
        )
        positive_classification_loss = classfication_loss[target_is_object.flatten()]
        negative_classification_loss = classfication_loss[~target_is_object.flatten()]
        sorted_loss, _ = negative_classification_loss.sort(descending=True)
        # Hard negative mining
        number_of_negative = torch.clamp(
            torch.tensor(3 * number_of_positive), max=sorted_loss.size(0)
        )
        mined_classification_loss = (
            sorted_loss[:number_of_negative].sum() + positive_classification_loss.sum()
        ) / (number_of_negative + number_of_positive)
        return mined_classification_loss, location_loss

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

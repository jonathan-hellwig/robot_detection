import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from object_detection_data import Encoder


class MultiClassJetNet(pl.LightningModule):
    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        self.encoder = encoder
        assert encoder.feature_map_height == 8 and encoder.feature_map_width == 10
        NUM_BOX_PARAMETERS = 4

        self.accuracy = MulticlassAccuracy(
            num_classes=encoder.num_classes + 1, average=None
        )

        self.batch_normalization_1 = nn.BatchNorm2d(1)
        self.conv2d_1 = nn.Conv2d(1, 16, 3, padding="same")

        self.batch_normalization_2 = nn.BatchNorm2d(16)
        self.depthwise_conv2d_1 = nn.Conv2d(
            16, 16, 3, stride=2, groups=16, bias=False, padding=1
        )
        self.conv2d_2 = nn.Conv2d(16, 24, 1, padding="same")

        self.batch_normalization_3 = nn.BatchNorm2d(24)
        self.depthwise_conv2d_2 = nn.Conv2d(
            24, 24, 3, groups=24, bias=False, padding="same"
        )
        self.conv2d_3 = nn.Conv2d(24, 16, 1, padding="same")

        self.batch_normalization_4 = nn.BatchNorm2d(16)
        self.depthwise_conv2d_3 = nn.Conv2d(
            16, 16, 3, groups=16, bias=False, padding="same"
        )
        self.conv2d_4 = nn.Conv2d(16, 20, 1, padding="same")

        self.batch_normalization_5 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_4 = nn.Conv2d(
            20, 20, 3, stride=2, groups=20, bias=False, padding=1
        )
        self.conv2d_5 = nn.Conv2d(20, 20, 1, padding="same")

        self.batch_normalization_6 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_5 = nn.Conv2d(
            20, 20, 3, groups=20, bias=False, padding="same"
        )
        self.conv2d_6 = nn.Conv2d(20, 20, 1, padding="same")

        self.batch_normalization_7 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_6 = nn.Conv2d(
            20, 20, 3, groups=20, bias=False, padding="same"
        )
        self.conv2d_7 = nn.Conv2d(20, 20, 1, padding="same")

        self.batch_normalization_8 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_7 = nn.Conv2d(
            20, 20, 3, groups=20, bias=False, padding="same"
        )
        self.conv2d_8 = nn.Conv2d(20, 20, 1, padding="same")

        self.batch_normalization_9 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_8 = nn.Conv2d(
            20, 20, 3, stride=2, groups=20, bias=False, padding=1
        )
        self.conv2d_9 = nn.Conv2d(20, 24, 1, padding="same")

        self.batch_normalization_10 = nn.BatchNorm2d(24)
        self.conv2d_10 = nn.Conv2d(24, 24, 3, padding="same")

        self.batch_normalization_11 = nn.BatchNorm2d(24)
        self.conv2d_11 = nn.Conv2d(24, 24, 3, padding="same")

        self.batch_normalization_12 = nn.BatchNorm2d(24)
        self.conv2d_12 = nn.Conv2d(24, 24, 3, padding="same")

        self.batch_normalization_13 = nn.BatchNorm2d(24)
        self.conv2d_13 = nn.Conv2d(24, 24, 3, padding="same")
        self.conv2d_14 = nn.Conv2d(
            24,
            (self.encoder.num_classes + 1 + NUM_BOX_PARAMETERS)
            * self.encoder.default_scalings.size(0),
            1,
            padding="same",
        )

    def layers(self):
        return [
            self.batch_normalization_1,
            self.conv2d_1,
            self.batch_normalization_2,
            self.depthwise_conv2d_1,
            self.conv2d_2,
            self.batch_normalization_3,
            self.depthwise_conv2d_2,
            self.conv2d_3,
            self.batch_normalization_4,
            self.depthwise_conv2d_3,
            self.conv2d_4,
            self.batch_normalization_5,
            self.depthwise_conv2d_4,
            self.conv2d_5,
            self.batch_normalization_6,
            self.depthwise_conv2d_5,
            self.conv2d_6,
            self.batch_normalization_7,
            self.depthwise_conv2d_6,
            self.conv2d_7,
            self.batch_normalization_8,
            self.depthwise_conv2d_7,
            self.conv2d_8,
            self.batch_normalization_9,
            self.depthwise_conv2d_8,
            self.conv2d_9,
            self.batch_normalization_10,
            self.conv2d_10,
            self.batch_normalization_11,
            self.conv2d_11,
            self.batch_normalization_12,
            self.conv2d_12,
            self.batch_normalization_13,
            self.conv2d_13,
            self.conv2d_14,
        ]

    def forward(self, x):
        for layer in self.layers():
            x = layer(x)
        return self._format_model_output(x)

    def _forward(self, x):
        for layer in self.layers():
            x = layer(x)
        return self._decode_model_ouput(x)

    def _format_model_output(self, output):
        # TODO: Extract the feature map size
        batch_size = output.size(0)
        NUM_BOX_PARAMETERS = 4
        output = output.permute((0, 2, 3, 1))
        output = output.reshape(
            (
                -1,
                self.encoder.feature_map_height,
                self.encoder.feature_map_width,
                self.encoder.default_scalings.size(0),
                NUM_BOX_PARAMETERS + self.encoder.num_classes + 1,
            )
        )
        predicted_boxes = output[:, :, :, :, 0:NUM_BOX_PARAMETERS].reshape(
            (batch_size, -1, NUM_BOX_PARAMETERS)
        )
        predicted_class_logits = output[:, :, :, :, NUM_BOX_PARAMETERS:].reshape(
            (-1, self.encoder.num_classes + 1)
        )
        return predicted_boxes, predicted_class_logits

    def _decode_model_ouput(self, output):
        NUM_BOX_PARAMETERS = 4

        output = output.permute((0, 2, 3, 1))
        output = output.reshape(
            (
                -1,
                self.encoder.feature_map_height,
                self.encoder.feature_map_width,
                self.encoder.default_scalings.size(0),
                NUM_BOX_PARAMETERS + self.encoder.num_classes + 1,
            )
        )
        return (
            output[:, :, :, :, 0:NUM_BOX_PARAMETERS],
            output[:, :, :, :, NUM_BOX_PARAMETERS:],
        )

    def training_step(self, batch, batch_idx):
        image, target_boxes, target_box_idx, target_classes = batch
        predicted_boxes, object_class_logits = self(image)
        selected_predicted_boxes = predicted_boxes[target_box_idx]
        selected_target_boxes = target_boxes[target_box_idx]
        # TODO: Check wether the permute operation gets handled correctly
        # TODO: Handle the case when there is no box!
        location_loss = F.smooth_l1_loss(
            selected_predicted_boxes, selected_target_boxes
        )
        classification_loss = F.cross_entropy(
            object_class_logits,
            target_classes.flatten(),
        )
        loss = location_loss + classification_loss
        acc = self.accuracy(object_class_logits, target_classes.flatten())
        self.log("train_accuracy/no_box", acc[0])
        self.log("train_accuracy/robot", acc[1])
        self.log("train_accuracy/ball", acc[2])
        self.log("train_accuracy/penalty", acc[3])
        self.log("train_accuracy/goal_post", acc[4])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target_boxes, target_box_idx, target_classes = batch
        predicted_boxes, object_class_logits = self(image)
        selected_predicted_boxes = predicted_boxes[target_box_idx]
        selected_target_boxes = target_boxes[target_box_idx]
        # TODO: Check wether the permute operation gets handled correctly
        # TODO: Handle the case when there is no box!
        location_loss = F.smooth_l1_loss(
            selected_predicted_boxes, selected_target_boxes
        )
        classification_loss = F.cross_entropy(
            object_class_logits, target_classes.flatten()
        )
        loss = location_loss + classification_loss
        acc = self.accuracy(object_class_logits, target_classes.flatten())
        self.log("val_accuracy/no_box", acc[0])
        self.log("val_accuracy/robot", acc[1])
        self.log("val_accuracy/ball", acc[2])
        self.log("val_accuracy/penalty", acc[3])
        self.log("val_accuracy/goal_post", acc[4])
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

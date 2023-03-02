import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from object_detection_data import Encoder


class DepthWise(nn.Module):
    def __init__(self, in_channels, out_channels, use_stride=False) -> None:
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

    def forward(self, x):
        x = self.batch_normalization(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        return x


class NormConv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.batch_normalization = nn.BatchNorm2d(in_channels)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, 3, bias=False, padding="same"
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batch_normalization(x)
        x = self.conv2d(x)
        x = self.relu(x)
        return x


class MultiClassJetNet(pl.LightningModule):
    def __init__(self, encoder: Encoder, learning_rate: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        assert encoder.feature_map_height == 8 and encoder.feature_map_width == 10
        NUM_BOX_PARAMETERS = 4

        self.accuracy = MulticlassAccuracy(
            num_classes=encoder.num_classes + 1, average=None
        )
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
            (self.encoder.num_classes + 1 + NUM_BOX_PARAMETERS)
            * self.encoder.default_scalings.size(0),
            1,
            padding="same",
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.depth_wise_backbone(x)
        x = self.classifier(x)
        x = self.output_layer(x)
        return self._format_model_output(x)

    def _format_model_output(self, output):
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
        image, target_boxes, target_masks, target_classes, _ = batch
        predicted_boxes, predicted_class_logits = self(image)
        mined_classification_loss, location_loss = self.loss(
            target_boxes,
            target_masks,
            target_classes,
            predicted_boxes,
            predicted_class_logits,
        )
        acc = self.accuracy(predicted_class_logits, target_classes.flatten())
        if self.encoder.num_classes == 4:
            self.log("train_accuracy/no_box", acc[0])
            self.log("train_accuracy/robot", acc[1])
            self.log("train_accuracy/ball", acc[2])
            self.log("train_accuracy/penalty", acc[3])
            self.log("train_accuracy/goal_post", acc[4])
        elif self.encoder.num_classes == 1:
            self.log("train_accuracy/no_object", acc[0])
            self.log("train_accuracy/object", acc[1])
        self.log("train_loss/classification", mined_classification_loss)
        self.log("train_loss/location", location_loss)
        return mined_classification_loss + location_loss

    def loss(
        self,
        target_boxes,
        target_masks,
        target_classes,
        predicted_boxes,
        predicted_class_logits,
    ):
        selected_predicted_boxes = predicted_boxes[target_masks]
        selected_target_boxes = target_boxes[target_masks]
        # TODO: Check whether the permute operation gets handled correctly
        # TODO: Handle the case when there is no box!
        batch_size = predicted_boxes.size(0)
        location_loss = (
            F.smooth_l1_loss(
                selected_predicted_boxes, selected_target_boxes, reduction="sum"
            )
            / batch_size
        )
        number_of_positive = target_masks.sum()
        classfication_loss = F.cross_entropy(
            predicted_class_logits,
            target_classes.flatten(),
            reduction="none",
        )
        positive_classification_loss = classfication_loss[target_masks.flatten()]
        negative_classification_loss = classfication_loss[~target_masks.flatten()]
        sorted_loss, _ = negative_classification_loss.sort(descending=True)
        number_of_negative = torch.clamp(
            3 * number_of_positive, max=sorted_loss.size(0)
        )
        mined_classification_loss = (
            sorted_loss[:number_of_negative].sum() + positive_classification_loss.sum()
        ) / batch_size
        # mined_classification_loss = positive_classification_loss.sum() / batch_size
        total_loss = mined_classification_loss + location_loss
        return mined_classification_loss, location_loss

    def validation_step(self, batch, batch_idx):
        image, target_boxes, target_masks, target_classes, _ = batch
        predicted_boxes, predicted_class_logits = self(image)
        mined_classification_loss, location_loss = self.loss(
            target_boxes,
            target_masks,
            target_classes,
            predicted_boxes,
            predicted_class_logits,
        )
        acc = self.accuracy(predicted_class_logits, target_classes.flatten())
        self.log("val_accuracy/no_box", acc[0])
        self.log("val_accuracy/robot", acc[1])
        self.log("val_accuracy/ball", acc[2])
        self.log("val_accuracy/penalty", acc[3])
        self.log("val_accuracy/goal_post", acc[4])
        self.log("train_loss/classification", mined_classification_loss)
        self.log("train_loss/location", location_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

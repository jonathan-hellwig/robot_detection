import torch
from torch import nn
import torch.nn.functional as F

import lightning
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import utils


class ObjectDetectionTask(lightning.LightningModule):
    def __init__(
        self,
        model,
        loss,
        encoder: utils.Encoder,
        learning_rate: float,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.mean_average_precision = MeanAveragePrecision()

    def _shared_eval_step(self, batch):
        images, encoded_target_bounding_boxes, encoded_target_classes = batch
        (
            encoded_predicted_bounding_boxes,
            encoded_predicted_class_logits,
        ) = self.model.forward(images)

        total_loss, location_loss, classification_loss = self.loss(
            encoded_target_bounding_boxes,
            encoded_target_classes,
            encoded_predicted_bounding_boxes,
            encoded_predicted_class_logits,
        )

        predicted_bounding_boxes = self.encoder.decode(encoded_predicted_bounding_boxes)
        target_bounding_boxes = self.encoder.decode(encoded_target_bounding_boxes)

        _, _, height, width = images.shape
        target_bounding_boxes = utils.convert_to_absolute_coordinates(
            target_bounding_boxes, (height, width)
        )
        predicted_bounding_boxes = utils.convert_to_absolute_coordinates(
            predicted_bounding_boxes, (height, width)
        )

        target_is_object = encoded_target_classes > 0
        target_bounding_boxes = target_bounding_boxes[target_is_object]
        target_classes = encoded_target_classes[target_is_object]

        _, _, num_classes = encoded_predicted_class_logits.shape
        predicted_class_scores = F.softmax(encoded_predicted_class_logits, dim=2)
        predicted_class = predicted_class_scores.argmax(dim=2)

        predictions = [
            dict(
                boxes=predicted_bounding_boxes.reshape((-1, 4)),
                scores=predicted_class_scores.reshape((-1, num_classes))[:, 1],
                labels=predicted_class.flatten(),
            )
        ]
        targets = [
            dict(
                boxes=target_bounding_boxes,
                labels=target_classes,
            )
        ]
        self.mean_average_precision(
            predictions,
            targets,
        )
        return (
            total_loss,
            location_loss,
            classification_loss,
        )

    def training_step(self, batch, batch_idx):
        total_loss, location_loss, classification_loss = self._shared_eval_step(batch)
        self.log_dict(
            {
                "train/total_loss": total_loss,
                "train/location_loss": location_loss,
                "train/classification_loss": classification_loss,
            },
            on_step=True,
            prog_bar=True,
        )
        return total_loss

    def on_train_end(self):
        self.log(
            "train/mean_average_precision", self.mean_average_precision.compute()["map"]
        )

    def validation_step(self, batch, batch_idx):
        (
            total_loss,
            location_loss,
            classification_loss,
        ) = self._shared_eval_step(batch)
        self.log_dict(
            {
                "validation/total_loss": total_loss,
                "validation/location_loss": location_loss,
                "validation/classification_loss": classification_loss,
            }
        )

        return total_loss

    def on_validation_epoch_end(self):
        self.log(
            "validation/mean_average_precision",
            self.mean_average_precision.compute()["map"],
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class SingleShotDetectorLoss(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        target_bounding_boxes: torch.Tensor,
        target_classes: torch.Tensor,
        predicted_bounding_boxes: torch.Tensor,
        predicted_class_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
        - target_bounding_boxes: (batch_size, num_target_boxes, 4)
        - target_classes: (batch_size, num_target_boxes)
        - predicted_bounding_boxes: (batch_size, num_target_boxes, 4)
        - predicted_class_logits: (batch_size, num_target_boxes, num_classes + 1)
        """
        assert target_bounding_boxes.dim() == 3
        assert target_classes.dim() == 2
        assert predicted_bounding_boxes.dim() == 3
        assert predicted_class_logits.dim() == 3
        assert target_bounding_boxes.size(0) == target_classes.size(0)
        assert target_bounding_boxes.size(1) == predicted_bounding_boxes.size(1)
        assert target_bounding_boxes.size(1) == predicted_class_logits.size(1)
        assert predicted_class_logits.size(2) >= target_classes.max() + 1

        target_is_object = target_classes > 0
        # TODO: Check this. Sum over the whole batch instead of each image
        number_of_positive = target_is_object.sum()
        number_of_positive = number_of_positive.clamp(min=1)
        number_of_negative = number_of_positive * 3

        location_loss = (
            F.smooth_l1_loss(
                predicted_bounding_boxes[target_is_object],
                target_bounding_boxes[target_is_object],
                reduction="sum",
            )
            / number_of_positive.sum()
        )

        batch_size, num_target_boxes, num_classes = predicted_class_logits.shape
        classification_loss = F.cross_entropy(
            predicted_class_logits.reshape(
                (batch_size * num_target_boxes, num_classes)
            ),
            target_classes.flatten(),
            reduction="none",
        )
        positive_classification_loss = classification_loss[target_is_object.flatten()]
        negative_classification_loss = classification_loss[~target_is_object.flatten()]
        sorted_loss, _ = negative_classification_loss.sort(descending=True)
        # Hard negative mining
        number_of_negative = torch.clamp(number_of_negative, max=sorted_loss.size(0))
        mined_classification_loss = (
            sorted_loss[:number_of_negative].sum() + positive_classification_loss.sum()
        ) / (number_of_negative + number_of_positive.sum())

        total_loss = location_loss + self.alpha * mined_classification_loss
        return total_loss, location_loss, mined_classification_loss


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


class JetNet(lightning.LightningModule):
    NUM_BOX_PARAMETERS = 4

    def __init__(
        self,
        num_classes: int,
        num_default_scalings: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_default_boxes = num_default_scalings

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
            (self.num_classes + 1 + self.NUM_BOX_PARAMETERS) * self.num_default_boxes,
            1,
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.depth_wise_backbone(x)
        x = self.classifier(x)
        x = self.output_layer(x)
        return self._process_model_output(x)

    def _process_model_output(
        self, model_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process model output to get predicted bounding boxes and predicted class logits

        Parameters:
            model_output: (batch_size, num_default_boxes * (num_classes + 1 + 4), feature_map_height, feature_map_height)
        """
        assert model_output.dim() == 4
        batch_size, _, feature_map_height, feature_map_width = model_output.shape
        model_output = model_output.permute((0, 2, 3, 1))
        model_output = model_output.reshape(
            (
                -1,
                feature_map_height,
                feature_map_width,
                self.num_default_boxes,
                self.NUM_BOX_PARAMETERS + self.num_classes + 1,
            )
        )
        predicted_bounding_boxes = model_output[
            ..., 0 : self.NUM_BOX_PARAMETERS
        ].reshape((batch_size, -1, self.NUM_BOX_PARAMETERS))
        predicted_class_logits = model_output[..., self.NUM_BOX_PARAMETERS :].reshape(
            (batch_size, -1, self.num_classes + 1)
        )
        return predicted_bounding_boxes, predicted_class_logits

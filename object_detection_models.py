import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
import pytorch_lightning as pl


class JetNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_normalization_1 = nn.BatchNorm2d(1)
        self.conv2d_1 = nn.Conv2d(1, 16, 3, padding='same')

        self.batch_normalization_2 = nn.BatchNorm2d(16)
        self.depthwise_conv2d_1 = nn.Conv2d(
            16, 16, 3, stride=2, groups=16, bias=False, padding=1)
        self.conv2d_2 = nn.Conv2d(16, 24, 1, padding='same')

        self.batch_normalization_3 = nn.BatchNorm2d(24)
        self.depthwise_conv2d_2 = nn.Conv2d(
            24, 24, 3, groups=24, bias=False, padding='same')
        self.conv2d_3 = nn.Conv2d(24, 16, 1, padding='same')

        self.batch_normalization_4 = nn.BatchNorm2d(16)
        self.depthwise_conv2d_3 = nn.Conv2d(
            16, 16, 3, groups=16, bias=False, padding='same')
        self.conv2d_4 = nn.Conv2d(16, 20, 1, padding='same')

        self.batch_normalization_5 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_4 = nn.Conv2d(
            20, 20, 3, stride=2, groups=20, bias=False, padding=1)
        self.conv2d_5 = nn.Conv2d(20, 20, 1, padding='same')

        self.batch_normalization_6 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_5 = nn.Conv2d(
            20, 20, 3, groups=20, bias=False, padding='same')
        self.conv2d_6 = nn.Conv2d(20, 20, 1, padding='same')

        self.batch_normalization_7 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_6 = nn.Conv2d(
            20, 20, 3, groups=20, bias=False, padding='same')
        self.conv2d_7 = nn.Conv2d(20, 20, 1, padding='same')

        self.batch_normalization_8 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_7 = nn.Conv2d(
            20, 20, 3, groups=20, bias=False, padding='same')
        self.conv2d_8 = nn.Conv2d(20, 20, 1, padding='same')

        self.batch_normalization_9 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_8 = nn.Conv2d(
            20, 20, 3, stride=2, groups=20, bias=False,  padding=1)
        self.conv2d_9 = nn.Conv2d(20, 24, 1, padding='same')

        self.batch_normalization_10 = nn.BatchNorm2d(24)
        self.conv2d_10 = nn.Conv2d(24, 24, 3, padding='same')

        self.batch_normalization_11 = nn.BatchNorm2d(24)
        self.conv2d_11 = nn.Conv2d(24, 24, 3, padding='same')

        self.batch_normalization_12 = nn.BatchNorm2d(24)
        self.conv2d_12 = nn.Conv2d(24, 24, 3, padding='same')

        self.batch_normalization_13 = nn.BatchNorm2d(24)
        self.conv2d_13 = nn.Conv2d(24, 24, 3, padding='same')
        self.conv2d_14 = nn.Conv2d(24, 24, 1, padding='same')

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
            self.conv2d_14
        ]

    def forward(self, x):
        x = self.batch_normalization_1(x)
        x = self.conv2d_1(x)

        x = self.batch_normalization_2(x)
        x = self.depthwise_conv2d_1(x)
        x = self.conv2d_2(x)

        x = self.batch_normalization_3(x)
        x = self.depthwise_conv2d_2(x)
        x = self.conv2d_3(x)

        x = self.batch_normalization_4(x)
        x = self.depthwise_conv2d_3(x)
        x = self.conv2d_4(x)

        x = self.batch_normalization_5(x)
        x = self.depthwise_conv2d_4(x)
        x = self.conv2d_5(x)

        x = self.batch_normalization_6(x)
        x = self.depthwise_conv2d_5(x)
        x = self.conv2d_6(x)

        x = self.batch_normalization_7(x)
        x = self.depthwise_conv2d_6(x)
        x = self.conv2d_7(x)

        x = self.batch_normalization_8(x)
        x = self.depthwise_conv2d_7(x)
        x = self.conv2d_8(x)

        x = self.batch_normalization_9(x)
        x = self.depthwise_conv2d_8(x)
        x = self.conv2d_9(x)

        x = self.batch_normalization_10(x)
        x = self.conv2d_10(x)

        x = self.batch_normalization_11(x)
        x = self.conv2d_11(x)

        x = self.batch_normalization_12(x)
        x = self.conv2d_12(x)

        x = self.batch_normalization_13(x)
        x = self.conv2d_13(x)
        x = self.conv2d_14(x)
        return x

    @classmethod
    def from_hdf5(cls, hdf5_path):
        tf_model = tf.keras.models.load_model(hdf5_path)
        torch_model = JetNet()

        for torch_layer, tf_layer in zip(torch_model.layers(), tf_model.layers[1:]):
            for torch_p, tf_p in zip(torch_layer.parameters(), tf_layer.trainable_weights):
                if torch_p.dim() != 1:
                    print(torch_p.shape)
                    print(tf_p.shape)
                    torch_p = torch.from_numpy(
                        tf_p.numpy()).permute((2, 3, 1, 0))
                else:
                    torch_p = torch.from_numpy(tf_p.numpy())
        return torch_model


class MultiClassJetNet(nn.Module):
    def __init__(self, num_classes, num_scalings) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_scalings = num_scalings
        NUM_BOX_PARAMETERS = 4

        self.batch_normalization_1 = nn.BatchNorm2d(1)
        self.conv2d_1 = nn.Conv2d(1, 16, 3, padding='same')

        self.batch_normalization_2 = nn.BatchNorm2d(16)
        self.depthwise_conv2d_1 = nn.Conv2d(
            16, 16, 3, stride=2, groups=16, bias=False, padding=1)
        self.conv2d_2 = nn.Conv2d(16, 24, 1, padding='same')

        self.batch_normalization_3 = nn.BatchNorm2d(24)
        self.depthwise_conv2d_2 = nn.Conv2d(
            24, 24, 3, groups=24, bias=False, padding='same')
        self.conv2d_3 = nn.Conv2d(24, 16, 1, padding='same')

        self.batch_normalization_4 = nn.BatchNorm2d(16)
        self.depthwise_conv2d_3 = nn.Conv2d(
            16, 16, 3, groups=16, bias=False, padding='same')
        self.conv2d_4 = nn.Conv2d(16, 20, 1, padding='same')

        self.batch_normalization_5 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_4 = nn.Conv2d(
            20, 20, 3, stride=2, groups=20, bias=False, padding=1)
        self.conv2d_5 = nn.Conv2d(20, 20, 1, padding='same')

        self.batch_normalization_6 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_5 = nn.Conv2d(
            20, 20, 3, groups=20, bias=False, padding='same')
        self.conv2d_6 = nn.Conv2d(20, 20, 1, padding='same')

        self.batch_normalization_7 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_6 = nn.Conv2d(
            20, 20, 3, groups=20, bias=False, padding='same')
        self.conv2d_7 = nn.Conv2d(20, 20, 1, padding='same')

        self.batch_normalization_8 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_7 = nn.Conv2d(
            20, 20, 3, groups=20, bias=False, padding='same')
        self.conv2d_8 = nn.Conv2d(20, 20, 1, padding='same')

        self.batch_normalization_9 = nn.BatchNorm2d(20)
        self.depthwise_conv2d_8 = nn.Conv2d(
            20, 20, 3, stride=2, groups=20, bias=False,  padding=1)
        self.conv2d_9 = nn.Conv2d(20, 24, 1, padding='same')

        self.batch_normalization_10 = nn.BatchNorm2d(24)
        self.conv2d_10 = nn.Conv2d(24, 24, 3, padding='same')

        self.batch_normalization_11 = nn.BatchNorm2d(24)
        self.conv2d_11 = nn.Conv2d(24, 24, 3, padding='same')

        self.batch_normalization_12 = nn.BatchNorm2d(24)
        self.conv2d_12 = nn.Conv2d(24, 24, 3, padding='same')

        self.batch_normalization_13 = nn.BatchNorm2d(24)
        self.conv2d_13 = nn.Conv2d(24, 24, 3, padding='same')
        self.conv2d_14 = nn.Conv2d(
            24, (self.num_classes + 1 + NUM_BOX_PARAMETERS) * self.num_scalings, 1, padding='same')

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
            self.conv2d_14
        ]

    def forward(self, x):
        for layer in self.layers():
            x = layer(x)
        return self._format_model_output(x)

    def _format_model_output(self, output):
        FEATURE_MAP_HEIGHT = 8
        FEATURE_MAP_WIDTH = 10
        NUM_BOX_PARAMETERS = 4
        reshaped_ouput = output.reshape(
            (-1, self.num_scalings, NUM_BOX_PARAMETERS + self.num_classes + 1, FEATURE_MAP_HEIGHT, FEATURE_MAP_WIDTH))
        predicted_boxes = reshaped_ouput[:, :,
                                         0:NUM_BOX_PARAMETERS, :, :].permute((0, 3, 4, 1, 2))
        object_class_logits = reshaped_ouput[:, :,
                                             NUM_BOX_PARAMETERS:, :, :].permute((0, 3, 4, 1, 2)).reshape((-1, self.num_classes + 1))
        return predicted_boxes, object_class_logits


class LightningMultiClassJetNet(pl.LightningModule):
    def __init__(self, num_classes, num_scalings) -> None:
        super().__init__()
        self.model = MultiClassJetNet(num_classes, num_scalings)

    def training_step(self, batch, batch_idx):
        image, target_boxes, target_mask, target_classes = batch
        selected_target_boxes = target_boxes[target_mask]
        predicted_boxes, object_class_logits = self.model(image)
        selected_predicted_boxes = predicted_boxes[target_mask]
        # TODO: Check wether the permute operation gets handled correctly by autodiff
        # TODO: Handle the case when there is no box!
        location_loss = F.smooth_l1_loss(
            selected_predicted_boxes, selected_target_boxes)
        classification_loss = F.cross_entropy(
            object_class_logits, target_classes.flatten())
        loss = location_loss + classification_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target_boxes, target_mask, target_classes = batch
        selected_target_boxes = target_boxes[target_mask]
        predicted_boxes, object_class_logits = self.model(image)
        selected_predicted_boxes = predicted_boxes[target_mask]
        # TODO: Check wether the permute operation gets handled correctly by autodiff
        # TODO: Handle the case when there is no box!
        location_loss = F.smooth_l1_loss(
            selected_predicted_boxes, selected_target_boxes)
        classification_loss = F.cross_entropy(
            object_class_logits, target_classes.flatten())
        loss = location_loss + classification_loss
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    hdf5_path = '../players_deeptector.h5'
    model = JetNet.from_hdf5(hdf5_path)
    torch.save(model, 'torch_weights.pt')

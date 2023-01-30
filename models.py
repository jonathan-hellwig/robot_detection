import torch
import torch.nn as nn
import tensorflow as tf


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
            # print(torch_layer.parameters())
            # print(tf_layer.trainable_weights)
            for torch_p, tf_p in zip(torch_layer.parameters(), tf_layer.trainable_weights):
                if torch_p.dim() != 1:
                    print(torch_p.shape)
                    print(tf_p.shape)
                    torch_p = torch.from_numpy(
                        tf_p.numpy()).permute((2, 3, 1, 0))
                else:
                    torch_p = torch.from_numpy(tf_p.numpy())
        return torch_model


if __name__ == "__main__":
    hdf5_path = '../players_deeptector.h5'
    model = JetNet.from_hdf5(hdf5_path)
    torch.save(model, 'torch_weights.pt')

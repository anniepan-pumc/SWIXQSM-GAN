"""SWIXQSM-GAN generator.

The project uses a compact 2D U-Net to translate one SWI slice into one
synthetic QSM slice. Inputs and outputs are expected to have shape
``(batch, 1, height, width)``.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions used in the generator encoder/decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

    def forward(self, x):
        return self.layers(x)


class UpSampleConcat(nn.Module):
    """Nearest-neighbor upsampling followed by skip-connection concatenation."""

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, skip):
        return torch.cat([self.up(x), skip], dim=1)


class UNetSmall(nn.Module):
    """Compact U-Net generator for SWI-to-QSM slice translation."""

    def __init__(self, num_channels=1, num_classes=1, normalize=True):
        super().__init__()
        features = [32, 64, 128, 256]

        self.down1 = ConvBlock(num_channels, features[0])
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(features[0]),
            ConvBlock(features[0], features[1]),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(features[1]),
            ConvBlock(features[1], features[2]),
        )
        self.bottom = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(features[2]),
            ConvBlock(features[2], features[3]),
            nn.BatchNorm2d(features[3]),
        )

        self.up1 = UpSampleConcat()
        self.upconv1 = nn.Sequential(
            ConvBlock(features[3] + features[2], features[2]),
            nn.BatchNorm2d(features[2]),
        )
        self.up2 = UpSampleConcat()
        self.upconv2 = nn.Sequential(
            ConvBlock(features[2] + features[1], features[1]),
            nn.BatchNorm2d(features[1]),
        )
        self.up3 = UpSampleConcat()
        self.upconv3 = nn.Sequential(
            ConvBlock(features[1] + features[0], features[0]),
            nn.BatchNorm2d(features[0]),
        )

        final_layers = [nn.Conv2d(features[0], num_classes, kernel_size=1)]
        if normalize:
            final_layers.append(nn.Sigmoid())
        self.final = nn.Sequential(*final_layers)

    def forward(self, x, return_features=False):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        bottom = self.bottom(down3)

        up1 = self.upconv1(self.up1(bottom, down3))
        up2 = self.upconv2(self.up2(up1, down2))
        up3 = self.upconv3(self.up3(up2, down1))

        if return_features:
            return up3
        return self.final(up3)

    def load_checkpoint(self, path, map_location="cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=False))
        return self


def build_generator(num_channels=1, num_classes=1, normalize=True):
    """Create the default SWIXQSM-GAN generator."""

    return UNetSmall(num_channels=num_channels, num_classes=num_classes, normalize=normalize)


Generator = UNetSmall

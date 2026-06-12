"""SWIXQSM-GAN discriminator."""

import torch.nn as nn


class Discriminator_ccgan(nn.Module):
    """PatchGAN-style discriminator for 2D QSM slices."""

    def __init__(self, input_shape=(1, 256, 256)):
        super().__init__()
        channels, height, width = input_shape
        if height != 256 or width != 256:
            raise ValueError("Discriminator_ccgan currently expects 256 x 256 inputs.")

        def block(in_channels, out_channels, stride, normalize):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_channels = channels
        for out_channels, stride, normalize in [
            (64, 2, False),
            (128, 2, True),
            (256, 2, True),
            (512, 1, True),
        ]:
            layers.extend(block(in_channels, out_channels, stride, normalize))
            in_channels = out_channels

        layers.extend(
            [
                nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_discriminator(input_shape=(1, 256, 256)):
    """Create the default SWIXQSM-GAN discriminator."""

    return Discriminator_ccgan(input_shape=input_shape)


Discriminator = Discriminator_ccgan

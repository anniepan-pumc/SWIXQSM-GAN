import torch
from torch import nn

class PatchDiscriminator(nn.Module):
    """New conditional discriminator; legacy discriminator weights are not loaded."""
    def __init__(self):
        super().__init__()
        layers, channels = [], 2
        for out in (32, 64, 128):
            layers += [nn.Conv2d(channels, out, 4, 2, 1), nn.LeakyReLU(0.2)]
            channels = out
        self.net = nn.Sequential(*layers, nn.Conv2d(128, 1, 3, padding=1))

    def forward(self, swi, qsm):
        return self.net(torch.cat((swi, qsm), dim=1))


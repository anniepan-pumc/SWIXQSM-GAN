"""Explicit architectures for the four prospective retraining comparisons.

These models do not change the legacy generator or load historical weights.
Every generator has an unrestricted linear QSM output.  The factory metadata
is JSON serializable and is intended to be saved with the training config.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .generator import Conv3x3Small, UNetSmall


MODEL_NAMES = ("swixqsm_gan", "fcn_gan_l1", "plain_unet", "pix2pix")
PIX2PIX_IMPLEMENTATION_URL = (
    "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py"
)
PIX2PIX_PAPER_URL = "https://arxiv.org/abs/1611.07004"


def _check_image(image, name):
    if (not isinstance(image, torch.Tensor) or image.ndim != 4
            or image.shape[1] != 1 or image.shape[0] < 1
            or min(image.shape[-2:]) < 32 or not image.is_floating_point()):
        raise ValueError(f"{name} must be a floating [N,1,H,W] tensor, H,W >= 32")
    if any(extent % 8 for extent in image.shape[-2:]):
        raise ValueError(f"{name} height and width must be multiples of 8")


def _check_pair(swi, qsm):
    _check_image(swi, "SWI")
    _check_image(qsm, "QSM")
    if swi.shape != qsm.shape or swi.device != qsm.device or swi.dtype != qsm.dtype:
        raise ValueError("SWI and QSM must have identical shape, dtype, and device")


class FCNGenerator(nn.Module):
    """Four-resolution encoder/decoder with no encoder-to-decoder skips.

    The channel widths, convolution blocks, ELU, dropout, pooling, and batch
    normalization follow UNetSmall; decoder concatenations are removed.
    Nearest-neighbour upsampling introduces no learned transposed convolution.
    """

    def __init__(self):
        super().__init__()
        self.down1 = Conv3x3Small(1, 32)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.BatchNorm2d(32),
                                   Conv3x3Small(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), nn.BatchNorm2d(64),
                                   Conv3x3Small(64, 128))
        self.bottom = nn.Sequential(nn.MaxPool2d(2), nn.BatchNorm2d(128),
                                    Conv3x3Small(128, 256), nn.BatchNorm2d(256))
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv3x3Small(256, 128), nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv3x3Small(128, 64), nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv3x3Small(64, 32), nn.BatchNorm2d(32),
        )
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, image):
        _check_image(image, "SWI")
        return self.final(self.decoder(self.bottom(self.down3(self.down2(self.down1(image))))))


class GlobalQSMDiscriminator(nn.Module):
    """QSM-only discriminator producing one unrestricted logit per slice.

    It deliberately repairs the historical two-output discriminator head for
    the prospective experiment.  It is not a historical-weight-compatible
    reproduction.  Its 3x3 convolution/InstanceNorm backbone and spatially
    specific flattened classifier retain the original design. InstanceNorm
    uses all spatial positions within each slice, without cross-slice batch
    statistics. The SWI argument is not a discriminator feature.
    """

    kind = "global_qsm_only"

    def __init__(self, image_size=256):
        super().__init__()
        self.image_size = _image_size(image_size)
        layers, channels = [], 1
        for index, (width, stride) in enumerate(zip((64, 128, 256, 512), (2, 2, 2, 1))):
            layers.append(nn.Conv2d(channels, width, 3, stride=stride, padding=1))
            if index > 0:
                layers.append(nn.InstanceNorm2d(width, affine=False, track_running_stats=False))
            layers.append(nn.LeakyReLU(0.2))
            channels = width
        self.features = nn.Sequential(*layers)
        self.spatial_logits = nn.Sequential(nn.Conv2d(512, 1, 3, padding=1), nn.LeakyReLU(0.2))
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear((self.image_size // 8) ** 2, 1)

    def forward(self, swi, qsm):
        _check_pair(swi, qsm)
        if tuple(qsm.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(f"Global discriminator requires configured square image_size={self.image_size}")
        values = self.spatial_logits(self.features(qsm))
        return self.classifier(self.flatten(values))


def _pix_down(in_channels, out_channels, first=False, innermost=False):
    layers = [] if first else [nn.LeakyReLU(0.2)]
    layers.append(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False))
    if not first and not innermost:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def _pix_up(in_channels, out_channels, final=False, dropout=False):
    layers = [nn.ReLU(), nn.ConvTranspose2d(in_channels, out_channels, 4,
                                           stride=2, padding=1, bias=final)]
    if not final:
        layers.append(nn.BatchNorm2d(out_channels))
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class Pix2PixQSMGenerator(nn.Module):
    """Eight-downsample, base-64 pix2pix U-Net adapted for signed QSM.

    This is an independent, explicit encoder/decoder implementation of the
    official unet_256 channel and skip topology.  The output Tanh is replaced
    by a linear head.  Right/bottom zero padding, never interpolation, permits
    original MRI sizes such as 416; the output is cropped to the original grid.
    """

    def __init__(self):
        super().__init__()
        encoder_widths = (1, 64, 128, 256, 512, 512, 512, 512, 512)
        self.encoders = nn.ModuleList(
            _pix_down(encoder_widths[i], encoder_widths[i + 1],
                      first=i == 0, innermost=i == 7)
            for i in range(8)
        )
        # The innermost up block has no dropout, as in the official model.
        # The following three 512-channel up blocks use dropout = 0.5.
        decoder_inputs = (512, 1024, 1024, 1024, 1024, 512, 256, 128)
        decoder_outputs = (512, 512, 512, 512, 256, 128, 64, 1)
        self.decoders = nn.ModuleList(
            _pix_up(decoder_inputs[i], decoder_outputs[i], final=i == 7,
                    dropout=i in (1, 2, 3))
            for i in range(8)
        )

    def forward(self, image):
        _check_image(image, "SWI")
        height, width = image.shape[-2:]
        padded = F.pad(image, (0, (-width) % 256, 0, (-height) % 256), value=0.)
        encoded = []
        value = padded
        for down in self.encoders:
            value = down(value)
            encoded.append(value)
        for index, up in enumerate(self.decoders):
            value = up(value)
            if index < 7:
                value = torch.cat((encoded[6 - index], value), dim=1)
        return value[..., :height, :width]


class Pix2Pix70Discriminator(nn.Module):
    """Conditional base-64 PatchGAN with nominal convolutional RF 70 x 70.

    BatchNorm matches the official baseline.  In training mode its statistics
    couple spatial locations and batch elements: 70 x 70 is the convolutional
    receptive field, not a claim of strict local dependence through BatchNorm.
    This model cannot use the legacy normalization-free patch-support helper.
    """

    kind = "conditional_patch70"

    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(2, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2)]
        for in_channels, out_channels, stride in ((64, 128, 2), (128, 256, 2),
                                                  (256, 512, 1)):
            layers.extend((nn.Conv2d(in_channels, out_channels, 4, stride=stride,
                                      padding=1, bias=False),
                           nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2)))
        layers.append(nn.Conv2d(512, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, swi, qsm):
        _check_pair(swi, qsm)
        return self.net(torch.cat((swi, qsm), dim=1))


def _count_parameters(module):
    if module is None:
        return {"registered": 0, "trainable": 0, "known_inactive_registered": 0,
                "active_trainable": 0}
    inactive = sum(parameter.numel() for name, parameter in module.named_parameters()
                   if isinstance(module, UNetSmall) and ".deconv." in name)
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {"registered": total, "trainable": trainable,
            "known_inactive_registered": inactive, "active_trainable": trainable - inactive}


def _image_size(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 32 or value % 8:
        raise ValueError("image_size must be an integer >= 32 and a multiple of 8")
    return value


def create_model(model_name, image_size=256):
    """Return ``(generator, discriminator_or_None, metadata)`` on CPU.

    Widths are fixed, including for tests; the public training interface cannot
    silently replace a baseline with a smaller smoke-test architecture.
    No historical checkpoint is loaded.  Initialization uses PyTorch defaults;
    callers may apply and record an explicit common initialization policy.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {', '.join(MODEL_NAMES)}")
    image_size = _image_size(image_size)
    if model_name in ("swixqsm_gan", "plain_unet"):
        generator = UNetSmall(normalize=False)
        discriminator = GlobalQSMDiscriminator(image_size) if model_name == "swixqsm_gan" else None
        generator_spec = {
            "class": "UNetSmall", "encoder_channels": [32, 64, 128, 256],
            "skip_connections": True, "normalization": "BatchNorm2d",
            "activation": "ELU", "dropout": 0.2,
            "upsampling": "nearest", "legacy_state_dict_names_preserved": True,
            "inactive_parameters_note": "Legacy UpSample.deconv parameters are registered but never used in forward",
        }
    elif model_name == "fcn_gan_l1":
        generator, discriminator = FCNGenerator(), GlobalQSMDiscriminator(image_size)
        generator_spec = {
            "class": "FCNGenerator", "encoder_channels": [32, 64, 128, 256],
            "skip_connections": False, "normalization": "BatchNorm2d",
            "activation": "ELU", "dropout": 0.2, "upsampling": "nearest",
        }
    else:
        generator, discriminator = Pix2PixQSMGenerator(), Pix2Pix70Discriminator()
        generator_spec = {
            "class": "Pix2PixQSMGenerator", "base_channels": 64,
            "downsampling_levels": 8,
            "encoder_channels": [64, 128, 256, 512, 512, 512, 512, 512],
            "skip_connections": True, "normalization": "BatchNorm2d",
            "encoder_activation": "LeakyReLU(0.2)", "decoder_activation": "ReLU",
            "dropout": {"probability": 0.5, "decoder_levels_from_innermost_zero_based": [1, 2, 3]},
            "upsampling": "ConvTranspose2d(kernel=4,stride=2,padding=1)",
            "sources": [PIX2PIX_PAPER_URL, PIX2PIX_IMPLEMENTATION_URL],
            "adaptations": ["one-channel SWI input and QSM output", "linear output replaces Tanh",
                            "right/bottom zero padding and final crop preserve the original voxel grid"],
        }
    generator_spec["output_head"] = "linear, unrestricted signed scalar per voxel"
    if discriminator is None:
        discriminator_spec = None
    elif isinstance(discriminator, GlobalQSMDiscriminator):
        discriminator_spec = {
            "class": "GlobalQSMDiscriminator", "kind": discriminator.kind,
            "conditional": False, "input_channels": 1,
            "feature_channels": [64, 128, 256, 512],
            "feature_convolutions": {"kernel": 3, "strides": [2, 2, 2, 1], "padding": 1},
            "final_convolution": {"in_channels": 512, "out_channels": 1, "kernel": 3, "padding": 1},
            "final_convolution_activation": "LeakyReLU(0.2)",
            "pooling": None, "flattened_spatial_features": (image_size // 8) ** 2,
            "classifier": f"Flatten + Linear({(image_size // 8) ** 2},1), no scalar activation",
            "normalization": "InstanceNorm2d after feature convolutions 2,3,4; affine=False, track_running_stats=False",
            "normalization_dependency": "Within-slice spatial statistics; no cross-sample batch statistics",
            "required_square_image_size": image_size,
            "output": "one unrestricted logit per slice, shape [N,1]",
            "forward_contract": "forward(swi,qsm); SWI is shape-checked and otherwise unused",
            "historical_compatibility": "revised scalar logit head; not the legacy two-output discriminator",
        }
    else:
        discriminator_spec = {
            "class": "Pix2Pix70Discriminator", "kind": discriminator.kind,
            "conditional": True, "input_channels": 2, "base_channels": 64,
            "feature_channels": [64, 128, 256, 512],
            "kernel_sizes": [4, 4, 4, 4, 4], "strides": [2, 2, 2, 1, 1],
            "paddings": [1, 1, 1, 1, 1], "nominal_convolutional_receptive_field": [70, 70],
            "normalization": "BatchNorm2d on intermediate feature layers",
            "batchnorm_dependency": "Training statistics couple spatial positions and batch elements; nominal RF is not strict local dependency",
            "output": "unrestricted conditional patch logits",
            "forward_contract": "forward(swi,qsm); channel concatenation [SWI,QSM]",
            "sources": [PIX2PIX_PAPER_URL, PIX2PIX_IMPLEMENTATION_URL],
        }
    metadata = {
        "schema": "swixqsm.prospective_training_model", "version": 1, "model_name": model_name,
        "image_size": image_size,
        "generator": generator_spec, "discriminator": discriminator_spec,
        "parameter_counts": {"generator": _count_parameters(generator),
                             "discriminator": _count_parameters(discriminator)},
        "input": {"channels": 1, "minimum_height_width": 32, "required_height_width_multiple": 8,
                  "configured_training_height_width": [image_size, image_size],
                  "global_discriminator_requires_configured_square": isinstance(discriminator, GlobalQSMDiscriminator)},
        "input_padding": ({"mode": "constant_zero", "sides": ["right", "bottom"],
                           "to_height_width_multiple": 256, "output_crop": "original height and width",
                           "interpolation": False}
                          if model_name == "pix2pix" else {"mode": "none", "interpolation": False}),
        "initialization": "PyTorch module defaults; no pretrained or historical weights loaded",
        "physical_output_units": "determined by the training normalization contract, not by model architecture",
    }
    return generator, discriminator, metadata

"""Receptive-field support and batch-normalized PatchGAN binary cross entropy.

Only the declared Sequential Conv2d/LeakyReLU topology is supported. This avoids
silently reusing an invalid receptive field if a spatial/global layer is added.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _topology(discriminator):
    network = getattr(discriminator, "net", None)
    if type(network) is not nn.Sequential:
        raise ValueError("Patch support requires discriminator.net to be a Sequential")
    layers, convolutions = [], []
    previous_channels = None
    for index, layer in enumerate(network):
        if type(layer) is nn.Conv2d:
            if layer.padding_mode != "zeros" or isinstance(layer.padding, str):
                raise ValueError("Patch support accepts explicit zero-padded Conv2d only")
            if previous_channels is not None and layer.in_channels != previous_channels:
                raise ValueError("Discriminator convolution channel topology is inconsistent")
            previous_channels = layer.out_channels
            record = {"index": index, "type": "Conv2d", "in_channels": layer.in_channels,
                      "out_channels": layer.out_channels, "kernel_size": list(layer.kernel_size),
                      "stride": list(layer.stride), "padding": list(layer.padding),
                      "dilation": list(layer.dilation), "groups": layer.groups,
                      "bias": layer.bias is not None, "padding_mode": "zeros"}
            layers.append(record)
            convolutions.append(layer)
        elif type(layer) is nn.LeakyReLU:
            if not math.isfinite(float(layer.negative_slope)):
                raise ValueError("LeakyReLU slope must be finite")
            layers.append({"index": index, "type": "LeakyReLU",
                           "negative_slope": float(layer.negative_slope), "inplace": layer.inplace})
        else:
            raise ValueError(f"Unsupported discriminator layer {index}: {type(layer).__name__}; "
                             "spatial/global/normalization changes require an explicit support rule")
    if not convolutions or convolutions[-1].out_channels != 1:
        raise ValueError("Patch support requires a one-channel convolutional logit output")
    return network, layers, convolutions


def _positive_support(value, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch tensor")
    if value.device.type == "meta":
        raise ValueError(f"{name} needs concrete values, not a meta-device tensor")
    if value.ndim != 4 or value.shape[1] != 1 or any(extent <= 0 for extent in value.shape):
        raise ValueError(f"{name} must have nonempty shape [N,1,H,W]")
    if value.is_complex() or value.dtype not in (
            torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"{name} must contain real numeric values")
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()) or bool((detached < 0).any()):
        raise ValueError(f"{name} must be finite and nonnegative")
    return detached > 0


@torch.no_grad()
def patch_support(mask, discriminator):
    """Return detached binary float32 support for the actual D.net topology.

    Positive mask values select valid input voxels; they are not probability
    weights. Each Conv2d propagates the logical OR over its actual kernel,
    stride, dilation, and zero-padding. A logit is valid iff any input voxel
    in its finite-network receptive field is valid. Implicit input/feature
    padding is always invalid. No resizing or centre-point approximation is used.
    """
    support = _positive_support(mask, "mask").to(dtype=torch.float32)
    network, _, convolutions = _topology(discriminator)
    devices = {parameter.device for parameter in network.parameters()}
    if devices and (len(devices) != 1 or support.device not in devices):
        raise ValueError("Mask and all discriminator parameters must be on the same device")
    for layer in convolutions:
        height_pad, width_pad = layer.padding
        padded = F.pad(support, (width_pad, width_pad, height_pad, height_pad), value=0.)
        try:
            support = F.max_pool2d(padded, kernel_size=layer.kernel_size,
                                   stride=layer.stride, dilation=layer.dilation,
                                   padding=0, ceil_mode=False)
        except RuntimeError as exc:
            raise ValueError(f"Mask shape is too small for the discriminator topology: {exc}") from exc
    return support.detach()


def adversarial_spec(discriminator, mode="valid_patch"):
    """Describe the precise spatial topology and adversarial reduction policy."""
    if not isinstance(mode, str) or mode not in {"valid_patch", "all_patches"}:
        raise ValueError("Adversarial mode must be valid_patch or all_patches")
    _, layers, convolutions = _topology(discriminator)
    receptive_field, stride, first_center = [1, 1], [1, 1], [.5, .5]
    for layer in convolutions:
        for axis in (0, 1):
            effective_kernel = (layer.kernel_size[axis]-1)*layer.dilation[axis]+1
            first_center[axis] += ((effective_kernel-1)/2-layer.padding[axis])*stride[axis]
            receptive_field[axis] += (effective_kernel-1)*stride[axis]
            stride[axis] *= layer.stride[axis]
    center_index = [value-.5 for value in first_center]
    return {"schema": "swixqsm.adversarial_loss_spec", "version": 1, "mode": mode,
            "topology": layers, "receptive_field": receptive_field, "stride": stride,
            "first_center_input_index": center_index,
            "first_center_from_input_edge": first_center,
            "nominal_input_index_offsets": {
                "minimum": [center-(size-1)/2 for center, size in zip(center_index, receptive_field)],
                "maximum": [center+(size-1)/2 for center, size in zip(center_index, receptive_field)]},
            "coordinate_order": "height,width; input pixel centres have indices 0,1,...",
            "boundary_policy": "Explicit Conv2d zero-padding is invalid at every feature-map stage; support is propagated layer by layer",
            "support_policy": ("any positive valid input in the actual receptive field"
                               if mode == "valid_patch" else "all logit locations, including pure background"),
            "support_version": 1, "mask_interpretation": "positive membership, no probability weighting",
            "reduction": "sum BCE over all selected patches in the entire batch divided by selected patch count",
            "slice_weighting": "pooled selected-patch mean; each slice contributes in proportion to its selected patch count under the declared mode",
            "empty_support": "raise ValueError", "targets": "same support for real, detached fake, and generator"}


def patch_bce(logits, target, weights):
    """Mean BCEWithLogits over positive support, pooled across the whole batch.

    This is not an equal-slice average. Positive weights denote membership,
    not probability weights; the detached membership count is the denominator.
    Excluded logits/targets are never evaluated, so background perturbations
    cannot affect either loss or gradients. All-zero support is an error.
    """
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != 1
            or any(extent <= 0 for extent in logits.shape) or not logits.is_floating_point()):
        raise ValueError("Logits must be a floating tensor with nonempty shape [N,1,H,W]")
    support = _positive_support(weights, "weights")
    if weights.shape != logits.shape or weights.device != logits.device:
        raise ValueError("Patch weights must match logit shape and device exactly")
    if not bool(support.any()):
        raise ValueError("Adversarial patch support is empty")
    selected_logits = logits[support]
    if not bool(torch.isfinite(selected_logits).all()):
        raise ValueError("Selected logits must be finite")
    if isinstance(target, torch.Tensor):
        if target.shape != logits.shape or target.device != logits.device or target.is_complex():
            raise ValueError("Target tensor must match logit shape and device exactly")
        selected_target = target.detach()[support].to(dtype=logits.dtype)
    else:
        try:
            target = float(target)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Target must be a scalar label or a tensor matching logits") from exc
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("Target values must be finite and within [0,1]")
        selected_target = torch.full_like(selected_logits, target)
    if (not bool(torch.isfinite(selected_target).all()) or bool((selected_target < 0).any())
            or bool((selected_target > 1).any())):
        raise ValueError("Selected target values must be finite and within [0,1]")
    return F.binary_cross_entropy_with_logits(selected_logits, selected_target, reduction="mean")

"""Explicit FP32 training objectives for the four prospective comparisons.

The historical engine remains unchanged.  A step performs one generator
forward and, when adversarial training is active, the canonical discriminator
update followed by the generator update.  An exception is fatal: in particular,
a failure in the later generator stage can follow an already completed D step.
The caller must not save that interrupted step as a successful checkpoint.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .normalization import QSMNormalization
from .patch_loss import patch_bce
from .training_diagnostics import masked_distribution


def _coefficient(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _check_objective(disc, lambda_l1, lambda_adv, mode):
    lambda_l1 = _coefficient(lambda_l1, "lambda_l1")
    lambda_adv = _coefficient(lambda_adv, "lambda_adv")
    if lambda_l1 + lambda_adv == 0:
        raise ValueError("At least one generator loss coefficient must be positive")
    if mode not in ("valid_patch", "all_patches"):
        raise ValueError("adversarial_mode must be valid_patch or all_patches")
    if disc is None and lambda_adv != 0:
        raise ValueError("A model without a discriminator requires lambda_adv = 0")
    return lambda_l1, lambda_adv


def _patch_topology(disc):
    network = getattr(disc, "net", None)
    if type(network) is not nn.Sequential or not len(network):
        raise ValueError("Conditional patch support requires a nonempty discriminator.net Sequential")
    layers, convolutions, channels = [], [], 2
    for index, layer in enumerate(network):
        if type(layer) is nn.Conv2d:
            if (layer.padding_mode != "zeros" or isinstance(layer.padding, str)
                    or layer.in_channels != channels):
                raise ValueError("Unsupported conditional discriminator convolution topology")
            channels = layer.out_channels
            convolutions.append(layer)
            layers.append({"index": index, "type": "Conv2d", "in_channels": layer.in_channels,
                           "out_channels": layer.out_channels, "kernel_size": list(layer.kernel_size),
                           "stride": list(layer.stride), "padding": list(layer.padding),
                           "dilation": list(layer.dilation), "groups": layer.groups,
                           "padding_mode": layer.padding_mode})
        elif type(layer) is nn.BatchNorm2d:
            if layer.num_features != channels:
                raise ValueError("BatchNorm feature count does not match the preceding convolution")
            layers.append({"index": index, "type": "BatchNorm2d", "num_features": layer.num_features,
                           "affine": layer.affine, "track_running_stats": layer.track_running_stats,
                           "eps": layer.eps, "momentum": layer.momentum})
        elif type(layer) is nn.LeakyReLU:
            if not math.isfinite(float(layer.negative_slope)):
                raise ValueError("LeakyReLU slope must be finite")
            layers.append({"index": index, "type": "LeakyReLU",
                           "negative_slope": float(layer.negative_slope)})
        else:
            raise ValueError(f"Unsupported conditional discriminator layer: {type(layer).__name__}")
    if not convolutions or channels != 1 or type(network[-1]) is not nn.Conv2d:
        raise ValueError("Conditional discriminator must end in one-channel convolutional logits")
    return layers, convolutions


def adversarial_spec(disc, mode="valid_patch"):
    """Describe actual loss selection and nominal convolutional support.

    BatchNorm is explicitly nonlocal during training.  The OR rule below only
    excludes loss locations with an empty direct convolutional mask footprint;
    it does not establish complete background-gradient isolation through BN.
    """
    if mode not in ("valid_patch", "all_patches"):
        raise ValueError("adversarial_mode must be valid_patch or all_patches")
    if disc is None:
        return {"kind": "none", "support_computed": False, "discriminator_forward_calls": 0}
    kind = getattr(disc, "kind", None)
    common = {
        "schema": "swixqsm.comparison_adversarial_support", "version": 1,
        "kind": kind, "mode": mode, "mask_interpretation": "mask > 0; binary membership",
        "input_policy": "SWI, real QSM and generated QSM are zeroed outside the same positive mask",
        "criterion": "BCEWithLogitsLoss", "discriminator_loss": "0.5 * (BCE(real,1) + BCE(detached_fake,0))",
        "generator_adversarial_loss": "BCE(fake,1)",
        "generator_forward_calls": 1, "discriminator_forward_calls_when_active": 3,
        "discriminator_update_order": "real + detached fake backward; D step; freeze D parameters; G adversarial forward/backward; G step",
        "batchnorm_during_generator_adversarial_forward": "training mode retained; running statistics update for all three D forwards",
        "empty_slice_policy": "reject every input slice without at least one positive mask voxel",
    }
    if kind == "global_qsm_only":
        common.update(
            conditional=False, output_shape="[N,1]", support="one logit per nonempty slice",
            reduction="equal nonempty-slice mean",
            normalization_dependency="InstanceNorm2d uses spatial statistics within each individual slice in both training and evaluation; no cross-sample BatchNorm statistics",
            batchnorm_during_generator_adversarial_forward="not applicable: global D has InstanceNorm, without running statistics or BatchNorm",
            background_gradient_isolation=False,
            nominal_convolutional_receptive_field=None,
            all_patches_difference="none: global D has one valid logit for every required nonempty slice",
        )
        return common
    if kind != "conditional_patch70":
        raise ValueError("Unknown discriminator kind for the prospective comparison")
    layers, convolutions = _patch_topology(disc)
    receptive_field, jump, center = [1, 1], [1, 1], [0., 0.]
    for layer in convolutions:
        for axis in (0, 1):
            effective = (layer.kernel_size[axis] - 1) * layer.dilation[axis] + 1
            center[axis] += ((effective - 1) / 2 - layer.padding[axis]) * jump[axis]
            receptive_field[axis] += (effective - 1) * jump[axis]
            jump[axis] *= layer.stride[axis]
    common.update(
        conditional=True, topology=layers,
        nominal_convolutional_receptive_field=receptive_field, convolutional_output_stride=jump,
        first_output_center_input_index=center,
        support=("any positive voxel in the finite direct convolutional receptive field"
                 if mode == "valid_patch" else "all output logit locations, including direct empty-mask footprints"),
        support_propagation="Conv2d-by-Conv2d logical OR with explicit zero padding; BatchNorm/LeakyReLU skipped only for the nominal convolutional footprint",
        reduction="pooled selected-logit mean across the full batch",
        normalization_dependency="BatchNorm training statistics couple spatial locations and batch elements",
        background_gradient_isolation=False,
        limitation="valid_patch excludes direct all-empty loss locations only; it does not isolate background or other slices from gradients through BatchNorm",
    )
    return common


def objective_spec(disc, model_meta, lambda_l1, lambda_adv, mode="valid_patch", qsm_scale=1., units=None):
    lambda_l1, lambda_adv = _check_objective(disc, lambda_l1, lambda_adv, mode)
    qsm_scale = _coefficient(qsm_scale, "qsm_scale")
    if qsm_scale == 0:
        raise ValueError("qsm_scale must be positive")
    physical_coefficient = lambda_l1 / qsm_scale
    if not math.isfinite(physical_coefficient):
        raise ValueError("lambda_l1 / qsm_scale must remain finite")
    return {
        "schema": "swixqsm.comparison_objective", "version": 1,
        "model_name": model_meta.get("model_name") if isinstance(model_meta, dict) else None,
        "generator_loss": "lambda_l1 * masked_l1 + lambda_adv * BCEWithLogits(fake,1)",
        "lambda_l1": lambda_l1, "lambda_adv": lambda_adv,
        "loss_value_space": "normalized_QSM",
        "l1_definition": "sum absolute normalized-QSM error over mask > 0 divided by positive voxel count",
        "qsm_scale": qsm_scale,
        **({"restored_value_l1_relation": "normalized L1 = original-storage-scale L1 / qsm_scale; offset cancels in paired errors",
             "lambda_l1_per_stored_unit": physical_coefficient,
             "unit_status": "unverified_original_storage"} if units == "stored" else
           {"physical_l1_relation": "normalized L1 = physical-unit L1 / qsm_scale; offset cancels in paired errors",
            "lambda_l1_per_physical_unit": physical_coefficient}),
        "plain_model_policy": "no discriminator, no discriminator optimizer, lambda_adv = 0",
        "warmup_policy": "lambda_adv = 0 means no discriminator forward, backward, support computation or update",
        "precision": "FP32 only; AMP and GradScaler are explicitly unsupported",
        "diagnostic_l1_batch": "weighted L1 batch mean (lambda_l1 * masked_l1), the denominator for the weighted adversarial/L1 loss-value ratio",
        "adversarial": adversarial_spec(disc, mode),
        "failure_policy": "nonfinite losses/gradients cause a fatal exception; G-stage failure may follow an already completed canonical D update; do not checkpoint interrupted steps",
    }


@torch.no_grad()
def nominal_patch_support(mask, disc):
    """Detached float32 OR support for the declared convolutional footprint."""
    if (not isinstance(mask, torch.Tensor) or mask.ndim != 4 or mask.shape[1] != 1
            or any(size < 1 for size in mask.shape) or mask.is_complex()
            or not bool(torch.isfinite(mask).all()) or bool((mask < 0).any())):
        raise ValueError("Support requires a finite, nonnegative [N,1,H,W] mask")
    _, convolutions = _patch_topology(disc)
    support = (mask > 0).to(torch.float32)
    for layer in convolutions:
        if layer.weight.device != support.device:
            raise ValueError("Mask and discriminator must share one device")
        height, width = layer.padding
        padded = F.pad(support, (width, width, height, height), value=0.)
        try:
            support = F.max_pool2d(padded, kernel_size=layer.kernel_size, stride=layer.stride,
                                   dilation=layer.dilation, padding=0)
        except RuntimeError as exc:
            raise ValueError("Mask dimensions are too small for the discriminator topology") from exc
    return support.detach()


def _optimizer_matches(optimizer, module, name):
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(f"{name} must be a torch optimizer")
    actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected = list(module.parameters())
    if (len(actual) != len({id(parameter) for parameter in actual})
            or {id(parameter) for parameter in actual} != {id(parameter) for parameter in expected}):
        raise ValueError(f"{name} must own exactly its model's parameters without duplicates or cross-model parameters")


def _finite_gradients(module, stage):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError(f"{stage}: no model gradients; refusing optimizer update")
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError(f"{stage}: nonfinite gradients; refusing optimizer update")


def _finite_scalar(value, stage):
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{stage}: nonfinite value; refusing the next optimizer update")


def _adversarial_bce(logits, target, support):
    if logits.shape != support.shape or logits.device != support.device:
        raise RuntimeError("Discriminator logits do not match the declared support")
    _finite_scalar(logits, "discriminator logits")
    if logits.ndim == 4:
        return patch_bce(logits, target, support)
    if logits.ndim != 2 or logits.shape[1] != 1 or not bool((support > 0).any()):
        raise RuntimeError("Global discriminator requires nonempty [N,1] support")
    selected = logits[support > 0]
    return F.binary_cross_entropy_with_logits(selected, torch.full_like(selected, target))


def comparison_step(gen, disc, opt_g, opt_d, x, y, mask, lambda_l1, lambda_adv, *,
                    adversarial_mode="valid_patch", normalization=None, amp="none", scaler=None):
    """One prospective FP32 training step; return detached additive diagnostics.

    Positive mask values are membership, not probability weights.  All slices
    must be nonempty.  The generator is evaluated once.  During GAN training,
    D sees real, detached fake, and frozen-parameter G fake in training mode;
    BatchNorm running statistics therefore update on all three D forwards.
    """
    lambda_l1, lambda_adv = _check_objective(disc, lambda_l1, lambda_adv, adversarial_mode)
    if amp != "none" or scaler is not None:
        raise ValueError("comparison_step supports FP32 only: amp='none', scaler=None")
    if normalization is not None and not isinstance(normalization, QSMNormalization):
        raise ValueError("normalization must be QSMNormalization or None")
    if (not all(isinstance(value, torch.Tensor) for value in (x, y, mask))
            or x.ndim != 4 or x.shape[1] != 1 or any(size < 1 for size in x.shape)
            or x.shape != y.shape or mask.shape != x.shape
            or x.device != y.device or mask.device != x.device
            or x.dtype != torch.float32 or y.dtype != torch.float32 or mask.is_complex()):
        raise ValueError("Training requires matching FP32 [N,1,H,W] SWI/QSM and a real mask on one device")
    if (not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all())
            or not bool(torch.isfinite(mask).all()) or bool((mask < 0).any())):
        raise ValueError("Training inputs, targets and masks must be finite; mask must be nonnegative")
    membership = mask > 0
    slice_support = membership.flatten(1).any(1)
    if not bool(slice_support.all()):
        raise ValueError("Every training slice must contain positive mask support")
    _optimizer_matches(opt_g, gen, "opt_g")
    if disc is None:
        if opt_d is not None:
            raise ValueError("A model without a discriminator must not have opt_d")
    elif lambda_adv > 0:
        _optimizer_matches(opt_d, disc, "opt_d")
        if {id(parameter) for parameter in gen.parameters()} & {id(parameter) for parameter in disc.parameters()}:
            raise ValueError("Generator and discriminator must not share parameters")

    active = lambda_adv > 0
    valid_support = weights = None
    if active:
        kind = getattr(disc, "kind", None)
        if kind == "global_qsm_only":
            valid_support = slice_support[:, None].to(torch.float32)
        elif kind == "conditional_patch70":
            valid_support = nominal_patch_support(membership, disc)
        else:
            raise ValueError("Unknown discriminator kind for active adversarial training")
        weights = valid_support if adversarial_mode == "valid_patch" else torch.ones_like(valid_support)
        if not bool((weights > 0).any()):
            raise ValueError("Adversarial support is empty")

    original_flags = None if disc is None else [parameter.requires_grad for parameter in disc.parameters()]
    opt_g.zero_grad(set_to_none=True)
    if disc is not None:
        disc.zero_grad(set_to_none=True)
    generator_updated = discriminator_updated = False
    try:
        gen.train()
        prediction = gen(x)
        if prediction.shape != y.shape or prediction.dtype != torch.float32:
            raise RuntimeError("Generator output must match FP32 target geometry")
        _finite_scalar(prediction, "generator prediction")
        l1 = (prediction[membership] - y[membership]).abs().mean()
        weighted_l1 = l1 * lambda_l1
        _finite_scalar(weighted_l1, "weighted L1")
        zero = prediction.new_zeros(())
        d_real = d_fake = adv = zero
        d_total = weighted_adv = zero
        real_logits = fake_logits = generator_logits = None
        if active:
            disc.train()
            disc.requires_grad_(True)
            opt_d.zero_grad(set_to_none=True)
            binary_mask = membership.to(x.dtype)
            masked_x, masked_y = x * binary_mask, y * binary_mask
            real_logits = disc(masked_x, masked_y)
            fake_logits = disc(masked_x, prediction.detach() * binary_mask)
            d_real = _adversarial_bce(real_logits, 1., weights)
            d_fake = _adversarial_bce(fake_logits, 0., weights)
            d_total = .5 * (d_real + d_fake)
            _finite_scalar(d_total, "discriminator loss")
            d_total.backward()
            _finite_gradients(disc, "discriminator backward")
            if any(parameter.grad is not None for parameter in gen.parameters()):
                raise RuntimeError("Detached discriminator update leaked gradients into the generator")
            opt_d.step()
            discriminator_updated = True
            opt_d.zero_grad(set_to_none=True)
            disc.requires_grad_(False)
            # Keep training mode: freezing parameters does not freeze BatchNorm.
            generator_logits = disc(masked_x, prediction * binary_mask)
            adv = _adversarial_bce(generator_logits, 1., weights)
            weighted_adv = lambda_adv * adv
        total = weighted_l1 + weighted_adv
        _finite_scalar(total, "generator loss")

        def observation(value, count, reduction):
            return {"sum": float(value.detach()) * count if count else 0.,
                    "count": count, "reduction": reduction}

        voxel_count = float(membership.sum())
        active_count = float(weights.sum()) if active else 0.
        reduction = ("nonempty_slice_mean" if active and weights.ndim == 2
                     else "valid_patch_mean" if adversarial_mode == "valid_patch" else "all_patch_mean")

        def probability(logits):
            return {"sum": float(logits.detach()[weights > 0].sigmoid().sum()) if active else 0.,
                    "count": active_count, "reduction": reduction}

        fields = {
            "l1": observation(l1, voxel_count, "valid_voxel_mean"),
            "weighted_l1": observation(weighted_l1, voxel_count, "valid_voxel_mean"),
            "adv": observation(adv, active_count, reduction),
            "weighted_adv": observation(weighted_adv, active_count, reduction),
            "d_real": observation(d_real, active_count, reduction),
            "d_fake": observation(d_fake, active_count, reduction),
            "d_total": observation(d_total, active_count, reduction),
            "g_total": observation(total, 1., "batch_mean"),
            "l1_batch": observation(weighted_l1, 1., "batch_mean"),
            "weighted_adv_batch": observation(weighted_adv, 1., "batch_mean"),
            "d_real_probability": probability(real_logits),
            "d_fake_probability": probability(fake_logits),
            "d_fake_for_g_probability": probability(generator_logits),
            "valid_patch_fraction": {"sum": float(valid_support.sum()) if active else 0.,
                                     "count": valid_support.numel() if active else 0, "reduction": "fraction"},
            "loss_patch_fraction": {"sum": active_count, "count": weights.numel() if active else 0,
                                    "reduction": "fraction"},
            "valid_voxel_fraction": {"sum": voxel_count, "count": mask.numel(), "reduction": "fraction"},
        }
        if any(not math.isfinite(field["sum"]) or not math.isfinite(field["count"])
               for field in fields.values()):
            raise RuntimeError("Nonfinite training diagnostic before generator update")
        distribution = masked_distribution(prediction, y, membership, normalization) if normalization is not None else None
        total.backward()
        _finite_gradients(gen, "generator backward after discriminator update" if active else "generator backward")
        opt_g.step()
        generator_updated = True
        return {
            "fields": fields, "distribution": distribution,
            "generator_updated": generator_updated, "discriminator_updated": discriminator_updated,
            "adversarial_active": active, "actual_lambda_l1": lambda_l1,
            "actual_lambda_adv": lambda_adv, "actual_adv_weight": lambda_adv,
            "adversarial_mode": adversarial_mode, "l1_batch_is_weighted": True,
            "losses": {"l1": float(l1.detach()), "weighted_l1": float(weighted_l1.detach()),
                       "adv": float(adv.detach()), "weighted_adv": float(weighted_adv.detach()),
                       "d_real": float(d_real.detach()), "d_fake": float(d_fake.detach()),
                       "d_total": float(d_total.detach()), "g_total": float(total.detach())},
            "forward_counts": {"generator": 1, "discriminator": 3 if active else 0},
        }
    except Exception:
        opt_g.zero_grad(set_to_none=True)
        raise
    finally:
        if disc is not None:
            disc.zero_grad(set_to_none=True)
            for parameter, flag in zip(disc.parameters(), original_flags):
                parameter.requires_grad_(flag)

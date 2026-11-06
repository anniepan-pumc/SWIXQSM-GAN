"""Small training primitives; no dependency on the original project modules."""
import math

import torch


from ..model.discriminator import PatchDiscriminator
from .normalization import QSMNormalization
from .patch_loss import patch_support, patch_bce
from .training_diagnostics import masked_distribution


def load_generator(model, path, reset_head=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "generator" in state:
        state = state["generator"]
    elif "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    # Strict loading deliberately rejects incompatible architectures.
    model.load_state_dict(state, strict=True)
    if reset_head:
        model.final[0].reset_parameters()


def masked_l1(pred, target, mask):
    return ((pred-target).abs()*mask).sum() / mask.sum().clamp_min(1)


def train_step(gen, disc, opt_g, opt_d, x, y, mask, adv_weight, *,
               adversarial_mode="valid_patch", return_diagnostics=False,
               diagnostics_normalization=None):
    """One G forward; the same spatial support weights all three GAN terms.

    Diagnostic distributions use this already-computed, pre-update prediction.
    The historical two-float return is preserved unless diagnostics are requested.
    """
    if not math.isfinite(adv_weight) or adv_weight < 0:
        raise ValueError("adv_weight must be finite and nonnegative")
    if adversarial_mode not in {"valid_patch", "all_patches"}:
        raise ValueError("Unknown adversarial patch policy")
    if (x.shape != y.shape or mask.shape != x.shape or x.ndim != 4 or x.shape[1] != 1
            or mask.device != x.device or y.device != x.device):
        raise ValueError("Training expects matching [N,1,H,W] tensors on one device")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Training inputs and targets must be finite")
    if not torch.isfinite(mask).all() or torch.any(mask < 0) or not torch.any(mask > 0):
        raise ValueError("Training support must be finite, nonnegative and nonempty")
    valid_patches = weights = None
    if adv_weight > 0 or return_diagnostics:
        valid_patches = patch_support(mask, disc)
        weights = valid_patches if adversarial_mode == "valid_patch" else torch.ones_like(valid_patches)
    opt_g.zero_grad(set_to_none=True)
    # Reuse one G forward so BatchNorm statistics are updated only once.
    pred = gen(x)
    if not torch.isfinite(pred).all():
        raise RuntimeError("Nonfinite prediction; refusing optimizer updates")
    l1 = masked_l1(pred, y, mask)
    if not torch.isfinite(l1):
        raise RuntimeError("Nonfinite masked L1; refusing optimizer updates")
    d_loss = pred.new_zeros(())
    real = fake = logits = d_real = d_fake = adv = None
    if adv_weight > 0:
        disc.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        real = disc(x*mask, y*mask)
        fake = disc(x*mask, pred.detach()*mask)
        d_real = patch_bce(real, 1., weights)
        d_fake = patch_bce(fake, 0., weights)
        d_loss = .5 * (d_real + d_fake)
        d_loss.backward()
        opt_d.step()
        opt_d.zero_grad(set_to_none=True)
        disc.requires_grad_(False)
    loss = l1
    if adv_weight > 0:
        logits = disc(x*mask, pred*mask)
        adv = patch_bce(logits, 1., weights)
        loss = loss + adv_weight * adv
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite generator loss; refusing generator update")
    details = None
    if return_diagnostics:
        patch_count = float(weights.sum())
        active_count = patch_count if adv_weight > 0 else 0.
        patch_reduction = "valid_patch_mean" if adversarial_mode == "valid_patch" else "all_patch_mean"

        def mean_field(value, count, reduction):
            return {"sum": float(value.detach()) * count if value is not None else 0.,
                    "count": count, "reduction": reduction}

        def probability_field(value):
            return {"sum": float(value.detach()[weights > 0].sigmoid().sum()) if value is not None else 0.,
                    "count": active_count, "reduction": patch_reduction}

        weighted_adv = adv * adv_weight if adv is not None else pred.new_zeros(())
        fields = {
            "l1": mean_field(l1, float(mask.sum()), "valid_voxel_mean"),
            "adv": mean_field(adv, active_count, patch_reduction),
            "weighted_adv": mean_field(weighted_adv if adv is not None else None, active_count, patch_reduction),
            "d_real": mean_field(d_real, active_count, patch_reduction),
            "d_fake": mean_field(d_fake, active_count, patch_reduction),
            "d_total": mean_field(d_loss if adv_weight > 0 else None, active_count, patch_reduction),
            "d_real_probability": probability_field(real),
            "d_fake_probability": probability_field(fake),
            "d_fake_for_g_probability": probability_field(logits),
            "g_total": mean_field(loss, 1., "batch_mean"),
            "l1_batch": mean_field(l1, 1., "batch_mean"),
            "weighted_adv_batch": mean_field(weighted_adv, 1., "batch_mean"),
            "valid_patch_fraction": {"sum": float(valid_patches.sum()), "count": valid_patches.numel(), "reduction": "fraction"},
            "loss_patch_fraction": {"sum": patch_count, "count": weights.numel(), "reduction": "fraction"},
            "valid_voxel_fraction": {"sum": int((mask > 0).sum()), "count": mask.numel(), "reduction": "fraction"},
        }
        details = {"fields": fields, "adversarial_active": adv_weight > 0,
                   "actual_adv_weight": float(adv_weight), "adversarial_mode": adversarial_mode,
                   "distribution": (masked_distribution(pred, y, mask, diagnostics_normalization)
                                    if diagnostics_normalization is not None else None)}
        if any(not math.isfinite(field["sum"]) or not math.isfinite(field["count"]) for field in fields.values()):
            raise RuntimeError("Nonfinite training diagnostic; refusing generator update")
    loss.backward()
    opt_g.step()
    disc.requires_grad_(True)
    result = (float(loss.detach()), float(d_loss.detach()))
    return (*result, details) if return_diagnostics else result


@torch.no_grad()
def validate(gen, loader, device, scale, output_units=None):
    """MAE in fixed training QSM units after reversing BOTH output and label.

    The loader must supply normalized labels under the same QSMNormalization.
    Pass the complete object (scale, offset, units) for prospective workflows;
    numeric scale is retained only for backward-compatible zero-offset callers.
    Validation/test never estimate new target normalization parameters here.
    """
    normalization = scale if isinstance(scale, QSMNormalization) else QSMNormalization(scale)
    gen.eval()
    totals = {}
    for x, y, mask, subjects in loader:
        x, y, mask = (a.to(device) for a in (x, y, mask))
        prediction = normalization.denormalize(gen(x), output_units=output_units)
        target = normalization.denormalize(y, output_units=output_units)
        errors = ((prediction-target).abs()*mask).sum((1, 2, 3)).cpu()
        counts = mask.sum((1, 2, 3)).cpu()
        for sid, error, count in zip(subjects, errors, counts):
            old = totals.setdefault(sid, [0., 0.])
            old[0] += float(error)
            old[1] += float(count)
    if not totals or any(count <= 0 for _, count in totals.values()):
        raise ValueError("Validation needs nonempty positive evaluation support per subject")
    patient_mae = {sid: error/count for sid, (error, count) in totals.items()}
    return sum(patient_mae.values())/len(patient_mae), patient_mae

"""Read-only masked distributions and additive, explicitly pooled loss logs.

These summaries do not alter optimization, choose lambda, select checkpoints,
infer historical discriminator losses, or establish ROI/lesion performance.
"""
import copy
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch

from .normalization import QSMNormalization


_STATS = ("mean", "sd", "p01", "p50", "p99", "min", "max")


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        # NumPy has no bfloat16 representation. Conversion remains detached and
        # never writes through the input tensor or changes model gradients.
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.cpu().numpy()
    value = np.asarray(value)
    if value.dtype.kind not in "fiub":
        raise ValueError("Diagnostics require real numeric arrays or tensors")
    return value


def _scalar(value, name):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a finite scalar")
        value = value.detach().cpu().item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        value = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _count_value(value):
    return int(value) if value.is_integer() else value


def _distribution_record(shape, count, normalization):
    transform = None if normalization is None else normalization.to_dict()
    if transform is not None:
        transform.update(scale=float(normalization.scale), offset=float(normalization.offset))
    return {
        "schema": "swixqsm.masked_distribution", "version": 1,
        "status": "empty_support" if count == 0 else "not_computed",
        "shape": list(shape), "n_elements": int(np.prod(shape)), "n_valid_voxels": int(count),
        "n_nonfinite_valid_pairs": 0,
        "mask_rule": "mask > 0; unweighted selected elements",
        "scope": "Positive-mask elements of this supplied array/batch only",
        "units": None if normalization is None else normalization.units,
        "value_space": ("as_supplied" if normalization is None else
                        "restored_stored_values" if normalization.units == "stored" else "physical"),
        **({"unit_status": "unverified_original_storage"}
           if normalization is not None and normalization.units == "stored" else {}),
        "normalization_applied": transform,
        "prediction": {key: None for key in _STATS},
        "target": {key: None for key in _STATS},
        "error": {"bias": None, "mae": None, "rmse": None},
        "prediction_to_target_sd_ratio": None, "sd_ratio_status": "empty_support" if count == 0 else "not_computed",
        "target_energy": None, "squared_error_energy": None,
        "nmse": None, "nmse_status": "empty_support" if count == 0 else "not_computed",
        "zero_baseline_nmse": None, "zero_baseline_status": "empty_support" if count == 0 else "not_computed",
        "sd_definition": "Population SD (ddof=0) over selected elements",
        "quantiles": {"probabilities": [.01, .5, .99], "method": "exact linear interpolation",
                      "scope": "This supplied array/batch only; never an epoch quantile"},
        "limitations": [
            "Masked global distributions do not establish regional agreement or lesion preservation.",
            "A low prediction/target SD ratio alone does not establish mode collapse.",
            "NMSE is squared error divided by uncentered target energy, not explained variance.",
            "No target-based normalization, re-referencing, parameter tuning or weight update is performed.",
        ],
    }


def _stats(values):
    quantiles = np.quantile(values, [.01, .5, .99], method="linear")
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=0)),
            "p01": float(quantiles[0]), "p50": float(quantiles[1]), "p99": float(quantiles[2]),
            "min": float(values.min()), "max": float(values.max())}


def masked_distribution(pred, target, mask, normalization=None):
    """Describe only positive-mask elements, with exact per-call quantiles.

    Inputs may be same-shape real NumPy arrays or tensors with at least three
    nonempty dimensions (including training [N,1,H,W] or evaluation [X,Y,Z]).
    Tensor inputs are detached. A supplied QSMNormalization is reversed before
    computing statistics. Without it, values remain as supplied and units are
    deliberately unspecified; the caller may annotate already physical data.
    Empty/nonfinite support or zero denominators are explicit JSON-safe states,
    never NaN/Infinity or silently dropped valid voxels.
    """
    if normalization is not None and not isinstance(normalization, QSMNormalization):
        raise ValueError("normalization must be QSMNormalization or None")
    pred, target, mask = (_numpy(value) for value in (pred, target, mask))
    if (pred.shape != target.shape or pred.shape != mask.shape or pred.ndim < 3
            or any(size == 0 for size in pred.shape)):
        raise ValueError("Prediction, target and mask must share a nonempty shape with at least three dimensions")
    if not np.isfinite(mask).all() or np.any(mask < 0):
        raise ValueError("Diagnostic mask must be finite and nonnegative")
    valid = mask > 0
    count = int(np.count_nonzero(valid))
    result = _distribution_record(pred.shape, count, normalization)
    if count == 0:
        return result
    selected_pred, selected_target = pred[valid], target[valid]
    finite = np.isfinite(selected_pred) & np.isfinite(selected_target)
    if not finite.all():
        result.update(status="nonfinite_valid_values", n_nonfinite_valid_pairs=int(np.count_nonzero(~finite)))
        result["sd_ratio_status"] = result["nmse_status"] = result["zero_baseline_status"] = "nonfinite_valid_values"
        return result
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            selected_pred = selected_pred.astype(np.float64)
            selected_target = selected_target.astype(np.float64)
            if normalization is not None:
                selected_pred = normalization.denormalize(selected_pred)
                selected_target = normalization.denormalize(selected_target)
            prediction_stats, target_stats = _stats(selected_pred), _stats(selected_target)
            error = selected_pred - selected_target
            squared_error = float(np.sum(error * error))
            target_energy = float(np.sum(selected_target * selected_target))
            error_stats = {"bias": float(np.mean(error)), "mae": float(np.mean(np.abs(error))),
                           "rmse": float(np.sqrt(squared_error / count))}
        if not all(math.isfinite(value) for value in [*prediction_stats.values(), *target_stats.values(),
                                                     *error_stats.values(), squared_error, target_energy]):
            raise FloatingPointError("Nonfinite diagnostic statistic")
    except (FloatingPointError, OverflowError):
        result["status"] = "numerical_overflow"
        result["sd_ratio_status"] = result["nmse_status"] = result["zero_baseline_status"] = "numerical_overflow"
        return result
    result.update(status="defined", prediction=prediction_stats, target=target_stats, error=error_stats,
                  target_energy=target_energy, squared_error_energy=squared_error)
    if target_stats["sd"] == 0:
        result["sd_ratio_status"] = "zero_target_sd"
    else:
        ratio = prediction_stats["sd"] / target_stats["sd"]
        result["sd_ratio_status"] = "defined" if math.isfinite(ratio) else "numerical_overflow"
        result["prediction_to_target_sd_ratio"] = ratio if math.isfinite(ratio) else None
    if target_energy == 0:
        result["nmse_status"] = result["zero_baseline_status"] = "zero_target_energy"
    else:
        nmse = squared_error / target_energy
        result["nmse_status"] = "defined" if math.isfinite(nmse) else "numerical_overflow"
        result["nmse"] = nmse if math.isfinite(nmse) else None
        result["zero_baseline_nmse"], result["zero_baseline_status"] = 1., "defined"
    return result


class EpochDiagnostics:
    """Aggregate named sum/count observations without inventing their reduction.

    ``update`` accepts {name: {sum, count, reduction}}. Names are intentionally
    extensible; each field keeps the same declared reduction across updates.
    Counts may be nonnegative fractional weights, preserving a loss's actual
    denominator. A zero count must accompany a zero sum and is not observed.
    Bounded distribution records remain separate per call; quantiles are never
    averaged or represented as whole-epoch quantiles.
    """
    def __init__(self, max_distribution_batches=4):
        if (isinstance(max_distribution_batches, bool) or not isinstance(max_distribution_batches, Integral)
                or max_distribution_batches < 0):
            raise ValueError("max_distribution_batches must be a nonnegative integer")
        self.max_distribution_batches = int(max_distribution_batches)
        self._fields = {}
        self._updates = 0
        self._distributions = []
        self._distribution_offers = 0

    def update(self, fields):
        if not isinstance(fields, Mapping) or not fields:
            raise ValueError("Diagnostic fields must be a nonempty mapping")
        pending = {}
        for name, observation in fields.items():
            if not isinstance(name, str) or not name or name != name.strip():
                raise ValueError("Diagnostic field names must be nonempty strings without surrounding whitespace")
            if not isinstance(observation, Mapping) or set(observation) != {"sum", "count", "reduction"}:
                raise ValueError(f"Diagnostic {name} requires exactly sum, count and reduction")
            total = _scalar(observation["sum"], f"{name}.sum")
            count = _scalar(observation["count"], f"{name}.count")
            reduction = observation["reduction"]
            if not isinstance(reduction, str) or not reduction.strip():
                raise ValueError(f"{name}.reduction must be an explicit nonempty string")
            reduction = reduction.strip()
            if count < 0 or (count == 0 and total != 0):
                raise ValueError(f"{name} needs nonnegative count and zero sum when count is zero")
            previous = self._fields.get(name, {"sum": 0., "count": 0., "reduction": reduction})
            if previous["reduction"] != reduction:
                raise ValueError(f"Diagnostic {name} reduction changed within the epoch")
            total, count = previous["sum"] + total, previous["count"] + count
            if not math.isfinite(total) or not math.isfinite(count):
                raise ValueError(f"Diagnostic {name} aggregate became nonfinite")
            pending[name] = {"sum": total, "count": count, "reduction": reduction}
        self._fields.update(pending)
        self._updates += 1

    def add_distribution(self, summary):
        """Keep at most max_distribution_batches full per-call summaries."""
        if (not isinstance(summary, Mapping) or summary.get("schema") != "swixqsm.masked_distribution"
                or type(summary.get("version")) is not int or summary["version"] != 1):
            raise ValueError("Expected a masked_distribution summary")
        count = summary.get("n_valid_voxels")
        if type(count) is not int or count < 0:
            raise ValueError("Distribution summary has invalid n_valid_voxels")
        # Keep only serializable finite records; never retain tensors or arrays.
        json.dumps(summary, allow_nan=False)
        self._distribution_offers += 1
        if len(self._distributions) >= self.max_distribution_batches:
            return False
        self._distributions.append(copy.deepcopy(dict(summary)))
        return True

    @staticmethod
    def _loss_ratio(metrics):
        batch_names = ("weighted_adv_batch", "l1_batch")
        use_batch = any(name in metrics for name in batch_names)
        numerator_name, denominator_name = batch_names if use_batch else ("weighted_adv", "l1")
        numerator, denominator = metrics.get(numerator_name), metrics.get(denominator_name)
        result = {"value": None, "status": "not_observed", "numerator_field": numerator_name,
                  "denominator_field": denominator_name,
                  "numerator_count": None if numerator is None else numerator["count"],
                  "denominator_count": None if denominator is None else denominator["count"],
                  "scope": "ratio_of_epoch_batch_means" if use_batch else "ratio_of_separately_pooled_loss_means",
                  "interpretation": "Numerical loss-value ratio only; not gradient contribution or an instruction to tune lambda"}
        if numerator is None or denominator is None:
            result["status"] = "missing_operand"
            return result
        if numerator["mean"] is None or denominator["mean"] is None:
            if any(value["status"] == "numerical_overflow" for value in (numerator, denominator)):
                result["status"] = "operand_numerical_overflow"
            return result
        if use_batch and (numerator["reduction"] != "batch_mean" or denominator["reduction"] != "batch_mean"):
            result["status"] = "incompatible_batch_reduction"
            return result
        if use_batch and numerator["count"] != denominator["count"]:
            result["status"] = "mismatched_batch_counts"
            return result
        if denominator["mean"] == 0:
            result["status"] = "zero_l1_and_weighted_adv" if numerator["mean"] == 0 else "zero_l1"
            return result
        if denominator["mean"] < 0:
            result["status"] = "negative_l1"
            return result
        value = numerator["mean"] / denominator["mean"]
        result["status"] = "defined" if math.isfinite(value) else "numerical_overflow"
        result["value"] = value if math.isfinite(value) else None
        return result

    def summary(self):
        metrics = {}
        for name, value in self._fields.items():
            count = value["count"]
            mean = value["sum"] / count if count > 0 else None
            status = "defined" if count > 0 else "not_observed"
            if mean is not None and not math.isfinite(mean):
                mean, status = None, "numerical_overflow"
            metrics[name] = {"sum": value["sum"], "count": _count_value(count),
                             "mean": mean, "status": status,
                             "reduction": value["reduction"]}
        return {"schema": "swixqsm.epoch_diagnostics", "version": 1,
                "n_update_calls": self._updates, "metrics": metrics,
                "weighted_adv_to_l1": self._loss_ratio(metrics),
                "distributions": {
                    "scope": "Caller-selected bounded array/batch samples; not full-epoch distributions or quantiles",
                    "sampling": "Caller-selected; not guaranteed random or representative",
                    "max_batches": self.max_distribution_batches, "n_batches_offered": self._distribution_offers,
                    "n_batches_recorded": len(self._distributions),
                    "n_valid_voxels_recorded": sum(value["n_valid_voxels"] for value in self._distributions),
                    "batches": copy.deepcopy(self._distributions),
                    "epoch_quantiles": None,
                },
                "limitations": [
                    "Sum/count pooling follows each field's explicit denominator; differently reduced losses are not interchangeable.",
                    "Loss-value ratios are not gradient-contribution measurements.",
                    "Sampled prediction SD or distributions alone do not establish mode collapse, regional agreement or lesion accuracy.",
                    "Diagnostics do not change validation-based selection, tune lambda on test data, or reconstruct historical discriminator losses.",
                ]}

"""Versioned, target-independent preprocessing and reversible QSM units."""
from dataclasses import asdict, dataclass
import math

import numpy as np


def unit_factor(source, target):
    # ``stored`` describes decoded NIfTI values with an unverified intensity
    # unit. It is deliberately disconnected from the physical conversion graph.
    if source == "stored" or target == "stored":
        if source == target:
            return 1.
        raise ValueError("Unverified stored values cannot be converted to or from ppm/ppb")
    factors = {"ppm": 1., "ppb": 0.001}
    if source not in factors or target not in factors:
        raise ValueError("QSM units must be explicitly ppm, ppb or stored")
    return factors[source] / factors[target]


def restored_value_metadata(units):
    """Describe reversed numerical normalization without certifying units."""
    unit_factor(units, units)
    if units == "stored":
        return {"Units": "stored", "ValueSpace": "restored_stored_values",
                "UnitStatus": "unverified_original_storage"}
    return {"Units": units, "ValueSpace": "physical_qsm"}


@dataclass(frozen=True)
class QSMNormalization:
    scale: float
    offset: float = 0.
    units: str = "ppm"

    def __post_init__(self):
        if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.offset):
            raise ValueError("QSM scale must be finite and positive; offset finite")
        unit_factor(self.units, self.units)

    def normalize(self, values, input_units=None):
        """No clipping, min-max, or per-patient target statistics. Tensor/array compatible."""
        values = values * unit_factor(input_units or self.units, self.units)
        return (values - self.offset) / self.scale

    def denormalize(self, values, output_units=None):
        return (values * self.scale + self.offset) * unit_factor(self.units, output_units or self.units)

    def to_dict(self):
        return {"method": "fixed_affine", "clip": False, **asdict(self)}

    @classmethod
    def from_dict(cls, spec):
        if set(spec) != {"method", "clip", "scale", "offset", "units"} or spec["method"] != "fixed_affine" or spec["clip"] is not False:
            raise ValueError("Unsupported or incomplete QSM normalization metadata")
        return cls(scale=spec["scale"], offset=spec["offset"], units=spec["units"])


SWI_SPEC = {"method": "brain_percentile", "percentiles": [0.5, 99.5], "clip": True, "outside_mask": 0.}


def normalize_swi(array, brain, spec=None, return_metadata=False):
    spec = SWI_SPEC if spec is None else spec
    if spec != SWI_SPEC:
        raise ValueError("Unsupported SWI normalization metadata")
    brain = np.asarray(brain, dtype=bool)
    if brain.shape != array.shape or not brain.any() or not np.isfinite(array).all():
        raise ValueError("SWI requires a matching nonempty brain mask and finite intensities")
    lo, hi = np.percentile(array[brain], spec["percentiles"])
    if hi <= lo:
        raise ValueError("Degenerate SWI intensity range")
    result = np.where(brain, np.clip((array-lo)/(hi-lo), 0, 1), 0).astype(np.float32)
    metadata = {**spec, "fitted_low": float(lo), "fitted_high": float(hi)}
    return (result, metadata) if return_metadata else result


def normalization_metadata(qsm):
    return {"version": 1, "swi": dict(SWI_SPEC), "qsm": qsm.to_dict(), "generator_output": "linear"}


def from_checkpoint(checkpoint):
    """Unknown legacy normalization cannot be inferred from state_dict or prediction range."""
    config = checkpoint.get("config", {})
    spec = checkpoint.get("normalization")
    if spec is None:
        # Explicit migration for the previous minimal-package format only.
        expected = "SWI brain p0.5-p99.5 -> [0,1]; QSM / fixed scale"
        if config.get("generator_output") != "linear" or config.get("normalization") != expected or "qsm_scale" not in config or "units" not in config:
            raise ValueError("Checkpoint has no trustworthy normalization metadata; use it only for training initialization")
        spec = normalization_metadata(QSMNormalization(config["qsm_scale"], units=config["units"]))
    if spec.get("version") != 1 or spec.get("generator_output") != "linear" or spec.get("swi") != SWI_SPEC:
        raise ValueError("Incompatible normalization version, SWI preprocessing or generator activation")
    qsm = QSMNormalization.from_dict(spec["qsm"])
    trained_qc = checkpoint.get("data_qc") or config.get("data_qc") or {}
    if qsm.units == "stored" and trained_qc.get("status") == "verified":
        raise ValueError("stored checkpoint cannot claim verified physical-unit training QC")
    if isinstance(config.get("normalization"), dict) and config["normalization"] != spec:
        raise ValueError("Conflicting checkpoint normalization blocks")
    if config.get("generator_output", "linear") != "linear":
        raise ValueError("Conflicting generator output metadata")
    for key, expected in (("qsm_scale", qsm.scale), ("qsm_offset", qsm.offset), ("units", qsm.units)):
        if key in config and config[key] != expected:
            raise ValueError(f"Conflicting checkpoint normalization: {key}")
    size = config.get("size")
    if not isinstance(size, int) or size < 32 or size % 8:
        raise ValueError("Checkpoint has invalid or missing spatial padding size")
    return qsm, spec

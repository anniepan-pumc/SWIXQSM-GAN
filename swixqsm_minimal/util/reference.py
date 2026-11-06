"""Inspect a declared QSM reference without re-referencing or changing data.

Numerical agreement with a target reference value is a consistency check. It
cannot prove anatomical mask identity, registration, or the truth of a declared
reconstruction/source. The caller supplies reconstruction-reliable support,
which may be broader than the final analysis mask (for example, for CSF).
"""
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import nibabel as nib
import numpy as np

from .normalization import unit_factor


_SPEC_KEYS = {"id", "method", "region", "target_ppb", "tolerance_ppb", "require_mask"}
_IDENTITY_KEYS = {"id", "method", "region", "target_ppb"}
_EXPECTED = (ValueError, TypeError, OSError, EOFError, KeyError, AttributeError,
             FloatingPointError, OverflowError, nib.filebasedimages.ImageFileError)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _number(value, name, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _identity_fields(value):
    identity = {"id": _text(value["id"], "reference id"),
                "method": _text(value["method"], "reference method"),
                "region": _text(value["region"], "reference region"),
                "target_ppb": _number(value["target_ppb"], "target_ppb")}
    if identity["method"] not in {"mean", "median"}:
        raise ValueError("Reference method must be mean or median")
    return identity


def validate_reference_spec(spec):
    """Return the normalized six-field contract; malformed specs raise ValueError."""
    if not isinstance(spec, Mapping) or set(spec) != _SPEC_KEYS:
        raise ValueError(f"Reference spec requires exactly {sorted(_SPEC_KEYS)}")
    result = _identity_fields(spec)
    result["tolerance_ppb"] = _number(spec["tolerance_ppb"], "tolerance_ppb", nonnegative=True)
    if type(spec["require_mask"]) is not bool:
        raise ValueError("require_mask must be a boolean")
    result["require_mask"] = spec["require_mask"]
    return result


def reference_identity(spec):
    """Identity excludes inspection tolerance and whether a mask is mandatory."""
    normalized = validate_reference_spec(spec)
    return {key: normalized[key] for key in ("id", "method", "region", "target_ppb")}


def assert_reference_identity(expected, actual):
    """Compare full specs or four-field checkpoint identities, returning identity.

    Tolerance and require_mask are inspection policy, not physical reference
    identity. They are checked when a full spec is supplied but are not compared.
    """
    def identity(value):
        if isinstance(value, Mapping) and set(value) == _IDENTITY_KEYS:
            return _identity_fields(value)
        return reference_identity(value)
    left, right = identity(expected), identity(actual)
    if left != right:
        raise ValueError("QSM reference identities disagree")
    return right


def _image_grid(image, name):
    shape = tuple(image.shape)
    affine = np.asarray(image.affine, dtype=np.float64)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError(f"{name} must be a nonempty 3D NIfTI image")
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0., 0., 0., 1.], atol=1e-8, rtol=0)
            or np.linalg.matrix_rank(affine[:3, :3]) != 3):
        raise ValueError(f"{name} has an invalid spatial affine")
    units = image.header.get_xyzt_units()[0]
    return shape, affine, units


def _volume(value, name, shape=None, mask=False):
    array = np.asarray(value)
    if (array.ndim != 3 or array.dtype.kind not in "fiub"
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be finite numeric 3D data")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape disagrees with the QSM grid")
    if mask:
        if np.any(array < 0) or not np.any(array > 0):
            raise ValueError(f"{name} must be nonnegative and nonempty")
        return array > 0
    return array


def inspect_reference(row, qsm_image, qsm_array, *, spec, qsm_units, brain_mask, reliable_mask):
    """Audit declarations and an optional reference mask; never modify QSM.

    Expected row/file/data errors are returned as failed inspection records.
    Invalid reference specs raise ValueError, since there is no valid contract
    against which to inspect data. ``reliable_mask`` must be chosen upstream
    from qsm_reconstruction_mask_path when provided, otherwise mask_path.
    A missing optional reference mask produces declared_only, not verified.
    """
    spec = validate_reference_spec(spec)
    identity = reference_identity(spec)
    result = {
        "passed": False, "status": "failed", "errors": [],
        "reference_identity": identity,
        "numerical": {"performed": False, "method": spec["method"], "units": "ppb",
                      "target_ppb": spec["target_ppb"], "tolerance_ppb": spec["tolerance_ppb"],
                      "n_voxels": None, "observed_ppb": None,
                      "absolute_difference_ppb": None, "within_tolerance": None},
        "provenance": {"declarations": {}, "source": None, "reference_mask_path": None,
                       "reference_mask_spatial_units": None, "qsm_spatial_units": None,
                       "qsm_input_units": None, "reliable_mask_declared_path": None,
                       "mask_interpretation": "Positive mask values select voxels; no probability weighting",
                       "operation": "Inspection only; no subtraction, centering, or QSM modification"},
        "limitations": [
            "Reference-mask mean/median near the target does not prove anatomical reference identity or the truth of the declared pipeline/source.",
            "Zero-valued QSM or padded background is not evidence of a valid reference region.",
            "Matching shape/affine does not prove anatomical registration.",
            "The reconstruction-reliable mask is supplied by the caller; it may be broader than the analysis mask.",
        ],
    }
    errors, provenance = result["errors"], result["provenance"]
    if not isinstance(row, Mapping):
        errors.append("Reference inspection row must be a mapping")
        row = {}
    for key, expected in (("qsm_reference", spec["id"]),
                          ("qsm_reference_method", spec["method"]),
                          ("qsm_reference_region", spec["region"])):
        try:
            declared = _text(row.get(key), key)
            provenance["declarations"][key] = declared
            if declared != expected:
                errors.append(f"{key} disagrees with the reference spec")
        except _EXPECTED as exc:
            errors.append(str(exc))
    try:
        provenance["source"] = _text(row.get("qsm_reference_source"), "qsm_reference_source")
    except _EXPECTED as exc:
        errors.append(str(exc))
    reliable_path = row.get("qsm_reconstruction_mask_path") or row.get("mask_path")
    if isinstance(reliable_path, (str, Path)):
        provenance["reliable_mask_declared_path"] = str(reliable_path)

    qsm_grid, qsm, brain, reliable, conversion = None, None, None, None, None
    try:
        qsm_grid = _image_grid(qsm_image, "QSM image")
        provenance["qsm_spatial_units"] = qsm_grid[2]
        if qsm_grid[2] == "unknown":
            result["limitations"].append("QSM spatial units are unknown; matching unknown headers do not establish physical units.")
    except _EXPECTED as exc:
        errors.append(f"QSM image: {exc}")
    try:
        qsm = _volume(qsm_array, "QSM values", None if qsm_grid is None else qsm_grid[0])
    except _EXPECTED as exc:
        errors.append(str(exc))
    shape = qsm_grid[0] if qsm_grid is not None else (qsm.shape if qsm is not None else None)
    for name, values in (("Brain mask", brain_mask), ("Reconstruction-reliable mask", reliable_mask)):
        try:
            checked = _volume(values, name, shape, mask=True)
            if name == "Brain mask":
                brain = checked
            else:
                reliable = checked
        except _EXPECTED as exc:
            errors.append(str(exc))
    try:
        conversion = unit_factor(qsm_units, "ppb")
        provenance["qsm_input_units"] = qsm_units
    except _EXPECTED as exc:
        errors.append(f"QSM units: {exc}")

    path_value = row.get("qsm_reference_mask_path")
    missing = path_value is None or (isinstance(path_value, str) and not path_value.strip())
    if missing:
        result["numerical"]["reason"] = "Reference mask absent; declaration only, not numerically verified"
        if spec["require_mask"]:
            errors.append("Reference mask is required but qsm_reference_mask_path is absent")
        if not errors:
            result["passed"], result["status"] = True, "declared_only"
        return result

    reference = None
    try:
        if not isinstance(path_value, (str, Path)):
            raise ValueError("qsm_reference_mask_path must be an already resolved absolute path")
        path = Path(path_value)
        if not path.is_absolute():
            raise ValueError("qsm_reference_mask_path must be an already resolved absolute path")
        path = path.resolve()
        provenance["reference_mask_path"] = str(path)
        mask_image = nib.load(str(path))
        mask_grid = _image_grid(mask_image, "Reference mask")
        provenance["reference_mask_spatial_units"] = mask_grid[2]
        reference = _volume(mask_image.get_fdata(dtype=np.float64), "Reference mask", shape, mask=True)
        if qsm_grid is None:
            raise ValueError("Reference mask grid cannot be checked against an invalid QSM image")
        if mask_grid[0] != qsm_grid[0] or not np.allclose(mask_grid[1], qsm_grid[1], atol=1e-4, rtol=0):
            raise ValueError("Reference mask shape/affine disagrees with the QSM grid")
        if mask_grid[2] != qsm_grid[2]:
            raise ValueError("Reference mask spatial units disagree with the QSM image")
        if brain is None or reliable is None:
            raise ValueError("Reference mask containment cannot be checked against invalid support masks")
        if np.any(reference & ~brain):
            raise ValueError("Reference mask includes voxels outside the input brain mask (possibly background/padding)")
        if np.any(reference & ~reliable):
            raise ValueError("Reference mask includes voxels outside reconstruction-reliable support")
    except _EXPECTED as exc:
        errors.append(f"Reference mask: {exc}")
        reference = None

    if reference is not None and qsm is not None and conversion is not None:
        try:
            with np.errstate(over="raise", invalid="raise"):
                values = qsm[reference].astype(np.float64) * conversion
                observed = float(np.mean(values) if spec["method"] == "mean" else np.median(values))
            difference = abs(observed - spec["target_ppb"])
            if not math.isfinite(observed) or not math.isfinite(difference):
                raise ValueError("Reference statistic or target difference is nonfinite")
            within = difference <= spec["tolerance_ppb"]
            result["numerical"].update(performed=True, n_voxels=int(values.size), observed_ppb=observed,
                                       absolute_difference_ppb=difference, within_tolerance=bool(within))
            if not within:
                errors.append("Reference statistic is outside the declared tolerance in ppb")
        except _EXPECTED as exc:
            errors.append(f"Reference numerical check: {exc}")
    if not errors and result["numerical"]["performed"]:
        result["passed"], result["status"] = True, "verified"
    return result

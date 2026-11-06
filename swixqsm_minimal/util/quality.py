"""Read-only, contract-driven data QC with explicit human registration review.

Geometry checks and optional alignment diagnostics do not prove registration.
QC never resamples, re-references, or changes the source images.
"""

import copy
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re

import nibabel as nib
import numpy as np
from scipy import ndimage

from .reference import inspect_reference, reference_identity, validate_reference_spec


MAIN_PATHS = ("swi_path", "qsm_path", "swi_mask_path", "mask_path")
QC_PATHS = (
    "qsm_reconstruction_mask_path", "qsm_reference_mask_path", "qsm_brain_mask_path",
    "registration_landmarks_path", "registration_image_path",
    "registration_transform_path", "transform_path",
)
IMAGE_PATHS = MAIN_PATHS + (
    "qsm_reconstruction_mask_path", "qsm_reference_mask_path",
    "qsm_brain_mask_path", "registration_image_path",
)
SPACE_FACTORS = {"mm": 1., "meter": 1000., "micron": .001}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value, name, *, minimum=None, maximum=None, strict_maximum=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if (not math.isfinite(value) or (minimum is not None and value < minimum)
            or (maximum is not None and (value >= maximum if strict_maximum else value > maximum))):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def _geometry_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("geometry must be an object")
    codes = spec.get("axis_codes")
    if not isinstance(codes, (list, tuple)) or len(codes) != 3:
        raise ValueError("geometry.axis_codes must contain exactly three axis codes")
    codes = list(codes)
    if not all(isinstance(code, str) for code in codes):
        raise ValueError("geometry.axis_codes must be axis-code strings")
    groups = [{"L", "R"}, {"P", "A"}, {"I", "S"}]
    if any(sum(code in group for code in codes) != 1 for group in groups):
        raise ValueError("geometry.axis_codes must include one left/right, anterior/posterior, inferior/superior axis")
    if spec.get("slice_axis") != 2 or isinstance(spec.get("slice_axis"), bool) or codes[2] not in {"S", "I"}:
        raise ValueError("slice_axis must be 2 with S/I as its axis code")
    limits = []
    for key in ("spacing_mm_min", "spacing_mm_max"):
        values = spec.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"geometry.{key} must contain three positive finite values")
        values = [_number(value, key, minimum=0.) for value in values]
        if any(value <= 0 for value in values):
            raise ValueError(f"geometry.{key} values must be positive")
        limits.append(values)
    if any(low > high for low, high in zip(*limits)):
        raise ValueError("geometry spacing minimum exceeds maximum")
    result = dict(axis_codes=codes, slice_axis=2,
                  spacing_mm_min=limits[0], spacing_mm_max=limits[1],
                  max_obliquity_deg=_number(spec.get("max_obliquity_deg"),
                                           "max_obliquity_deg", minimum=0., maximum=90., strict_maximum=True))
    if "spatial_units_if_unknown" in spec:
        units = spec["spatial_units_if_unknown"]
        if units not in SPACE_FACTORS:
            raise ValueError("spatial_units_if_unknown must be mm, meter, or micron")
        result["spatial_units_if_unknown"] = units
    return result


def validate_contract(contract):
    """Return the canonical version-1 contract, rejecting implicit geometry."""
    if not isinstance(contract, dict) or contract.get("schema") != "swixqsm.data_contract":
        raise ValueError("Expected schema swixqsm.data_contract")
    if type(contract.get("version")) is not int or contract.get("version") != 1:
        raise ValueError("Only data contract version 1 is supported")
    geometry = _geometry_spec(contract.get("geometry"))
    reference = validate_reference_spec(contract.get("reference"))
    registration = contract.get("registration")
    if not isinstance(registration, dict) or registration.get("require_review") is not True:
        raise ValueError("registration.require_review must explicitly be true")
    normalized_registration = {"require_review": True}
    if "min_mask_dice" in registration:
        normalized_registration["min_mask_dice"] = _number(
            registration["min_mask_dice"], "min_mask_dice", minimum=0., maximum=1.)
    if "max_landmark_error_mm" in registration:
        normalized_registration["max_landmark_error_mm"] = _number(
            registration["max_landmark_error_mm"], "max_landmark_error_mm", minimum=0.)
    return dict(schema="swixqsm.data_contract", version=1, geometry=geometry,
                reference=reference, registration=normalized_registration)


def load_contract(path):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_contract(json.load(stream))


def resolve_qc_paths(rows, manifest_path):
    """Copy manifest rows, resolving main and optional QC paths beside the CSV."""
    parent = Path(manifest_path).expanduser().resolve().parent
    resolved = []
    for original in rows:
        row = dict(original)
        for key in MAIN_PATHS + QC_PATHS:
            value = row.get(key)
            if value is not None and str(value).strip():
                path = Path(str(value)).expanduser()
                row[key] = str((parent / path if not path.is_absolute() else path).resolve())
        resolved.append(row)
    return resolved


def inspect_geometry(image, spec):
    """Inspect an image against geometry only, without modifying it.

    Obliquity is the acute angle of the current grid's slice plane normal to
    the RAS superior axis. It does not recover original acquisition/head angle to B0.
    """
    spec = _geometry_spec(spec)
    errors = []
    result = {"passed": False, "errors": errors, "shape": list(image.shape),
              "slice_axis": 2, "obliquity_definition": "slice_normal_to_RAS_superior_acute_angle"}
    if len(image.shape) != 3 or any(extent == 0 for extent in image.shape):
        errors.append("Image must be nonempty 3D")
    try:
        array = image.get_fdata(dtype=np.float64)
        if not np.isfinite(array).all():
            errors.append("Image contains nonfinite values")
    except (ValueError, TypeError, OSError, EOFError) as exc:
        errors.append(f"Cannot read image values: {exc}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        errors.append("Image affine must be a finite 4x4 matrix")
        return result
    if not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        errors.append("Image affine has an invalid homogeneous last row")
    basis = affine[:3, :3]
    lengths = np.linalg.norm(basis, axis=0)
    if (not np.isfinite(lengths).all() or np.any(lengths <= 0)
            or np.linalg.matrix_rank(basis) < 3):
        errors.append("Image affine is singular or has invalid voxel lengths")
        return result
    directions = basis / lengths
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4, rtol=0):
        errors.append("Image affine is not orthogonal; shear is unsupported")
    qform = sform = None
    qcode = scode = 0
    try:
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        errors.append(f"Cannot validate qform/sform: {exc}")
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    try:
        original_units = image.header.get_xyzt_units()[0]
    except (KeyError, ValueError) as exc:
        errors.append(f"Invalid NIfTI spatial-unit code: {exc}")
        return result
    units = original_units
    if units == "unknown":
        units = spec.get("spatial_units_if_unknown")
        if units is None:
            errors.append("Unknown spatial units require an explicit contract assumption")
    if units not in SPACE_FACTORS:
        errors.append(f"Unsupported spatial units: {original_units}")
        return result
    factor = SPACE_FACTORS[units]
    if qcode and scode:
        qform_mm, sform_mm = qform.copy(), sform.copy()
        qform_mm[:3] *= factor
        sform_mm[:3] *= factor
        if not np.allclose(qform_mm, sform_mm, atol=1e-4, rtol=0):
            errors.append("Coded qform and sform conflict")
    if len(zooms) != 3 or not np.allclose(zooms*factor, lengths*factor, atol=1e-5, rtol=1e-5):
        errors.append("Header voxel sizes disagree with affine column lengths")
    affine_mm = affine.copy()
    affine_mm[:3] *= factor
    spacing_mm = lengths * factor
    result.update(spatial_units=units, header_spatial_units=original_units,
                  spatial_units_assumed=original_units == "unknown",
                  spacing_mm=spacing_mm.tolist(), affine_mm=affine_mm.tolist())
    if any(value < low-1e-6 or value > high+1e-6
           for value, low, high in zip(spacing_mm, spec["spacing_mm_min"], spec["spacing_mm_max"])):
        errors.append("Voxel spacing is outside the contract's millimetre bounds")
    codes = list(nib.aff2axcodes(affine))
    result["axis_codes"] = codes
    if codes != spec["axis_codes"]:
        errors.append(f"Axis codes {codes} do not match required {spec['axis_codes']}")
    normal = np.cross(directions[:, 0], directions[:, 1])
    normal /= np.linalg.norm(normal)
    angle = math.degrees(math.acos(float(np.clip(abs(normal[2]), 0., 1.))))
    result["slice_normal_obliquity_deg"] = angle
    if angle > spec["max_obliquity_deg"] + 1e-6:
        errors.append("Slice normal obliquity exceeds the contract limit")
    result["passed"] = not errors
    return result


def _normalize_row(row):
    normalized = {}
    for key, value in dict(row).items():
        if isinstance(value, Path):
            value = str(value)
        normalized[str(key)] = value
    for key in MAIN_PATHS + QC_PATHS:
        if normalized.get(key) is not None:
            normalized[key] = (str(Path(str(normalized[key])).expanduser().resolve())
                               if str(normalized[key]).strip() else "")
    return normalized


def _case_key(row):
    return _digest({key: row.get(key) for key in ("subject_id", "split", "swi_path")})


def _mask(array, name):
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    mask = array > 0
    if not mask.any():
        raise ValueError(f"{name} must be nonempty")
    return mask


def _same_grid(first, second):
    return (first.get("shape") == second.get("shape")
            and first.get("spatial_units") == second.get("spatial_units")
            and "affine_mm" in first and "affine_mm" in second
            and np.allclose(first["affine_mm"], second["affine_mm"], atol=1e-4, rtol=0))


def _alignment_masks(brain, qsm_brain, spacing):
    dice = 2. * np.count_nonzero(brain & qsm_brain) / (brain.sum() + qsm_brain.sum())
    structure = ndimage.generate_binary_structure(3, 1)
    first = brain & ~ndimage.binary_erosion(brain, structure=structure, border_value=0)
    second = qsm_brain & ~ndimage.binary_erosion(qsm_brain, structure=structure, border_value=0)
    first_to_second = ndimage.distance_transform_edt(~second, sampling=spacing)[first]
    second_to_first = ndimage.distance_transform_edt(~first, sampling=spacing)[second]
    return {"dice": float(dice),
            "symmetric_mean_surface_distance_mm": float((first_to_second.sum()+second_to_first.sum())
                                                         / (len(first_to_second)+len(second_to_first))),
            "symmetric_max_surface_distance_mm": float(max(first_to_second.max(), second_to_first.max()))}


def _landmarks(path):
    with Path(path).open(encoding="utf-8") as stream:
        evidence = json.load(stream)
    if not isinstance(evidence, dict) or evidence.get("coordinate_system") != "ras_mm":
        raise ValueError("Landmarks must explicitly use coordinate_system='ras_mm'")
    fixed = np.asarray(evidence.get("fixed"), dtype=np.float64)
    moving = np.asarray(evidence.get("moving"), dtype=np.float64)
    if (fixed.ndim != 2 or fixed.shape[1:] != (3,) or len(fixed) == 0
            or moving.shape != fixed.shape or not np.isfinite(fixed).all() or not np.isfinite(moving).all()):
        raise ValueError("Landmarks must be finite, matched nonempty Nx3 fixed/moving points")
    distances = np.linalg.norm(moving-fixed, axis=1)
    return {"coordinate_system": "ras_mm", "n": len(distances),
            "rms_mm": float(np.sqrt(np.mean(distances**2))), "max_mm": float(distances.max()),
            "interpretation": "independently identified corresponding landmarks after registration"}


def inspect_case(row, contract, *, default_qsm_units=None):
    """Inspect one case, recording expected input failures without skipping them."""
    contract = validate_contract(contract)
    row = _normalize_row(row)
    units = row.get("qsm_units", default_qsm_units)
    errors = []
    paths = {key: row[key] for key in MAIN_PATHS + QC_PATHS if row.get(key)}
    case = {"key": _case_key(row), "subject_id": row.get("subject_id"), "split": row.get("split"),
            "passed": False, "errors": errors, "paths": paths, "qsm_units": units,
            "metadata_sha256": _digest({"row": {key: value for key, value in row.items()
                                                 if not key.startswith("prediction_")},
                                        "effective_qsm_units": units}),
            "file_sha256": {}, "geometry": {}, "reference": None, "registration": {}}
    if (not isinstance(row.get("subject_id"), str) or not row["subject_id"].strip()
            or not isinstance(row.get("split"), str) or row["split"] not in {"train", "val", "test"}):
        errors.append("Case needs subject_id and a train/val/test split")
    valid_units = isinstance(units, str) and units in {"ppm", "ppb"}
    if not valid_units:
        errors.append("QSM units must be explicitly ppm/ppb in row or default_qsm_units")
    for key in MAIN_PATHS:
        if not row.get(key):
            errors.append(f"Missing required path: {key}")
    for key, path in paths.items():
        try:
            case["file_sha256"][key] = _file_hash(path)
        except (OSError, ValueError) as exc:
            errors.append(f"Cannot hash {key}: {exc}")
    images, arrays = {}, {}
    for key in IMAGE_PATHS:
        if not row.get(key):
            continue
        try:
            image = nib.load(row[key])
            images[key] = image
            arrays[key] = image.get_fdata(dtype=np.float64)
            geometry = inspect_geometry(image, contract["geometry"])
            case["geometry"][key] = geometry
            errors.extend(f"{key}: {error}" for error in geometry["errors"])
        except (OSError, ValueError, TypeError, EOFError, nib.filebasedimages.ImageFileError) as exc:
            errors.append(f"Cannot inspect {key}: {exc}")
    base_geometry = case["geometry"].get("swi_path")
    if base_geometry:
        for key, geometry in case["geometry"].items():
            if key != "swi_path" and not _same_grid(base_geometry, geometry):
                errors.append(f"{key} is not on the same physical grid/units as SWI")
    brain = reliable = valid = None
    for key in ("swi_mask_path", "mask_path", "qsm_reconstruction_mask_path",
                "qsm_reference_mask_path", "qsm_brain_mask_path"):
        if key in arrays:
            try:
                mask = _mask(arrays[key], key)
                if key == "swi_mask_path":
                    brain = mask
                elif key == "mask_path":
                    valid = mask
                elif key == "qsm_reconstruction_mask_path":
                    reliable = mask
            except ValueError as exc:
                errors.append(str(exc))
    if brain is not None and valid is not None and brain.shape == valid.shape:
        if np.any(valid & ~brain):
            errors.append("Valid reference mask must be contained in SWI brain mask")
    if reliable is None:
        reliable = valid
    elif valid is not None and reliable.shape == valid.shape and np.any(valid & ~reliable):
        errors.append("Reconstruction reliability mask must contain the valid evaluation mask")
    if ("qsm_path" in images and "qsm_path" in arrays
            and brain is not None and reliable is not None and valid_units):
        try:
            case["reference"] = inspect_reference(
                row, images["qsm_path"], arrays["qsm_path"], spec=contract["reference"],
                qsm_units=units, brain_mask=brain, reliable_mask=reliable)
            if not case["reference"].get("passed"):
                errors.extend(f"reference: {error}" for error in case["reference"].get("errors", ["Reference check failed"]))
        except (OSError, ValueError, TypeError, EOFError, nib.filebasedimages.ImageFileError) as exc:
            errors.append(f"Cannot inspect reference: {exc}")
    else:
        errors.append("Reference check lacks valid QSM, masks, or declared units")
    registration = case["registration"]
    registration["review"] = row.get("registration_review")
    registration["reviewer"] = row.get("registration_reviewer", "")
    registration["review_date"] = row.get("registration_review_date", "")
    registration["limitation"] = "Automatic geometry and alignment diagnostics do not replace human registration review"
    review = registration["review"]
    if not isinstance(review, str) or review not in {"accepted", "rejected", "pending"}:
        errors.append("registration_review must be accepted, rejected, or pending")
    elif review != "accepted":
        errors.append(f"Human registration review is {review}")
    if "registration_reviewer" not in row or "registration_review_date" not in row:
        errors.append("Registration reviewer and ISO review-date fields are required")
    if review == "accepted":
        if not isinstance(registration["reviewer"], str) or not registration["reviewer"].strip():
            errors.append("Accepted registration needs a named reviewer")
        review_date = registration["review_date"]
        try:
            if not isinstance(review_date, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", review_date) is None:
                raise ValueError("Expected YYYY-MM-DD")
            date.fromisoformat(review_date)
        except ValueError:
            errors.append("Accepted registration needs a valid ISO YYYY-MM-DD review date")
    registration_spec = contract["registration"]
    registration["mask_alignment"] = None
    independent_path = row.get("qsm_brain_mask_path")
    source_value = row.get("qsm_brain_mask_source", "")
    independent_source = source_value.strip() if isinstance(source_value, str) else ""
    if independent_path:
        if not independent_source:
            errors.append("Independent QSM brain mask requires qsm_brain_mask_source declaration")
        try:
            if any(Path(independent_path).samefile(row[key]) for key in ("swi_mask_path", "mask_path") if row.get(key)):
                raise ValueError("QSM brain mask must be independent of input/valid masks, not the same file")
            if (brain is not None and "qsm_brain_mask_path" in arrays and base_geometry
                    and _same_grid(base_geometry, case["geometry"].get("qsm_brain_mask_path", {}))):
                qsm_brain = _mask(arrays["qsm_brain_mask_path"], "qsm_brain_mask_path")
                registration["mask_alignment"] = _alignment_masks(brain, qsm_brain, base_geometry["spacing_mm"])
                registration["mask_alignment"]["declared_source"] = independent_source
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"Cannot inspect independent brain-mask alignment: {exc}")
    elif independent_source:
        errors.append("qsm_brain_mask_source has no independent QSM brain-mask path")
    if "min_mask_dice" in registration_spec:
        alignment = registration["mask_alignment"]
        if alignment is None:
            errors.append("Configured min_mask_dice requires independent QSM brain-mask evidence")
        elif alignment["dice"] < registration_spec["min_mask_dice"]:
            errors.append("Independent brain-mask Dice is below the contract threshold")
    registration["landmarks"] = None
    if row.get("registration_landmarks_path"):
        try:
            registration["landmarks"] = _landmarks(row["registration_landmarks_path"])
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"Cannot inspect landmark evidence: {exc}")
    if "max_landmark_error_mm" in registration_spec:
        if registration["landmarks"] is None:
            errors.append("Configured max_landmark_error_mm requires matched landmark evidence")
        elif registration["landmarks"]["max_mm"] > registration_spec["max_landmark_error_mm"]:
            errors.append("Maximum post-registration landmark error exceeds the contract threshold")
    case["passed"] = not errors and bool(case["reference"] and case["reference"].get("passed"))
    return case


def build_report(rows, contract, *, default_qsm_units=None):
    """Inspect every supplied row and bind its inputs, evidence, and metadata."""
    contract = validate_contract(contract)
    cases = [inspect_case(row, contract, default_qsm_units=default_qsm_units) for row in rows]
    keys = [case["key"] for case in cases]
    for case in cases:
        if keys.count(case["key"]) > 1:
            case["errors"].append("Duplicate case key in report")
            case["passed"] = False
    return {"schema": "swixqsm.data_qc_report", "version": 1,
            "status": "passed" if cases and all(case["passed"] for case in cases) else "failed",
            "contract": contract, "contract_sha256": _digest(contract),
            "default_qsm_units": default_qsm_units,
            "data_reference": reference_identity(contract["reference"]),
            "cases": cases, "n_cases": len(cases),
            "n_passed": sum(case["passed"] for case in cases),
            "limitations": ["Human review is required; automated metrics do not establish successful anatomical registration"]}


def verify_report(rows, path, *, reference_id=None):
    """Recheck selected rows only and reject stale, failed, or mismatched QC.

    Other report cases and other splits' images are never opened. Additional
    provenance/preview fields are allowed, but all deterministic case core
    fields must match the re-inspection of the selected inputs.
    """
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("schema") != "swixqsm.data_qc_report" or report.get("version") != 1:
        raise ValueError("Unsupported data QC report schema/version")
    contract = validate_contract(report.get("contract"))
    if report.get("contract_sha256") != _digest(contract):
        raise ValueError("QC contract hash does not match its canonical contract")
    expected_reference = reference_identity(contract["reference"])
    if report.get("data_reference") != expected_reference:
        raise ValueError("QC report data_reference disagrees with its contract")
    if reference_id is not None and expected_reference["id"] != reference_id:
        raise ValueError("QC reference ID does not match the required reference")
    recorded = {}
    for case in report.get("cases", []):
        if not isinstance(case, dict) or not case.get("key") or case["key"] in recorded:
            raise ValueError("QC report contains invalid or duplicate case keys")
        recorded[case["key"]] = case
    selected = list(rows)
    if not selected:
        raise ValueError("No selected cases to verify")
    verified_keys = []
    for row in selected:
        current = inspect_case(row, contract, default_qsm_units=report.get("default_qsm_units"))
        if not current["passed"]:
            raise ValueError(f"Selected case {current['subject_id']} fails current QC: {'; '.join(current['errors'])}")
        old = recorded.get(current["key"])
        if old is None or not old.get("passed"):
            raise ValueError(f"Selected case {current['subject_id']} has no passing QC record")
        for key, value in current.items():
            if key not in old or _canonical(old[key]) != _canonical(value):
                raise ValueError(f"Selected case {current['subject_id']} QC changed: {key}")
        if current["key"] in verified_keys:
            raise ValueError("Duplicate selected case")
        verified_keys.append(current["key"])
    return {"status": "verified", "report_sha256": _file_hash(path), "report_path": str(path),
            "contract_sha256": _digest(contract), "contract": copy.deepcopy(contract),
            "default_qsm_units": report.get("default_qsm_units"),
            "data_reference": expected_reference, "selected_case_keys": verified_keys,
            "n_cases": len(verified_keys)}

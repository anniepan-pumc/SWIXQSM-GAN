"""Train-only voxel baselines for an explicitly prepared common anatomical space.

Matching grids are required but do not establish anatomical registration. This
module never registers or resamples images, chooses a susceptibility reference,
or uses validation/test images to fit a baseline.
"""
import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import load_volume, require_grid
from .normalization import SWI_SPEC, normalize_swi, unit_factor


SCHEMA = "swixqsm.voxel_baselines"
SCHEMA_VERSION = 1
GRID_ATOL = 1e-4
_ARRAY_KEYS = {"train_mean", "slope", "intercept", "support", "affine", "metadata_json"}
_IMAGE_KEYS = ("swi_path", "qsm_path", "swi_mask_path", "mask_path")
_SPATIAL_UNITS = {"unknown", "meter", "mm", "micron"}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty explicit declaration")
    return value.strip()


def _path(value):
    return str(Path(_text(str(value), "image path")).expanduser().resolve())


def _declarations(row, reference_id, space_id):
    for key, expected in (("qsm_reference", reference_id), ("space_id", space_id)):
        if key in row and _text(row[key], key) != expected:
            raise ValueError(f"Manifest {key} disagrees with the explicit declaration")


def _affine(value):
    value = np.asarray(value, dtype=np.float64)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0., 0., 0., 1.], atol=1e-8, rtol=0)
            or np.linalg.matrix_rank(value[:3, :3]) != 3):
        raise ValueError("Grid affine must be a finite nonsingular 4x4 spatial transform")
    return value


def _binary_mask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite and match the image shape")
    return value > 0


def _json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_metadata(metadata, shape):
    if not isinstance(metadata, dict):
        raise ValueError("Baseline metadata must be a JSON object")
    # Reject NaN, non-JSON values, or an incompatible preprocessing contract.
    json.dumps(metadata, allow_nan=False)
    if metadata.get("schema") != SCHEMA or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported baseline artifact schema")
    unit_factor(metadata.get("units"), metadata.get("units"))
    if metadata["units"] == "stored":
        raise ValueError("Anatomical baseline artifacts require physical ppm/ppb units; stored is unsupported")
    _text(metadata.get("reference_id"), "reference_id")
    _text(metadata.get("space_id"), "space_id")
    if metadata.get("shape") != list(shape):
        raise ValueError("Baseline metadata shape disagrees with its arrays")
    if metadata.get("grid_spatial_units") not in _SPATIAL_UNITS:
        raise ValueError("Baseline spatial units are missing or unsupported")
    if metadata.get("swi_normalization") != SWI_SPEC:
        raise ValueError("Baseline SWI normalization is missing or incompatible")
    for key in ("training_subject_ids", "training_swi_paths"):
        values = metadata.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Invalid baseline {key}")
        for value in values:
            if _text(value, key) != value:
                raise ValueError(f"Invalid whitespace in baseline {key}")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate baseline {key}")
        if key == "training_swi_paths" and any(_path(value) != value for value in values):
            raise ValueError("Baseline training paths must be canonical absolute paths")
    for key in ("training_manifest_sha256", "fit_data_sha256"):
        value = metadata.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid baseline {key}")
    if not isinstance(metadata.get("fit"), dict):
        raise ValueError("Missing baseline fit details")


@dataclass(frozen=True)
class BaselineBundle:
    train_mean: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray
    support: np.ndarray
    affine: np.ndarray
    metadata: dict

    def __post_init__(self):
        shape = np.shape(self.train_mean)
        if len(shape) != 3 or any(n <= 0 for n in shape):
            raise ValueError("Baseline arrays must be nonempty 3D volumes")
        for key in ("train_mean", "slope", "intercept"):
            array = np.asarray(getattr(self, key))
            if (array.shape != shape or array.dtype.kind not in "fiu"
                    or not np.isfinite(array).all()):
                raise ValueError("Baseline coefficient arrays must be finite and share a 3D shape")
            array = np.array(array, dtype=np.float64, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, key, array)
        support = np.asarray(self.support)
        if support.shape != shape or support.dtype.kind != "b" or not support.any():
            raise ValueError("Baseline support must be a nonempty matching boolean volume")
        support = support.copy()
        support.setflags(write=False)
        object.__setattr__(self, "support", support)
        affine = _affine(self.affine).copy()
        affine.setflags(write=False)
        object.__setattr__(self, "affine", affine)
        metadata = copy.deepcopy(self.metadata)
        _validate_metadata(metadata, shape)
        object.__setattr__(self, "metadata", metadata)

    @property
    def shape(self):
        return self.train_mean.shape

    @property
    def units(self):
        return self.metadata["units"]

    @property
    def reference_id(self):
        return self.metadata["reference_id"]

    @property
    def space_id(self):
        return self.metadata["space_id"]

    def assert_independent(self, rows):
        """Audit the selected evaluation rows, including every repeated visit.

        Canonical paths and declared patient IDs are checked. Renamed copies,
        unlinked aliases, or earlier pretrained-model exposure require an
        external identity/provenance audit.
        """
        subjects = set(self.metadata["training_subject_ids"])
        paths = set(self.metadata["training_swi_paths"])
        for row in rows:
            sid = _text(row.get("subject_id"), "subject_id")
            path = _path(row["swi_path"])
            if sid in subjects or path in paths:
                raise ValueError(f"Baseline training/evaluation leakage for subject {sid}")

    def predict(self, method, swi, brain, affine, *, mask=None, reference_id, space_id):
        """Predict physical susceptibility without consulting evaluation targets.

        The requested evaluation mask (or the whole input brain when omitted)
        must be fully supported by training. The caller must also compare its
        NIfTI header units with metadata['grid_spatial_units']; this array API
        has no header from which to infer or convert spatial units.
        """
        if method not in {"train_mean", "linear"}:
            raise ValueError("Baseline method must be train_mean or linear")
        if (_text(reference_id, "reference_id") != self.reference_id
                or _text(space_id, "space_id") != self.space_id):
            raise ValueError("Evaluation reference_id/space_id disagrees with the fitted baseline")
        swi = np.asarray(swi)
        if swi.shape != self.shape or swi.ndim != 3 or not np.isfinite(swi).all():
            raise ValueError("Evaluation SWI must be finite and match the baseline grid")
        affine = _affine(affine)
        if not np.allclose(affine, self.affine, atol=GRID_ATOL, rtol=0):
            raise ValueError("Evaluation affine disagrees with the baseline grid; no resampling is performed")
        brain = _binary_mask(brain, self.shape, "SWI brain mask")
        if not brain.any():
            raise ValueError("SWI brain mask must be nonempty")
        requested = brain if mask is None else _binary_mask(mask, self.shape, "Evaluation mask")
        if not requested.any() or np.any(requested & ~brain):
            raise ValueError("Evaluation mask must be nonempty and contained in the SWI brain mask")
        if np.any(requested & ~self.support):
            count = int(np.count_nonzero(requested & ~self.support))
            raise ValueError(f"Evaluation mask includes {count} voxels outside training coverage; refusing to shrink it")
        if method == "linear":
            x = normalize_swi(swi, brain, spec=self.metadata["swi_normalization"])
            prediction = self.intercept + self.slope * x
        else:
            prediction = self.train_mean
        # This does not change the caller's requested mask. Unrequested values
        # outside fitted support are not validated baseline predictions.
        result = np.where(brain & self.support, prediction, 0.)
        if not np.isfinite(result).all():
            raise ValueError("Baseline prediction is nonfinite")
        return result


def fit_baselines(rows, *, units="ppb", default_qsm_units=None, reference_id, space_id):
    """Fit train-only per-voxel mean and OLS in declared common anatomical space.

    Each training visit receives 1/n_training_visits_for_that_subject. At each
    voxel these weights are normalized over visits with valid target support.
    Consequently, people with the same complete coverage have equal total
    weight, while partially covered people contribute only their valid visits.
    The declared common space must be prepared and quality-controlled upstream.
    """
    unit_factor(units, units)
    if units == "stored":
        raise ValueError("Anatomical baseline fitting does not support unverified stored units")
    if default_qsm_units is not None:
        unit_factor(default_qsm_units, units)
    reference_id = _text(reference_id, "reference_id")
    space_id = _text(space_id, "space_id")
    train_rows = [dict(row) for row in rows if row.get("split") == "train"]
    if not train_rows:
        raise ValueError("At least one train row is required to fit baselines")
    visits = Counter(_text(row.get("subject_id"), "subject_id") for row in train_rows)
    seen_paths, file_hashes, records = set(), {}, []
    common_ref, spatial_units = None, None
    weight_sum = mean_x = mean_y = m2_x = covariance = None
    for row in train_rows:
        _declarations(row, reference_id, space_id)
        sid = _text(row.get("subject_id"), "subject_id")
        paths = {key: _path(row[key]) for key in _IMAGE_KEYS}
        if paths["swi_path"] in seen_paths:
            raise ValueError("Duplicate training SWI path; each visit must be unique")
        seen_paths.add(paths["swi_path"])
        input_units = row["qsm_units"] if "qsm_units" in row else default_qsm_units
        conversion = unit_factor(input_units, units)
        ref, swi = load_volume(paths["swi_path"])
        _affine(ref.affine)
        this_units = ref.header.get_xyzt_units()[0]
        arrays = []
        for key in ("qsm_path", "swi_mask_path", "mask_path"):
            image, array = load_volume(paths[key])
            require_grid(ref, image)
            if image.header.get_xyzt_units()[0] != this_units:
                raise ValueError("Training SWI/reference/masks must have identical spatial units")
            arrays.append(array)
        if common_ref is None:
            common_ref, spatial_units = ref, this_units
            weight_sum, mean_x, mean_y, m2_x, covariance = (
                np.zeros(swi.shape, dtype=np.float64) for _ in range(5))
        else:
            require_grid(common_ref, ref)
            if this_units != spatial_units:
                raise ValueError("All training visits must use identical grid spatial units")
        qsm, brain_values, valid_values = arrays
        brain = brain_values > 0
        valid = valid_values > 0
        if not valid.any() or np.any(valid & ~brain):
            raise ValueError("Training reference mask must be nonempty and contained in the SWI brain mask")
        x = normalize_swi(swi, brain).astype(np.float64)
        y = qsm.astype(np.float64) * conversion
        if not np.isfinite(y).all():
            raise ValueError("QSM unit conversion produced nonfinite values")
        weight = valid.astype(np.float64) / visits[sid]
        new_weight = weight_sum + weight
        ratio = np.divide(weight, new_weight, out=np.zeros_like(weight), where=new_weight > 0)
        delta_x, delta_y = x - mean_x, y - mean_y
        mean_x += ratio * delta_x
        mean_y += ratio * delta_y
        # Weighted online central moments avoid subtracting nearly equal sums.
        m2_x += weight * delta_x * (x - mean_x)
        covariance += weight * delta_x * (y - mean_y)
        weight_sum = new_weight
        for path in paths.values():
            if path not in file_hashes:
                file_hashes[path] = _file_hash(path)
        records.append({"subject_id": sid, "split": "train", **paths,
                        "qsm_units": input_units, "qsm_reference": reference_id,
                        "space_id": space_id, "visit_weight": 1. / visits[sid]})
    support = weight_sum > 0
    slope = np.divide(covariance, m2_x, out=np.zeros_like(covariance), where=support & (m2_x > 0))
    intercept = mean_y - slope * mean_x
    metadata = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "units": units, "reference_id": reference_id, "space_id": space_id,
        "shape": list(mean_y.shape), "grid_spatial_units": spatial_units,
        "swi_normalization": copy.deepcopy(SWI_SPEC),
        "training_subject_ids": sorted(visits), "training_swi_paths": sorted(seen_paths),
        "training_manifest_sha256": _json_hash(records),
        "fit_data_sha256": _json_hash({"training_rows": records, "files": file_hashes}),
        "training_rows": records, "training_file_sha256": file_hashes,
        "fit": {
            "methods": ["train_mean", "linear"], "n_subjects": len(visits),
            "n_visits": len(train_rows), "visits_per_subject": dict(sorted(visits.items())),
            "weighting": "Each visit is weighted by 1/n_training_visits_for_subject; weights are normalized over valid visits at each voxel",
            "linear": "Per-voxel weighted OLS with intercept using SWI-only brain-percentile normalization",
            "constant_x": "Nonpositive weighted x central moment gives slope=0 and intercept=train_mean",
            "uncovered": "support=false; evaluation requests outside support are rejected",
            "target_transform": "Explicit units only; no target percentiles, clipping, centering or calibration",
            "registration": "User-declared common anatomical space; matching grids do not prove registration; no registration or resampling performed",
            "unknown_spatial_units": "Unknown is permitted only when all training headers agree; it remains an explicit upstream assumption",
        },
    }
    return BaselineBundle(mean_y, slope, intercept, support, common_ref.affine, metadata)


def save_baselines(bundle, path):
    """Save numeric arrays and a JSON metadata scalar in a non-pickle NPZ."""
    if not isinstance(bundle, BaselineBundle):
        raise TypeError("Expected a BaselineBundle")
    # Revalidate after any caller edits to its mutable provenance dictionary.
    _validate_metadata(bundle.metadata, bundle.shape)
    _affine(bundle.affine)
    for array in (bundle.train_mean, bundle.slope, bundle.intercept):
        if array.shape != bundle.shape or not np.isfinite(array).all():
            raise ValueError("Cannot save malformed or nonfinite baseline coefficient arrays")
    if bundle.support.shape != bundle.shape or bundle.support.dtype.kind != "b" or not bundle.support.any():
        raise ValueError("Cannot save malformed baseline support")
    serialized = json.dumps(bundle.metadata, indent=2, sort_keys=True, allow_nan=False)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Baseline artifact output must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            np.savez_compressed(stream, train_mean=bundle.train_mean, slope=bundle.slope,
                                intercept=bundle.intercept, support=bundle.support,
                                affine=bundle.affine, metadata_json=np.asarray(serialized))
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def load_baselines(path):
    """Load a fully validated numeric artifact; object/pickle arrays are forbidden."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != _ARRAY_KEYS:
            raise ValueError("Baseline artifact has missing or unexpected fields")
        raw = archive["metadata_json"]
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("Baseline metadata must be a Unicode JSON scalar")
        metadata = json.loads(raw.item())
        return BaselineBundle(archive["train_mean"], archive["slope"], archive["intercept"],
                              archive["support"], archive["affine"], metadata)

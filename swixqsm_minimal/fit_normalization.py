"""Fit one reversible QSM normalization from training subjects only.

Each subject has equal weight. Within a subject, reliable voxels from all visits
are pooled, rather than counting each visit as an independent person. Image data
are decoded as float32 and moments are accumulated in float64 after explicit
unit conversion. No validation/test image or target statistic is needed.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

from .util.data import load_volume, read_manifest, require_grid
from .util.independence_gate import add_independence_arguments, check_independence
from .util.normalization import QSMNormalization, normalization_metadata, unit_factor
from .util.quality_gate import (
    add_quality_arguments, bind_input_units, check_data_quality, prepare_quality_rows,
)


SCHEMA = "swixqsm.fitted_normalization"
VERSION = 1
METHODS = ("mean_std", "zero_rms")
WEIGHTING = "equal_subjects__pooled_valid_voxels_within_subject"
IMAGE_KEYS = ("swi_path", "qsm_path", "swi_mask_path", "mask_path")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _training_rows(rows, default_qsm_units=None):
    # Select before inspecting units, resolving input paths or opening images.
    selected = [dict(row) for row in rows if row.get("split") == "train"]
    if not selected:
        raise ValueError("No training rows; normalization cannot use val/test")
    for row in selected:
        if not isinstance(row.get("global_subject_id", ""), str):
            raise ValueError("global_subject_id must be a string when supplied")
    have_global_ids = [bool(row.get("global_subject_id", "").strip()) for row in selected]
    if any(have_global_ids) and not all(have_global_ids):
        raise ValueError("Training global_subject_id must be present for every visit or none")
    prepared, seen_swi, seen_qsm, subject_globals = [], set(), set(), {}
    for row in selected:
        sid = _text(row.get("subject_id"), "subject_id")
        gid = row.get("global_subject_id", "").strip()
        if gid and sid in subject_globals and subject_globals[sid] != gid:
            raise ValueError("A training subject_id maps to multiple global subjects")
        subject_globals[sid] = gid
        units = row.get("qsm_units", default_qsm_units)
        unit_factor(units, units)  # An absent/blank unit must never be guessed.
        record = dict(subject_id=sid, global_subject_id=gid,
                      case_id=str(row.get("case_id", "")), split="train", qsm_units=units)
        for key in IMAGE_KEYS:
            raw = row.get(key)
            if raw is None:
                raise ValueError(f"{key} must be an explicit input path")
            value = _text(str(raw), key)
            record[key] = str(Path(value).expanduser().resolve())
        if record["swi_path"] in seen_swi or record["qsm_path"] in seen_qsm:
            raise ValueError("Duplicate training SWI/QSM path; do not count one visit twice")
        seen_swi.add(record["swi_path"])
        seen_qsm.add(record["qsm_path"])
        prepared.append(record)
    return sorted(prepared, key=lambda row: _canonical(row))


def training_identity(rows, default_qsm_units=None):
    """Order-independent hash of train identities, visits, input paths and units.

    This compares declarations, not undisclosed person identities or contents.
    No files from any split are opened by this function.
    """
    return _digest(_training_rows(rows, default_qsm_units))


def _subject_key(row):
    return ("global:" + row["global_subject_id"] if row["global_subject_id"]
            else "subject:" + row["subject_id"])


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_snapshot(path):
    path = Path(path)
    before = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected a regular training input: {path}")
    digest = _file_hash(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Training input changed while hashing: {path}")
    return dict(path=str(path), sha256=digest, size_bytes=after.st_size,
                mtime_ns=after.st_mtime_ns)


def _combine(left, right):
    """Stable population moments (count, mean, sum of squared deviations)."""
    if left is None:
        return right
    n1, mean1, m21 = left
    n2, mean2, m22 = right
    n = n1 + n2
    delta = mean2 - mean1
    return n, mean1 + delta * (n2 / n), m21 + m22 + delta * delta * n1 * n2 / n


def _statistics(inputs, method, units):
    subjects = {}
    for item in inputs:
        moments = item["moments"]
        count = item["valid_voxel_count"]
        if type(count) is not int or count <= 0:
            raise ValueError("Every training visit needs positive reliable-voxel counts")
        mean, variance = moments["mean"], moments["population_variance"]
        if (not math.isfinite(mean) or not math.isfinite(variance) or variance < 0):
            raise ValueError("Invalid training moment metadata")
        key = _subject_key(item)
        entry = subjects.setdefault(key, dict(moments=None, n_visits=0))
        entry["moments"] = _combine(entry["moments"], (count, mean, variance * count))
        entry["n_visits"] += 1
    summaries = []
    for key, entry in sorted(subjects.items()):
        count, mean, m2 = entry["moments"]
        summaries.append(dict(subject_key=key, n_visits=entry["n_visits"],
                              valid_voxel_count=count, mean=mean,
                              population_variance=m2 / count))
    population_mean = math.fsum(item["mean"] for item in summaries) / len(summaries)
    population_variance = math.fsum(
        item["population_variance"] + (item["mean"] - population_mean) ** 2
        for item in summaries) / len(summaries)
    offset = population_mean if method == "mean_std" else 0.
    squared_scale = population_variance + (population_mean - offset) ** 2
    if not math.isfinite(squared_scale) or squared_scale <= 0:
        raise ValueError("Zero or nonfinite fitted scale; inspect units, mask and target variance")
    normalization = QSMNormalization(math.sqrt(squared_scale), offset, units)
    return normalization, summaries, population_mean, population_variance


def fit_training_normalization(rows, *, units, method="mean_std", default_qsm_units=None,
                               data_qc=None, data_independence=None, manifest_path=None):
    """Return a traceable artifact without writing files or touching holdout images.

    The CLI supplies existing QC/independence gates. Direct API calls without
    those reports remain explicitly unverified. ``mean_std`` uses population
    std, not the std of subject means; ``zero_rms`` fixes offset to zero.
    """
    unit_factor(units, units)
    if units == "stored" and data_qc and data_qc.get("status") == "verified":
        raise ValueError("stored normalization requires unverified data, not a physical-unit QC contract")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if default_qsm_units is None and data_qc and data_qc.get("status") == "verified":
        default_qsm_units = data_qc.get("default_qsm_units")
    train = _training_rows(rows, default_qsm_units)
    for row in train:
        unit_factor(row["qsm_units"], units)
    source = Path(manifest_path).expanduser().resolve() if manifest_path is not None else None
    manifest_snapshot = _file_snapshot(source) if source is not None else None
    if source is not None:
        source_rows = prepare_quality_rows(read_manifest(source, validate_inputs=False), source)
        if training_identity(source_rows, default_qsm_units) != training_identity(train):
            raise ValueError("Provided rows do not match the training membership in manifest_path")
    inputs = []
    for row in train:
        files = {key: _file_snapshot(row[key]) for key in IMAGE_KEYS}
        reference, qsm = load_volume(row["qsm_path"])
        mask_image, mask = load_volume(row["mask_path"])
        require_grid(reference, mask_image)
        brain_image, brain = load_volume(row["swi_mask_path"])
        require_grid(reference, brain_image)
        # The estimator needs no SWI intensities, but both masks and the input
        # must be prepared on the same grid as the training target.
        require_grid(reference, nib.load(row["swi_path"]))
        if np.any(brain < 0) or not np.any(brain > 0):
            raise ValueError("SWI brain mask must be nonnegative and nonempty")
        if np.any(mask < 0):
            raise ValueError("Reference reliability mask must be nonnegative")
        valid = mask > 0
        if not valid.any():
            raise ValueError("Reference reliability mask is empty")
        if np.any(valid & ~(brain > 0)):
            raise ValueError("Reference reliability mask must be contained in the SWI brain mask")
        values = np.asarray(qsm[valid], dtype=np.float64) * unit_factor(row["qsm_units"], units)
        mean = float(values.mean(dtype=np.float64))
        centered = values - mean
        variance = float(np.dot(centered, centered) / values.size)
        if not np.isfinite(values).all() or not math.isfinite(variance):
            raise ValueError("QSM unit conversion/statistics overflow")
        # Bind the actual decoded values to immutable primary source bytes.
        for key in IMAGE_KEYS:
            if _file_snapshot(row[key]) != files[key]:
                raise RuntimeError(f"Training input changed during normalization fit: {row[key]}")
        inputs.append({**{key: row[key] for key in ("subject_id", "global_subject_id", "case_id", "split", "qsm_units")},
                       "files": files, "valid_voxel_count": int(values.size),
                       "moments": dict(mean=mean, population_variance=variance)})
    # Catch an earlier visit being modified while a later visit is processed.
    for item in inputs:
        for entry in item["files"].values():
            final = Path(entry["path"]).stat()
            if (final.st_size, final.st_mtime_ns) != (entry["size_bytes"], entry["mtime_ns"]):
                raise RuntimeError(f"Training input changed during normalization fit: {entry['path']}")
    if source is not None and _file_snapshot(source) != manifest_snapshot:
        raise RuntimeError("Manifest changed during normalization fit")
    normalization, summaries, mean, variance = _statistics(inputs, method, units)
    artifact = dict(
        schema=SCHEMA, version=VERSION, method=method,
        **({"value_space": "restored_stored_values", "unit_status": "unverified_original_storage"}
           if units == "stored" else {}),
        normalization=normalization_metadata(normalization),
        fit_selection=dict(split="train", weighting=WEIGHTING,
                           n_subjects=len(summaries), n_visits=len(inputs),
                           n_valid_voxels=sum(item["valid_voxel_count"] for item in inputs),
                           train_identity_sha256=training_identity(train),
                           default_input_qsm_units=default_qsm_units),
        statistics=dict(canonical_units=units, decoded_dtype="float32", accumulation_dtype="float64",
                        variance_ddof=0, subject_equal_mean=mean,
                        subject_equal_population_variance=variance),
        subject_statistics=summaries, inputs=inputs,
        data_qc=deepcopy(data_qc) if data_qc is not None else {
            "status": "unverified", "reason": "No QC gate supplied to the fitting API"},
        data_independence=deepcopy(data_independence) if data_independence is not None else {
            "status": "unverified", "reason": "No independence gate supplied to the fitting API"},
        data_reference=deepcopy(data_qc.get("data_reference")) if data_qc else None,
        provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                        manifest_path=str(source) if source is not None else None,
                        manifest_sha256=manifest_snapshot["sha256"] if manifest_snapshot is not None else None,
                        fitter_source_sha256=_file_hash(__file__),
                        command="python -m swixqsm_minimal.fit_normalization",
                        image_access="Only split=train primary inputs are opened or hashed; QSM, reliability mask and SWI brain mask are decoded; SWI is checked by header only"),
        limitations=[
            "Newly fitted preprocessing for prospective training; does not recover an unknown legacy checkpoint scale or normalization.",
            "Declared units and reference must be verified independently; fitted offset is numeric centering, not physical re-referencing.",
            "QC and independence status are retained; fitting does not certify registration, identity, clinical validity or an unrecorded training history.",
            "All visits of each subject share one subject weight; voxels within a subject are pooled with equal voxel weight.",
        ],
    )
    _validate_artifact(artifact)
    return artifact


def _validate_artifact(artifact):
    if not isinstance(artifact, dict):
        raise ValueError("Fitted normalization must be a JSON object")
    _canonical(artifact)
    if artifact.get("schema") != SCHEMA or type(artifact.get("version")) is not int or artifact["version"] != VERSION:
        raise ValueError("Unsupported fitted-normalization schema/version")
    method = artifact.get("method")
    if method not in METHODS:
        raise ValueError("Unsupported fitted-normalization method")
    spec = artifact.get("normalization", {})
    normalization = QSMNormalization.from_dict(spec.get("qsm", {}))
    if normalization.units == "stored" and artifact.get("data_qc", {}).get("status") == "verified":
        raise ValueError("stored fitted normalization cannot claim verified physical-unit QC")
    if normalization.units == "stored" and (artifact.get("value_space") != "restored_stored_values"
            or artifact.get("unit_status") != "unverified_original_storage"):
        raise ValueError("stored fitted normalization must retain unverified original-storage metadata")
    if spec != normalization_metadata(normalization):
        raise ValueError("Unsupported SWI/output normalization contract")
    selection = artifact.get("fit_selection", {})
    if selection.get("split") != "train" or selection.get("weighting") != WEIGHTING:
        raise ValueError("Normalization was not fitted using the supported train-only weighting")
    inputs = artifact.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Fitted normalization has no training input provenance")
    rows = []
    for item in inputs:
        if item.get("split") != "train" or set(item.get("files", {})) != set(IMAGE_KEYS):
            raise ValueError("Invalid fitted training membership/input roles")
        row = {key: item[key] for key in ("subject_id", "global_subject_id", "case_id", "split", "qsm_units")}
        unit_factor(row["qsm_units"], normalization.units)
        for role in IMAGE_KEYS:
            entry = item["files"][role]
            path = entry.get("path")
            if (not isinstance(path, str) or not Path(path).is_absolute()
                    or str(Path(path).resolve()) != path):
                raise ValueError("Training input provenance requires canonical absolute paths")
            digest = entry.get("sha256", "")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    or type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0
                    or type(entry.get("mtime_ns")) is not int):
                raise ValueError("Invalid training input hash/stat provenance")
            row[role] = path
        rows.append(row)
    if training_identity(rows) != selection.get("train_identity_sha256"):
        raise ValueError("Fitted training identity hash is inconsistent")
    expected, subjects, mean, variance = _statistics(inputs, method, normalization.units)
    for actual, fitted in ((normalization.offset, expected.offset), (normalization.scale, expected.scale)):
        if not math.isclose(actual, fitted, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("Fitted normalization disagrees with its recorded training statistics")
    if (selection.get("n_subjects") != len(subjects) or selection.get("n_visits") != len(inputs)
            or selection.get("n_valid_voxels") != sum(item["valid_voxel_count"] for item in inputs)
            or artifact.get("subject_statistics") != subjects):
        raise ValueError("Fitted subject/visit/voxel counts or moments are inconsistent")
    stats = artifact.get("statistics", {})
    expected_stats = dict(canonical_units=normalization.units, decoded_dtype="float32", accumulation_dtype="float64",
                          variance_ddof=0, subject_equal_mean=mean, subject_equal_population_variance=variance)
    if stats != expected_stats:
        raise ValueError("Fitted population statistics are inconsistent")
    for key in ("data_qc", "data_independence"):
        if not isinstance(artifact.get(key), dict) or not artifact[key].get("status"):
            raise ValueError(f"Missing {key} status")
    return normalization, rows


def save_fitted_normalization(artifact, path):
    """Validate then atomically publish a new JSON; never overwrite any file/link."""
    _validate_artifact(artifact)
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Fitted-normalization output must end in .json")
    payload = json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".normalization-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication fails for existing files or symlinks.
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_fitted_normalization(path, *, units=None, rows=None, verify_inputs=True):
    """Return (QSMNormalization, artifact), checking train declarations and bytes.

    ``rows`` may contain holdouts; only train rows are inspected. Setting
    ``verify_inputs=False`` explicitly skips rehashing, e.g. for an embedded
    immutable checkpoint copy; it does not upgrade any QC/independence status.
    """
    with Path(path).open(encoding="utf-8") as stream:
        artifact = json.load(stream)
    normalization, _ = _validate_artifact(artifact)
    if units is not None and units != normalization.units:
        raise ValueError("Requested canonical units conflict with fitted normalization")
    if rows is not None:
        identity = training_identity(rows, artifact["fit_selection"].get("default_input_qsm_units"))
        if identity != artifact["fit_selection"]["train_identity_sha256"]:
            raise ValueError("Fitted normalization belongs to different training identities/paths/units")
    if verify_inputs:
        for item in artifact["inputs"]:
            for entry in item["files"].values():
                actual = _file_snapshot(entry["path"])
                if actual["sha256"] != entry["sha256"] or actual["size_bytes"] != entry["size_bytes"]:
                    raise ValueError(f"Fitted normalization training input changed: {entry['path']}")
    return normalization, artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New fitted-normalization JSON; will not overwrite")
    parser.add_argument("--units", choices=["ppm", "ppb", "stored"], required=True, help="Output/normalization units; stored preserves unverified original values")
    parser.add_argument("--method", choices=METHODS, default="mean_std")
    parser.add_argument("--qsm-units", choices=["ppm", "ppb", "stored"], help="Explicit input fallback only if CSV lacks qsm_units")
    parser.add_argument("--reference-id", help="Actual QSM reference convention; never inferred from fitted offset")
    add_quality_arguments(parser)
    add_independence_arguments(parser)
    args = parser.parse_args()
    if args.units == "stored" and args.qc_report:
        parser.error("stored normalization requires explicit unverified data; physical QC cannot certify unknown units")
    if os.path.lexists(args.output):
        parser.error("Output already exists")
    # Full-manifest identity checks are metadata-only. Do not stat/decode holdouts.
    rows = prepare_quality_rows(read_manifest(args.manifest, validate_inputs=False), args.manifest)
    train = [row for row in rows if row["split"] == "train"]
    if not train:
        parser.error("No training rows")
    data_independence = check_independence(rows, args, content_splits=("train",), check_initialization=False)
    data_qc = check_data_quality(train, args, reference_id=args.reference_id)
    train = bind_input_units(train, data_qc, requested_fallback=args.qsm_units)
    artifact = fit_training_normalization(
        train, units=args.units, method=args.method, default_qsm_units=args.qsm_units,
        data_qc=data_qc, data_independence=data_independence, manifest_path=args.manifest)
    save_fitted_normalization(artifact, args.output)
    fitted = artifact["normalization"]["qsm"]
    print(f"Saved train-only {args.method}: offset={fitted['offset']:.9g}, scale={fitted['scale']:.9g} "
          f"{args.units}; {artifact['fit_selection']['n_subjects']} subjects to {args.output}", flush=True)


if __name__ == "__main__":
    main()

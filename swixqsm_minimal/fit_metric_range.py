"""Fit one fixed PSNR/SSIM range from train QSM and positive reliability masks."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import numpy as np

from .fit_normalization import _file_snapshot, _training_rows, training_identity
from .util.data import load_volume, read_manifest, require_grid
from .util.independence_gate import add_independence_arguments, check_independence
from .util.normalization import unit_factor
from .util.quality_gate import add_quality_arguments, bind_input_units, check_data_quality, prepare_quality_rows


SCHEMA = "swixqsm.train_metric_range"
UNITS = ("ppm", "ppb", "stored")


def fit_metric_range(rows, *, units, method="train_minmax", default_qsm_units=None,
                     data_qc=None, data_independence=None, manifest_path=None):
    """Read only train qsm_path/mask_path; never fit on holdout distributions.

    Units must be explicit. ``stored`` permits no physical-unit conversion and
    always remains unverified storage scale. This does not clip any values.
    """
    if method != "train_minmax":
        raise ValueError("Only method=train_minmax is supported")
    unit_factor(units, units)
    train = _training_rows(rows, default_qsm_units)
    source = Path(manifest_path).expanduser().resolve() if manifest_path is not None else None
    manifest_before = _file_snapshot(source) if source is not None else None
    if source is not None:
        original = prepare_quality_rows(read_manifest(source, validate_inputs=False), source)
        if training_identity(original, default_qsm_units) != training_identity(train):
            raise ValueError("Provided train rows differ from the manifest train identities")
    low, high, inputs = math.inf, -math.inf, []
    for row in train:
        files = {key: _file_snapshot(row[key]) for key in ("qsm_path", "mask_path")}
        image, qsm = load_volume(row["qsm_path"])
        mask_image, mask = load_volume(row["mask_path"])
        require_grid(image, mask_image)
        if image.header.get_xyzt_units()[0] != mask_image.header.get_xyzt_units()[0]:
            raise ValueError("QSM and reliability-mask spatial units disagree")
        if np.any(mask < 0) or not np.any(mask > 0):
            raise ValueError("Reliability mask must be nonnegative and nonempty")
        values = np.asarray(qsm[mask > 0], dtype=np.float64) * unit_factor(row["qsm_units"], units)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite QSM after unit conversion")
        minimum, maximum = float(values.min()), float(values.max())
        low, high = min(low, minimum), max(high, maximum)
        for key in files:
            if _file_snapshot(row[key]) != files[key]:
                raise RuntimeError("Training input changed during metric-range fit: " + row[key])
        inputs.append({**{key: row[key] for key in ("subject_id", "global_subject_id", "case_id", "split", "qsm_units")},
                       "files": files, "valid_voxel_count": int(values.size),
                       "min": minimum, "max": maximum})
    width = high - low
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Training metric range must have a finite positive width; constant targets are invalid")
    for item in inputs:
        for entry in item["files"].values():
            stat = Path(entry["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (entry["size_bytes"], entry["mtime_ns"]):
                raise RuntimeError("Earlier training input changed during metric-range fit")
    if source is not None and _file_snapshot(source) != manifest_before:
        raise RuntimeError("Manifest changed during metric-range fit")
    return dict(schema=SCHEMA, version=1, method=method, units=units,
                unit_status="unverified" if units == "stored" else "declared_not_independently_calibrated",
                data_range=width, train_min=low, train_max=high,
                train_identity_sha256=training_identity(train),
                manifest_sha256=manifest_before["sha256"] if manifest_before else None,
                manifest_path=str(source) if source else None,
                fit_selection=dict(split="train", support="mask_path > 0", n_cases=len(inputs),
                                   n_valid_voxels=sum(item["valid_voxel_count"] for item in inputs),
                                   default_input_qsm_units=default_qsm_units),
                inputs=inputs, data_qc=deepcopy(data_qc) if data_qc else {"status": "unverified"},
                data_independence=deepcopy(data_independence) if data_independence else {"status": "unverified"},
                provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                image_access="Only train qsm_path/mask_path are decoded or hashed by the estimator"),
                limitations=["Global train min/max is sensitive to outliers; no clipping or percentile trimming is applied.",
                             "This fixed range is a metric parameter, not normalization, QSM re-referencing or calibration.",
                             "Val/test distributions are not used; their values may exceed the fitted training bounds.",
                             "stored means unverified original storage scale, never ppm/ppb; declarations/QC status are not upgraded."])


def save_metric_range(artifact, path):
    """Publish a new JSON; never replace existing files or symbolic links."""
    if (artifact.get("schema") != SCHEMA or type(artifact.get("version")) is not int or artifact["version"] != 1
            or artifact.get("method") != "train_minmax" or artifact.get("units") not in UNITS):
        raise ValueError("Invalid train metric-range artifact")
    low, high, width = (artifact[key] for key in ("train_min", "train_max", "data_range"))
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (low, high, width)) or width <= 0 or width != high - low):
        raise ValueError("Metric-range bounds and positive width must agree")
    payload = json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New JSON, never overwritten")
    parser.add_argument("--units", choices=UNITS, required=True)
    parser.add_argument("--qsm-units", choices=UNITS, help="Explicit fallback only when source units are absent")
    parser.add_argument("--method", choices=["train_minmax"], default="train_minmax")
    add_quality_arguments(parser)
    add_independence_arguments(parser)
    args = parser.parse_args(argv)
    if os.path.lexists(args.output):
        parser.error("Output already exists")
    rows = prepare_quality_rows(read_manifest(args.manifest, validate_inputs=False), args.manifest)
    train = [row for row in rows if row["split"] == "train"]
    if not train:
        parser.error("No training rows")
    independence = check_independence(rows, args, content_splits=("train",), check_initialization=False)
    qc = check_data_quality(train, args)
    train = bind_input_units(train, qc, requested_fallback=args.qsm_units)
    artifact = fit_metric_range(train, units=args.units, method=args.method, default_qsm_units=args.qsm_units,
                                data_qc=qc, data_independence=independence, manifest_path=args.manifest)
    save_metric_range(artifact, args.output)
    print(f"Saved fixed train-only range={artifact['data_range']:.9g} {args.units}: {args.output}")


if __name__ == "__main__":
    main()

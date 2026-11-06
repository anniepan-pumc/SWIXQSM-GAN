"""Create a new patient-level train/val split from train, preserving test rows.

Run from the repository root: python -m swixqsm_minimal.split_training --help.
Only CSV metadata are read; image headers, arrays, QC evidence and reference
files are not opened. This is a new development split, not a historical split
reconstruction or an independence/registration-quality audit.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from .util.data import read_manifest
from .util.quality_gate import prepare_quality_rows


MAIN_PATHS = ("swi_path", "qsm_path", "swi_mask_path", "mask_path")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _absolute_path(value, parent):
    path = Path(value).expanduser()
    return str((parent/path if not path.is_absolute() else path).resolve())


def _resolved_metadata(rows, originals, manifest):
    """Resolve known QC fields and declared *_path / JSON *_paths metadata.

    Re-resolve from original CSV values so leading '~' also works for the
    primary paths (the legacy manifest parser does not expand it itself).
    No referenced file is opened or checked for existence.
    """
    resolved = prepare_quality_rows(rows, manifest)
    path_fields = set()
    for row, original in zip(resolved, originals):
        for key, value in original.items():
            if not value or not value.strip():
                continue
            if key.endswith("_path"):
                row[key] = _absolute_path(value, manifest.parent)
                path_fields.add(key)
            elif key.endswith("_paths"):
                try:
                    paths = json.loads(value)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{key} must be a JSON array of path strings") from exc
                if not isinstance(paths, list) or any(not isinstance(path, str) or not path.strip()
                                                      for path in paths):
                    raise ValueError(f"{key} must be a JSON array of nonempty path strings")
                row[key] = json.dumps([_absolute_path(path, manifest.parent) for path in paths],
                                      ensure_ascii=False)
                path_fields.add(key)
    paths = [row["swi_path"] for row in resolved]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate resolved SWI path; keep one reference per input")
    return resolved, sorted(path_fields)


def create_split(manifest, output, *, val_fraction=0.1, seed=42):
    """Write manifest.csv and a hash-bound metadata-only split_record.json."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if isinstance(val_fraction, (bool, np.bool_)):
        raise ValueError("val_fraction must be finite and strictly between 0 and 1")
    try:
        val_fraction = float(val_fraction)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("val_fraction must be finite and strictly between 0 and 1") from exc
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be finite and strictly between 0 and 1")

    manifest = Path(manifest).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Split output already exists; use a new directory")
    source_bytes = manifest.read_bytes()
    reader = csv.DictReader(io.StringIO(source_bytes.decode("utf-8"), newline=""))
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)) or any(not field for field in fields):
        raise ValueError("Manifest needs unique, nonempty CSV column names")
    originals = list(reader)
    if not originals or any(None in row or any(value is None for value in row.values()) for row in originals):
        raise ValueError("Manifest is empty or has rows with a mismatched number of columns")
    required = {"subject_id", "global_subject_id", "split", *MAIN_PATHS}
    if not required.issubset(fields):
        raise ValueError(f"Manifest requires {sorted(required)}")

    subject_global, subject_split, group_split, group_aliases = {}, {}, {}, {}
    for row in originals:
        for key in ("subject_id", "global_subject_id"):
            value = row[key]
            if not value or value != value.strip():
                raise ValueError(f"Every row needs a nonempty {key} without surrounding whitespace")
        if row["split"] not in {"train", "test"}:
            raise ValueError("Input must contain only original train/test rows; existing val is not re-split")
        if any(not row[key].strip() for key in MAIN_PATHS):
            raise ValueError("All four primary image-path fields must be nonempty")
        sid, gid, split = row["subject_id"], row["global_subject_id"], row["split"]
        if sid in subject_global and subject_global[sid] != gid:
            raise ValueError(f"subject_id maps to multiple global_subject_id values: {sid}")
        if sid in subject_split and subject_split[sid] != split:
            raise ValueError(f"subject_id crosses train/test splits: {sid}")
        if gid in group_split and group_split[gid] != split:
            raise ValueError(f"global_subject_id crosses train/test splits: {gid}")
        subject_global[sid], subject_split[sid], group_split[gid] = gid, split, split
        group_aliases.setdefault(gid, set()).add(sid)

    rows = read_manifest(manifest, validate_inputs=False)
    if manifest.read_bytes() != source_bytes:
        raise RuntimeError("Source manifest changed while being parsed; no split was written")
    rows, path_fields = _resolved_metadata(rows, originals, manifest)
    train_ids = sorted(gid for gid, split in group_split.items() if split == "train")
    test_ids = sorted(gid for gid, split in group_split.items() if split == "test")
    if len(train_ids) < 2:
        raise ValueError("Original training pool needs at least two unique global_subject_id groups")
    requested_n_val = math.ceil(len(train_ids) * val_fraction)
    n_val = min(len(train_ids)-1, max(1, requested_n_val))
    shuffled = np.random.default_rng(int(seed)).permutation(np.asarray(train_ids, dtype=object)).tolist()
    validation_ids = set(shuffled[:n_val])
    test_before = [dict(row) for row in rows if row["split"] == "test"]
    result = [{**row, "split": "val" if row["split"] == "train" and
               row["global_subject_id"] in validation_ids else row["split"]} for row in rows]
    test_after = [row for row in result if row["split"] == "test"]
    if test_before != test_after:
        raise RuntimeError("Internal error: protected test metadata changed")
    aliases = {gid: sorted(group_aliases[gid]) for gid in sorted(group_split)}
    split_groups = {split: sorted({row["global_subject_id"] for row in result if row["split"] == split})
                    for split in ("train", "val", "test")}
    record = {
        "schema": "swixqsm.new_training_split", "version": 1,
        "source_manifest_path": str(manifest),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "grouping_key": "global_subject_id",
        "grouping_scope": "All visits and subject_id aliases sharing a global_subject_id stay together, across cohorts",
        "seed": int(seed), "val_fraction_requested": val_fraction,
        "algorithm": "Sort original train global_subject_id strings; numpy.default_rng(seed).permutation; first n_val groups become val",
        "numpy_version": np.__version__, "bit_generator": "PCG64",
        "rounding": "n_val = min(n_train_groups - 1, max(1, ceil(n_train_groups * val_fraction)))",
        "n_val_before_clamp": requested_n_val,
        "original_train_group_ids": train_ids, "original_test_group_ids": test_ids,
        "group_ids": split_groups,
        "group_counts": {split: len(ids) for split, ids in split_groups.items()},
        "row_counts": {split: sum(row["split"] == split for row in result) for split in split_groups},
        "subject_id_aliases_by_global_subject_id": aliases,
        "test_membership_unchanged": True, "test_rows_metadata_unchanged_after_path_resolution": True,
        "test_resolved_rows_sha256_before": _digest(test_before),
        "test_resolved_rows_sha256_after": _digest(test_after),
        "absolute_path_columns": path_fields,
        "path_policy": "Resolve nonempty *_path columns and JSON-array *_paths columns against the source CSV parent, preserving all other fields",
        "output_row_order": "Same as source CSV; only eligible original train split values change",
        "image_header_reads": 0, "image_array_reads": 0,
        "read_scope": "CSV metadata only; no input-file existence checks, image/header reads, QC evidence or reference-file reads",
        "scope": "New development split for new training; not a reconstruction of the historical training split",
        "limitations": [
            "global_subject_id correctness is supplied by the user; undeclared aliases are not discovered",
            "Metadata grouping is not a content-duplicate, registration, unit or reference audit; run the existing audits before training",
            "Only the new val split may select epochs/hyperparameters; original test and external cohorts remain evaluation-only",
            "Existing test performance has already informed manuscript revision; do not call new comparisons a completely feedback-naive first test",
        ],
    }
    destination = io.StringIO(newline="")
    writer = csv.DictWriter(destination, fieldnames=fields)
    writer.writeheader()
    writer.writerows(result)
    manifest_bytes = destination.getvalue().encode("utf-8")
    record["output_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    # mkdir is the exclusive publication claim: an existing empty directory is
    # also rejected, rather than potentially replaced by a staging rename.
    output.mkdir(parents=True, exist_ok=False)
    with (output/"manifest.csv").open("xb") as stream:
        stream.write(manifest_bytes)
    with (output/"split_record.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Original CSV containing train/test only")
    parser.add_argument("--output", required=True, help="New, nonexisting split directory")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of original training patient groups")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        record = create_split(args.manifest, args.output, val_fraction=args.val_fraction, seed=args.seed)
    except (ValueError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"New patient split {record['group_counts']}; protected test unchanged; {Path(args.output)/'manifest.csv'}")


if __name__ == "__main__":
    main()

"""Prepare explicit SWI-only input masks from declared skull-stripped volumes.

This is support extraction, not anatomical segmentation or registration QC.
QSM label intensities are never opened. A fixed optional intersection prepares
training/evaluation validity before any model or error-based comparison.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from scipy import ndimage


ROLES = ("swi_path", "qsm_path", "swi_mask_path", "mask_path")
DROP_METADATA_FIELDS = {"split", "source_case_path", "participant_id", "directory_present",
                        "case_origin", "role_candidates_json", "issues_json", "complete_required_files"}
GRID_ATOL = 1e-4


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat(path):
    value = Path(path).stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns,
            "inode": value.st_ino, "device": value.st_dev}


def _resolve_paths(row, parent):
    result = dict(row)
    for key, value in row.items():
        if not value or not value.strip():
            continue
        if key.endswith("_path"):
            path = Path(value).expanduser()
            result[key] = str((parent/path if not path.is_absolute() else path).resolve())
        elif key.endswith("_paths"):
            try:
                paths = json.loads(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{key} must be a JSON array of path strings") from exc
            if not isinstance(paths, list) or any(not isinstance(path, str) or not path.strip() for path in paths):
                raise ValueError(f"{key} must be a JSON array of nonempty path strings")
            result[key] = json.dumps([str((parent/Path(path).expanduser()
                                          if not Path(path).expanduser().is_absolute()
                                          else Path(path).expanduser()).resolve()) for path in paths], ensure_ascii=False)
    return result


def _read_rows(manifest, output, policy):
    data = manifest.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames
    required = {"case_id", "original_split", "subject_id", "global_subject_id", *ROLES}
    if (not fields or len(fields) != len(set(fields)) or any(not field for field in fields)
            or not required.issubset(fields)):
        raise ValueError(f"Manifest needs unique columns including {sorted(required)}")
    rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Manifest is empty or has malformed CSV rows")
    groups, subjects, sid_to_gid, case_ids, paths = {}, {}, {}, set(), {}
    resolved = []
    for row in rows:
        row = _resolve_paths(row, manifest.parent)
        for key in ("case_id", "subject_id", "global_subject_id"):
            if not row[key] or row[key] != row[key].strip():
                raise ValueError(f"Every case needs a nonempty {key} without surrounding whitespace")
        split = row["original_split"]
        if split not in ("train", "test") or row.get("split", split) != split:
            raise ValueError("Input must retain matching original train/test assignments; validate is test")
        if row["case_id"] in case_ids:
            raise ValueError("Duplicate case_id, including across original splits")
        case_ids.add(row["case_id"])
        for key, seen in (("global_subject_id", groups), ("subject_id", subjects)):
            if row[key] in seen and seen[row[key]] != split:
                raise ValueError(f"{key} crosses original train/test boundary")
            seen[row[key]] = split
        if row["subject_id"] in sid_to_gid and sid_to_gid[row["subject_id"]] != row["global_subject_id"]:
            raise ValueError("One subject_id maps to multiple global identities")
        sid_to_gid[row["subject_id"]] = row["global_subject_id"]
        # No image is opened until every required file in the whole table exists.
        for key in ("swi_path", "mask_path") if policy == "intersect" else ("swi_path",):
            if not row[key] or not Path(row[key]).is_file():
                raise FileNotFoundError(f"Required {key} missing for {row['case_id']}: {row[key]}")
        for key in ROLES:
            value = row[key]
            if value:
                if output == Path(value) or output in Path(value).parents:
                    raise ValueError("A source path lies inside the requested output directory")
                if (key, value) in paths:
                    raise ValueError(f"Duplicate {key} across cases")
                paths[(key, value)] = row["case_id"]
        for key, value in row.items():
            values = ([value] if key.endswith("_path") else json.loads(value)
                      if key.endswith("_paths") and value.strip() else [])
            if any(value and (output == Path(value) or output in Path(value).parents) for value in values):
                raise ValueError("A source or evidence path lies inside the requested output directory")
        if policy == "intersect" and row["swi_path"] == row["mask_path"]:
            raise ValueError("SWI and reference validity roles must not reuse one file")
        resolved.append(row)
    return resolved, fields, data


def _load_finite(path, role):
    image = nib.load(path)
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ValueError(f"{role} must be a NIfTI image")
    values = image.get_fdata(dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"{role} must be a finite scaled float32 3D volume")
    affine = np.asarray(image.affine)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], (0., 0., 0., 1.), rtol=0., atol=1e-8)
            or np.linalg.det(affine[:3, :3]) == 0):
        raise ValueError(f"{role} has an invalid spatial affine")
    return image, values


def _save_mask(path, values, source):
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_intent("none")
    header["cal_min"], header["cal_max"] = 0., 1.
    header["descrip"] = b"SWI support mask; not anatomical segmentation"
    image = type(source)(np.asarray(values, dtype=np.uint8), source.affine, header=header)
    qform, qcode = source.get_qform(coded=True)
    sform, scode = source.get_sform(coded=True)
    image.set_qform(qform, int(qcode))
    image.set_sform(sform, int(scode))
    image.header.set_slope_inter(1., 0.)
    nib.save(image, path)


def prepare_swi_masks(manifest, output, *, skull_stripped=False, fill_holes=True,
                      valid_mask_policy="intersect"):
    """Return a report containing final absolute metadata_csv and report_path.

    Input is builder's original_train_test.draft.csv, with all rows retained.
    skull_stripped must be literal True: no inference from an intensity range.
    SWI uses scaled float32 finite nonzero support (including negative values),
    optionally 3D binary_fill_holes with SciPy's default 6-connectivity. There
    is no dilation, intensity threshold, QSM-label read or segmentation model.

    intersect reads only the original mask_path (besides SWI), checks grid and
    nonnegative finite values and saves (reference_valid > 0) & input_support.
    preserve leaves mask_path untouched and never opens/hashes its contents.
    No case is dropped. An incomplete or failed case aborts the entire output.
    All files are staged then published together; metadata paths name the final
    directory. Source files are never rewritten, and later data QC is required.
    """
    if skull_stripped is not True:
        raise ValueError("Explicit skull_stripped=True is required; do not infer a brain mask from unstripped SWI")
    if type(fill_holes) is not bool or valid_mask_policy not in ("intersect", "preserve"):
        raise ValueError("fill_holes must be boolean and valid_mask_policy intersect or preserve")
    manifest = Path(manifest).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Mask output already exists; use a new directory")
    if output == manifest or output in manifest.parents:
        raise ValueError("Manifest source conflicts with requested output")
    rows, fields, manifest_bytes = _read_rows(manifest, output, valid_mask_policy)
    source_paths = {row["swi_path"] for row in rows}
    if valid_mask_policy == "intersect":
        source_paths.update(row["mask_path"] for row in rows)
    before = {path: _stat(path) for path in sorted(source_paths)}
    source_hashes = {path: _file_sha(path) for path in sorted(source_paths)}
    if any(_stat(path) != before[path] for path in before):
        raise RuntimeError("Source metadata changed during hashing; refusing to prepare masks")
    algorithm = {"input": "scaled float32 SWI", "support": "finite values != 0, including negative values",
                 "fill_holes_3d": fill_holes, "hole_connectivity": 6 if fill_holes else None,
                 "dilation": False, "intensity_threshold": None, "qsm_label_intensities_used": False,
                 "valid_mask_policy": valid_mask_policy, "grid_affine_absolute_tolerance": GRID_ATOL}
    cases, prepared_rows = [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".swixqsm-swi-masks-", dir=output.parent) as temporary:
        staging = Path(temporary)/"prepared"
        staging.mkdir()
        components = set()
        for row in rows:
            case_id, split = row["case_id"], row["original_split"]
            component = "case_"+hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:20]
            if (split, component) in components:
                raise ValueError("Output case-name collision")
            components.add((split, component))
            relative = Path(split)/component
            working = staging/relative
            final = output/relative
            working.mkdir(parents=True)
            image, swi = _load_finite(row["swi_path"], "SWI")
            raw_support = swi != 0
            raw_n = int(raw_support.sum())
            if raw_n == 0:
                raise ValueError(f"Empty SWI nonzero support for {case_id}")
            support = ndimage.binary_fill_holes(raw_support) if fill_holes else raw_support
            n_support = int(support.sum())
            input_mask_path = final/"swi_brain_mask.nii.gz"
            _save_mask(working/input_mask_path.name, support, image)
            case = {"case_id": case_id, "original_split": split,
                    "subject_id": row["subject_id"], "global_subject_id": row["global_subject_id"],
                    "swi_path": row["swi_path"], "swi_sha256": source_hashes[row["swi_path"]],
                    "shape": list(swi.shape), "affine": image.affine.tolist(),
                    "spatial_units": image.header.get_xyzt_units()[0],
                    "raw_nonzero_voxels": raw_n, "input_mask_voxels": n_support,
                    "holes_filled_voxels": n_support-raw_n,
                    "input_mask_fraction": n_support/int(np.prod(swi.shape)),
                    "input_mask_path": str(input_mask_path),
                    "input_mask_sha256": _file_sha(working/input_mask_path.name),
                    "original_valid_mask_path": row["mask_path"],
                    "original_valid_mask_sha256": None, "original_valid_voxels": None,
                    "final_valid_voxels": None, "excluded_valid_voxels": None}
            new_row = {key: value for key, value in row.items() if key not in DROP_METADATA_FIELDS}
            new_row["swi_mask_path"] = str(input_mask_path)
            new_row["input_mask_source"] = (
                "SWI-only scaled nonzero support"+(" with 3D hole filling" if fill_holes else "")+
                "; user declares original SWI already skull-stripped; provenance: "+str(final/"swi_brain_mask.json"))
            if valid_mask_policy == "intersect":
                valid_image, valid = _load_finite(row["mask_path"], "Reference validity mask")
                if (valid.shape != swi.shape or not np.allclose(valid_image.affine, image.affine, atol=GRID_ATOL, rtol=0.)
                        or valid_image.header.get_xyzt_units()[0] != image.header.get_xyzt_units()[0]):
                    raise ValueError(f"Reference validity mask grid differs from SWI for {case_id}")
                if np.any(valid < 0):
                    raise ValueError(f"Reference validity mask must be nonnegative for {case_id}")
                membership = valid > 0
                original_n = int(membership.sum())
                intersected = membership & support
                final_n = int(intersected.sum())
                if original_n == 0 or final_n == 0:
                    raise ValueError(f"Empty original or intersected reference validity for {case_id}")
                valid_path = final/"reference_valid_intersect_swi.nii.gz"
                _save_mask(working/valid_path.name, intersected, image)
                new_row["mask_path"] = str(valid_path)
                case.update(original_valid_mask_sha256=source_hashes[row["mask_path"]],
                            original_valid_voxels=original_n, final_valid_voxels=final_n,
                            excluded_valid_voxels=original_n-final_n,
                            final_valid_mask_path=str(valid_path),
                            final_valid_mask_sha256=_file_sha(working/valid_path.name))
            else:
                case.update(final_valid_mask_path=row["mask_path"], final_valid_mask_sha256=None,
                            preserve_note="Original mask was not opened, hashed or changed; later QC must check containment")
            provenance = {"schema": "swixqsm.swi_support_mask_provenance", "version": 1,
                          "skull_stripped_declaration": True, "algorithm": algorithm,
                          "case": case, "anatomical_segmentation": False,
                          "qsm_reliability_validated": False, "registration_qc_passed": False,
                          "limitations": "Nonzero/filled SWI support is not an anatomical segmentation; intersection does not establish QSM reconstruction reliability or registration"}
            serialized_provenance = json.dumps(provenance, indent=2, allow_nan=False)+"\n"
            (working/"swi_brain_mask.json").write_text(serialized_provenance, encoding="utf-8")
            case["provenance_path"] = str(final/"swi_brain_mask.json")
            case["provenance_sha256"] = _file_sha(working/"swi_brain_mask.json")
            if valid_mask_policy == "intersect":
                (working/"reference_valid_intersect_swi.json").write_text(serialized_provenance, encoding="utf-8")
                case["valid_mask_provenance_path"] = str(final/"reference_valid_intersect_swi.json")
                case["valid_mask_provenance_sha256"] = _file_sha(working/"reference_valid_intersect_swi.json")
            cases.append(case)
            prepared_rows.append(new_row)
        if manifest.read_bytes() != manifest_bytes or any(_stat(path) != before[path] for path in before):
            raise RuntimeError("Source manifest/image metadata changed during preparation; no output published")
        metadata_fields = [key for key in fields if key not in DROP_METADATA_FIELDS]
        if "input_mask_source" not in metadata_fields:
            metadata_fields.append("input_mask_source")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=metadata_fields)
        writer.writeheader()
        writer.writerows(prepared_rows)
        metadata_bytes = stream.getvalue().encode("utf-8")
        (staging/"metadata.csv").write_bytes(metadata_bytes)
        report = {"schema": "swixqsm.swi_support_masks", "version": 1,
                  "status": "prepared_not_qc_verified", "manifest_path": str(manifest),
                  "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "metadata_csv": str(output/"metadata.csv"),
                  "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                  "report_path": str(output/"report.json"), "n_cases": len(cases),
                  "case_counts": {split: sum(case["original_split"] == split for case in cases) for split in ("train", "test")},
                  "skull_stripped_declaration": True, "algorithm": algorithm,
                  "source_sha256": source_hashes, "source_stat_before_after_unchanged": True,
                  "stat_check_limitation": "Source SHA256 recorded before decoding; later stat signatures checked, no claim to detect same-stat concurrent rewrites",
                  "test_membership_preserved": True, "qsm_label_intensities_read": False,
                  "dropped_metadata_columns": sorted(set(fields) & DROP_METADATA_FIELDS),
                  "cases": cases,
                  "limitations": ["User declaration that source SWI is already skull-stripped is required, not independently established",
                                  "No anatomical segmentation, registration, QSM unit or susceptibility-zero verification",
                                  "Intersection policy is fixed before training/error evaluation; not selected from model performance",
                                  "Regenerate QC/content-independence reports for the new mask paths before training"]}
        (staging/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError("Output appeared during mask preparation; refusing to overwrite it")
        staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New directory; all cases publish together")
    parser.add_argument("--skull-stripped", action="store_true", help="Actual user declaration; required to execute")
    parser.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--valid-mask-policy", choices=["intersect", "preserve"], default="intersect")
    args = parser.parse_args()
    try:
        report = prepare_swi_masks(args.manifest, args.output, skull_stripped=args.skull_stripped,
                                   fill_holes=args.fill_holes, valid_mask_policy=args.valid_mask_policy)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": report["status"], "n_cases": report["n_cases"],
                      "metadata_csv": report["metadata_csv"], "report_path": report["report_path"]}))


if __name__ == "__main__":
    main()

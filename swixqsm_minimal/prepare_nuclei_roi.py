"""Bind NextBrain RN/SN/DN labels to a separate, deferred evaluation ROI CSV.

Only already registered integer labels are encoded. No registration or intensity
resampling occurs; incompatible evaluation/native grids are refused.
"""
import argparse
import csv
import json
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

from .evaluate_native_nuclei import REGIONS, resolve, sha, write_csv


RIGHT_OFFSET = 1_000_000


def build(cases_path, manifest_path, output, *, split, target, source_root, grid_column="qsm_path"):
    cases_path, manifest_path, output = Path(cases_path).resolve(), Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if split not in {"val", "test"} or target not in {"gre", "swi"}:
        raise ValueError("Select val/test and gre/swi")
    original_manifest = manifest_path.read_bytes()
    cases_bytes = cases_path.read_bytes()
    cases = json.loads(cases_bytes)
    known = {}
    for row in cases:
        if row["case_id"] in known:
            if row["registration_report"] != known[row["case_id"]]["registration_report"]:
                raise ValueError("Different ROI registration for the same case")
        known[row["case_id"]] = row
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        selected = [row for row in csv.DictReader(stream) if row["split"] == split]
    if not selected or len({r["case_id"] for r in selected}) != len(selected):
        raise ValueError("Selected split must have unique nonempty cases")
    spec = {"schema": "swixqsm.roi_spec", "version": 1, "id": "nextbrain_allen_RN_SN_DN_15_native_v1",
            "definition": "NextBrain AllenAtlasLUT RN385, SNr310, SNc352, DN721; SN union310+352. Left target IDs unchanged; right target IDs add 1_000_000. Other atlas labels omitted. Left/right/bilateral are separate ROI definitions, not independent participants.",
            "label_source": "NextBrain AllenAtlasLUT; original hemisphere atlas labels and hash-bound scanner-identity v2 native-grid registration; anatomical and registration QC pending",
            "min_coverage": 1., "min_voxels": 1, "regions": []}
    for name, ids in REGIONS.items():
        for side, offsets in (("L", [0]), ("R", [RIGHT_OFFSET]), ("bilateral", [0, RIGHT_OFFSET])):
            spec["regions"].append({"name": name + "_" + side,
                                    "labels": [label + offset for offset in offsets for label in ids]})
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger, overlay = {}, []
    with tempfile.TemporaryDirectory(prefix=".nuclei-labels-", dir=output.parent) as tmp:
        stage = Path(tmp) / "results"; stage.mkdir(); (stage / "labels").mkdir()
        for row in selected:
            entry = {"case_id": row["case_id"], "roi_labels_path": "", "roi_label_source": "",
                     "roi_review": "pending", "roi_reviewer": "", "roi_review_date": ""}
            if row["case_id"] in known:
                case = known[row["case_id"]]
                rp = resolve(case["registration_report"], source_root)
                rep = json.loads(rp.read_text()); digest = sha(rp); ledger[str(rp)] = digest
                if (rep["case_id"] != row["case_id"] or rep["registration_status"] != "candidate_completed_pending_QC"
                        or rep.get("protocol_revision") != "scanner_identity_initialization_v2"):
                    raise ValueError("Incomplete or rejected ROI registration")
                if rep["subject_id"] != row.get("subject_id") or rep["cohort"] != row.get("cohort"):
                    raise ValueError("Evaluation participant/cohort differs from the registered ROI source")
                qsm_path = resolve(row[grid_column], manifest_path.parent)
                source = rep["inputs"]["qsm_grid" if target == "gre" else "swi"]
                if sha(qsm_path) != source["sha256"]:
                    raise ValueError("Evaluation grid image is not the hash-bound registration target; bind the actual native qsm_path/real swi_path")
                ledger[str(qsm_path)] = source["sha256"]
                grid = nib.load(qsm_path)
                encoded = np.zeros(grid.shape, dtype=np.int32)
                for side, offset in (("left", 0), ("right", RIGHT_OFFSET)):
                    source = rep["targets"][target]["labels"][side]
                    path = resolve(source["path"], source_root)
                    if sha(path) != source["sha256"]:
                        raise ValueError("Registered label hash changed")
                    ledger[str(path)] = source["sha256"]
                    image = nib.load(path)
                    if image.shape != grid.shape or not np.allclose(image.affine, grid.affine, rtol=0, atol=1e-5):
                        raise ValueError("Evaluation and native label grids differ; use dual-native evaluation rather than copying headers")
                    unit_a, unit_b = image.header.get_xyzt_units()[0], grid.header.get_xyzt_units()[0]
                    if unit_a in {"mm", "meter", "micron"} and unit_b in {"mm", "meter", "micron"} and unit_a != unit_b:
                        raise ValueError("Native label/target spatial units disagree")
                    array = image.get_fdata(dtype=np.float32, caching="unchanged")
                    if not np.isfinite(array).all() or not np.array_equal(array, np.rint(array)) or np.any(array < 0):
                        raise ValueError("Invalid registered integer labels")
                    wanted = np.isin(array, [385, 310, 352, 721])
                    if np.any((encoded > 0) & wanted):
                        raise ValueError("Target nuclei overlap between hemisphere labels")
                    encoded[wanted] = array[wanted].astype(np.int32) + offset
                filename = sha(rp)[:12] + "_nuclei.nii.gz"
                header = grid.header.copy(); header.set_data_dtype(np.int32)
                image = nib.Nifti1Image(encoded, grid.affine, header)
                image.set_qform(grid.get_qform(), int(grid.header["qform_code"]))
                image.set_sform(grid.get_sform(), int(grid.header["sform_code"]))
                nib.save(image, stage / "labels" / filename)
                entry["roi_labels_path"] = "labels/" + filename
                entry["roi_label_source"] = "NextBrain AllenAtlasLUT; registration_report_sha256=" + digest + "; pending_anatomical_and_registration_QC"
            overlay.append(entry)
        if manifest_path.read_bytes() != original_manifest or cases_path.read_bytes() != cases_bytes:
            raise ValueError("Input manifest changed during preparation")
        if any(sha(p) != digest for p, digest in ledger.items()):
            raise ValueError("Input labels/registration changed during preparation")
        write_csv(stage / "roi_manifest.csv", overlay)
        (stage / "roi_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        report = {"schema": "swixqsm.nextbrain_roi_binding", "version": 1, "split": split, "target_grid": target,
                  "grid_column": grid_column,
                  "input_manifest": str(manifest_path), "input_manifest_sha256": sha(manifest_path),
                  "cases_manifest": str(cases_path), "cases_manifest_sha256": sha(cases_path),
                  "n_cases": len(overlay), "n_with_labels": sum(bool(r["roi_labels_path"]) for r in overlay),
                  "n_missing_labels_retained": sum(not r["roi_labels_path"] for r in overlay),
                  "sources": ledger, "review_policy": "deferred", "anatomical_qc_status": "pending",
                  "registration_qc_status": "pending", "input_manifest_modified": False,
                  "registration_or_intensity_resampling": False, "right_offset": RIGHT_OFFSET}
        (stage / "binding_report.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--target", choices=["gre", "swi"], default="gre")
    parser.add_argument("--grid-column", default="qsm_path", help="Header grid column in the manifest; use swi_path only for a separate native-prediction label binding")
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.cases, args.evaluation_manifest, args.output, split=args.split,
                   target=args.target, source_root=args.source_root, grid_column=args.grid_column)
    print(json.dumps({key: result[key] for key in ("n_cases", "n_with_labels", "n_missing_labels_retained")}))


if __name__ == "__main__":
    main()

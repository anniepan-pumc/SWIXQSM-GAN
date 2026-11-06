"""Deferred RN/SN/DN scalars in two independently registered native grids.

This deliberately does not calculate voxel errors between GRE and SWI grids.
It accepts hash-bound NextBrain registration and prediction records, without
training, intensity fitting, header repair or claims of anatomical acceptance.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

from .util.agreement import paired_agreement


REGIONS = {"RN": (385,), "SN": (310, 352), "SNr": (310,), "SNc": (352,), "DN": (721,)}
FRAME = "dual_native_grid_ROI_means_not_voxel_errors"
FLAGS = {"units": "stored", "value_scale": "original_stored_scale_unverified",
         "unit_status": "unverified_original_storage", "reference_zero_status": "unverified",
         "roi_review_policy": "deferred", "anatomical_qc_status": "pending",
         "registration_qc_status": "pending", "formal_accuracy_claim": False,
         "research_only": True, "measurement": FRAME}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def resolve(path, base):
    path = Path(path).expanduser()
    path = (path if path.is_absolute() else Path(base) / path).resolve()
    if "primary_cohort" in str(path).lower():
        raise ValueError("This external pilot command does not access restricted cohort imaging")
    return path


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cohort"], row["model"], row["region"], row["side"])].append(row)
    summaries = []
    for (cohort, model, region, side), values in sorted(groups.items()):
        valid = [v for v in values if v["status"] == "numeric_candidate_QC_pending"]
        reference = [v["reference_mean_raw"] for v in valid]
        prediction = [v["prediction_mean_raw"] for v in valid]
        agreement = paired_agreement(reference, prediction, bootstrap_reps=0)
        result = {"cohort": cohort, "model": model, "region": region, "side": side,
                  "n_expected_participants": len(values), "n_defined_participants": len(valid),
                  "n_excluded_participants": len(values) - len(valid),
                  "n_paired_comparison_participants": sum(v.get("paired_comparison_eligible", False) for v in values), **FLAGS}
        for key in ("reference_mean_raw", "prediction_mean_raw", "mean_difference_raw", "roi_mean_abs_error_raw",
                    "roi_mean_squared_error_raw", "zero_roi_mean_abs_error_raw"):
            array = np.asarray([v[key] for v in valid], dtype=float)
            result[key + "_mean"] = float(array.mean()) if len(array) else None
            result[key + "_sd"] = float(array.std(ddof=1)) if len(array) > 1 else None
        result["roi_mean_error_rmse_raw"] = (float(np.sqrt(np.mean([v["roi_mean_squared_error_raw"] for v in valid])))
                                             if valid else None)
        result["fraction_better_than_numeric_zero"] = (sum(v["roi_mean_abs_error_raw"] < v["zero_roi_mean_abs_error_raw"]
                                                         for v in valid) / len(valid) if valid else None)
        for key, value in agreement["estimates"].items():
            result[key] = value
            result[key + "_status"] = agreement["metric_status"][key]
        result["confidence_intervals_status"] = "disabled_small_pilot_pending_QC"
        result["aggregation"] = "one_case_per_participant_equal_weight_separate_cohorts_regions_sides"
        summaries.append(result)
    return summaries


def evaluate(cases, output, *, source_root, reason, expected_models=None):
    if not reason.strip() or not cases:
        raise ValueError("Nonempty cases and an explicit deferred-review reason are required")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    ledger, rows, seen, participants, model_cases = {}, [], set(), {}, defaultdict(set)
    case_registration = {}

    def record(path, expected=None):
        path = resolve(path, source_root)
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError("Source hash changed: " + str(path))
        ledger[str(path)] = digest
        return path

    def load(path, expected=None):
        path = record(path, expected)
        image = nib.load(path)
        array = image.get_fdata(dtype=np.float32, caching="unchanged")
        if array.ndim != 3 or not np.isfinite(array).all() or not np.isfinite(image.affine).all():
            raise ValueError("Nonfinite or non-3D image: " + str(path))
        return image, array

    def same_grid(first, second):
        if first.shape != second.shape or not np.allclose(first.affine, second.affine, rtol=0, atol=1e-5):
            raise ValueError("ROI labels are not on their declared native target grid")
        known = {"mm", "meter", "micron"}
        a, b = first.header.get_xyzt_units()[0], second.header.get_xyzt_units()[0]
        if a in known and b in known and a != b:
            raise ValueError("Native-grid spatial units disagree")

    for case in cases:
        model = case.get("model", "historical_gen_new_19_fixed_profile")
        if not isinstance(model, str) or not model.strip() or model == "zero":
            raise ValueError("A nonempty model name other than reserved zero is required")
        identity = case["case_id"]
        if (model, identity) in seen:
            raise ValueError("Duplicate model/case")
        seen.add((model, identity)); model_cases[model].add(identity)
        participant = (case["cohort"], case["subject_id"])
        if participant in participants and participants[participant] != identity:
            raise ValueError("Repeated visits require explicit aggregation before this pilot command")
        participants[participant] = identity
        report_path = record(case["registration_report"])
        if identity in case_registration and case_registration[identity] != ledger[str(report_path)]:
            raise ValueError("Models must use the same reference, registration and ROI support per case")
        case_registration[identity] = ledger[str(report_path)]
        report = json.loads(report_path.read_text())
        if (report["case_id"] != identity or report["cohort"] != case["cohort"]
                or report["subject_id"] != case["subject_id"]
                or report["registration_status"] != "candidate_completed_pending_QC"
                or report.get("protocol_revision") != "scanner_identity_initialization_v2"):
            raise ValueError("Wrong/incomplete registration or rejected v1 registration")
        provenance_path = record(case["prediction_provenance"])
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("units", "stored") not in {"stored", "original_stored_scale_unverified"}:
            raise ValueError("This command requires stored prediction values without inferred physical conversion")
        checkpoint = provenance.get("actual_new_checkpoint_sha256") or provenance.get("checkpoint_sha256")
        if (not checkpoint or len(checkpoint) != 64 or provenance.get("cohort") != case["cohort"]
                or provenance.get("subject_id") != case["subject_id"]):
            raise ValueError("Prediction identity/checkpoint binding missing")
        if case.get("expected_checkpoint_sha256") and checkpoint != case["expected_checkpoint_sha256"]:
            raise ValueError("Unexpected checkpoint")
        ref_record = report["inputs"]["qsm_grid"]
        ref_image, reference = load(ref_record["path"], ref_record["sha256"])
        prediction_image, prediction = load(provenance["prediction_path"], provenance["prediction_sha256"])
        # A prediction belongs to the real SWI grid; never fix headers to force this.
        swi_image = nib.load(record(report["inputs"]["swi"]["path"], report["inputs"]["swi"]["sha256"]))
        same_grid(prediction_image, swi_image)
        mask_record = report["inputs"]["qsm_reliability_mask"]
        mask_image, reliability = load(mask_record["path"], mask_record["sha256"])
        same_grid(ref_image, mask_image)
        if np.any(reliability < 0) or not np.any(reliability > 0):
            raise ValueError("Invalid reference reliability mask")
        labels = {}
        for target, image in (("gre", ref_image), ("swi", prediction_image)):
            labels[target] = []
            for side in ("left", "right"):
                entry = report["targets"][target]["labels"][side]
                label_image, label = load(entry["path"], entry["sha256"])
                same_grid(label_image, image)
                if np.any(label < 0) or not np.array_equal(label, np.rint(label)):
                    raise ValueError("Invalid integer labels")
                labels[target].append(label)
            wanted = [v for ids in REGIONS.values() for v in ids]
            if np.any(np.isin(labels[target][0], wanted) & np.isin(labels[target][1], wanted)):
                raise ValueError("Overlapping target nuclei between hemispheres")
        for region, ids in REGIONS.items():
            for side, indices in (("L", [0]), ("R", [1]), ("bilateral", [0, 1])):
                rm = np.logical_or.reduce([np.isin(labels["gre"][i], ids) for i in indices])
                pm = np.logical_or.reduce([np.isin(labels["swi"][i], ids) for i in indices])
                nr, npred = int(rm.sum()), int(pm.sum())
                coverage = float(np.count_nonzero(rm & (reliability > 0)) / nr) if nr else None
                rv, pv = reference[rm].astype(np.float64), prediction[pm].astype(np.float64)
                refmean = float(rv.mean()) if nr else None
                predmean = float(pv.mean()) if npred else None
                defined = nr > 0 and npred > 0
                component_counts = {target: {hemisphere + ":" + str(label): int(np.count_nonzero(labels[target][i] == label))
                                      for i in indices for hemisphere in (["L"] if i == 0 else ["R"]) for label in ids}
                                    for target in ("gre", "swi")}
                reference_complete = all(component_counts["gre"].values())
                prediction_complete = all(component_counts["swi"].values())
                zero_eligible = nr > 0 and reference_complete and coverage == 1.
                eligible = defined and zero_eligible and prediction_complete
                common = {"case_id": identity, "cohort": case["cohort"], "subject_id": case["subject_id"],
                          "region": region, "side": side, "label_ids": "+".join(map(str, ids)),
                          "n_reference_voxels": nr, "n_prediction_voxels": npred,
                          "reference_reliability_coverage": coverage, **FLAGS,
                          "component_counts": json.dumps(component_counts, sort_keys=True),
                          "reference_all_components_present": reference_complete,
                          "prediction_all_components_present": prediction_complete,
                          "paired_comparison_eligible": eligible,
                          "status": "numeric_candidate_QC_pending" if eligible else "missing_or_incomplete_reliable_ROI",
                          "reference_mean_raw": refmean,
                          "zero_roi_mean_abs_error_raw": abs(refmean) if nr else None,
                          "reference_std_raw": float(rv.std()) if nr else None,
                          "registration_report_sha256": ledger[str(report_path)]}
                for method, mean, method_sha in ((model, predmean, checkpoint), ("zero", 0. if nr else None, "")):
                    if method == "zero" and any(r["case_id"] == identity and r["region"] == region
                                                 and r["side"] == side and r["model"] == "zero" for r in rows):
                        continue
                    diff = mean - refmean if mean is not None and refmean is not None else None
                    rows.append({**common, "model": method, "checkpoint_sha256": method_sha,
                                 "n_prediction_voxels": nr if method == "zero" else npred,
                                 "prediction_support_grid": "native_reference_GRE" if method == "zero" else "native_prediction_SWI",
                                 "prediction_all_components_present": reference_complete if method == "zero" else prediction_complete,
                                 "status": ("numeric_candidate_QC_pending" if (zero_eligible if method == "zero" else eligible)
                                            else "missing_or_incomplete_reliable_ROI"),
                                 "prediction_mean_raw": mean, "mean_difference_raw": diff,
                                 "roi_mean_abs_error_raw": abs(diff) if diff is not None else None,
                                 "roi_mean_squared_error_raw": diff * diff if diff is not None else None,
                                 "prediction_std_raw": float(pv.std()) if method != "zero" and npred else (0. if method == "zero" else None),
                                 "prediction_provenance_sha256": ledger[str(provenance_path)] if method != "zero" else "",
                                 "zero_support_note": "numeric_zero_on_reference_ROI_no_physical_zero_claim" if method == "zero" else ""})
    selected = list(expected_models or model_cases)
    if set(selected) != set(model_cases) or any(model_cases[m] != next(iter(model_cases.values())) for m in model_cases):
        raise ValueError("Every expected model must provide exactly the same pilot cases")
    for model in model_cases:
        if len({r["checkpoint_sha256"] for r in rows if r["model"] == model}) != 1:
            raise ValueError("Mixed checkpoints within one model")
    checkpoint_models = {next(r["checkpoint_sha256"] for r in rows if r["model"] == model) for model in model_cases}
    if len(checkpoint_models) != len(model_cases):
        raise ValueError("Different model names cannot relabel identical checkpoint bytes")
    for path, digest in ledger.items():
        if sha(path) != digest:
            raise ValueError("Source changed while extracting: " + path)
    report = {"schema": "swixqsm.deferred_native_nuclei", "version": 1, **FLAGS,
              "allow_unverified_data": reason, "n_cases": len(participants), "n_models_available": len(model_cases),
              "models": selected, "n_measurements_including_zero": len(rows), "regions": REGIONS,
              "source_files": ledger, "training_performed": False, "restricted_cohort_access": False,
              "extractor_sha256": sha(Path(__file__)),
              "cases_content_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
              "no_voxel_error_across_grids": True, "confidence_intervals": None,
              "limitations": "Descriptive pilot numeric agreement only. Separate native supports; unverified units/zero and deferred anatomical/registration QC. No calibration. Do not pool cohorts, hemispheres, subnuclei or repeat versions as independent people. SN=SNr310 union SNc352 in NextBrain AllenAtlasLUT, RN385, DN721. Eight-model results require eight real predictions; absent models are never imputed."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-nuclei-", dir=output.parent) as tmp:
        stage = Path(tmp) / "results"; stage.mkdir()
        write_csv(stage / "native_roi_per_case.csv", rows)
        write_csv(stage / "native_roi_summary.csv", summarize(rows))
        (stage / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="JSON list: case/cohort/subject, registration_report, prediction_provenance, optional model/expected_checkpoint_sha256")
    parser.add_argument("--source-root", type=Path, default=Path.cwd(), help="Base for relative paths inside existing source records")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-unverified-data", required=True)
    parser.add_argument("--expected-model", action="append", help="Require every named model and identical case sets; repeat for all eight when available")
    args = parser.parse_args()
    report = evaluate(json.loads(args.cases.read_text()), args.output, source_root=args.source_root,
                      reason=args.allow_unverified_data, expected_models=args.expected_model)
    print(json.dumps({k: report[k] for k in ("n_cases", "n_models_available", "n_measurements_including_zero")}))


if __name__ == "__main__":
    main()

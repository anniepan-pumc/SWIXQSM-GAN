"""Evaluate physical QSM NIfTI predictions and train-only baselines on one split."""
import argparse
import csv
import hashlib
import io
import json
import platform
import tempfile
from pathlib import Path

import nibabel
import numpy as np
import scipy

from .util.baselines import load_baselines
from .util.data import read_manifest
from .util.evaluation import evaluate_rows, resolve_prediction_paths, write_csv
from .util.nifti import save_qsm
from .util.normalization import restored_value_metadata
from .util.quality_gate import (add_quality_arguments, check_data_quality,
                               prepare_quality_rows, check_prediction_references, bind_input_units,
                               prediction_sources)
from .util.reference import assert_reference_identity
from .util.independence_gate import (add_independence_arguments, check_independence,
                                    check_prediction_independence, assert_source_independence)
from .util.selection_record import inspect_selection_record
from .util.training_diagnostics import masked_distribution
from .util.regions import RegionEvaluator


def roi_rows_with_overlay(rows, path):
    """Bind ROI-only metadata without changing the audited evaluation rows."""
    if path is None:
        return [dict(row) for row in rows], None
    path = Path(path).expanduser().resolve()
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames
    allowed = {"case_id", "roi_labels_path", "roi_label_source", "roi_review", "roi_review_status",
               "roi_reviewer", "roi_review_date"}
    if (not fields or len(fields) != len(set(fields)) or "case_id" not in fields
            or set(fields) - allowed):
        raise ValueError("ROI manifest requires unique case_id/ROI-only columns; core input, unit and identity fields are forbidden")
    overlay = {}
    for record in reader:
        if None in record or any(value is None for value in record.values()):
            raise ValueError("ROI manifest CSV rows must match their header")
        case = record.get("case_id", "")
        if not case or case != case.strip() or case in overlay:
            raise ValueError("ROI manifest case_id must be nonempty, unique and without surrounding whitespace")
        review, alias = record.get("roi_review", ""), record.get("roi_review_status", "")
        if review.strip() and alias.strip() and review.strip() != alias.strip():
            raise ValueError("ROI manifest roi_review and roi_review_status conflict")
        record["roi_review"] = review.strip() or alias.strip()
        overlay[case] = record
    expected = {row["case_id"] for row in rows}
    if len(expected) != len(rows) or set(overlay) != expected:
        raise ValueError("ROI manifest must match every selected evaluation case_id exactly once, with no missing or extra cases")
    merged = []
    for row in rows:
        item = dict(row)
        for field in ("roi_labels_path", "roi_label_source", "roi_review", "roi_reviewer", "roi_review_date"):
            item[field] = overlay[row["case_id"]].get(field, "")
        label = item["roi_labels_path"].strip()
        if label:
            label = Path(label).expanduser()
            item["roi_labels_path"] = str((path.parent / label if not label.is_absolute() else label).resolve())
        merged.append(item)
    source = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "n_rows": len(overlay),
              "columns": fields, "scope": "ROI-only overlay; core evaluation manifest and checkpoint/identity binding unchanged",
              "selected_case_ids_sha256": hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest()}
    return merged, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New, nonexisting results directory")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--units", choices=["ppm", "ppb", "stored"], default="ppb", help="Output units; stored is unverified original numerical scale")
    parser.add_argument("--qsm-units", choices=["ppm", "ppb", "stored"], help="Fallback only when qsm_units column is absent")
    parser.add_argument("--reference-id", required=True, help="Declared common susceptibility zero-reference, never fitted here")
    parser.add_argument("--prediction-units", choices=["ppm", "ppb", "stored"], help="Must agree with prediction CSV/sidecar when present")
    parser.add_argument("--prediction-reference-id", help="Required for predictions without a reference declaration")
    parser.add_argument("--prediction-name", default="prediction")
    parser.add_argument("--expected-prediction-phase", choices=["l1", "warmup", "gan"],
                        help="Require recorded current-run model phase; GAN must have adversarial updates")
    parser.add_argument("--selection-record", help="Document frozen checkpoint selection without requiring validation or retraining")
    parser.add_argument("--baselines", help="Train-only .npz fitted by swixqsm_minimal.fit_baselines")
    parser.add_argument("--space-id", help="Declared common anatomical space; required by spatial baselines")
    parser.add_argument("--spatial-units", choices=["mm", "meter", "micron"], help="Explicit assumption for unknown NIfTI spatial units")
    parser.add_argument("--data-range", type=float, help="Fixed PSNR/SSIM range in --units; absent means those metrics are undefined")
    parser.add_argument("--ssim-window", type=int, default=7)
    parser.add_argument("--hfen-sigma-mm", type=float, default=1.5)
    parser.add_argument("--save-predictions", action="store_true", help="Also save physical evaluated volumes for visual QC")
    parser.add_argument("--distribution-diagnostics", action="store_true",
                        help="Read-only physical distributions on each case/method's evaluation mask; no model forward or training")
    parser.add_argument("--roi-spec", help="Prespecified region labels/coverage JSON; manifest supplies ROI paths, sources and review records")
    parser.add_argument("--roi-manifest", help="Optional ROI-only CSV, exactly one row per selected case_id; does not modify the audited evaluation manifest")
    parser.add_argument("--roi-review-policy", choices=["accepted", "deferred"], default="accepted",
                        help="accepted requires recorded label review; deferred computes research-candidate values only and requires explicit unverified data")
    parser.add_argument("--roi-bootstrap-reps", type=int, default=2000,
                        help="Paired-participant percentile CI replicates for ROI agreement; 0 disables CIs")
    parser.add_argument("--roi-bootstrap-seed", type=int, default=42)
    add_quality_arguments(parser)
    add_independence_arguments(parser)
    args = parser.parse_args()
    if args.roi_manifest and not args.roi_spec:
        parser.error("--roi-manifest requires --roi-spec")
    if args.roi_review_policy == "deferred" and (not args.roi_spec or args.qc_report
            or not isinstance(args.allow_unverified_data, str) or not args.allow_unverified_data.strip()):
        parser.error("deferred ROI requires --roi-spec and a nonempty --allow-unverified-data reason; verified QC reports cannot be used")
    if args.units == "stored" and (args.qc_report or args.baselines
            or args.roi_spec and args.roi_review_policy != "deferred"):
        parser.error("stored evaluation requires unverified data; ROI requires explicit deferred policy and anatomical baselines remain unsupported")
    if args.roi_bootstrap_reps < 0 or not 0 <= args.roi_bootstrap_seed < 2**64:
        parser.error("ROI bootstrap repetitions must be nonnegative; seed must be in [0,2**64)")
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; use a new evaluation directory")
    manifest = Path(args.manifest).resolve()
    all_rows = prepare_quality_rows(read_manifest(manifest), manifest)
    data_independence = check_independence(all_rows, args, content_splits=(args.split,), check_initialization=False)
    rows = resolve_prediction_paths([row for row in all_rows if row["split"] == args.split], manifest)
    rows = prepare_quality_rows(rows, manifest)
    data_qc = check_data_quality(rows, args, reference_id=args.reference_id)
    data_reference = data_qc.get("data_reference")
    rows = bind_input_units(rows, data_qc, requested_fallback=args.qsm_units)
    if data_qc["status"] == "verified":
        check_prediction_references(rows, data_reference)
    check_prediction_independence(rows, data_independence, expected_phase=args.expected_prediction_phase)
    model_selection = (inspect_selection_record(args.selection_record, rows) if args.selection_record
                       else {"status": "not_recorded", "limitations": "Selection history cannot be inferred from the checkpoint filename"})
    sources = prediction_sources(rows)
    for index, row in enumerate(rows):
        row["case_id"] = row.get("case_id") or f"{row['subject_id']}_row{index + 1:06d}"
    roi_rows, roi_manifest_source = roi_rows_with_overlay(rows, args.roi_manifest) if args.roi_spec else (None, None)
    regions = (RegionEvaluator(roi_rows, args.roi_spec, manifest, units=args.units,
                              reference_id=args.reference_id, bootstrap_reps=args.roi_bootstrap_reps,
                              seed=args.roi_bootstrap_seed, review_policy=args.roi_review_policy,
                              allow_unverified_data=args.allow_unverified_data, data_qc=data_qc,
                              roi_manifest_source=roi_manifest_source) if args.roi_spec else None)
    bundle = load_baselines(args.baselines) if args.baselines else None
    if bundle is not None and data_independence["status"] == "verified":
        assert_source_independence(data_independence, bundle.metadata.get("data_independence"), name="Baseline")
    if bundle is not None and data_qc["status"] == "verified":
        if bundle.metadata.get("data_qc", {}).get("status") != "verified":
            raise ValueError("Baseline artifact has no verified data QC; refit using --qc-report or explicitly select legacy evaluation")
        assert_reference_identity(data_reference, bundle.metadata.get("data_reference"))
    case_rows = {row["case_id"]: row for row in rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Failed evaluations leave no partially successful result directory.
    with tempfile.TemporaryDirectory(prefix=".swixqsm-evaluation-", dir=output.parent) as tmp:
        staging = Path(tmp) / "results"
        staging.mkdir()
        saved = []
        distributions = []

        def diagnose(case_id, method, target, prediction, mask, units):
            statistics = masked_distribution(prediction, target, mask)
            metadata = restored_value_metadata(units)
            statistics.update(units=units, value_space=metadata["ValueSpace"])
            if units == "stored":
                statistics["unit_status"] = metadata["UnitStatus"]
            distributions.append({"case_id": case_id, "subject_id": case_rows[case_id]["subject_id"],
                                  "split": args.split, "method": method,
                                  "statistics": statistics})

        def save(case_id, method, values, reference, units, reference_id, space_id):
            token = hashlib.sha256((case_id + "\0" + method).encode()).hexdigest()[:20]
            filename = f"{token}.nii.gz"
            source = sources.get(case_rows[case_id].get("prediction_path"), {}) if method == args.prediction_name else {}
            save_qsm(staging / "predictions" / filename, values, reference,
                     {**restored_value_metadata(units), "ReferenceID": reference_id, "SpaceID": space_id,
                      "Method": method, "CaseID": case_id, "PostHocCalibration": False,
                      "DataReference": data_reference, "DataQC": data_qc,
                      "EvaluationDataQC": data_qc,
                      "DataIndependence": data_independence, "EvaluationDataIndependence": data_independence,
                      "EvaluationModelSelection": model_selection,
                      **source,
                      "EvaluationMaskPath": case_rows[case_id]["mask_path"],
                      "InputBrainMaskPath": case_rows[case_id]["swi_mask_path"],
                      "Validity": "Only the declared evaluation mask is scored; zero outside brain/training support is padding, not validated tissue"})
            saved.append({"case_id": case_id, "method": method,
                          "prediction_path": str((output / "predictions" / filename).resolve())})

        results = evaluate_rows(rows, units=args.units, qsm_units=args.qsm_units,
                                reference_id=args.reference_id, spatial_units=args.spatial_units,
                                space_id=args.space_id, baseline_bundle=bundle,
                                prediction_units=args.prediction_units,
                                prediction_reference_id=args.prediction_reference_id,
                                prediction_name=args.prediction_name, data_range=args.data_range,
                                ssim_window=args.ssim_window, hfen_sigma_mm=args.hfen_sigma_mm,
                                geometry_contract=data_qc.get("contract", {}).get("geometry"),
                                diagnostic_callback=diagnose if args.distribution_diagnostics else None,
                                region_callback=regions.collect if regions is not None else None,
                                roi_review_policy=args.roi_review_policy,
                                save_prediction=save if args.save_predictions else None)
        write_csv(staging / "per_case.csv", results["cases"])
        write_csv(staging / "per_subject.csv", results["subjects"])
        parameters = {key: value for key, value in results["cases"][0].get("parameters", {}).items()
                      if key not in {"spacing_mm", "hfen_radius_voxels"}}
        config = vars(args).copy()
        config.update(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      baseline_sha256=hashlib.sha256(Path(args.baselines).read_bytes()).hexdigest() if args.baselines else None,
                      python_version=platform.python_version(), numpy_version=np.__version__,
                      scipy_version=scipy.__version__, nibabel_version=nibabel.__version__,
                      aggregation="voxel-pooled visits per participant; equal-participant mean and sample SD",
                      zero_baseline=("numeric zero in original stored scale, independent of neural target offset"
                                     if args.units == "stored" else "physical zero, independent of neural target offset"),
                      value_space=restored_value_metadata(args.units)["ValueSpace"],
                      unit_status="unverified_original_storage" if args.units == "stored" else "declared_physical_units",
                      calibration="no clipping, target-based scaling, re-referencing or post-hoc fitting",
                      data_qc=data_qc, data_reference=data_reference,
                      data_independence=data_independence,
                      model_selection=model_selection,
                      prediction_provenance=sources,
                      roi_manifest_source=roi_manifest_source,
                      metrics_parameters=parameters)
        payload = {"config": config, "summary": results["summary"]}
        (staging / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        (staging / "manifest_resolved.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        if regions is not None:
            regional = regions.finish()
            write_csv(staging / "roi_per_case.csv", regional["cases"])
            write_csv(staging / "roi_per_subject.csv", regional["subjects"])
            write_csv(staging / "roi_summary.csv", regional["summary"])
            (staging / "roi_agreement.json").write_text(json.dumps({
                "schema": "swixqsm.frozen_roi_agreement", "version": 1,
                "units": args.units, "reference_id": args.reference_id,
                "analysis_status": "estimated" if any(r["status"] == "ok" for r in regional["subjects"]) else "no_eligible_participants",
                "roi_review_policy": args.roi_review_policy,
                "QCStatus": data_qc["status"],
                "UnitStatus": "unverified_original_storage" if args.units == "stored" else "declared_physical_units",
                "ReferenceStatus": "unverified" if args.roi_review_policy == "deferred" or data_qc["status"] != "verified" else "verified_by_data_qc",
                "value_space": restored_value_metadata(args.units)["ValueSpace"],
                "research_only": args.roi_review_policy == "deferred", "formal_accuracy_claim": False,
                "roi_manifest_source": roi_manifest_source,
                "data_qc": data_qc, "data_reference": data_reference,
                "data_independence": data_independence, "model_selection": model_selection,
                "prediction_provenance": sources, "provenance": regional["provenance"],
                "configuration": {"bootstrap_reps": args.roi_bootstrap_reps, "bootstrap_seed": args.roi_bootstrap_seed,
                                  "visit_policy": "all expected visits eligible, then voxel-pooled participant ROI mean",
                                  "roi_review_policy": args.roi_review_policy,
                                  "model_forward_or_training": False},
                "agreement": regional["agreement"],
            }, indent=2, allow_nan=False) + "\n")
        if saved:
            write_csv(staging / "saved_predictions.csv", saved)
        if args.distribution_diagnostics:
            (staging / "distribution_diagnostics.json").write_text(json.dumps({
                "schema": "swixqsm.frozen_distribution_diagnostics", "version": 1,
                "units": args.units, "support": "Each case's declared common evaluation mask",
                "scope": "Exact per-case/method valid-voxel distributions; not a pooled participant or whole-cohort quantile",
                "model_selection": model_selection, "prediction_provenance": sources,
                "limitations": "Read-only prediction statistics cannot recover historical adversarial losses or establish lesion/ROI accuracy",
                "cases": distributions,
            }, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError("Evaluation output appeared while computing; refusing to replace it")
        staging.rename(output)
    print(f"Evaluated {len(rows)} volumes. Results: {output.resolve()}")


if __name__ == "__main__":
    main()

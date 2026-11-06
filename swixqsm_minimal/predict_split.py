"""Predict one val/test split once, then use evaluation_manifest.csv with evaluate.

Inference decodes SWI and its brain mask only. A verified QC report is reused
through exact selected-row metadata and file hashes; its QSM arrays/reference
statistics are not recomputed here. Full paired QC is rechecked by evaluate.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile

from .test.infer import infer_case, load_inference_context
from .util.data import read_manifest
from .util.independence_gate import (add_independence_arguments, assert_source_independence,
                                    check_independence)
from .util.normalization import unit_factor
from .util.quality import (IMAGE_PATHS, MAIN_PATHS, QC_PATHS, _case_key, _digest,
                          _normalize_row, validate_contract)
from .util.quality_gate import (add_quality_arguments, bind_input_units, check_data_quality,
                               prepare_quality_rows)
from .util.reference import assert_reference_identity, reference_identity


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_qc(rows, report_path, *, reference_id=None):
    """Rebind prior passing QC to identical bytes, without decoding QSM values."""
    report_path = Path(report_path).expanduser().resolve()
    report_sha = file_sha256(report_path)
    report = json.loads(report_path.read_text())
    if (not isinstance(report, dict) or report.get("schema") != "swixqsm.data_qc_report"
            or type(report.get("version")) is not int or report["version"] != 1):
        raise ValueError("Unsupported data QC report schema/version")
    contract = validate_contract(report.get("contract"))
    reference = reference_identity(contract["reference"])
    if report.get("contract_sha256") != _digest(contract) or report.get("data_reference") != reference:
        raise ValueError("QC report contract hash or reference identity is inconsistent")
    if reference_id is not None and reference_id != reference["id"]:
        raise ValueError("QC reference ID does not match the required reference")
    recorded = {}
    for case in report.get("cases", []):
        if not isinstance(case, dict) or not case.get("key") or case["key"] in recorded:
            raise ValueError("QC report contains invalid or duplicate case keys")
        recorded[case["key"]] = case
    selected_keys, verified_files = [], {}
    for original in rows:
        row = _normalize_row(original)
        key = _case_key(row)
        old = recorded.get(key)
        if old is None or old.get("passed") is not True or old.get("errors") != []:
            raise ValueError("Selected case has no passing frozen QC record: " + row["subject_id"])
        if key in selected_keys:
            raise ValueError("Duplicate selected QC case")
        units = row.get("qsm_units", report.get("default_qsm_units"))
        unit_factor(units, units)
        metadata = {"row": {k: v for k, v in row.items() if not k.startswith("prediction_")},
                    "effective_qsm_units": units}
        paths = {k: row[k] for k in MAIN_PATHS + QC_PATHS if row.get(k)}
        if (old.get("metadata_sha256") != _digest(metadata) or old.get("qsm_units") != units
                or old.get("paths") != paths or set(old.get("file_sha256", {})) != set(paths)):
            raise ValueError("Selected case QC metadata/paths changed: " + row["subject_id"])
        for path_key, path in paths.items():
            digest = file_sha256(path)
            if old["file_sha256"][path_key] != digest:
                raise ValueError("Selected case QC input hash changed: " + path)
            if path in verified_files and verified_files[path] != digest:
                raise ValueError("Selected QC input changed while checking it: " + path)
            verified_files[path] = digest
        for image_key in IMAGE_PATHS:
            if row.get(image_key) and old.get("geometry", {}).get(image_key, {}).get("passed") is not True:
                raise ValueError("Frozen QC lacks passing geometry: " + image_key)
        reference_check = old.get("reference") or {}
        if reference_check.get("passed") is not True:
            raise ValueError("Frozen QC lacks a passing reference check")
        assert_reference_identity(reference, reference_check.get("reference_identity"))
        selected_keys.append(key)
    if not selected_keys:
        raise ValueError("No selected cases to verify")
    if file_sha256(report_path) != report_sha:
        raise RuntimeError("QC report changed while being checked")
    return dict(status="verified", report_path=str(report_path), report_sha256=report_sha,
                contract=contract, contract_sha256=_digest(contract), data_reference=reference,
                default_qsm_units=report.get("default_qsm_units"), selected_case_keys=selected_keys,
                n_cases=len(selected_keys), verified_files=verified_files,
                verification_scope="frozen passing QC report: selected metadata and exact file SHA256; inference rechecks SWI/brain-mask geometry",
                qsm_voxels_decoded_here=False, reference_statistics_recomputed_here=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output", required=True, help="New directory, atomically published after every case succeeds")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-units", choices=["ppm", "ppb", "stored"])
    parser.add_argument("--qsm-units", choices=["ppm", "ppb", "stored"], help="Fallback for absent source qsm_units; not a target transformation")
    parser.add_argument("--reference-id", help="Consistency declaration only; never re-reference predictions")
    parser.add_argument("--expected-phase", choices=["l1", "warmup", "gan"])
    add_quality_arguments(parser)
    add_independence_arguments(parser)
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error("Output already exists; choose a new prediction directory")
    manifest = Path(args.manifest).expanduser().resolve()
    manifest_sha = file_sha256(manifest)
    all_rows = prepare_quality_rows(read_manifest(manifest, validate_inputs=False), manifest)
    rows = [row for row in all_rows if row["split"] == args.split]
    if not rows:
        raise ValueError("No cases in the selected validation/test split")
    case_ids = [row.get("case_id") for row in rows if row.get("case_id")]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Selected case_id values must be unique")
    for row in all_rows:
        if row.get("roi_labels_path") and not Path(row["roi_labels_path"]).is_absolute():
            raise ValueError("Relative roi_labels_path cannot move with a hash-bound manifest; prepare absolute paths and re-audit first")
    independence = check_independence(all_rows, args, content_splits=(args.split,), check_initialization=False)
    qc = (verify_frozen_qc(rows, args.qc_report, reference_id=args.reference_id) if args.qc_report
          else check_data_quality(rows, args, reference_id=args.reference_id))
    bound = bind_input_units(rows, qc, requested_fallback=args.qsm_units)
    for row in bound:
        unit_factor(row["qsm_units"], row["qsm_units"])
    context = load_inference_context(args.checkpoint, args.device, args.expected_phase)
    unit_factor(context["normalization"].units, args.output_units or context["normalization"].units)
    for row in bound:
        unit_factor(row["qsm_units"], context["normalization"].units)
    source, trained_qc = context["data_reference"], context["trained_qc"]
    if qc["status"] == "verified":
        if trained_qc.get("status") != "verified":
            raise ValueError("Checkpoint has no verified training data QC; explicitly select unverified inference")
        assert_reference_identity(qc["data_reference"], source)
    if args.reference_id is not None and (not args.reference_id.strip()
            or source is not None and args.reference_id != source["id"]):
        raise ValueError("Reference ID conflicts with the checkpoint or is empty")
    reference = source["id"] if source is not None else args.reference_id
    if reference is None:
        declared = {row["qsm_reference"] for row in rows if row.get("qsm_reference")}
        if len(declared) == 1:
            reference = declared.pop()
        else:
            raise ValueError("Batch evaluation requires one known reference ID; provide --reference-id for an unverified checkpoint")
    for row in rows:
        if row.get("qsm_reference") and row["qsm_reference"] != reference:
            raise ValueError("Selected QSM reference declaration conflicts with checkpoint/inference reference")
    if independence.get("status") == "verified":
        assert_source_independence(independence,
            context["checkpoint"].get("data_independence") or context["config"].get("data_independence"), name="Checkpoint")
    units = args.output_units or context["normalization"].units
    input_hashes = dict(qc.get("verified_files", {}))
    for row in rows:
        for key in ("swi_path", "swi_mask_path"):
            path = row[key]
            digest = file_sha256(path)
            if path in input_hashes and input_hashes[path] != digest:
                raise RuntimeError("Inference input changed after QC: " + path)
            input_hashes[path] = digest
    evaluation_rows = copy.deepcopy(all_rows)
    for row in evaluation_rows:
        for key in list(row):
            if key.startswith("prediction_"):
                row[key] = ""
    predictions = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".swixqsm-predict-split-", dir=output.parent) as temporary:
        staging = Path(temporary) / "results"
        (staging / "predictions").mkdir(parents=True)
        for index, row in enumerate(evaluation_rows):
            if row["split"] != args.split:
                continue
            case_id = row.get("case_id") or f"{row['subject_id']}_row{index + 1:06d}"
            token = hashlib.sha256(json.dumps([case_id, row["swi_path"]], ensure_ascii=False).encode()).hexdigest()[:24]
            name = token + ".nii.gz"
            case_qc = {key: value for key, value in qc.items() if key not in {"verified_files", "selected_case_keys", "n_cases"}}
            case_qc["n_cases"] = 1
            if qc["status"] == "verified":
                case_qc["selected_case_keys"] = [_case_key(row)]
                case_qc["verified_files"] = {row[key]: input_hashes[row[key]]
                    for key in MAIN_PATHS + QC_PATHS if row.get(key)}
            metadata = infer_case(context, row["swi_path"], row["swi_mask_path"], staging / "predictions" / name,
                output_units=units, reference_id=reference, allow_unverified_data=args.allow_unverified_data,
                extra_metadata={"CaseID": case_id, "InputSWIPath": row["swi_path"],
                    "InputBrainMaskPath": row["swi_mask_path"],
                    "InputSWISHA256": input_hashes[row["swi_path"]],
                    "InputBrainMaskSHA256": input_hashes[row["swi_mask_path"]],
                    "InferenceDataQC": case_qc, "InferenceDataIndependence": independence,
                    "InferenceManifestSHA256": manifest_sha, "InferenceSplit": args.split})
            final_path = str(output / "predictions" / name)
            row.update(prediction_path=final_path, prediction_units=units, prediction_reference=reference or "")
            if source is not None:
                row.update(prediction_reference_method=source["method"], prediction_reference_region=source["region"],
                           prediction_reference_target_ppb=source["target_ppb"])
            predictions.append({"case_id": case_id, "subject_id": row["subject_id"], "prediction_path": final_path,
                                "units": units, "reference_id": metadata["ReferenceID"]})
        fields = list(dict.fromkeys(key for row in evaluation_rows for key in row))
        with (staging / "evaluation_manifest.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(evaluation_rows)
        if file_sha256(manifest) != manifest_sha:
            raise RuntimeError("Manifest changed during inference")
        for path, digest in input_hashes.items():
            if file_sha256(path) != digest:
                raise RuntimeError("Selected input changed during inference: " + path)
        if args.qc_report and file_sha256(args.qc_report) != qc["report_sha256"]:
            raise RuntimeError("QC report changed during inference")
        if args.independence_report and file_sha256(args.independence_report) != independence["report_sha256"]:
            raise RuntimeError("Independence report changed during inference")
        payload = dict(schema="swixqsm.split_predictions", version=1, split=args.split,
                       checkpoint_path=str(Path(args.checkpoint).expanduser().resolve()),
                       checkpoint_sha256=context["checkpoint_sha256"], manifest_path=str(manifest),
                       manifest_sha256=manifest_sha, evaluation_manifest_sha256=file_sha256(staging / "evaluation_manifest.csv"),
                       n_predictions=len(predictions), n_manifest_rows=len(all_rows), units=units, reference_id=reference,
                       data_qc=qc, data_independence=independence, training_state=context["training_state"],
                       predictions=predictions, training_performed=False, qsm_voxels_decoded=False,
                       value_space="restored_stored_values" if units == "stored" else "physical_qsm",
                       unit_status="unverified_original_storage" if units == "stored" else "declared_physical_units",
                       limitation=("Predictions restore original numerical label values only; units, susceptibility reference and registration remain unverified; evaluate supports numeric zero and voxel metrics, not anatomical baselines/ROI"
                                   if units == "stored" else "Predictions are physical QSM; run evaluate for target metrics, full paired QC, baselines and optional ROI analysis"))
        (staging / "prediction_run.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output appeared during inference; refusing to replace it")
        staging.rename(output)
    print(f"Predicted {len(predictions)} {args.split} volumes in {units}; {output / 'evaluation_manifest.csv'}")
    return payload


if __name__ == "__main__":
    main()

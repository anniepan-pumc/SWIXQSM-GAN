"""SWI-only external-pilot inference and dual-native nuclei comparison.

Requires new checkpoints containing their actual normalization metadata. It does
not reconstruct missing preprocessing for legacy weights, or read restricted-cohort images.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from .compare_models import REVIEW_MODELS
from .evaluate_native_nuclei import evaluate, resolve, sha
from .test.infer import infer_case, load_inference_context
from .util.nifti import sidecar_path


def run(cases_path, models_path, output, *, source_root, reason, device="cpu", expected_models=None):
    if not reason.strip():
        raise ValueError("An explicit reason for deferred QC/units is required")
    output, cases_path, models_path = Path(output).resolve(), Path(cases_path).resolve(), Path(models_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    case_sha, models_sha = sha(cases_path), sha(models_path)
    cases = json.loads(cases_path.read_text())
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Supply one registration per pilot case, without duplicate models")
    with models_path.open(newline="", encoding="utf-8-sig") as stream:
        models = list(csv.DictReader(stream))
    expected_models = list(expected_models or REVIEW_MODELS)
    if (not models or len({m["model"] for m in models}) != len(models)
            or set(expected_models) != {m["model"] for m in models}):
        raise ValueError("Checkpoint CSV must contain every expected model exactly once")
    # Validate all identities, source hashes and checkpoint paths before inference.
    inputs, input_hashes = [], {}
    for case in cases:
        rp = resolve(case["registration_report"], source_root)
        rep = json.loads(rp.read_text())
        if (rep["case_id"] != case["case_id"] or rep["cohort"] != case["cohort"]
                or rep["subject_id"] != case["subject_id"] or rep["registration_status"] != "candidate_completed_pending_QC"
                or rep.get("protocol_revision") != "scanner_identity_initialization_v2"):
            raise ValueError("Incomplete/wrong native registration")
        swi, mask = rep["inputs"]["swi"], rep["targets"]["swi"]["mapped_brain"]
        pair = []
        for entry in (swi, mask):
            path = resolve(entry["path"], source_root)
            if sha(path) != entry["sha256"]:
                raise ValueError("External inference input hash changed")
            input_hashes[str(path)] = entry["sha256"]
            pair.append(path)
        inputs.append((case, rp, sha(rp), pair))
    checkpoint_hashes = set()
    for model in models:
        if model.get("expected_phase") not in {"l1", "gan"}:
            raise ValueError("Declare each checkpoint's expected_phase as l1 or gan; warmup is not a final GAN")
        checkpoint_path = Path(model["checkpoint_path"]).expanduser()
        # A pretrained weight file is permitted; restricted-cohort image access is barred.
        model["checkpoint_path"] = str((checkpoint_path if checkpoint_path.is_absolute()
                                        else models_path.parent / checkpoint_path).resolve())
        if not Path(model["checkpoint_path"]).is_file():
            raise FileNotFoundError(model["checkpoint_path"])
        digest = sha(model["checkpoint_path"])
        if digest in checkpoint_hashes:
            raise ValueError("Distinct model comparisons require distinct checkpoint bytes, not relabeled copies")
        checkpoint_hashes.add(digest)
        model["checkpoint_sha256"] = digest
    output.mkdir(parents=True)
    products, completed = [], []
    status_path = output / "run_status.json"

    def status(state, error=None):
        status_path.write_text(json.dumps({"schema": "swixqsm.external_nuclei_prediction_run", "version": 1,
                    "status": state, "completed_predictions": completed, "error": error,
                    "cases_sha256": case_sha, "checkpoint_manifest_sha256": models_sha,
                    "expected_models": expected_models, "n_cases": len(cases), "training_performed": False,
                    "restricted_cohort_images_read": False, "review_policy": "deferred", "allow_unverified_data": reason,
                    "brain_mask_source": "same hash-bound T1 SuperSynth brain physically mapped to real SWI; registration QC pending; no reference QSM intensities used for inference",
                    "external_training_overlap_status": "not_independently_verified"}, indent=2) + "\n")
    status("running")
    try:
        for model in models:
            name = model["model"]
            context = load_inference_context(model["checkpoint_path"], device, model["expected_phase"])
            if context["checkpoint_sha256"] != model["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed after preflight")
            if context["config"].get("recipe") != name:
                raise ValueError("Checkpoint recipe does not match its comparison model name")
            if context["normalization"].units != "stored":
                raise ValueError("Unverified reference storage cannot be compared with a physical checkpoint by inventing a conversion")
            for case, rp, rp_sha, (swi, mask) in inputs:
                if sha(rp) != rp_sha:
                    raise ValueError("Registration report changed during inference")
                if any(sha(path) != input_hashes[str(path)] for path in (swi, mask)):
                    raise ValueError("SWI/mask changed before inference")
                token = hashlib.sha256((name + "\0" + case["case_id"]).encode()).hexdigest()[:20]
                directory = output / "predictions" / token
                directory.mkdir(parents=True)
                destination = directory / "prediction.nii.gz"
                metadata = infer_case(context, swi, mask, destination, output_units="stored",
                                      reference_id="unverified_original_reference", allow_unverified_data=reason,
                                      extra_metadata={"ExternalCaseID": case["case_id"], "Cohort": case["cohort"],
                                                      "SubjectID": case["subject_id"], "ROIRegistrationReportSHA256": rp_sha})
                if any(sha(path) != input_hashes[str(path)] for path in (swi, mask)):
                    raise ValueError("SWI/mask changed during inference")
                provenance = {"case_id": case["case_id"], "cohort": case["cohort"], "subject_id": case["subject_id"],
                              "model": name, "units": "stored", "prediction_path": str(destination),
                              "prediction_sha256": sha(destination), "checkpoint_sha256": context["checkpoint_sha256"],
                              "checkpoint_path": model["checkpoint_path"], "expected_phase": model["expected_phase"],
                              "sidecar_path": str(sidecar_path(destination)), "sidecar_sha256": sha(sidecar_path(destination)),
                              "registration_report_sha256": rp_sha, "input_swi_sha256": sha(swi), "input_brain_mask_sha256": sha(mask),
                              "normalization": metadata["Normalization"], "inverse_transform_applied_once": True,
                              "training_performed": False, "roi_review_policy": "deferred"}
                pp = directory / "prediction_provenance.json"
                pp.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
                products.append({"case_id": case["case_id"], "cohort": case["cohort"], "subject_id": case["subject_id"],
                                 "model": name, "registration_report": str(rp), "prediction_provenance": str(pp),
                                 "expected_checkpoint_sha256": context["checkpoint_sha256"]})
                completed.append({"model": name, "case_id": case["case_id"], "prediction_provenance": str(pp)})
                status("running")
            del context
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if sha(cases_path) != case_sha or sha(models_path) != models_sha:
            raise ValueError("Input case/checkpoint list changed during inference")
        (output / "evaluation_cases.json").write_text(json.dumps(products, indent=2) + "\n")
        result = evaluate(products, output / "metrics", source_root=source_root, reason=reason, expected_models=expected_models)
        status("completed_QC_pending")
        return result
    except Exception as error:
        status("failed_partial_predictions_preserved", {"type": type(error).__name__, "message": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path, help="CSV: model,checkpoint_path,expected_phase; all eight models by default")
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--expected-model", action="append")
    parser.add_argument("--allow-unverified-data", required=True)
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    result = run(args.cases, args.checkpoints, args.output, source_root=args.source_root,
                 reason=args.allow_unverified_data, device=args.device, expected_models=args.expected_model)
    print(json.dumps({k: result[k] for k in ("n_cases", "n_models_available", "n_measurements_including_zero")}))


if __name__ == "__main__":
    main()

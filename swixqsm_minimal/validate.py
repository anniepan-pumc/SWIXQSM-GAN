"""Validate/test a new checkpoint in physical QSM units, without fitting or training."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
from torch.utils.data import DataLoader

from .util.data import PairedSlices, read_manifest
from .util.engine import validate
from .util.independence_gate import (add_independence_arguments, assert_prediction_phase,
                                    assert_source_independence, check_independence)
from .util.model_loading import generator_from_checkpoint, load_checkpoint_snapshot
from .util.normalization import from_checkpoint
from .util.normalization_workflow import physical_output_metadata
from .util.quality import inspect_geometry, validate_contract
from .util.quality_gate import (add_quality_arguments, bind_input_units, check_data_quality,
                               prepare_quality_rows)
from .util.reference import assert_reference_identity


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output", required=True, help="New JSON file; no overwrite")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output-units", choices=["ppm", "ppb", "stored"], help="Default: checkpoint units")
    p.add_argument("--qsm-units", choices=["ppm", "ppb", "stored"], help="Explicit fallback when source qsm_units is absent")
    p.add_argument("--reference-id", help="Consistency check only; never re-reference the data")
    p.add_argument("--expected-phase", choices=["l1", "warmup", "gan"])
    add_quality_arguments(p)
    add_independence_arguments(p)
    args = p.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        p.error("Output already exists")
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    checkpoint, checkpoint_sha = load_checkpoint_snapshot(args.checkpoint)
    norm, spec = from_checkpoint(checkpoint)
    norm.denormalize(0., args.output_units)  # Reject unsupported stored/physical conversions before data reads.
    config = checkpoint["config"]
    assert_prediction_phase({"TrainingState": checkpoint.get("training_state")}, args.expected_phase)
    manifest_sha = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    all_rows = prepare_quality_rows(read_manifest(args.manifest, validate_inputs=False), args.manifest)
    independence = check_independence(all_rows, args, content_splits=(args.split,), check_initialization=False)
    rows = [row for row in all_rows if row["split"] == args.split]
    if not rows:
        raise ValueError("No rows in the selected validation/test split")
    qc = check_data_quality(rows, args, reference_id=args.reference_id)
    rows = bind_input_units(rows, qc, requested_fallback=args.qsm_units)
    trained_qc = checkpoint.get("data_qc") or config.get("data_qc") or {}
    trained_reference = checkpoint.get("data_reference") or config.get("data_reference")
    if qc.get("status") == "verified":
        if trained_qc.get("status") != "verified":
            raise ValueError("Checkpoint has no verified training data QC; explicitly select unverified evaluation")
        assert_reference_identity(qc.get("data_reference"), trained_reference)
        geometry = validate_contract(trained_qc["contract"])["geometry"]
        for row in rows:
            for key in ("swi_path", "qsm_path", "swi_mask_path", "mask_path"):
                if not inspect_geometry(nib.load(row[key]), geometry)["passed"]:
                    raise ValueError("Validation/test grid violates the checkpoint training contract: " + row[key])
    if args.reference_id is not None and trained_reference is not None:
        if args.reference_id != trained_reference["id"]:
            raise ValueError("Reference ID conflicts with the checkpoint; numeric offset does not reset it")
    if independence.get("status") == "verified":
        assert_source_independence(independence,
                                  checkpoint.get("data_independence") or config.get("data_independence"),
                                  name="Checkpoint")
    has_global = [bool(row.get("global_subject_id", "").strip()) for row in rows]
    if any(has_global) and not all(has_global):
        raise ValueError("Provide global_subject_id for every selected visit or none")
    dataset_class = PairedSlices
    if all(has_global):
        from .train_models import GlobalParticipantSlices
        dataset_class = GlobalParticipantSlices
    dataset = dataset_class(rows, args.split, config["size"], norm)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = generator_from_checkpoint(checkpoint).to(args.device).eval()
    score, subjects = validate(model, loader, args.device, norm, output_units=args.output_units)
    if hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest() != manifest_sha:
        raise RuntimeError("Manifest changed during validation/test")
    payload = {
        "schema": "swixqsm.checkpoint_stored_validation" if norm.units == "stored" else "swixqsm.checkpoint_physical_validation", "version": 1,
        "split": args.split, "training_performed": False, "normalization_fitted_here": False,
        **physical_output_metadata(norm, args.output_units), "normalization": spec,
        "normalization_source": config.get("normalization_source"),
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "reference_id": trained_reference.get("id") if trained_reference else args.reference_id,
        "data_qc": qc, "data_independence": independence,
        "aggregation": "voxel-pooled visits within participant; equal-participant mean MAE",
        "identity": "global_subject_id" if all(has_global) else "subject_id",
        "n_subjects": len(subjects), "n_visits": len(rows),
        "mae": score, "per_subject_mae": subjects,
        "limitation": ("MAE in restored original stored values only; units/reference/registration remain unverified. Save predictions and use evaluate for numeric-zero and voxel metrics. ROI/anatomical baselines are unsupported."
                       if norm.units == "stored" else "MAE only. For zero/ROI/SSIM/HFEN comparisons, save physical predictions with infer and use evaluate."),
    }
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(serialized)
    print(f"{args.split} MAE={score:.8g} {payload['Units']}; {len(subjects)} subjects; {output.resolve()}")


if __name__ == "__main__":
    main()

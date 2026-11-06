"""Restore an explicitly declared raw network-output NIfTI with its own new checkpoint.

infer already saves restored QSM values. This utility is only for separately exported,
normalized floating-point network outputs. It cannot recover unknown legacy
normalization or undo clipping, uint8 quantization or a nonlinear activation.
"""
import argparse
import hashlib
from pathlib import Path

from .util.data import load_volume, require_grid
from .util.evaluation import prediction_metadata
from .util.nifti import save_qsm
from .util.normalization import from_checkpoint
from .util.model_loading import load_checkpoint_snapshot
from .util.normalization_workflow import physical_output_metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="Raw floating-point network output, before inverse normalization")
    p.add_argument("--input-value-space", choices=["network_normalized"], required=True,
                   help="Explicit declaration; never use this on physical predictions from infer")
    p.add_argument("--swi-mask", required=True, help="Set only outside-brain padding to numeric zero")
    p.add_argument("--output", required=True)
    p.add_argument("--output-units", choices=["ppm", "ppb", "stored"])
    args = p.parse_args(argv)
    checkpoint, sha = load_checkpoint_snapshot(args.checkpoint)
    norm, spec = from_checkpoint(checkpoint)
    norm.denormalize(0., args.output_units)
    source = prediction_metadata(args.input)
    if source.get("ValueSpace") not in {None, "network_normalized"}:
        raise ValueError("Input is already physical or has incompatible value-space metadata; refusing double inverse")
    if source.get("InverseNormalization", {}).get("applied") is True:
        raise ValueError("Inverse normalization was already applied")
    if source.get("Units") not in {None, "dimensionless"}:
        raise ValueError("Normalized network outputs must not declare restored ppm/ppb/stored units")
    if source.get("CheckpointSHA256") not in {None, sha}:
        raise ValueError("Input was exported with a different checkpoint")
    if source.get("Normalization") not in (None, spec):
        raise ValueError("Input and checkpoint normalization disagree")
    input_sha = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    image, values = load_volume(args.input)
    if image.get_data_dtype().kind != "f":
        raise ValueError("Input must store floating-point network outputs; integer/PIL/uint8 information cannot be restored")
    mask_ref, mask = load_volume(args.swi_mask)
    require_grid(image, mask_ref)
    if not (mask > 0).any():
        raise ValueError("A nonempty input brain mask is required")
    values = norm.denormalize(values, output_units=args.output_units)
    values[mask <= 0] = 0
    if hashlib.sha256(Path(args.input).read_bytes()).hexdigest() != input_sha:
        raise RuntimeError("Raw network output changed while restoring scale")
    config = checkpoint.get("config", {})
    reference = checkpoint.get("data_reference") or config.get("data_reference")
    save_qsm(args.output, values, image, {
        **physical_output_metadata(norm, args.output_units), "Normalization": spec,
        "NormalizationSource": config.get("normalization_source"), "CheckpointSHA256": sha,
        "ReferenceID": reference.get("id") if reference else config.get("reference_id"),
        "DataReference": reference,
        "DataIndependence": checkpoint.get("data_independence") or config.get("data_independence"),
        "TrainingState": checkpoint.get("training_state"),
        "DataQC": {"status": "unverified", "reason": "Restoration only; no registration/geometry/ROI QC performed"},
        "Source": {"path": str(Path(args.input).resolve()),
                   "sha256": input_sha,
                   "value_space_declaration": "sidecar" if source.get("ValueSpace") else "explicit_cli",
                   "checkpoint_binding": "sidecar" if source.get("CheckpointSHA256") else "user_declaration"},
        "OutsideBrainMask": 0, "PostHocCalibration": False,
    })
    print(f"Restored QSM values in {args.output_units or norm.units}: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

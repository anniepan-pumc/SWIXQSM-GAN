"""SWI-only inference using the preprocessing stored with a new checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ..util.data import load_volume, require_grid, normalize_swi, pad_slice
from ..util.model_loading import generator_from_checkpoint, load_checkpoint_snapshot
from ..util.normalization import from_checkpoint
from ..util.normalization_workflow import physical_output_metadata
from ..util.nifti import save_qsm, sidecar_path
from ..util.quality import inspect_geometry, validate_contract
from ..util.reference import assert_reference_identity
from ..util.independence_gate import assert_prediction_phase


@torch.no_grad()
def predict_volume(gen, swi, mask, size, normalization, device="cpu", swi_spec=None, output_units=None):
    """Return restored QSM values, never normalized network output. Invert once.

    qsm = (network_output * training_scale + training_offset), followed only
    by the requested ppm/ppb conversion. ``stored`` restores only the unverified
    numerical label scale and cannot convert to ppm/ppb. Signed values survive.
    No reference QSM or test-set target statistics enter this calculation.
    """
    gen.eval()
    x, fitted = normalize_swi(swi, mask > 0, spec=swi_spec, return_metadata=True)
    result = np.empty_like(x)
    for z in range(x.shape[2]):
        tensor = pad_slice(x[:, :, z], size)[None].to(device)
        normalized = gen(tensor)[0, 0, :x.shape[1], :x.shape[0]]
        result[:, :, z] = normalization.denormalize(normalized, output_units=output_units).cpu().numpy().T
    result[mask <= 0] = 0
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite prediction")
    return result, fitted


def load_inference_context(checkpoint_path, device="cpu", expected_phase=None):
    """Load/check the checkpoint once; safely reuse the generator across cases."""
    checkpoint, checkpoint_sha = load_checkpoint_snapshot(checkpoint_path)
    training_state = checkpoint.get("training_state")
    assert_prediction_phase({"TrainingState": training_state}, expected_phase)
    normalization, spec = from_checkpoint(checkpoint)
    config = checkpoint["config"]
    trained_qc = checkpoint.get("data_qc") or config.get("data_qc") or {}
    if normalization.units == "stored" and trained_qc.get("status") == "verified":
        raise ValueError("stored checkpoint cannot claim verified physical-unit training QC")
    data_reference = checkpoint.get("data_reference") or config.get("data_reference")
    if data_reference is not None:
        assert_reference_identity(data_reference, data_reference)
    if trained_qc.get("status") == "verified" and data_reference is not None:
        assert_reference_identity(validate_contract(trained_qc["contract"])["reference"], data_reference)
    gen = generator_from_checkpoint(checkpoint).to(device).eval()
    return dict(checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha, config=config,
                generator=gen, device=device, normalization=normalization, spec=spec,
                trained_qc=trained_qc, data_reference=data_reference, training_state=training_state)


def infer_case(context, swi_path, swi_mask_path, output, *, output_units=None,
               reference_id=None, allow_unverified_data=None, extra_metadata=None):
    """Save one restored prediction; only SWI and its brain mask are decoded."""
    if Path(output).exists() or sidecar_path(output).exists():
        raise FileExistsError("Output already exists")
    checkpoint, config = context["checkpoint"], context["config"]
    normalization, spec = context["normalization"], context["spec"]
    normalization.denormalize(0., output_units)  # Validate conversion before inference or data reads.
    trained_qc, data_reference = context["trained_qc"], context["data_reference"]
    if reference_id is not None and (not reference_id.strip()
            or data_reference is not None and reference_id.strip() != data_reference["id"]):
        raise ValueError("Reference ID conflicts with checkpoint provenance or is empty")
    ref, swi = load_volume(swi_path)
    mask_ref, mask = load_volume(swi_mask_path)
    require_grid(ref, mask_ref)
    input_qc = {"status": "unverified", "training_qc_status": trained_qc.get("status", "not_recorded")}
    if allow_unverified_data is not None:
        if not allow_unverified_data.strip():
            raise ValueError("A nonempty reason is required for unverified inference")
        input_qc["reason"] = allow_unverified_data.strip()
    elif trained_qc.get("status") == "verified":
        contract = validate_contract(trained_qc["contract"])
        geometry = inspect_geometry(ref, contract["geometry"])
        mask_geometry = inspect_geometry(mask_ref, contract["geometry"])
        if not geometry["passed"] or not mask_geometry["passed"]:
            raise ValueError(f"Inference grid violates training data contract: {geometry.get('errors')}; {mask_geometry.get('errors')}")
        if ref.header.get_xyzt_units()[0] != mask_ref.header.get_xyzt_units()[0]:
            raise ValueError("SWI and brain-mask spatial units disagree")
        if not isinstance(data_reference, dict):
            raise ValueError("Verified checkpoint has no reference provenance")
        input_qc.update(status="geometry_verified", geometry=geometry, mask_geometry=mask_geometry,
                        limitations="SWI-only geometry check; no inference reference QSM or cross-acquisition registration is available")
    else:
        raise ValueError("Legacy/unverified checkpoint requires --allow-unverified-data REASON; its reference and geometry cannot be inferred")
    result, fitted = predict_volume(context["generator"], swi, mask, config["size"], normalization,
                                    context["device"], spec["swi"], output_units)
    metadata = {
        **physical_output_metadata(normalization, output_units), "Normalization": spec,
        "NormalizationSource": config.get("normalization_source"),
        "ReferenceID": data_reference["id"] if data_reference is not None else reference_id,
        "DataReference": data_reference, "DataQC": input_qc,
        "DataIndependence": checkpoint.get("data_independence") or config.get("data_independence") or {"status": "not_recorded"},
        "TrainingState": context["training_state"], "CheckpointRole": checkpoint.get("checkpoint_role", "not_recorded"),
        "TrainingObjective": checkpoint.get("training_objective") or config.get("training_objective"),
        "TrainingDiagnostics": checkpoint.get("training_diagnostics"),
        "Initialization": config.get("initialization"),
        "ModelSpec": checkpoint.get("model_spec") or config.get("model_spec"),
        "ExperimentRecipe": config.get("recipe"),
        "TrainingQCReportSHA256": trained_qc.get("report_sha256"),
        "SWINormalizationApplied": fitted, "OutsideBrainMask": 0,
        "CheckpointSHA256": context["checkpoint_sha256"],
    }
    if extra_metadata:
        if set(extra_metadata) & set(metadata):
            raise ValueError("Extra inference metadata cannot override physical/checkpoint provenance")
        metadata.update(extra_metadata)
    save_qsm(output, result, ref, metadata)
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--swi", required=True)
    p.add_argument("--swi-mask", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-units", choices=["ppm", "ppb", "stored"], help="Default: checkpoint parameter units; inverse transform is always applied once")
    p.add_argument("--allow-unverified-data", metavar="REASON", help="Explicitly accept legacy/mismatched geometry as unverified")
    p.add_argument("--reference-id", help="Legacy declaration or consistency check; does not re-reference predictions")
    p.add_argument("--expected-phase", choices=["l1", "warmup", "gan"],
                   help="Require this recorded current-run phase; GAN also requires actual adversarial updates")
    args = p.parse_args(argv)
    if Path(args.output).exists() or sidecar_path(args.output).exists():
        p.error("Output already exists")
    context = load_inference_context(args.checkpoint, args.device, args.expected_phase)
    infer_case(context, args.swi, args.swi_mask, args.output, output_units=args.output_units,
               reference_id=args.reference_id, allow_unverified_data=args.allow_unverified_data)


if __name__ == "__main__":
    main()

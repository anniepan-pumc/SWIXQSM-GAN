"""Bind one train-fitted transform to prospective training and physical outputs."""
import hashlib
from pathlib import Path

from .normalization import QSMNormalization, restored_value_metadata
from .independence_gate import assert_source_independence
from .reference import assert_reference_identity
from .quality_gate import bind_input_units


def add_training_normalization_arguments(parser, *, required=False):
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--normalization-file", help="Train-only JSON from swixqsm_minimal.fit_normalization")
    group.add_argument("--qsm-scale", type=float, help="Manually declared positive fixed scale in --units")
    parser.add_argument("--qsm-offset", type=float, help="Manual fixed centering; defaults to zero. Cannot accompany --normalization-file")
    parser.add_argument("--qsm-units", choices=["ppm", "ppb", "stored"], help="Source-label fallback if CSV lacks qsm_units; stored preserves unverified numerical values; fitted file/QC declaration is inherited when available")


def bind_training_units(rows, args, data_qc):
    """The canonical output unit must never override a fitted source unit."""
    fallback = getattr(args, "qsm_units", None)
    path = getattr(args, "normalization_file", None)
    if path:
        from ..fit_normalization import load_fitted_normalization
        _, artifact = load_fitted_normalization(path, units=args.units, verify_inputs=False)
        fitted_fallback = artifact["fit_selection"].get("default_input_qsm_units")
        if fallback is not None and fitted_fallback is not None and fallback != fitted_fallback:
            raise ValueError("Source --qsm-units conflicts with the fitted normalization input-unit declaration")
        if fallback is None:
            fallback = fitted_fallback
    elif fallback is None and data_qc.get("status") != "verified":
        # Preserve the original manual-scale CLI, with an explicit recorded unit.
        fallback = args.units
    return bind_input_units(rows, data_qc, requested_fallback=fallback)


def resolve_training_normalization(args, rows, data_qc, data_independence):
    """Never fit on the fly; recheck training identities, units and image hashes."""
    if args.units == "stored" and data_qc.get("status") == "verified":
        raise ValueError("stored values cannot use a verified physical-unit QC contract; explicitly select unverified data")
    path = getattr(args, "normalization_file", None)
    if not path:
        norm = QSMNormalization(args.qsm_scale,
                                0. if args.qsm_offset is None else args.qsm_offset, args.units)
        return norm, {"kind": "manual_fixed", "normalization": norm.to_dict(),
                      "limitation": "Manual parameters are a declaration, not a verified train-only fit"}
    if args.qsm_scale is not None or args.qsm_offset is not None:
        raise ValueError("--normalization-file cannot accompany --qsm-scale or --qsm-offset")
    from ..fit_normalization import load_fitted_normalization
    path = Path(path).resolve()
    artifact_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    norm, artifact = load_fitted_normalization(path, units=args.units, rows=rows, verify_inputs=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact_sha:
        raise RuntimeError("Normalization artifact changed while verifying training inputs")
    if data_qc.get("status") == "verified":
        if artifact.get("data_qc", {}).get("status") != "verified":
            raise ValueError("Normalization fit has no verified data QC; refit using --qc-report or explicitly select unverified data")
        assert_reference_identity(data_qc.get("data_reference"), artifact.get("data_reference"))
    if data_independence.get("status") == "verified":
        assert_source_independence(data_independence, artifact.get("data_independence"), name="Normalization fit")
    return norm, {"kind": "train_fitted", "path": str(path),
                  "sha256": artifact_sha,
                  "schema": artifact["schema"], "version": artifact["version"],
                  "method": artifact["method"], "fit_selection": artifact["fit_selection"],
                  "normalization": norm.to_dict(),
                  "data_qc": artifact.get("data_qc"), "data_reference": artifact.get("data_reference"),
                  "data_independence": artifact.get("data_independence")}


def physical_output_metadata(normalization, output_units=None):
    """Describe restored values; these must not be denormalized again.

    The historical function name remains compatible. ``stored`` restores only
    the decoded numerical label scale and never declares physical QSM units.
    """
    units = output_units or normalization.units
    # Validate the requested conversion even when metadata is produced alone.
    normalization.denormalize(0., output_units=units)
    value_metadata = restored_value_metadata(units)
    return {**value_metadata,
            "InverseNormalization": {
                "applied": True, "input_value_space": "network_normalized",
                "output_value_space": value_metadata["ValueSpace"], "clip": False,
                "formula": "(network_output * scale + offset) * unit_factor(parameter_units, output_units)",
                "scale": normalization.scale, "offset": normalization.offset,
                "parameter_units": normalization.units, "output_units": units}}

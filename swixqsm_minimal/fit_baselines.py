"""Fit train-only mean/linear maps in a declared, pre-registered common space."""
import argparse
import hashlib
from pathlib import Path

from .util.baselines import fit_baselines, save_baselines
from .util.data import read_manifest
from .util.quality_gate import add_quality_arguments, check_data_quality, prepare_quality_rows, bind_input_units
from .util.independence_gate import add_independence_arguments, check_independence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New .npz artifact; will not overwrite")
    parser.add_argument("--units", choices=["ppm", "ppb"], default="ppb")
    parser.add_argument("--qsm-units", choices=["ppm", "ppb"],
                        help="Explicit fallback only for a CSV without qsm_units")
    parser.add_argument("--reference-id", required=True, help="Actual susceptibility reference convention")
    parser.add_argument("--space-id", required=True, help="Prepared common anatomical space; matching grids alone are insufficient")
    add_quality_arguments(parser)
    add_independence_arguments(parser)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists")
    rows = prepare_quality_rows(read_manifest(args.manifest), args.manifest)
    data_independence = check_independence(rows, args, content_splits=("train",), check_initialization=False)
    consumed_rows = [row for row in rows if row["split"] == "train"]
    data_qc = check_data_quality(consumed_rows, args, reference_id=args.reference_id)
    consumed_rows = bind_input_units(consumed_rows, data_qc, requested_fallback=args.qsm_units)
    bundle = fit_baselines(consumed_rows, units=args.units, default_qsm_units=args.qsm_units,
                           reference_id=args.reference_id, space_id=args.space_id)
    bundle.metadata["provenance"] = {
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "command": "python -m swixqsm_minimal.fit_baselines",
        "image_access": "Only rows with split=train are loaded or hashed by fitting",
    }
    bundle.metadata["data_qc"] = data_qc
    bundle.metadata["data_reference"] = data_qc.get("data_reference")
    bundle.metadata["data_independence"] = data_independence
    save_baselines(bundle, output)
    print(f"Saved train-only baselines from {bundle.metadata['fit']['n_subjects']} subjects "
          f"({bundle.metadata['fit']['n_visits']} visits) in {bundle.units} to {output}", flush=True)


if __name__ == "__main__":
    main()

"""Inspect prepared pairs without changing registration, orientation or QSM values."""
import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path

from .util.data import read_manifest
from .util.qc_preview import write_preview
from .util.quality import build_report, load_contract, resolve_qc_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--contract", required=True, help="Explicit geometry/reference/registration policy JSON")
    parser.add_argument("--output", required=True, help="New directory for report.json and preview.html")
    parser.add_argument("--split", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--qsm-units", choices=["ppm", "ppb"], help="Fallback when qsm_units is absent")
    parser.add_argument("--preview-limit", type=int, default=12)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("QC output already exists")
    if args.preview_limit < 0:
        parser.error("preview-limit must be nonnegative")
    # Keep structural/leakage checks, but let per-case QC record missing files
    # and invalid units instead of aborting before a failed report is written.
    rows = read_manifest(args.manifest, validate_inputs=False)
    rows = resolve_qc_paths([row for row in rows if row["split"] in args.split], args.manifest)
    if not rows:
        parser.error("No cases in the selected split(s)")
    contract = load_contract(args.contract)
    report = build_report(rows, contract, default_qsm_units=args.qsm_units)
    report["provenance"] = {
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "contract_path": str(Path(args.contract).resolve()),
        "contract_file_sha256": hashlib.sha256(Path(args.contract).read_bytes()).hexdigest(),
        "qsm_units_fallback": args.qsm_units,
        "selected_splits": args.split, "python_version": platform.python_version(),
        "scope": "Prepared image QC; no registration, resampling, reorientation or susceptibility re-referencing performed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".swixqsm-qc-", dir=output.parent) as tmp:
        staging = Path(tmp) / "results"
        staging.mkdir()
        (staging / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (staging / "manifest_resolved.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        write_preview(staging / "preview.html", rows, report["cases"], limit=args.preview_limit)
        if output.exists():
            raise FileExistsError("QC destination appeared while inspecting; refusing to replace it")
        staging.rename(output)
    passed = sum(bool(case["passed"]) for case in report["cases"])
    print(f"QC cases accepted: {passed}/{len(rows)}. Report: {(output/'report.json').resolve()}")
    if passed != len(rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

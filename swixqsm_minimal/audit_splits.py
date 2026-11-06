"""Audit all splits and declared pretrained exposure before model construction."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

from .util.independence import MAIN_PATHS, build_independence_report
from .util.quality import resolve_qc_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New directory containing the audit report")
    parser.add_argument("--checkpoint", help="Actual pretrained G initialization checkpoint")
    parser.add_argument("--history", help="Checkpoint-bound initialization exposure history JSON")
    args = parser.parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error("Audit output already exists; use a new directory")
    # Do not call training's strict read_manifest: alias/duplicate/missing-file
    # failures belong in the persisted audit instead of aborting before it.
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in MAIN_PATHS:
            if row.get(key) and row[key].strip():
                path = Path(row[key]).expanduser()
                row[key] = str((manifest.parent/path if not path.is_absolute() else path).resolve())
    # Consumer entry points resolve these optional metadata paths too. Keep the
    # full-manifest identity identical without opening optional QC evidence here.
    rows = resolve_qc_paths(rows, manifest)
    report = build_independence_report(rows, checkpoint_path=args.checkpoint, history_path=args.history)
    report["provenance"] = {"manifest_path": str(manifest),
                            "manifest_file_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                            "read_scope": "All manifest splits: decoded SWI and primary-file SHA256; no training or model selection"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".swixqsm-independence-", dir=output.parent) as temporary:
        staging = Path(temporary)/"results"
        staging.mkdir()
        (staging/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        (staging/"manifest_resolved.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError("Audit destination appeared; refusing to overwrite it")
        staging.rename(output)
    print(f"Independence audit {report['status']}: {output/'report.json'}", flush=True)
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

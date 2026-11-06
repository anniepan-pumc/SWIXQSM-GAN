"""Explicit, auditable entry-point gating for verified and legacy datasets."""
import sys
import math
import hashlib
from pathlib import Path

from .quality import resolve_qc_paths, verify_report
from .reference import assert_reference_identity
from .normalization import unit_factor


def add_quality_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qc-report", help="Hash-bound report.json produced by swixqsm_minimal.qc")
    group.add_argument("--allow-unverified-data", metavar="REASON",
                       help="Explicit legacy opt-out; recorded as unverified, never QC passed")


def check_data_quality(rows, args, *, reference_id=None):
    if args.qc_report:
        return verify_report(rows, args.qc_report, reference_id=reference_id)
    reason = args.allow_unverified_data
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("An explicit nonempty reason is required to bypass data QC")
    print("Data QC is UNVERIFIED. " + reason.strip(), file=sys.stderr)
    references = sorted({row["qsm_reference"] for row in rows if row.get("qsm_reference")})
    return {"status": "unverified", "reason": reason.strip(),
            "reference_id_declaration": reference_id,
            "manifest_reference_declarations": references,
            "n_cases": len(rows),
            "limitations": "No verified registration/reference/geometry contract; existing grid checks alone do not establish alignment"}


def prepare_quality_rows(rows, manifest):
    return resolve_qc_paths(rows, manifest)


def bind_input_units(rows, data_qc, requested_fallback=None):
    """Carry the inspected label-unit interpretation into the consuming path."""
    bound = []
    for original in rows:
        row = dict(original)
        if data_qc.get("status") == "verified" and (requested_fallback == "stored" or row.get("qsm_units") == "stored"):
            raise ValueError("Verified QC requires physical ppm/ppb units; stored values require explicit unverified data")
        if "qsm_units" not in row:
            units = (data_qc.get("default_qsm_units") if data_qc["status"] == "verified"
                     else requested_fallback)
            if data_qc["status"] == "verified" and requested_fallback is not None and requested_fallback != units:
                raise ValueError("Runtime QSM-unit fallback conflicts with the verified QC report")
            if data_qc["status"] == "verified" and units == "stored":
                raise ValueError("Verified QC cannot supply an unverified stored-unit fallback")
            unit_factor(units, units)
            row["qsm_units"] = units
        bound.append(row)
    return bound


def check_prediction_references(rows, data_reference):
    """A matching name alone is insufficient for strictly checked predictions."""
    from .evaluation import prediction_metadata
    for row in rows:
        if not row.get("prediction_path"):
            continue
        metadata = prediction_metadata(row["prediction_path"])
        source_qc = metadata.get("PredictionDataQC", metadata.get("DataQC"))
        if source_qc is not None:
            if not isinstance(source_qc, dict):
                raise ValueError("Prediction QC provenance must be a JSON object")
            status = source_qc.get("status")
            if not isinstance(status, str) or status not in {"verified", "geometry_verified", "passed", "not_recorded"}:
                raise ValueError(f"Prediction was generated with {status or 'unknown'} QC; use explicit legacy evaluation rather than relabelling it as verified")
        declared = metadata.get("DataReference")
        if declared is not None:
            assert_reference_identity(data_reference, declared)
        original_reference = metadata.get("PredictionDataReference")
        if original_reference is not None:
            assert_reference_identity(data_reference, original_reference)
            if declared is None:
                declared = original_reference
        columns = ("prediction_reference", "prediction_reference_method",
                   "prediction_reference_region", "prediction_reference_target_ppb")
        provided = [row.get(key) for key in columns]
        if any(value not in (None, "") for value in provided):
            # A legacy reference-id column alone can coexist with a complete sidecar.
            if declared is None or any(row.get(key) not in (None, "") for key in columns[1:]):
                if not all(value not in (None, "") for value in provided):
                    raise ValueError("Strict prediction QC requires method, region and target_ppb, not only a reference name")
                try:
                    target = float(row["prediction_reference_target_ppb"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("Invalid prediction_reference_target_ppb") from exc
                if not math.isfinite(target):
                    raise ValueError("prediction_reference_target_ppb must be finite")
                identity = dict(zip(("id", "method", "region", "target_ppb"), provided))
                identity["target_ppb"] = target
                assert_reference_identity(data_reference, identity)
                declared = identity
        if declared is None:
            raise ValueError("Prediction has no full reference provenance; provide DataReference or explicit method/region/target declarations, or use an auditable legacy opt-out")


def prediction_sources(rows):
    """Keep original prediction provenance separate from evaluation-pair QC."""
    from .evaluation import prediction_metadata
    from .nifti import sidecar_path

    def digest(path):
        value = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    sources = {}
    for row in rows:
        path = row.get("prediction_path")
        if not path:
            continue
        metadata = prediction_metadata(path)
        sidecar = sidecar_path(path)
        sources[path] = {"prediction_path": path, "prediction_sha256": digest(path),
                         "sidecar_sha256": digest(sidecar) if sidecar.exists() else None,
                         "PredictionDataQC": metadata.get("PredictionDataQC", metadata.get("DataQC", {"status": "not_recorded"})),
                         "PredictionDataReference": metadata.get("PredictionDataReference", metadata.get("DataReference")),
                         "PredictionReferenceID": metadata.get("PredictionReferenceID", metadata.get("ReferenceID")),
                         "PredictionDataIndependence": metadata.get("PredictionDataIndependence", metadata.get("DataIndependence", {"status": "not_recorded"})),
                         "PredictionTrainingState": metadata.get("PredictionTrainingState", metadata.get("TrainingState")),
                         "PredictionTrainingObjective": metadata.get("PredictionTrainingObjective", metadata.get("TrainingObjective")),
                         "PredictionTrainingDiagnostics": metadata.get("PredictionTrainingDiagnostics", metadata.get("TrainingDiagnostics")),
                         "PredictionCheckpointRole": metadata.get("PredictionCheckpointRole", metadata.get("CheckpointRole", "not_recorded")),
                         "PredictionInitialization": metadata.get("PredictionInitialization", metadata.get("Initialization")),
                         "PredictionModelSelection": metadata.get("PredictionModelSelection", metadata.get("EvaluationModelSelection")),
                         "SourceCheckpointSHA256": metadata.get("CheckpointSHA256", metadata.get("SourceCheckpointSHA256")),
                         "TrainingQCReportSHA256": metadata.get("TrainingQCReportSHA256")}
    return sources

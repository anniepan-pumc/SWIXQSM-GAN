"""Keep split/initialization evidence separate from geometric data QC."""
import hashlib
import sys
from pathlib import Path

from .independence import verify_independence_report


def add_independence_arguments(parser, *, required=False):
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--independence-report", help="report.json produced by swixqsm_minimal.audit_splits")
    group.add_argument("--allow-unverified-independence", metavar="REASON",
                       help="Record missing identity/pretraining evidence as unverified; never proves independence")


def check_independence(rows, args, *, checkpoint_path=None,
                       content_splits=("train", "val"), check_initialization=True):
    if args.independence_report:
        return verify_independence_report(rows, args.independence_report,
                                          checkpoint_path=checkpoint_path,
                                          content_splits=content_splits,
                                          check_initialization=check_initialization)
    reason = args.allow_unverified_independence
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("A nonempty reason is required for unverified independence")
    reason = reason.strip() if reason is not None else "No independence report supplied"
    print("Data independence is UNVERIFIED. " + reason, file=sys.stderr)
    checkpoint_hash = None
    if checkpoint_path is not None:
        digest = hashlib.sha256()
        with Path(checkpoint_path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_hash = digest.hexdigest()
    return {"status": "unverified", "reason": reason,
            "initialization": {"kind": ("pretrained" if checkpoint_path is not None else "random")
                               if check_initialization else "not_assessed",
                               "checkpoint_sha256": checkpoint_hash,
                               "history_coverage": "not_audited"},
            "limitations": "Subject/path checks and image QC do not establish global identity or absence of historical holdout exposure"}


def assert_source_independence(expected, source, *, name):
    if not isinstance(source, dict) or source.get("status") != "verified":
        raise ValueError(f"{name} has no verified independence provenance; provide an audited source or explicitly select unverified independence")
    identity = expected.get("manifest_identity_sha256")
    if not identity or source.get("manifest_identity_sha256") != identity:
        raise ValueError(f"{name} was audited against a different identity/split manifest")


def assert_prediction_phase(metadata, expected):
    if expected is None:
        return
    state = metadata.get("PredictionTrainingState", metadata.get("TrainingState"))
    if not isinstance(state, dict) or state.get("current_run_phase") != expected:
        raise ValueError(f"Expected current-run {expected} prediction, but checkpoint phase is missing or different")
    if expected == "gan":
        count = state.get("adversarial_generator_updates")
        if type(count) is not int or count <= 0:
            raise ValueError("GAN prediction has no recorded current-run adversarial generator updates")


def check_prediction_independence(rows, expected, *, expected_phase=None):
    from .evaluation import prediction_metadata
    for row in rows:
        if not row.get("prediction_path"):
            continue
        metadata = prediction_metadata(row["prediction_path"])
        if expected.get("status") == "verified":
            source = metadata.get("PredictionDataIndependence", metadata.get("DataIndependence"))
            assert_source_independence(expected, source, name="Prediction")
        assert_prediction_phase(metadata, expected_phase)

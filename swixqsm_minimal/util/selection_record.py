"""Bind a frozen model's evaluation to its documented selection procedure.

This checks files and consistency of declarations. It does not infer how a
model was selected from its weights, and never chooses or changes a model.
"""
import hashlib
import json
from pathlib import Path
import re

from .evaluation import prediction_metadata


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_selection_record(path, rows):
    path = Path(path).expanduser().resolve()
    record = json.loads(path.read_text())
    required = {"schema", "version", "checkpoint_path", "checkpoint_sha256", "selection_basis",
                "test_feedback_used", "validation_used", "source", "evidence_files"}
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError(f"Model selection record requires exactly {sorted(required)}")
    if (record["schema"] != "swixqsm.model_selection_record"
            or type(record["version"]) is not int or record["version"] != 1):
        raise ValueError("Unsupported model selection record schema/version")
    sha = record["checkpoint_sha256"]
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha) is None:
        raise ValueError("checkpoint_sha256 must be a 64-digit SHA256")
    sha = sha.lower()
    basis = record["selection_basis"]
    if not isinstance(basis, str) or basis not in {"fixed_before_test", "training_only", "test_based", "unknown"}:
        raise ValueError("Unsupported selection_basis")
    feedback = record["test_feedback_used"]
    if feedback is not None and type(feedback) is not bool:
        raise ValueError("test_feedback_used must be true, false, or null when unknown")
    if type(record["validation_used"]) is not bool:
        raise ValueError("validation_used must explicitly be boolean")
    if not isinstance(record["source"], str) or not record["source"].strip():
        raise ValueError("Model selection source must be nonempty")
    if (basis in {"fixed_before_test", "training_only"} and feedback is True
            or basis == "test_based" and feedback is False):
        raise ValueError("selection_basis conflicts with test_feedback_used")

    def resolve(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Checkpoint and evidence paths must be nonempty strings")
        target = Path(value).expanduser()
        return (path.parent / target if not target.is_absolute() else target).resolve()

    checkpoint = resolve(record["checkpoint_path"])
    if _hash(checkpoint) != sha:
        raise ValueError("Frozen checkpoint bytes do not match the model selection record")
    if not isinstance(record["evidence_files"], list):
        raise ValueError("evidence_files must be a list of actual logs or protocol files")
    evidence = []
    for value in record["evidence_files"]:
        target = resolve(value)
        evidence.append({"path": str(target), "sha256": _hash(target)})
    predictions = [row for row in rows if row.get("prediction_path")]
    if not predictions or len(predictions) != len(rows):
        raise ValueError("A model selection record must bind predictions for every evaluated case")
    if any(row.get("split") != "test" for row in predictions):
        raise ValueError("Held-out model selection records require split=test evaluation")
    for row in predictions:
        metadata = prediction_metadata(row["prediction_path"])
        hashes = [metadata[key] for key in ("CheckpointSHA256", "SourceCheckpointSHA256")
                  if metadata.get(key) is not None]
        if not hashes or any(not isinstance(value, str) or value.lower() != sha for value in hashes):
            raise ValueError("Prediction checkpoint provenance does not match the frozen model selection record")
        state = metadata.get("PredictionTrainingState", metadata.get("TrainingState"))
        if (isinstance(state, dict) and state.get("selection_split") == "val"
                and state.get("validated_epoch_indices") and record["validation_used"] is False):
            raise ValueError("Prediction records observed validation; validation_used=false conflicts with its provenance")
        previous = metadata.get("PredictionModelSelection", metadata.get("EvaluationModelSelection"))
        if previous is not None:
            if not isinstance(previous, dict):
                raise ValueError("Previous model selection provenance must be a JSON object")
            if previous.get("checkpoint_sha256") not in (None, sha):
                raise ValueError("Previous model selection checkpoint disagrees with prediction provenance")
            if (previous.get("status") == "development_evaluation" or previous.get("test_feedback_used") is True) and feedback is not True:
                raise ValueError("Previously recorded test-based selection cannot be relabelled held-out or unknown")
            if previous.get("validation_used") is True and record["validation_used"] is False:
                raise ValueError("Previously recorded validation cannot be removed from the model selection history")
    status = ("development_evaluation" if feedback is True or basis == "test_based" else
              "heldout_by_declaration" if feedback is False and basis in {"fixed_before_test", "training_only"}
              else "unverified")
    return {"status": status, "record_path": str(path), "record_sha256": _hash(path),
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha,
            "selection_basis": basis, "test_feedback_used": feedback,
            "validation_used": record["validation_used"], "source": record["source"].strip(),
            "evidence_files": evidence,
            "evidence_level": "declaration_with_hashed_files" if evidence else "declaration_only",
            "weight_updates_performed": False,
            "limitations": [
                "Absence of a validation split does not itself invalidate a test set; model selection must be fixed independently of test feedback.",
                "This checks byte-level binding and declared selection consistency, not the truth of historical decisions or completeness of ancestor training identities.",
                "Additional metrics on a frozen model do not create an L1-only ablation or change which data participated in development.",
            ]}

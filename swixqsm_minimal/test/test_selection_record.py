import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ..util.nifti import sidecar_path
from ..util.selection_record import inspect_selection_record


class SelectionRecordTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "frozen.pt"
        self.checkpoint.write_bytes(b"Synthetic immutable checkpoint; not a real model")
        self.sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.log = self.root / "training.log"
        self.log.write_text("Synthetic declaration evidence; no real subjects or performance")
        self.prediction = self.root / "prediction.nii.gz"
        sidecar_path(self.prediction).write_text(json.dumps({"CheckpointSHA256": self.sha}))
        self.rows = [{"subject_id": "synthetic_test", "split": "test", "prediction_path": str(self.prediction)}]
        self.path = self.root / "selection.json"
        self.record = {"schema": "swixqsm.model_selection_record", "version": 1,
                       "checkpoint_path": "frozen.pt", "checkpoint_sha256": self.sha,
                       "selection_basis": "fixed_before_test", "test_feedback_used": False,
                       "validation_used": False, "source": "Synthetic authored procedure",
                       "evidence_files": ["training.log"]}

    def inspect(self):
        self.path.write_text(json.dumps(self.record))
        return inspect_selection_record(self.path, self.rows)

    def test_no_validation_is_allowed_for_frozen_heldout_test(self):
        before = self.checkpoint.read_bytes()
        result = self.inspect()
        self.assertEqual(result["status"], "heldout_by_declaration")
        self.assertFalse(result["validation_used"])
        self.assertFalse(result["weight_updates_performed"])
        self.assertEqual(result["evidence_files"][0]["sha256"], hashlib.sha256(self.log.read_bytes()).hexdigest())
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.record["selection_basis"] = "training_only"
        self.assertEqual(self.inspect()["status"], "heldout_by_declaration")

    def test_test_based_selection_and_unknown_are_never_relabelled_heldout(self):
        self.record.update(selection_basis="test_based", test_feedback_used=True)
        self.assertEqual(self.inspect()["status"], "development_evaluation")
        self.record.update(selection_basis="unknown", test_feedback_used=None)
        self.assertEqual(self.inspect()["status"], "unverified")
        self.record.update(selection_basis="fixed_before_test", test_feedback_used=None)
        self.assertEqual(self.inspect()["status"], "unverified")

    def test_checkpoint_and_prediction_hashes_must_match(self):
        sidecar_path(self.prediction).write_text(json.dumps({"CheckpointSHA256": "f" * 64}))
        with self.assertRaisesRegex(ValueError, "Prediction checkpoint"):
            self.inspect()
        sidecar_path(self.prediction).write_text(json.dumps({"SourceCheckpointSHA256": self.sha}))
        self.assertEqual(self.inspect()["checkpoint_sha256"], self.sha)
        self.checkpoint.write_bytes(b"Changed frozen model")
        with self.assertRaisesRegex(ValueError, "Frozen checkpoint"):
            self.inspect()

    def test_missing_prediction_origin_and_inconsistent_declarations_rejected(self):
        sidecar_path(self.prediction).write_text("{}")
        with self.assertRaises(ValueError):
            self.inspect()
        sidecar_path(self.prediction).write_text(json.dumps({"CheckpointSHA256": self.sha}))
        for changes in ({"selection_basis": "test_based", "test_feedback_used": False},
                        {"test_feedback_used": True}, {"validation_used": None}, {"version": True}):
            original = dict(self.record)
            self.record.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.inspect()
            self.record = original
        self.rows.append({"subject_id": "synthetic_other", "split": "test"})
        with self.assertRaises(ValueError):
            self.inspect()

    def test_declarations_without_logs_are_not_claimed_verified(self):
        self.record["evidence_files"] = []
        result = self.inspect()
        self.assertEqual(result["evidence_level"], "declaration_only")
        self.assertNotEqual(result["status"], "verified")

    def test_heldout_scope_and_known_validation_evidence_cannot_be_overridden(self):
        self.rows[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "split=test"):
            self.inspect()
        self.rows[0]["split"] = "test"
        sidecar_path(self.prediction).write_text(json.dumps({"CheckpointSHA256": self.sha,
            "TrainingState": {"selection_split": "val", "validated_epoch_indices": [0]}}))
        with self.assertRaisesRegex(ValueError, "observed validation"):
            self.inspect()
        self.record["validation_used"] = True
        self.assertEqual(self.inspect()["status"], "heldout_by_declaration")

    def test_prior_development_selection_is_preserved(self):
        original = {"status": "development_evaluation", "test_feedback_used": True,
                    "checkpoint_sha256": self.sha, "validation_used": False}
        metadata = {"CheckpointSHA256": self.sha, "PredictionModelSelection": original,
                    "EvaluationModelSelection": {"status": "heldout_by_declaration"}}
        sidecar_path(self.prediction).write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Previously recorded test-based"):
            self.inspect()
        self.record.update(selection_basis="test_based", test_feedback_used=True)
        self.assertEqual(self.inspect()["status"], "development_evaluation")
        original.update(status="heldout_by_declaration", test_feedback_used=False, validation_used=True)
        self.record.update(selection_basis="fixed_before_test", test_feedback_used=False)
        sidecar_path(self.prediction).write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Previously recorded validation"):
            self.inspect()


if __name__ == "__main__":
    unittest.main()

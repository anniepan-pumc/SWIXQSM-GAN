import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from ..audit_splits import main as audit_main
from ..util.independence import build_independence_report, verify_independence_report
from ..util.quality import resolve_qc_paths


class IndependenceAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.shape = (4, 5, 3)
        self.sequence = 0
        self.rows = [self.make_case("p1", "global-1", "train"),
                     self.make_case("p2", "global-2", "val"),
                     self.make_case("p3", "global-3", "test")]
        self.checkpoint = self.root/"initial.pt"
        self.checkpoint.write_bytes(b"synthetic checkpoint bytes for hash binding, not a model")

    def tearDown(self):
        self.temporary.cleanup()

    def write_image(self, path, array, *, affine=None, slope=None, intercept=None):
        image = nib.Nifti1Image(array, np.eye(4) if affine is None else affine)
        image.header.set_xyzt_units("mm")
        if slope is not None:
            image.header.set_slope_inter(slope, intercept)
        nib.save(image, path)
        return str(path)

    def make_case(self, sid, gid, split):
        self.sequence += 1
        directory = self.root/f"case-{self.sequence}"
        directory.mkdir()
        row = {"subject_id": sid, "global_subject_id": gid, "split": split,
               "case_id": f"visit-{self.sequence}", "cohort": "synthetic"}
        swi = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)+self.sequence*100
        for key, array in (("swi_path", swi), ("qsm_path", np.zeros(self.shape, dtype=np.float32)),
                           ("swi_mask_path", np.ones(self.shape, dtype=np.float32)),
                           ("mask_path", np.ones(self.shape, dtype=np.float32))):
            row[key] = self.write_image(directory/f"{key}.nii.gz", array)
        return row

    def history(self, *, coverage="complete", exposures=None, **changes):
        data = {"schema": "swixqsm.initialization_history", "version": 1,
                "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                "coverage": coverage, "source": "Synthetic known exposure ledger",
                "exposures": [] if exposures is None else exposures}
        data.update(changes)
        path = self.root/"history.json"
        path.write_text(json.dumps(data))
        return path

    def save_report(self, report):
        path = self.root/"report.json"
        path.write_text(json.dumps(report, allow_nan=False))
        return path

    def manifest(self, rows):
        path = self.root/"manifest.csv"
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_random_initialization_and_full_manifest_verification(self):
        report = build_independence_report(self.rows)
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(report["initialization"]["kind"], "random")
        compact = verify_independence_report(self.rows, self.save_report(report))
        self.assertEqual(compact["status"], "verified")
        self.assertEqual(compact["n_content_cases"], 2)
        self.assertEqual(compact["manifest_identity_sha256"], report["manifest_identity_sha256"])
        self.assertNotIn("cases", compact)
        # All QSM maps are identical zeros; only SWI is used for identity duplicates.
        self.assertEqual(len(report["cases"]), 3)

    def test_multiple_visits_allowed_but_aliases_even_within_split_fail(self):
        visit = self.make_case("p1", "global-1", "train")
        report = build_independence_report(self.rows+[visit])
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(report["n_subjects"], 3)
        alias = self.make_case("renamed-p1", "global-1", "train")
        report = build_independence_report(self.rows+[alias])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("aliases" in error for error in report["errors"]))
        conflicting = self.make_case("p1", "different-global", "train")
        self.assertEqual(build_independence_report(self.rows+[conflicting])["status"], "failed")

    def test_train_test_without_validation_is_auditable_without_retraining(self):
        rows = [self.rows[0], self.rows[2]]
        report = build_independence_report(rows)
        self.assertEqual(report["status"], "passed")
        compact = verify_independence_report(rows, self.save_report(report))
        self.assertEqual(compact["n_content_cases"], 1)
        compact = verify_independence_report(rows, self.save_report(report),
                                             content_splits=("test",), check_initialization=False)
        self.assertEqual(compact["n_content_cases"], 1)

    def test_global_identity_cannot_cross_splits_or_be_missing(self):
        repeated = self.make_case("p1", "global-1", "test")
        report = build_independence_report(self.rows+[repeated])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("crosses splits" in error for error in report["errors"]))
        rows = copy.deepcopy(self.rows)
        rows[0].pop("global_subject_id")
        self.assertEqual(build_independence_report(rows)["status"], "failed")

    def test_same_swi_path_cannot_be_repeated_for_visits(self):
        row = {**self.rows[0], "case_id": "duplicate-visit"}
        report = build_independence_report(self.rows+[row])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("Repeated SWI path" in error for error in report["errors"]))

    def test_decoded_duplicate_survives_header_and_gzip_rewrite_and_scaling(self):
        first, second = self.rows[:2]
        raw = np.arange(np.prod(self.shape), dtype=np.int16).reshape(self.shape)
        self.write_image(first["swi_path"], raw, slope=2., intercept=-20.)
        other = self.root/"float_rewritten.nii"
        affine = np.eye(4)
        affine[0, 3] = 23.
        self.write_image(other, raw.astype(np.float32)*2-20, affine=affine)
        rows = [first, {**second, "swi_path": str(other)}, self.rows[2]]
        report = build_independence_report(rows)
        self.assertEqual(report["status"], "failed")
        a, b = report["cases"][:2]
        self.assertNotEqual(a["file_sha256"]["swi_path"], b["file_sha256"]["swi_path"])
        self.assertEqual(a["swi_fingerprint"], b["swi_fingerprint"])

    def test_fingerprint_normalizes_signed_zero_endianness_and_array_order(self):
        array = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
        modified = np.asfortranarray(array.astype(">f4"))
        modified[0, 0, 0] = -0.
        self.write_image(self.rows[0]["swi_path"], array)
        self.write_image(self.rows[1]["swi_path"], modified)
        report = build_independence_report(self.rows)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cases"][0]["swi_fingerprint"], report["cases"][1]["swi_fingerprint"])

    def test_pretrained_without_history_and_unknown_empty_history_are_unverified(self):
        report = build_independence_report(self.rows, checkpoint_path=self.checkpoint)
        self.assertEqual(report["status"], "unverified")
        self.assertEqual(report["initialization"]["coverage"], "unknown")
        report = build_independence_report(self.rows, checkpoint_path=self.checkpoint,
                                          history_path=self.history(coverage="unknown"))
        self.assertEqual(report["status"], "unverified")
        with self.assertRaisesRegex(ValueError, "completely passed"):
            verify_independence_report(self.rows, self.save_report(report), checkpoint_path=self.checkpoint)

    def test_complete_history_allows_current_training_membership(self):
        history = self.history(exposures=[{"global_subject_id": "global-1", "role": "train"}])
        report = build_independence_report(self.rows, checkpoint_path=self.checkpoint, history_path=history)
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(len(report["initialization"]["training_exposures"]), 1)
        compact = verify_independence_report(self.rows, self.save_report(report), checkpoint_path=self.checkpoint)
        self.assertEqual(compact["initialization"]["coverage"], "complete")

    def test_every_prior_holdout_role_is_exposure(self):
        for gid in ("global-2", "global-3"):
            for role in ("train", "selection", "evaluation", "unknown"):
                with self.subTest(gid=gid, role=role):
                    history = self.history(exposures=[{"global_subject_id": gid, "role": role}])
                    report = build_independence_report(self.rows, checkpoint_path=self.checkpoint, history_path=history)
                    self.assertEqual(report["status"], "failed")
                    self.assertEqual(report["initialization"]["holdout_exposures"][0]["role"], role)

    def test_history_schema_source_and_checkpoint_binding_are_strict(self):
        changes = [{"checkpoint_sha256": "0"*64}, {"coverage": "probably_complete"}, {"source": ""},
                   {"version": True}, {"exposures": [{"global_subject_id": "", "role": "train"}]},
                   {"exposures": [{"global_subject_id": "someone", "role": "validation"}]},
                   {"exposures": [{"global_subject_id": "someone", "role": "train", "extra": "ignored?"}]}]
        for change in changes:
            with self.subTest(change=change):
                report = build_independence_report(self.rows, checkpoint_path=self.checkpoint,
                                                  history_path=self.history(**change))
                self.assertEqual(report["status"], "failed")
        report = build_independence_report(self.rows, history_path=self.history())
        self.assertEqual(report["status"], "failed")

    def test_changed_history_or_actual_checkpoint_invalidates_verification(self):
        history = self.history()
        report = build_independence_report(self.rows, checkpoint_path=self.checkpoint, history_path=history)
        path = self.save_report(report)
        history.write_text(history.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "history_sha256"):
            verify_independence_report(self.rows, path, checkpoint_path=self.checkpoint)
        self.history()
        self.checkpoint.write_bytes(b"different initializer")
        with self.assertRaises(ValueError):
            verify_independence_report(self.rows, path, checkpoint_path=self.checkpoint)

    def test_training_requires_random_pretrained_source_match(self):
        path = self.save_report(build_independence_report(self.rows))
        with self.assertRaisesRegex(ValueError, "random/pretrained"):
            verify_independence_report(self.rows, path, checkpoint_path=self.checkpoint)
        report = build_independence_report(self.rows, checkpoint_path=self.checkpoint, history_path=self.history())
        path = self.save_report(report)
        with self.assertRaisesRegex(ValueError, "random/pretrained"):
            verify_independence_report(self.rows, path)

    def test_baseline_evaluation_mode_does_not_read_checkpoint_but_keeps_history_binding(self):
        history = self.history()
        report = build_independence_report(self.rows, checkpoint_path=self.checkpoint, history_path=history)
        path = self.save_report(report)
        self.checkpoint.unlink()
        compact = verify_independence_report(self.rows, path, check_initialization=False, content_splits=("train",))
        self.assertEqual(compact["status"], "verified")
        self.assertFalse(compact["initialization_checked_against_current_checkpoint"])
        history.write_text(history.read_text()+"\n")
        with self.assertRaises(ValueError):
            verify_independence_report(self.rows, path, check_initialization=False)

    def test_unselected_test_contents_never_reopened_but_all_metadata_bound(self):
        path = self.save_report(build_independence_report(self.rows))
        for key in ("swi_path", "qsm_path", "swi_mask_path", "mask_path"):
            Path(self.rows[2][key]).unlink()
        verify_independence_report(self.rows, path)
        with self.assertRaises(ValueError):
            verify_independence_report(self.rows, path, content_splits=("test",))
        changed = copy.deepcopy(self.rows)
        changed[2]["cohort"] = "a changed held-out declaration"
        with self.assertRaisesRegex(ValueError, "Full manifest"):
            verify_independence_report(changed, path)

    def test_prediction_columns_and_row_order_do_not_change_membership_binding(self):
        report = build_independence_report(self.rows)
        path = self.save_report(report)
        changed = [{**row, "prediction_path": "/nonexistent/prediction.nii.gz", "prediction_units": "ppm"}
                   for row in reversed(self.rows)]
        compact = verify_independence_report(changed, path)
        self.assertEqual(compact["manifest_identity_sha256"], report["manifest_identity_sha256"])
        with self.assertRaises(ValueError):
            verify_independence_report(self.rows[:2], path)

    def test_primary_file_change_and_empty_content_selection_are_rejected(self):
        path = self.save_report(build_independence_report(self.rows))
        with self.assertRaises(ValueError):
            verify_independence_report(self.rows, path, content_splits=())
        self.write_image(self.rows[0]["qsm_path"], np.ones(self.shape, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "file_sha256"):
            verify_independence_report(self.rows, path)

    def test_cli_writes_failed_or_unverified_reports_before_exit(self):
        manifest = self.manifest(self.rows)
        output = self.root/"unverified-cli"
        argv = ["audit", "--manifest", str(manifest), "--output", str(output), "--checkpoint", str(self.checkpoint)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            audit_main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertEqual(json.loads((output/"report.json").read_text())["status"], "unverified")
        self.assertTrue((output/"manifest_resolved.json").is_file())
        aliased = self.rows+[self.make_case("alias", "global-1", "train")]
        manifest = self.manifest(aliased)
        output = self.root/"failed-cli"
        argv = ["audit", "--manifest", str(manifest), "--output", str(output)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            audit_main()
        self.assertEqual(json.loads((output/"report.json").read_text())["status"], "failed")

    def test_missing_and_truncated_input_are_reported_not_silently_skipped(self):
        Path(self.rows[0]["mask_path"]).unlink()
        bad = self.root/"truncated.nii"
        self.write_image(bad, np.ones(self.shape, dtype=np.float32))
        with bad.open("r+b") as stream:
            stream.truncate(352)
        self.rows[1]["swi_path"] = str(bad)
        report = build_independence_report(self.rows)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["cases"][0]["errors"])
        self.assertTrue(report["cases"][1]["errors"])
        json.dumps(report, allow_nan=False)

    def test_cli_resolves_auxiliary_metadata_like_consuming_workflows(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["qsm_reference_mask_path"] = "relative_reference_mask.nii.gz"
            row["registration_landmarks_path"] = "relative_landmarks.json"
        manifest = self.manifest(rows)
        output = self.root/"relative-auxiliary-cli"
        argv = ["audit", "--manifest", str(manifest), "--output", str(output)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            audit_main()
        prepared = resolve_qc_paths(rows, manifest)
        compact = verify_independence_report(prepared, output/"report.json")
        self.assertEqual(compact["status"], "verified")

    def test_tampered_failed_history_and_unselected_case_schema_cannot_pass(self):
        report = build_independence_report(self.rows)
        report["cases"][2]["swi_fingerprint"]["sha256"] = report["cases"][0]["swi_fingerprint"]["sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate decoded"):
            verify_independence_report(self.rows, self.save_report(report), check_initialization=False)
        report = build_independence_report(self.rows)
        report["status"] = "unverified"
        with self.assertRaisesRegex(ValueError, "completely passed"):
            verify_independence_report(self.rows, self.save_report(report), check_initialization=False)


if __name__ == "__main__":
    unittest.main()

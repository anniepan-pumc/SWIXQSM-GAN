import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..util.quality import build_report, verify_report
from ..util.reference import assert_reference_identity
from ..util.quality_gate import bind_input_units, check_prediction_references, prepare_quality_rows
from ..util.nifti import save_qsm, sidecar_path
from ..util.reference import reference_identity
from ..util.independence import build_independence_report
from ..util.independence_gate import check_prediction_independence


class QualityWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shape = (32, 40, 2)
        self.affine = np.diag([.8, .9, 1.5, 1.])
        self.contract = {
            "schema": "swixqsm.data_contract", "version": 1,
            "geometry": {"axis_codes": ["R", "A", "S"], "slice_axis": 2,
                         "spacing_mm_min": [.79, .89, 1.49],
                         "spacing_mm_max": [.81, .91, 1.51], "max_obliquity_deg": 1.},
            "reference": {"id": "synthetic_reference", "method": "mean", "region": "synthetic_ROI",
                          "target_ppb": 0., "tolerance_ppb": .001, "require_mask": True},
            "registration": {"require_review": True, "min_mask_dice": .99, "max_landmark_error_mm": .1},
        }
        self.rows = []
        for index, split in enumerate(("train", "val", "test")):
            row = {"subject_id": f"synthetic_{split}", "split": split, "case_id": split,
                   "global_subject_id": f"global_synthetic_{split}",
                   "qsm_units": "ppm", "qsm_reference": "synthetic_reference",
                   "qsm_reference_method": "mean", "qsm_reference_region": "synthetic_ROI",
                   "qsm_reference_source": "Synthetic generated labels; no real imaging",
                   "space_id": "synthetic_space", "registration_review": "accepted",
                   "registration_reviewer": "synthetic_test", "registration_review_date": "2026-10-17",
                   "qsm_brain_mask_source": "Synthetic independently generated all-brain reference mask"}
            swi = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape) + index
            qsm = (swi / 10000 - .1).copy()
            reference_mask = np.zeros(self.shape, dtype=np.float32)
            reference_mask[2:6, 3:7, :] = 1
            qsm[reference_mask > 0] = 0
            for key, values in {"swi_path": swi, "qsm_path": qsm,
                                "swi_mask_path": np.ones(self.shape), "mask_path": np.ones(self.shape),
                                "qsm_brain_mask_path": np.ones(self.shape),
                                "qsm_reference_mask_path": reference_mask,
                                "registration_image_path": swi * .7 + 10}.items():
                row[key] = str(self.save(f"{split}_{key}.nii.gz", values))
            landmarks = self.root / f"{split}_landmarks.json"
            points = [[0, 0, 0], [1, 2, 3], [4, 5, 6]]
            landmarks.write_text(json.dumps({"coordinate_system": "ras_mm", "fixed": points, "moving": points}))
            row["registration_landmarks_path"] = str(landmarks)
            self.rows.append(row)
        self.manifest = self.root / "manifest.csv"
        self.write_manifest()
        self.contract_path = self.root / "contract.json"
        self.contract_path.write_text(json.dumps(self.contract))
        self.report_path = self.root / "report.json"
        report = build_report(self.rows, self.contract)
        self.assertEqual(report["n_passed"], 3, report)
        self.report_path.write_text(json.dumps(report, allow_nan=False))

    def save(self, name, values, affine=None):
        image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), self.affine if affine is None else affine)
        image.header.set_xyzt_units("mm")
        path = self.root / name
        nib.save(image, path)
        return path

    def write_manifest(self):
        fields = sorted(set().union(*(row.keys() for row in self.rows)))
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def run_cli(self, module, *args, success=True):
        result = subprocess.run([sys.executable, "-m", f"swixqsm_minimal.{module}", *map(str, args)],
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def test_qc_cli_preview_and_pending_report(self):
        hashes = {row["qsm_path"]: hashlib.sha256(Path(row["qsm_path"]).read_bytes()).hexdigest() for row in self.rows}
        output = self.root / "qc"
        self.run_cli("qc", "--manifest", self.manifest, "--contract", self.contract_path,
                     "--output", output, "--preview-limit", 1)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["n_passed"], 3)
        self.assertIn("data:image/png;base64,", (output / "preview.html").read_text())
        for path, digest in hashes.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
        self.rows[-1]["registration_review"] = "pending"
        self.write_manifest()
        pending = self.root / "pending_qc"
        result = self.run_cli("qc", "--manifest", self.manifest, "--contract", self.contract_path,
                              "--output", pending, "--preview-limit", 0, success=False)
        self.assertEqual(result.returncode, 2)
        self.assertTrue((pending / "report.json").exists())
        self.assertEqual(json.loads((pending / "report.json").read_text())["n_passed"], 2)
        Path(self.rows[0]["swi_path"]).write_bytes(b"Synthetic corrupt NIfTI")
        corrupt = self.root / "corrupt_qc"
        result = self.run_cli("qc", "--manifest", self.manifest, "--contract", self.contract_path,
                              "--output", corrupt, "--preview-limit", 1, success=False)
        self.assertEqual(result.returncode, 2)
        self.assertTrue((corrupt / "report.json").exists())
        self.assertIn("无法生成可比预览", (corrupt / "preview.html").read_text())

    def test_stale_report_and_selected_split_isolation(self):
        # Test-image changes must not be read by train/val or train-only gates.
        Path(self.rows[-1]["qsm_path"]).write_bytes(b"invalid unconsumed test image")
        info = verify_report(self.rows[:2], self.report_path)
        self.assertEqual(info["n_cases"], 2)
        with self.assertRaises(ValueError):
            verify_report(self.rows[-1:], self.report_path)
        changed = dict(self.rows[0], qsm_reference_source="different_pipeline")
        with self.assertRaises(ValueError):
            verify_report([changed], self.report_path)

    def test_truncated_qsm_payload_produces_failed_report(self):
        truncated = self.save("truncated_qsm.nii", np.zeros(self.shape))
        truncated.write_bytes(truncated.read_bytes()[:352])
        self.rows[0]["qsm_path"] = str(truncated)
        self.write_manifest()
        output = self.root / "truncated_qc"
        result = self.run_cli("qc", "--manifest", self.manifest, "--contract", self.contract_path,
                              "--output", output, "--preview-limit", 1, success=False)
        self.assertEqual(result.returncode, 2)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["n_passed"], 2)
        self.assertTrue(any("Cannot inspect qsm_path" in error for error in report["cases"][0]["errors"]))

    def test_missing_files_and_invalid_units_are_case_failures(self):
        Path(self.rows[-1]["qsm_path"]).unlink()
        selected = self.root / "train_only_qc"
        self.run_cli("qc", "--manifest", self.manifest, "--contract", self.contract_path,
                     "--output", selected, "--split", "train", "--preview-limit", 0)
        self.assertEqual(json.loads((selected / "report.json").read_text())["n_passed"], 1)
        self.rows[0]["qsm_units"] = "unknown"
        self.write_manifest()
        output = self.root / "missing_file_qc"
        result = self.run_cli("qc", "--manifest", self.manifest, "--contract", self.contract_path,
                              "--output", output, "--preview-limit", 0, success=False)
        self.assertEqual(result.returncode, 2)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["n_passed"], 1)
        self.assertTrue(any("QSM units" in error for error in report["cases"][0]["errors"]))
        self.assertTrue(any("Cannot hash qsm_path" in error for error in report["cases"][-1]["errors"]))

    def test_verified_train_infer_fit_and_evaluate_reference_chain(self):
        experiment = self.root / "experiment"
        independence = self.root / "independence.json"
        report = build_independence_report(prepare_quality_rows(self.rows, self.manifest))
        self.assertEqual(report["status"], "passed", report)
        independence.write_text(json.dumps(report, allow_nan=False))
        self.run_cli("train", "--manifest", self.manifest, "--output", experiment,
                     "--qc-report", self.report_path, "--units", "ppm", "--qsm-scale", ".1",
                     "--independence-report", independence,
                     "--size", 40, "--epochs", 1, "--batch-size", 2, "--adv-weight", 0, "--device", "cpu")
        checkpoint = torch.load(experiment / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["data_qc"]["status"], "verified")
        self.assertEqual(checkpoint["data_independence"]["status"], "verified")
        self.assertEqual(checkpoint["training_state"]["current_run_phase"], "l1")
        assert_reference_identity(self.contract["reference"], checkpoint["data_reference"])
        row = self.rows[-1]
        prediction = self.root / "prediction.nii.gz"
        self.run_cli("infer", "--checkpoint", experiment / "best.pt", "--swi", row["swi_path"],
                     "--swi-mask", row["swi_mask_path"], "--output", prediction, "--expected-phase", "l1")
        metadata = json.loads(prediction.with_suffix("").with_suffix(".json").read_text())
        self.assertEqual(metadata["DataQC"]["status"], "geometry_verified")
        self.assertEqual(metadata["DataIndependence"]["status"], "verified")
        self.assertEqual(metadata["ReferenceID"], "synthetic_reference")
        assert_reference_identity(self.contract["reference"], metadata["DataReference"])
        mismatched_affine = np.diag([1.2, .9, 1.5, 1.])
        mismatch_swi = self.save("mismatch_swi.nii.gz", nib.load(row["swi_path"]).get_fdata(), mismatched_affine)
        mismatch_mask = self.save("mismatch_mask.nii.gz", np.ones(self.shape), mismatched_affine)
        mismatch_output = self.root / "mismatch_prediction.nii.gz"
        mismatch = self.run_cli("infer", "--checkpoint", experiment / "best.pt", "--swi", mismatch_swi,
                               "--swi-mask", mismatch_mask, "--output", mismatch_output, success=False)
        self.assertIn("violates training data contract", mismatch.stderr)
        self.assertFalse(mismatch_output.exists())
        row["prediction_path"] = str(prediction)
        self.write_manifest()
        artifact = self.root / "baselines.npz"
        self.run_cli("fit_baselines", "--manifest", self.manifest, "--output", artifact,
                     "--independence-report", independence,
                     "--qc-report", self.report_path, "--reference-id", "synthetic_reference",
                     "--space-id", "synthetic_space")
        output = self.root / "evaluation"
        self.run_cli("evaluate", "--manifest", self.manifest, "--output", output,
                     "--independence-report", independence, "--expected-prediction-phase", "l1",
                     "--qc-report", self.report_path, "--baselines", artifact,
                     "--reference-id", "synthetic_reference", "--space-id", "synthetic_space",
                     "--data-range", 500)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["config"]["data_qc"]["status"], "verified")
        self.assertEqual(summary["config"]["data_independence"]["status"], "verified")
        self.assertEqual(set(summary["summary"]), {"prediction", "zero", "train_mean", "linear"})
        # Matching reference ID with a changed method must still be rejected.
        metadata["DataReference"]["method"] = "median"
        prediction.with_suffix("").with_suffix(".json").write_text(json.dumps(metadata))
        failed = self.root / "bad_reference_evaluation"
        self.run_cli("evaluate", "--manifest", self.manifest, "--output", failed,
                     "--qc-report", self.report_path, "--reference-id", "synthetic_reference", success=False)
        self.assertFalse(failed.exists())

    def test_default_entrypoints_require_report_or_explicit_legacy_reason(self):
        result = self.run_cli("fit_baselines", "--manifest", self.manifest,
                              "--output", self.root / "missing.npz", "--reference-id", "synthetic_reference",
                              "--space-id", "synthetic_space", success=False)
        self.assertIn("--qc-report", result.stderr)
        artifact = self.root / "legacy.npz"
        self.run_cli("fit_baselines", "--manifest", self.manifest, "--output", artifact,
                     "--reference-id", "synthetic_reference", "--space-id", "synthetic_space",
                     "--allow-unverified-data", "Synthetic legacy migration test")
        from ..util.baselines import load_baselines
        self.assertEqual(load_baselines(artifact).metadata["data_qc"]["status"], "unverified")

    def test_unverified_prediction_origin_is_rejected_and_preserved(self):
        row = self.rows[-1]
        prediction = self.root / "legacy_prediction.nii.gz"
        reference = reference_identity(self.contract["reference"])
        source_qc = {"status": "unverified", "reason": "Synthetic legacy origin"}
        metadata = {"Units": "ppm", "ReferenceID": reference["id"],
                    "DataReference": reference, "DataQC": source_qc,
                    "DataIndependence": {"status": "unverified", "reason": "Synthetic unknown initialization history"},
                    "TrainingState": {"current_run_phase": "gan", "adversarial_generator_updates": 1},
                    "CheckpointSHA256": "synthetic_checkpoint_hash"}
        image = nib.load(row["qsm_path"])
        save_qsm(prediction, image.get_fdata(), image, metadata)
        row["prediction_path"] = str(prediction)
        self.write_manifest()
        for status in ("failed", "pending", "rejected", "unexpected_status"):
            sidecar_path(prediction).write_text(json.dumps({**metadata, "DataQC": {"status": status}}))
            with self.subTest(status=status), self.assertRaises(ValueError):
                check_prediction_references([row], reference)
        sidecar_path(prediction).write_text(json.dumps(metadata))
        rejected = self.root / "rejected_origin"
        result = self.run_cli("evaluate", "--manifest", self.manifest, "--output", rejected,
                              "--qc-report", self.report_path, "--reference-id", reference["id"],
                              success=False)
        self.assertIn("unverified QC", result.stderr)
        self.assertFalse(rejected.exists())

        output = self.root / "legacy_evaluation"
        self.run_cli("evaluate", "--manifest", self.manifest, "--output", output,
                     "--reference-id", reference["id"], "--save-predictions",
                     "--allow-unverified-data", "Synthetic migration audit")
        summary = json.loads((output / "summary.json").read_text())
        provenance = summary["config"]["prediction_provenance"][str(prediction.resolve())]
        self.assertEqual(provenance["PredictionDataQC"], source_qc)
        self.assertEqual(provenance["PredictionDataReference"], reference)
        self.assertTrue(provenance["prediction_sha256"])
        with (output / "saved_predictions.csv").open() as stream:
            saved = next(item for item in csv.DictReader(stream) if item["method"] == "prediction")
        saved_metadata = json.loads(sidecar_path(saved["prediction_path"]).read_text())
        self.assertEqual(saved_metadata["PredictionDataQC"], source_qc)
        self.assertEqual(saved_metadata["EvaluationDataQC"]["status"], "unverified")
        self.assertEqual(saved_metadata["PredictionDataReference"], reference)
        self.assertEqual(saved_metadata["PredictionDataIndependence"], metadata["DataIndependence"])
        self.assertEqual(saved_metadata["PredictionTrainingState"], metadata["TrainingState"])
        self.assertEqual(saved_metadata["SourceCheckpointSHA256"], "synthetic_checkpoint_hash")
        # An evaluation's own status must never hide the original prediction status.
        saved_metadata["DataQC"] = {"status": "verified"}
        sidecar_path(saved["prediction_path"]).write_text(json.dumps(saved_metadata))
        with self.assertRaisesRegex(ValueError, "unverified QC"):
            check_prediction_references([saved], reference)
        saved_metadata["DataIndependence"] = {"status": "verified", "manifest_identity_sha256": "synthetic_split"}
        sidecar_path(saved["prediction_path"]).write_text(json.dumps(saved_metadata))
        with self.assertRaisesRegex(ValueError, "no verified independence"):
            check_prediction_independence([saved], saved_metadata["DataIndependence"])

    def test_verified_unit_fallback_is_not_reinterpreted_as_output_units(self):
        for row in self.rows:
            del row["qsm_units"]
        self.write_manifest()
        report = build_report(self.rows, self.contract, default_qsm_units="ppm")
        self.assertEqual(report["n_passed"], 3)
        self.report_path.write_text(json.dumps(report, allow_nan=False))
        info = verify_report(self.rows[:1], self.report_path)
        bound = bind_input_units(self.rows[:1], info)
        self.assertEqual(bound[0]["qsm_units"], "ppm")
        with self.assertRaisesRegex(ValueError, "unit fallback conflicts"):
            bind_input_units(self.rows[:1], info, requested_fallback="ppb")
        artifact = self.root / "fallback_baselines.npz"
        self.run_cli("fit_baselines", "--manifest", self.manifest, "--output", artifact,
                     "--qc-report", self.report_path, "--reference-id", "synthetic_reference",
                     "--space-id", "synthetic_space", "--units", "ppb")
        from ..util.baselines import load_baselines
        bundle = load_baselines(artifact)
        expected = nib.load(self.rows[0]["qsm_path"]).get_fdata() * 1000
        np.testing.assert_allclose(bundle.train_mean, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

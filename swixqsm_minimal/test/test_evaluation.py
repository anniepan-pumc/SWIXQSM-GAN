import copy
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

from ..util.baselines import fit_baselines, load_baselines
from ..util.evaluation import evaluate_rows, resolve_prediction_paths


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(42)
        self.shape = (21, 21, 21)
        self.rows = []
        for i, (sid, split, value) in enumerate([
                ("train_a", "train", 15.), ("train_b", "train", 25.),
                ("test_a", "test", 10.), ("test_a", "test", 30.),
                ("test_b", "test", 100.)]):
            mask = np.ones(self.shape, dtype=np.float32)
            if i == 2:
                mask[:, :, 10:] = 0
            row = {"subject_id": sid, "split": split, "case_id": str(i),
                   "qsm_units": "ppm", "qsm_reference": "ventricular_csf",
                   "space_id": "synthetic_common_space", "cohort": "synthetic"}
            arrays = {"swi_path": self.rng.uniform(1, 100, self.shape).astype(np.float32),
                      "qsm_path": np.full(self.shape, value / 1000, dtype=np.float32),
                      "swi_mask_path": np.ones(self.shape, dtype=np.float32), "mask_path": mask}
            for key, data in arrays.items():
                row[key] = str(self.save(f"case_{i}_{key}.nii.gz", data))
            self.rows.append(row)
        self.test_rows = self.rows[2:]
        self.kwargs = {"units": "ppb", "reference_id": "ventricular_csf", "data_range": 500.}
        self.manifest = self.root / "manifest.csv"
        self.write_manifest(self.rows)

    def save(self, name, values, affine=None, units="mm"):
        path = self.root / name
        image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), np.eye(4) if affine is None else affine)
        image.header.set_xyzt_units(units)
        nib.save(image, path)
        return path

    def write_manifest(self, rows):
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)

    def fit(self):
        return fit_baselines(self.rows, units="ppb", reference_id="ventricular_csf",
                             space_id="synthetic_common_space")

    def add_prediction(self, row, *, units="ppb", reference="ventricular_csf"):
        values = nib.load(row["qsm_path"]).get_fdata() * (1000 if units == "ppb" else 1)
        path = self.save(f"prediction_{row['case_id']}.nii.gz", values)
        sidecar = path.with_suffix("").with_suffix(".json")
        sidecar.write_text(json.dumps({"Units": units, "ReferenceID": reference,
                                      "SpaceID": "synthetic_common_space"}))
        row["prediction_path"] = str(path)
        return sidecar

    def test_zero_is_physical_zero_and_subjects_are_not_visit_weighted(self):
        result = evaluate_rows(self.test_rows, **self.kwargs)
        self.assertEqual(len(result["cases"]), 3)
        self.assertEqual(len(result["subjects"]), 2)
        for case in result["cases"]:
            self.assertEqual(case["method"], "zero")
            self.assertEqual(case["pred_mean"], 0.)
            self.assertEqual(case["nmse"], 1.)
        first, second = result["subjects"]
        n1, n2 = (r["n_voxels"] for r in result["cases"][:2])
        expected_a = (10 * n1 + 30 * n2) / (n1 + n2)
        self.assertAlmostEqual(first["mae"], expected_a, places=4)
        self.assertEqual(first["n_cases"], 2)
        self.assertAlmostEqual(second["mae"], 100., places=4)
        self.assertAlmostEqual(result["summary"]["zero"]["all"]["metrics"]["mae"]["mean"],
                               (expected_a + 100) / 2, places=4)
        json.dumps(result, allow_nan=False)

    def test_units_sidecars_and_reference_conflicts(self):
        row = copy.deepcopy(self.test_rows[0])
        self.add_prediction(row)
        result = evaluate_rows([row], space_id="synthetic_common_space", **self.kwargs)
        prediction = [r for r in result["cases"] if r["method"] == "prediction"][0]
        self.assertLess(prediction["mae"], 1e-5)
        with self.assertRaisesRegex(ValueError, "Conflicting prediction units"):
            evaluate_rows([row], prediction_units="ppm", space_id="synthetic_common_space", **self.kwargs)
        with self.assertRaisesRegex(ValueError, "Conflicting prediction reference"):
            evaluate_rows([row], prediction_reference_id="whole_brain", **self.kwargs)
        with self.assertRaisesRegex(ValueError, "Conflicting space ID"):
            evaluate_rows([row], space_id="other_space", **self.kwargs)
        row["qsm_reference"] = "whole_brain"
        with self.assertRaisesRegex(ValueError, "Conflicting QSM reference"):
            evaluate_rows([row], **self.kwargs)

    def test_no_unit_or_reference_is_guessed(self):
        row = copy.deepcopy(self.test_rows[0])
        self.add_prediction(row)
        Path(row["prediction_path"]).with_suffix("").with_suffix(".json").unlink()
        with self.assertRaisesRegex(ValueError, "prediction units"):
            evaluate_rows([row], **self.kwargs)
        with self.assertRaisesRegex(ValueError, "prediction reference ID"):
            evaluate_rows([row], prediction_units="ppb", **self.kwargs)
        del row["prediction_path"]
        del row["qsm_units"]
        with self.assertRaisesRegex(ValueError, "qsm_units"):
            evaluate_rows([row], **self.kwargs)

    def test_baselines_use_identical_support_and_reject_training_members(self):
        bundle = self.fit()
        result = evaluate_rows(self.test_rows, baseline_bundle=bundle, **self.kwargs)
        self.assertEqual({r["method"] for r in result["cases"]}, {"zero", "train_mean", "linear"})
        for case_id in ("2", "3", "4"):
            self.assertEqual(len({r["n_voxels"] for r in result["cases"] if r["case_id"] == case_id}), 1)
        with self.assertRaises(ValueError):
            evaluate_rows([self.rows[0]], baseline_bundle=bundle, **self.kwargs)
        missing_space = copy.deepcopy(self.test_rows)
        for row in missing_space:
            del row["space_id"]
        with self.assertRaisesRegex(ValueError, "evaluation space ID"):
            evaluate_rows(missing_space, baseline_bundle=bundle, **self.kwargs)

    def test_geometry_spatial_units_and_masks_are_checked(self):
        row = copy.deepcopy(self.test_rows[0])
        values = np.ones(self.shape)
        affine = np.eye(4)
        affine[0, 3] = 3
        row["qsm_path"] = str(self.save("shifted.nii.gz", values, affine))
        with self.assertRaisesRegex(ValueError, "registered"):
            evaluate_rows([row], **self.kwargs)
        row = copy.deepcopy(self.test_rows[0])
        row["mask_path"] = str(self.save("empty.nii.gz", np.zeros(self.shape)))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            evaluate_rows([row], **self.kwargs)
        row = copy.deepcopy(self.test_rows[0])
        for key in ("swi_path", "qsm_path", "mask_path", "swi_mask_path"):
            data = nib.load(row[key]).get_fdata()
            row[key] = str(self.save(f"unknown_{key}.nii.gz", data, units="unknown"))
        with self.assertRaisesRegex(ValueError, "Unknown NIfTI spatial units"):
            evaluate_rows([row], **self.kwargs)
        evaluate_rows([row], spatial_units="mm", **self.kwargs)

    def test_partial_predictions_rejected_and_missing_range_reported(self):
        rows = copy.deepcopy(self.test_rows)
        self.add_prediction(rows[0])
        with self.assertRaisesRegex(ValueError, "every case"):
            resolve_prediction_paths(rows, self.manifest)
        result = evaluate_rows(self.test_rows, reference_id="ventricular_csf")
        self.assertIsNone(result["subjects"][0]["ssim"])
        reason = result["summary"]["zero"]["all"]["metrics"]["ssim"]
        self.assertEqual(reason["n_undefined"], 2)
        self.assertEqual(reason["undefined_reasons"], {"missing_data_range": 2})

    def test_cli_fit_evaluate_and_saved_physical_predictions(self):
        for row in self.test_rows:
            self.add_prediction(row)
        self.write_manifest(self.rows)
        artifact = self.root / "baselines.npz"
        common = [sys.executable, "-m"]
        fit = common + ["swixqsm_minimal.fit_baselines", "--manifest", str(self.manifest),
                        "--output", str(artifact), "--units", "ppb",
                        "--reference-id", "ventricular_csf", "--space-id", "synthetic_common_space",
                        "--allow-unverified-data", "First-stage synthetic fixture"]
        subprocess.run(fit, check=True, capture_output=True, text=True)
        self.assertEqual(load_baselines(artifact).metadata["units"], "ppb")
        output = self.root / "evaluation"
        command = common + ["swixqsm_minimal.evaluate", "--manifest", str(self.manifest),
                            "--output", str(output), "--baselines", str(artifact),
                            "--reference-id", "ventricular_csf", "--space-id", "synthetic_common_space",
                            "--data-range", "500", "--save-predictions",
                            "--allow-unverified-data", "First-stage synthetic fixture"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(set(summary["summary"]), {"zero", "train_mean", "linear", "prediction"})
        self.assertEqual(summary["summary"]["zero"]["all"]["n_subjects"], 2)
        with (output / "per_case.csv").open() as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 12)
        self.assertIn("spacing_mm", json.loads(records[0]["parameters"]))
        for case_id in ("2", "3", "4"):
            self.assertEqual(len({r["n_voxels"] for r in records if r["case_id"] == case_id}), 1)
        with (output / "saved_predictions.csv").open() as stream:
            saved = list(csv.DictReader(stream))
        for item in saved:
            path = Path(item["prediction_path"])
            self.assertTrue(path.exists())
            if item["method"] == "zero":
                np.testing.assert_array_equal(nib.load(path).get_fdata(), 0.)
            metadata = json.loads(path.with_suffix("").with_suffix(".json").read_text())
            self.assertEqual(metadata["Units"], "ppb")
            self.assertEqual(metadata["ReferenceID"], "ventricular_csf")
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertNotEqual(subprocess.run(fit, capture_output=True).returncode, 0)
        # A later prediction-declaration error must not publish partial baseline results.
        sidecar = Path(self.test_rows[-1]["prediction_path"]).with_suffix("").with_suffix(".json")
        metadata = json.loads(sidecar.read_text())
        metadata["ReferenceID"] = "incompatible_reference"
        sidecar.write_text(json.dumps(metadata))
        failed = self.root / "failed_evaluation"
        failure_command = [str(failed) if item == str(output) else item for item in command]
        self.assertNotEqual(subprocess.run(failure_command, capture_output=True).returncode, 0)
        self.assertFalse(failed.exists())
        self.assertEqual(list(self.root.glob(".swixqsm-evaluation-*")), [])

    def test_frozen_checkpoint_evaluation_needs_no_validation_or_retraining(self):
        self.assertNotIn("val", {row["split"] for row in self.rows})
        checkpoint = self.root / "frozen.pt"
        checkpoint.write_bytes(b"Synthetic pre-existing frozen checkpoint; not a real model")
        before = checkpoint.read_bytes()
        sha = hashlib.sha256(before).hexdigest()
        for row in self.test_rows:
            sidecar = self.add_prediction(row)
            metadata = json.loads(sidecar.read_text())
            metadata["CheckpointSHA256"] = sha
            sidecar.write_text(json.dumps(metadata))
        self.write_manifest(self.rows)
        record = self.root / "selection.json"
        record.write_text(json.dumps({"schema": "swixqsm.model_selection_record", "version": 1,
                                      "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha,
                                      "selection_basis": "fixed_before_test", "test_feedback_used": False,
                                      "validation_used": False, "source": "Synthetic declared procedure",
                                      "evidence_files": []}))
        output = self.root / "frozen_evaluation"
        command = [sys.executable, "-m", "swixqsm_minimal.evaluate", "--manifest", str(self.manifest),
                   "--output", str(output), "--selection-record", str(record), "--save-predictions",
                   "--distribution-diagnostics",
                   "--reference-id", "ventricular_csf", "--data-range", "500",
                   "--allow-unverified-data", "Synthetic first-stage fixture",
                   "--allow-unverified-independence", "Synthetic history not supplied"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        selection = summary["config"]["model_selection"]
        self.assertEqual(selection["status"], "heldout_by_declaration")
        self.assertFalse(selection["validation_used"])
        self.assertFalse(selection["weight_updates_performed"])
        self.assertEqual(summary["config"]["data_independence"]["status"], "unverified")
        self.assertEqual(summary["config"]["data_independence"]["initialization"]["kind"], "not_assessed")
        self.assertEqual(checkpoint.read_bytes(), before)
        distributions = json.loads((output / "distribution_diagnostics.json").read_text())
        self.assertEqual(distributions["units"], "ppb")
        self.assertEqual(len(distributions["cases"]), 6)
        self.assertEqual(distributions["model_selection"]["status"], "heldout_by_declaration")
        for case in distributions["cases"]:
            stats = case["statistics"]
            self.assertEqual(stats["units"], "ppb")
            self.assertEqual(stats["value_space"], "physical_qsm")
            self.assertEqual(stats["status"], "defined")
            if case["method"] == "prediction":
                self.assertLess(stats["error"]["mae"], 1e-5)
                self.assertAlmostEqual(stats["prediction"]["p50"], stats["target"]["p50"], places=5)
            else:
                self.assertEqual(stats["prediction"]["p99"], 0.)
        with (output / "saved_predictions.csv").open() as stream:
            saved = next(row for row in csv.DictReader(stream) if row["method"] == "prediction")
        sidecar = Path(saved["prediction_path"]).with_suffix("").with_suffix(".json")
        saved_metadata = json.loads(sidecar.read_text())
        self.assertEqual(saved_metadata["EvaluationModelSelection"]["checkpoint_sha256"], sha)
        self.assertIsNone(saved_metadata["PredictionTrainingObjective"])
        self.assertIsNone(saved_metadata["PredictionTrainingDiagnostics"])


if __name__ == "__main__":
    unittest.main()

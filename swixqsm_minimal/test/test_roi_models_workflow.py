"""Eight-reviewer-model ROI integration using real CLI parsers and tiny maps.

Each model receives one synthetic prediction on the same single held-out case.
Marker files supply distinct synthetic checkpoint identifiers; they are never
loaded as weights. No inference, optimizer, registration or training is run.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

from .. import compare_models, compare_roi_models, evaluate
from ..util.data import read_manifest
from ..util.independence import build_independence_report
from ..util.nifti import sidecar_path
from ..util.quality_gate import prepare_quality_rows


class ROIModelsWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="swix-eight-roi-workflow-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.models = compare_models.REVIEW_MODELS
        self.shape = (10, 10, 3)
        self.affine = np.diag([.8, .9, 2., 1.])
        raw = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
        self.target = np.asarray(raw / 20 - 4, dtype=np.float32)
        self.labels = np.zeros(self.shape, dtype=np.float32)
        for index in range(4):
            self.labels[index * 2:(index + 1) * 2] = index + 1
        self.core = dict(case_id="synthetic_case", subject_id="synthetic_person", global_subject_id="synthetic_global",
                         cohort="synthetic_cohort", split="test", qsm_units="stored",
                         qsm_reference="unverified_original_reference")
        for role, values in dict(swi_path=raw + 1, qsm_path=self.target, swi_mask_path=np.ones(self.shape),
                                 mask_path=np.ones(self.shape)).items():
            self.core[role] = str(self.save(role + ".nii.gz", values))
        self.core_manifest = self.root / "frozen_core.csv"
        self.write_csv(self.core_manifest, [self.core])
        self.label_path = self.save("actual_synthetic_labels.nii.gz", self.labels)
        self.roi_manifest = self.root / "roi_test.csv"
        self.write_csv(self.roi_manifest, [dict(case_id=self.core["case_id"], roi_labels_path=self.label_path.name,
            roi_label_source="Synthetic integer labels only; IDs are not an anatomical atlas",
            roi_review="pending", roi_reviewer="", roi_review_date="")])
        self.regions = [dict(name="RN_L", labels=[1]), dict(name="RN_R", labels=[2]), dict(name="RN_whole", labels=[1, 2]),
                        dict(name="SN_L", labels=[3]), dict(name="SN_R", labels=[4]), dict(name="SN_whole", labels=[3, 4]),
                        dict(name="whole", labels=[1, 2, 3, 4])]
        self.spec = self.root / "nuclear_spec.json"
        self.spec.write_text(json.dumps(dict(schema="swixqsm.roi_spec", version=1, id="synthetic_7_regions",
            definition="Synthetic L/R and label unions, no anatomical accuracy claim", label_source="Synthetic known arrays",
            regions=self.regions, min_coverage=1., min_voxels=1)))
        report = build_independence_report(prepare_quality_rows(read_manifest(self.core_manifest), self.core_manifest))
        self.assertEqual(report["status"], "passed", report)
        self.independence = self.root / "independence.json"
        self.independence.write_text(json.dumps(report, allow_nan=False))
        self.identity_sha = report["manifest_identity_sha256"]
        self.evaluations, self.predictions, self.markers, self.manifests = {}, {}, {}, {}
        error = np.ones(self.shape, dtype=np.float32)
        error[:2, :, 0] = -2.
        for index, model in enumerate(self.models):
            marker = self.root / (model + "_synthetic_checkpoint_marker.txt")
            marker.write_text("Synthetic test identifier only; never model weights: " + model)
            self.markers[model] = marker
            prediction = np.asarray(self.target + (index + 1) / 4 * error, dtype=np.float32)
            path = self.save(model + "_prediction.nii.gz", prediction)
            self.predictions[model] = prediction
            sidecar_path(path).write_text(json.dumps(dict(Units="stored", ValueSpace="restored_stored_values",
                UnitStatus="unverified_original_storage", ReferenceID="unverified_original_reference",
                DataQC=dict(status="unverified", reason="Synthetic fixture; no anatomical/units/reference review"),
                DataIndependence=dict(status="verified", manifest_identity_sha256=self.identity_sha),
                CheckpointSHA256=self.sha(marker), SyntheticCheckpointScope="Marker bytes only; no weights or training history")))
            manifest = self.root / (model + "_evaluation.csv")
            self.write_csv(manifest, [dict(self.core, prediction_path=str(path), prediction_units="stored",
                                           prediction_reference="unverified_original_reference")])
            self.manifests[model] = manifest
            self.evaluations[model] = self.root / ("evaluation_" + model)

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def save(self, name, values):
        path = self.root / name
        image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), self.affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        return path

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def read_csv(path):
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def run_evaluation(self, model):
        arguments = ["evaluate", "--manifest", str(self.manifests[model]), "--output", str(self.evaluations[model]),
                     "--split", "test", "--units", "stored", "--qsm-units", "stored", "--prediction-units", "stored",
                     "--reference-id", "unverified_original_reference", "--data-range", "40",
                     "--allow-unverified-data", "Synthetic deferred ROI/unit/reference review",
                     "--independence-report", str(self.independence), "--roi-spec", str(self.spec),
                     "--roi-manifest", str(self.roi_manifest), "--roi-review-policy", "deferred",
                     "--roi-bootstrap-reps", "20", "--roi-bootstrap-seed", "42"]
        with mock.patch.object(sys, "argv", arguments):
            evaluate.main()

    def test_real_eight_model_evaluation_to_global_and_roi_comparison(self):
        original_files = {path: self.sha(path) for path in self.root.iterdir() if path.is_file()}
        # The complete flow consumes already written synthetic predictions.
        with mock.patch("torch.load", side_effect=AssertionError("No checkpoint loading")), \
                mock.patch("swixqsm_minimal.test.infer.predict_volume", side_effect=AssertionError("No model inference")), \
                mock.patch("swixqsm_minimal.train_models.main", side_effect=AssertionError("No training")):
            for model in self.models:
                self.run_evaluation(model)
            flags = [item for model in self.models for item in ("--evaluation", f"{model}={self.evaluations[model]}")]
            global_output, roi_output = self.root / "comparison_global", self.root / "comparison_roi"
            global_report = compare_models.main([*flags, "--output", str(global_output)])
            roi_report = compare_roi_models.main([*flags, "--output", str(roi_output)])

        self.assertEqual(global_report["available_models"], list(self.models))
        self.assertEqual(roi_report["available_models"], list(self.models))
        self.assertEqual(global_report["missing_models"], {})
        self.assertEqual(roi_report["missing_models"], {})
        global_counts = {"all_models_per_case.csv": 16, "all_models_per_subject.csv": 16,
                         "model_metrics_long.csv": 416, "model_metrics_wide.csv": 32}
        roi_counts = {"all_models_roi_per_case.csv": 112, "all_models_roi_per_subject.csv": 112,
                      "roi_model_metrics_long.csv": 3136, "roi_model_metrics_wide.csv": 224}
        self.assertEqual(global_report["row_counts"], global_counts)
        self.assertEqual(roi_report["row_counts"], roi_counts)
        for output, counts in ((global_output, global_counts), (roi_output, roi_counts)):
            for filename, count in counts.items():
                self.assertEqual(len(self.read_csv(output / filename)), count, filename)
        global_cases = self.read_csv(global_output / "all_models_per_case.csv")
        roi_cases = self.read_csv(roi_output / "all_models_roi_per_case.csv")
        self.assertEqual(sum(row["method_role"] == "prediction" for row in global_cases), 8)
        self.assertEqual(sum(row["method_role"] == "zero" for row in global_cases), 8)
        self.assertEqual(sum(row["method_role"] == "prediction" for row in roi_cases), 56)
        self.assertEqual({row["roi_review_status"] for row in roi_cases}, {"pending"})
        self.assertEqual({row["UnitStatus"] for row in roi_cases}, {"unverified_original_storage"})
        self.assertTrue(all(row["research_only"] == "True" and row["formal_accuracy_claim"] == "False" for row in roi_cases))
        self.assertEqual({row["data_independence_status"] for row in global_cases}, {"verified"})
        self.assertEqual({row["qc_status"] for row in roi_cases}, {"unverified"})
        self.assertEqual({row["case_id"] for row in roi_cases}, {self.core["case_id"]})
        global_wide = self.read_csv(global_output / "model_metrics_wide.csv")
        roi_long = self.read_csv(roi_output / "roi_model_metrics_long.csv")
        for model in self.models:
            errors = self.predictions[model].astype(np.float64) - self.target.astype(np.float64)
            point = next(row for row in global_wide if row["model"] == model and row["method_role"] == "prediction" and row["group"] == "all")
            self.assertEqual((point["n_subjects"], point["n_cases"]), ("1", "1"))
            self.assertAlmostEqual(float(point["mae__mean"]), float(np.abs(errors).mean()), places=10)
            self.assertEqual(point["mae__std"], "")
            left = next(row for row in roi_cases if row["model"] == model and row["method_role"] == "prediction" and row["region"] == "RN_L")
            selection = self.labels == 1
            self.assertAlmostEqual(float(left["difference"]), float(errors[selection].mean()), places=10)
            self.assertAlmostEqual(float(left["mae"]), float(np.abs(errors[selection]).mean()), places=10)
            self.assertGreater(float(left["mae"]), abs(float(left["difference"])))
            icc = next(row for row in roi_long if row["model"] == model and row["method_role"] == "prediction"
                       and row["region"] == "RN_L" and row["group_type"] == "overall" and row["metric"] == "icc_a1")
            self.assertEqual((icc["n"], icc["estimate"], icc["ci_lower"], icc["ci_upper"]), ("1", "", "", ""))
            self.assertEqual(icc["metric_units"], "dimensionless")
            self.assertIn("insufficient", icc["metric_status"])

        # Both exporters bind the same native evaluation artifacts, and the
        # ROI exporter adds its spec/overlay/actual array/support hashes.
        global_sources = {row["model"]: row for row in global_report["sources"]}
        roi_sources = {row["model"]: row for row in roi_report["sources"]}
        checkpoint_shas = set()
        for model in self.models:
            native = self.evaluations[model]
            global_source, roi_source = global_sources[model], roi_sources[model]
            for filename in compare_models.FILES:
                self.assertEqual(global_source["file_sha256"][filename], roi_source["file_sha256"][filename])
                self.assertEqual(global_source["file_sha256"][filename], self.sha(native / filename))
            for filename in compare_roi_models.ROI_FILES:
                self.assertEqual(roi_source["file_sha256"][filename], self.sha(native / filename))
            whole = json.loads((native / "summary.json").read_text())
            regional = json.loads((native / "roi_agreement.json").read_text())
            for field in ("data_qc", "data_reference", "data_independence", "prediction_provenance"):
                self.assertEqual(whole["config"][field], regional[field])
            self.assertEqual(regional["roi_manifest_source"], whole["config"]["roi_manifest_source"])
            self.assertEqual(regional["roi_manifest_source"], regional["provenance"]["roi_manifest"])
            self.assertEqual(regional["roi_manifest_source"]["sha256"], self.sha(self.roi_manifest))
            self.assertEqual(regional["provenance"]["specification"]["sha256"], self.sha(self.spec))
            self.assertEqual(regional["provenance"]["cases"][0]["label_sha256"], self.sha(self.label_path))
            self.assertEqual(whole["config"]["data_independence"]["manifest_identity_sha256"], self.identity_sha)
            self.assertNotIn("roi_labels_path", json.loads((native / "manifest_resolved.json").read_text())[0])
            checkpoint_shas.add(global_source["flags"]["source_checkpoint_sha256"])
        self.assertEqual(len(checkpoint_shas), 8)
        for region in self.regions:
            entries = [row for row in roi_cases if row["region"] == region["name"]]
            for field in compare_roi_models.HASH_FIELDS:
                self.assertEqual(len({row[field] for row in entries}), 1, field)
        for path, old_sha in original_files.items():
            self.assertEqual(self.sha(path), old_sha, str(path))
        self.assertEqual(list(self.root.glob(".swixqsm-*")), [])


if __name__ == "__main__":
    unittest.main()

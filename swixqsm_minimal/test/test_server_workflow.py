"""Upload workflow probes with synthetic NIfTI and temporary output roots only."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

from ..server_workflow import Workflow


class ServerWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name).resolve() / "bundle"
        self.bundle.mkdir()
        self.run = self.bundle.parent / "new_run"
        self.data = self.bundle.parent / "flat_data"
        self.data.mkdir()
        self.ids = self.bundle / "ids"
        self.ids.mkdir()
        (self.ids / "train.txt").write_text("100\n200\n")
        (self.ids / "test.txt").write_text("300\n")
        for number in ("100", "200", "300"):
            case = self.data / number
            case.mkdir()
            raw = np.arange(32 * 32 * 2, dtype=np.float32).reshape(32, 32, 2) + int(number) / 100
            qsm = np.sin(raw / 100) * 50
            qsm -= qsm.mean()
            for name, values in {"SWI.nii.gz": raw, "QSM.nii.gz": qsm,
                                 "QSM_mask.nii.gz": np.ones_like(raw), "reference_mask.nii.gz": np.ones_like(raw)}.items():
                image = nib.Nifti1Image(values, np.diag([1., 1., 2., 1.]))
                image.header.set_xyzt_units("mm")
                nib.save(image, case / name)
        self.config = {
            "schema": "swixqsm.server_training", "version": 1,
            "data_root": str(self.data), "work_dir": str(self.run / "prepared"), "model": "plain_unet",
            "manifest": {"layout": "flat", "original_train_ids": "ids/train.txt", "original_test_ids": "ids/test.txt",
                         "expected_train": 2, "expected_test": 1, "val_fraction": .5, "seed": 42, "qsm_units": "ppb",
                         "metadata_csv": str(self.run / "review/reviewed_case_metadata.csv")},
            "input_masks": {"enabled": False, "skull_stripped": True, "fill_holes": True, "valid_mask_policy": "intersect"},
            "quality": {"contract": str(self.run / "review/data_contract.json")},
            "independence": {"run_audit": True},
            "training": {"units": "ppb", "qsm_scale": 100., "qsm_offset": 0., "size": 32,
                         "epochs": 1, "warmup_epochs": 0, "batch_size": 2, "device": "cpu", "num_threads": 1,
                         "lr": .0002, "lr_disc": .0001, "lr_policy": "plateau", "min_lr": 1e-6, "seed": 42},
        }
        (self.bundle / "config.json").write_text(json.dumps(self.config))
        self.settings_path = self.bundle / "settings.json"
        self.settings_path.write_text(json.dumps({"schema": "swixqsm.upload_settings", "version": 1,
                 "run_root": str(self.run), "prepare_config": "config.json", "normalization_method": "mean_std", "data_range": 400.}))
        templates = self.bundle / "templates"
        templates.mkdir()
        self.actual_contract = {"schema": "swixqsm.data_contract", "version": 1,
            "geometry": {"axis_codes": ["R", "A", "S"], "slice_axis": 2,
                         "spacing_mm_min": [1., 1., 2.], "spacing_mm_max": [1., 1., 2.], "max_obliquity_deg": 0.},
            "reference": {"id": "synthetic_ref", "method": "mean", "region": "synthetic_full_brain",
                          "target_ppb": 0., "tolerance_ppb": 1e-4, "require_mask": True},
            "registration": {"require_review": True}}
        template = json.loads(json.dumps(self.actual_contract))
        template["reference"]["id"] = None
        (templates / "data_contract.TEMPLATE.json").write_text(json.dumps(template))
        self.workflow = Workflow(self.settings_path)
        self.args = argparse.Namespace(label="final", model="plain_unet", preview_limit=0,
                                      header_limit=0, data_range=None, dry_run=False)

    def do(self, stage):
        return self.workflow.execute(stage, self.args)

    def fill_actual_synthetic_review(self):
        self.workflow.contract.write_text(json.dumps(self.actual_contract))
        with self.workflow.metadata.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            row.update(registration_review="accepted", registration_reviewer="Synthetic fixture author",
                       registration_review_date="2026-10-18", qsm_reference="synthetic_ref",
                       qsm_reference_method="mean", qsm_reference_region="synthetic_full_brain",
                       qsm_reference_source="Synthetic test data explicitly mean-centered before saving",
                       qsm_reference_target_ppb="0", qsm_reference_mask_path=str(self.data / row["case_id"] / "reference_mask.nii.gz"))
        with self.workflow.metadata.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)

    def install_test_copy(self):
        source = Path(__file__).resolve().parents[1]
        shutil.copytree(source, self.bundle / "swixqsm_minimal", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    def test_dry_run_never_reads_images_or_creates_a_run(self):
        self.args.dry_run = True
        with mock.patch("nibabel.load", side_effect=AssertionError("No image reads")), \
                mock.patch("subprocess.run", side_effect=AssertionError("No execution")):
            for stage in ("inventory", "masks", "init-review", "headers", "prepare", "fit", "train", "validate",
                          "test-qc", "test", "predict-test", "evaluate-test"):
                self.do(stage)
        self.assertFalse(self.run.exists())

    def test_path_mismatch_or_data_output_rejected(self):
        settings = json.loads(self.settings_path.read_text())
        settings["run_root"] = str(self.data / "wrong")
        self.settings_path.write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, "outside"):
            Workflow(self.settings_path)

    def test_deferred_roi_overlay_and_eight_model_comparison_commands(self):
        config = json.loads(json.dumps(self.config))
        config["manifest"].pop("metadata_csv")
        config["manifest"]["qsm_units"] = config["training"]["units"] = "stored"
        config["quality"] = {"allow_unverified_reason": "User deferred ROI and units review"}
        (self.bundle / "config.json").write_text(json.dumps(config))
        settings = json.loads(self.settings_path.read_text())
        settings.update(quality_mode="deferred", roi_evaluation={"enabled": True,
                        "spec": "nuclei.json", "manifests": {"test": "roi_test.csv"},
                        "review_policy": "deferred", "bootstrap_reps": 0})
        self.settings_path.write_text(json.dumps(settings))
        workflow = Workflow(self.settings_path)
        self.args.model = "reviewer_suite"
        commands = workflow.commands("evaluate-test", self.args, planning=True)
        self.assertEqual(len(commands), 8)
        for command in commands:
            self.assertIn("--roi-manifest", command)
            self.assertEqual(command[command.index("--roi-review-policy") + 1], "deferred")
            self.assertIn("evaluation_manifest.csv", " ".join(command))
        self.assertFalse(any("--roi-manifest" in c for c in workflow.commands("evaluate-val", self.args, planning=True)))
        comparison = workflow.commands("compare-test", self.args, planning=True)
        self.assertEqual(len(comparison), 2)
        self.assertIn("swixqsm_minimal.compare_models", comparison[0])
        self.assertIn("swixqsm_minimal.compare_roi_models", comparison[1])
        self.assertEqual(comparison[1].count("--expected-model"), 8)
        self.assertEqual(workflow.commands("compare-roi-test", self.args, planning=True), comparison[1:])
        with self.assertRaisesRegex(ValueError, "requires enabled"):
            workflow.commands("compare-roi-val", self.args, planning=True)
        self.assertFalse(self.run.exists())

    def test_complete_synthetic_upload_route_through_physical_evaluation(self):
        self.install_test_copy()
        self.do("inventory")
        self.do("masks")
        self.do("init-review")
        with self.workflow.metadata.open(newline="") as stream:
            initial = list(csv.DictReader(stream))
        self.assertEqual({r["registration_review"] for r in initial}, {"pending"})
        self.assertEqual({r["registration_reviewer"] for r in initial}, {""})
        with self.assertRaises(ValueError):
            self.do("precheck")
        self.fill_actual_synthetic_review()
        with mock.patch("nibabel.Nifti1Image.get_fdata", side_effect=AssertionError("Header-only stage must not read arrays")):
            self.do("headers")
        self.do("precheck")
        self.do("prepare")
        with self.assertRaises(FileNotFoundError):
            self.do("train")
        self.do("test-qc")
        self.do("fit")
        fitted = json.loads(self.workflow.normfile.read_text())
        self.assertEqual(fitted["fit_selection"]["n_subjects"], 1)
        self.do("train")
        self.do("validate")
        self.do("test")
        self.do("predict-val")
        self.do("evaluate-val")
        self.do("predict-test")
        self.do("evaluate-test")
        for split in ("val", "test"):
            evaluation = json.loads((self.run / "evaluation" / f"plain_unet_{split}_final" / "summary.json").read_text())
            self.assertIn("zero", evaluation["summary"])
            self.assertIn("plain_unet", evaluation["summary"])
            self.assertEqual(evaluation["config"]["units"], "ppb")
        frozen_manifest = Path(self.workflow.prepared_record()["artifacts"]["manifest"])
        frozen_bytes = frozen_manifest.read_bytes()
        review_bytes = self.workflow.metadata.read_bytes()
        self.workflow.metadata.write_text(self.workflow.metadata.read_text().replace("accepted", "rejected", 1))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.workflow.prepared_record()
        self.workflow.metadata.write_bytes(review_bytes)
        self.assertEqual(frozen_manifest.read_bytes(), frozen_bytes)
        changed = json.loads(json.dumps(self.config))
        changed["training"]["lr"] = .0003
        config_path = self.bundle / "config.json"
        config_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.workflow.prepared_record()
        config_path.write_text(json.dumps(self.config))
        self.workflow.prepared_record()
        with self.assertRaises(FileExistsError):
            self.do("init-review")

    def test_deferred_qc_stored_scale_training_and_separate_comparison_csvs(self):
        config = json.loads(json.dumps(self.config))
        config["manifest"].pop("metadata_csv")
        config["manifest"]["qsm_units"] = "stored"
        config["training"]["units"] = "stored"
        config["input_masks"]["enabled"] = True
        config["quality"] = {"allow_unverified_reason": "Synthetic fixture: manual QC and source units are deferred"}
        (self.bundle / "config.json").write_text(json.dumps(config))
        settings = json.loads(self.settings_path.read_text())
        settings.update(quality_mode="deferred", default_model="reviewer_suite", data_range=None,
                        reference_id="unverified_original_reference")
        self.settings_path.write_text(json.dumps(settings))
        self.workflow = Workflow(self.settings_path)
        self.install_test_copy()
        self.do("prepare")
        self.assertFalse(self.workflow.contract.exists())
        self.do("fit")
        self.do("range")
        self.do("train")
        self.do("validate")
        self.do("test")
        for split in ("val", "test"):
            self.do("predict-" + split)
            self.do("evaluate-" + split)
            self.do("compare-" + split)
            directory = self.run / "comparison" / f"{split}_final"
            for name in ("all_models_per_case.csv", "all_models_per_subject.csv",
                         "model_metrics_long.csv", "model_metrics_wide.csv"):
                self.assertTrue((directory / name).is_file(), name)
            with (directory / "all_models_per_case.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(row["units"] == "stored" for row in rows))
            self.assertTrue(all(row["qc_status"] == "unverified" for row in rows))
        self.assertFalse((self.run / "test_qc_final").exists())


if __name__ == "__main__":
    unittest.main()

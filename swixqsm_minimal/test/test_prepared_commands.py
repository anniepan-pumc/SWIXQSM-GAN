"""Prepared command generation reads metadata only and never launches jobs."""
import copy
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest import mock

from ..prepared_commands import commands_from_prepared, main
from ..run_server import _canonical_sha, load_config


class PreparedCommandsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.work = self.root / "prepared run"
        self.work.mkdir()
        self.manifest = self.work / "manifest.csv"
        self.manifest.write_text("subject_id,split\nsynthetic,train\n")
        self.config = {"data_root": str(self.root / "data"), "work_dir": str(self.work),
                       "model": "swixqsm_gan", "manifest": {"qsm_units": "ppb", "layout": "flat"},
                       "training": {"units": "ppm", "qsm_scale": .1, "qsm_offset": .02,
                                    "lr": .0002, "lr_disc": .0001, "lr_policy": "plateau",
                                    "plateau_patience": 5, "plateau_factor": .5, "plateau_threshold": 0,
                                    "min_lr": 1e-6, "seed": 42, "device": "cuda", "warmup_epochs": 3}}
        signature = _canonical_sha({k: v for k, v in self.config.items() if k != "model"})
        self.prepared = {"schema": "swixqsm.server_prepared", "version": 1, "status": "prepared",
                         "config_prepare_signature": signature,
                         "artifacts": {"manifest": str(self.manifest)},
                         "fixed_file_sha256": {str(self.manifest): hashlib.sha256(self.manifest.read_bytes()).hexdigest()},
                         "quality_args": ["--allow-unverified-data", "Synthetic reason; do not execute a shell"],
                         "independence_args": ["--allow-unverified-independence", "Synthetic only"]}
        self.plan = {"schema": "swixqsm.server_plan", "version": 1,
                     "config_prepare_signature": signature, "resolved_configuration": self.config}
        (self.work / "prepared.json").write_text(json.dumps(self.prepared))
        (self.work / "server_plan.json").write_text(json.dumps(self.plan))
        self.norm = self.root / "normalization file.json"  # Not yet fitted; planning must not open it.

    def test_fit_and_training_inherit_gates_and_replace_manual_scale(self):
        with mock.patch("nibabel.load", side_effect=AssertionError("No image reads")), \
                mock.patch("subprocess.run", side_effect=AssertionError("Never launch a process")):
            fit, = commands_from_prepared(self.work, self.norm)
            commands = commands_from_prepared(self.work, self.norm, stage="train", model="reviewer_suite")
        self.assertEqual(fit[fit.index("--qsm-units") + 1], "ppb")
        self.assertEqual(fit[fit.index("--units") + 1], "ppm")
        self.assertEqual(len(commands), 8)
        for command in commands:
            self.assertNotIn("--qsm-scale", command)
            self.assertNotIn("--qsm-offset", command)
            self.assertEqual(command[command.index("--normalization-file") + 1], str(self.norm))
            self.assertEqual(command[command.index("--lr") + 1], "0.0002")
            self.assertEqual(command[command.index("--lr-policy") + 1], "plateau")
            self.assertEqual(command[command.index("--allow-unverified-data") + 1], self.prepared["quality_args"][1])
            self.assertEqual(shlex.split(shlex.join(str(a) for a in command)), [str(a) for a in command])
        self.assertFalse(self.norm.exists())
        self.assertFalse((self.root / "revision_models_v1").exists())

    def test_validation_defaults_to_val_and_phase_specific_checkpoint(self):
        gan, = commands_from_prepared(self.work, self.norm, stage="validate")
        l1, = commands_from_prepared(self.work, self.norm, stage="validate", model="plain_unet", split="test")
        self.assertTrue(gan[gan.index("--checkpoint") + 1].endswith("/swixqsm_gan/best_gan.pt"))
        self.assertEqual(gan[gan.index("--split") + 1], "val")
        self.assertTrue(l1[l1.index("--checkpoint") + 1].endswith("/plain_unet/best_l1.pt"))
        self.assertEqual(l1[l1.index("--split") + 1], "test")

    def test_changed_manifest_or_plan_and_output_in_data_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            commands_from_prepared(self.work, self.norm, stage="train", output_root=self.root / "data" / "models")
        with self.assertRaisesRegex(ValueError, "outside"):
            commands_from_prepared(self.work, self.root / "data" / "normalization.json", stage="fit")
        original = self.manifest.read_bytes()
        self.manifest.write_bytes(original + b"changed\n")
        with self.assertRaisesRegex(ValueError, "manifest bytes changed"):
            commands_from_prepared(self.work, self.norm)
        self.manifest.write_bytes(original)
        plan = copy.deepcopy(self.plan)
        plan["resolved_configuration"]["training"]["lr"] = .1
        (self.work / "server_plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "configuration disagree"):
            commands_from_prepared(self.work, self.norm)

    def test_supplied_server_profile_parses_with_local_metadata_only(self):
        profile = Path(__file__).resolve().parents[1] / "server_training.pxq-flat.example.json"
        config = json.loads(profile.read_text())
        self.assertEqual(config["data_root"], "/path/to/swi_qsm_nifti")
        self.assertTrue(config["work_dir"].endswith("/res-1018/revision_prepared_v1"))
        self.assertEqual(config["manifest"]["val_fraction"], .1)
        config["data_root"], config["work_dir"] = str(self.root / "source"), str(self.root / "new-run")
        config["manifest"].pop("metadata_csv")
        for key in ("original_train_ids", "original_test_ids"):
            config["manifest"][key] = str((profile.parent / config["manifest"][key]).resolve())
        config["quality"] = {"allow_unverified_reason": "Synthetic parser-only validation"}
        config["independence"] = {"allow_unverified_reason": "Synthetic parser-only validation"}
        path = self.root / "server_config.json"
        path.write_text(json.dumps(config))
        with mock.patch("nibabel.load", side_effect=AssertionError("No image reads")):
            resolved = load_config(path)
        self.assertEqual(resolved["training"]["lr"], .0002)
        self.assertEqual(resolved["training"]["lr_disc"], .0001)
        self.assertEqual(resolved["training"]["lr_policy"], "plateau")
        self.assertEqual(resolved["manifest"]["layout"], "flat")

    def test_test_requires_explicit_covered_qc_without_downgrading_prepared_report(self):
        prepared = copy.deepcopy(self.prepared)
        prepared["quality_args"] = ["--qc-report", str(self.work / "train_val_qc.json")]
        (self.work / "prepared.json").write_text(json.dumps(prepared))
        with self.assertRaisesRegex(ValueError, "covering test"):
            commands_from_prepared(self.work, self.norm, stage="validate", split="test")
        report = self.root / "test_qc" / "report.json"
        command, = commands_from_prepared(self.work, self.norm, stage="validate", split="test", qc_report=report)
        self.assertEqual(command[command.index("--qc-report") + 1], str(report))
        self.assertNotIn(str(self.work / "train_val_qc.json"), command)
        self.assertNotIn("--allow-unverified-data", command)
        self.assertIn(self.prepared["independence_args"][1], command)
        with self.assertRaisesRegex(ValueError, "only supported"):
            commands_from_prepared(self.work, self.norm, stage="train", qc_report=report)


if __name__ == "__main__":
    unittest.main()

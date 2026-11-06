"""Exercise the real training/save path with controlled validation rankings."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import torch

from .. import train
from ..util.independence_gate import assert_prediction_phase, assert_source_independence


class TrainingWorkflowTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for index, split in enumerate(("train", "val", "test")):
            row = {"subject_id": f"synthetic_{split}", "global_subject_id": f"global_{split}", "split": split}
            raw = np.arange(32 * 32, dtype=np.float32).reshape(32, 32, 1) + index
            for key, values in {"swi_path": raw, "qsm_path": raw / 10000 - .1,
                                "swi_mask_path": np.ones_like(raw), "mask_path": np.ones_like(raw)}.items():
                path = self.root / f"{split}_{key}.nii.gz"
                image = nib.Nifti1Image(values, np.diag([.8, .8, 2., 1.]))
                image.header.set_xyzt_units("mm")
                nib.save(image, path)
                row[key] = str(path)
            self.rows.append(row)
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def args(self, output, *extra):
        return ["train", "--manifest", str(self.manifest), "--output", str(output),
                "--size", "32", "--qsm-scale", ".1", "--units", "ppm", "--batch-size", "1",
                "--device", "cpu", "--allow-unverified-data", "Synthetic phase test",
                "--allow-unverified-independence", "Synthetic explicit legacy identity history", *extra]

    def test_warmup_global_best_and_actual_gan_best_are_distinct(self):
        output = self.root / "phase_run"
        # No test image may enter training or model selection.
        Path(self.rows[-1]["qsm_path"]).write_bytes(b"Unconsumed synthetic test data")
        scores = [(1., {"synthetic_val": 1.}), (2., {"synthetic_val": 2.})]
        with mock.patch.object(sys, "argv", self.args(output, "--epochs", "2", "--warmup-epochs", "1", "--adv-weight", ".01")), \
                mock.patch.object(train, "validate", side_effect=scores):
            train.main()
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        gan = torch.load(output / "best_gan.pt", map_location="cpu", weights_only=True)
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(best["training_state"]["current_run_phase"], "warmup")
        self.assertEqual(best["checkpoint_role"], "global_best")
        self.assertTrue((output / "best_warmup.pt").exists())
        self.assertFalse((output / "best_l1.pt").exists())
        self.assertEqual(gan["epoch"], 2)
        self.assertEqual(gan["training_state"]["current_run_phase"], "gan")
        self.assertEqual(gan["checkpoint_role"], "current_run_gan_best")
        self.assertEqual(gan["val_mae"], 2.)
        self.assertEqual(gan["checkpoint_selection_score"], 2.)
        self.assertEqual(gan["best_val_mae"], 1.)
        self.assertEqual(gan["best_val_mae_scope"], "global_current_run")
        self.assertEqual(gan["training_state"]["generator_updates"], 2)
        self.assertEqual(gan["training_state"]["discriminator_updates"], 1)
        self.assertEqual(gan["training_state"]["adversarial_generator_updates"], 1)
        self.assertTrue(any(not torch.equal(best["discriminator"][key], gan["discriminator"][key])
                            for key in gan["discriminator"]))
        selection = json.loads((output / "selection.json").read_text())
        self.assertEqual(selection["training_state"]["global_best"]["phase"], "warmup")
        self.assertEqual(selection["training_state"]["best_by_phase"]["gan"]["score"], 2.)
        self.assertEqual(selection["data_independence"]["status"], "unverified")
        self.assertEqual(gan["training_objective"]["adversarial"]["mode"], "valid_patch")
        self.assertEqual(gan["training_objective"]["adversarial"]["receptive_field"], [38, 38])
        history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
        self.assertIsNone(history[0]["training_diagnostics"]["metrics"]["adv"]["mean"])
        self.assertEqual(history[0]["training_diagnostics"]["metrics"]["d_total"]["status"], "not_observed")
        self.assertGreater(history[1]["training_diagnostics"]["metrics"]["adv"]["count"], 0)
        self.assertEqual(gan["training_diagnostics"]["distributions"]["n_batches_recorded"], 1)
        self.assertEqual(gan["sampling_policy"]["mode"], "volume_grouped")
        self.assertEqual(gan["sampling_policy"]["num_workers"], 0)
        self.assertEqual(gan["config"]["indexing_io"]["train"]["index_volume_decodes"], 1)
        self.assertEqual(gan["config"]["indexing_io"]["train"]["prepared_volume_decodes"], 0)
        self.assertEqual(gan["training_io"]["train"]["read_delta"]["prepared_volume_decodes"], 0)
        self.assertEqual(gan["training_io"]["train"]["cache_counter_delta"]["hits"], 1)
        self.assertEqual(history[0]["training_io"]["train"]["read_delta"]["prepared_volume_decodes"], 4)
        self.assertEqual(selection["training_io"], gan["training_io"])
        sampled = gan["training_diagnostics"]["distributions"]["batches"][0]
        self.assertEqual(sampled["units"], "ppm")
        self.assertEqual(sampled["subject_ids"], ["synthetic_train"])

        wrong = self.root / "wrong_phase.nii.gz"
        common = ["--swi", self.rows[1]["swi_path"], "--swi-mask", self.rows[1]["swi_mask_path"],
                  "--allow-unverified-data", "Synthetic phase check", "--expected-phase", "gan"]
        result = subprocess.run([sys.executable, "-m", "swixqsm_minimal.infer", "--checkpoint", str(output / "best.pt"),
                                 "--output", str(wrong), *common], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkpoint phase", result.stderr)
        self.assertFalse(wrong.exists())
        prediction = self.root / "gan_prediction.nii.gz"
        result = subprocess.run([sys.executable, "-m", "swixqsm_minimal.infer", "--checkpoint", str(output / "best_gan.pt"),
                                 "--output", str(prediction), *common], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metadata = json.loads(prediction.with_suffix("").with_suffix(".json").read_text())
        assert_prediction_phase(metadata, "gan")
        self.assertEqual(metadata["DataIndependence"]["status"], "unverified")
        self.assertEqual(metadata["CheckpointRole"], "current_run_gan_best")
        self.assertEqual(metadata["TrainingObjective"], gan["training_objective"])
        self.assertEqual(metadata["TrainingDiagnostics"], gan["training_diagnostics"])

    def test_l1_only_saves_l1_best_and_never_claims_adversarial_updates(self):
        output = self.root / "l1_run"
        with mock.patch.object(sys, "argv", self.args(output, "--epochs", "1", "--warmup-epochs", "10", "--adv-weight", "0",
                                                       "--diagnostic-batches", "0", "--adversarial-mode", "all_patches",
                                                       "--sampling-mode", "slice_random")):
            train.main()
        checkpoint = torch.load(output / "best_l1.pt", map_location="cpu", weights_only=True)
        state = checkpoint["training_state"]
        self.assertEqual(state["current_run_phase"], "l1")
        self.assertEqual(state["generator_updates"], 1)
        self.assertEqual(state["adversarial_generator_updates"], 0)
        self.assertEqual(state["discriminator_updates"], 0)
        self.assertFalse((output / "best_gan.pt").exists())
        self.assertFalse((output / "best_warmup.pt").exists())
        self.assertEqual(checkpoint["training_objective"]["adversarial"]["mode"], "all_patches")
        self.assertIsNone(checkpoint["training_diagnostics"]["metrics"]["d_total"]["mean"])
        self.assertEqual(checkpoint["training_diagnostics"]["distributions"]["n_batches_recorded"], 0)
        self.assertEqual(checkpoint["sampling_policy"]["mode"], "slice_random")
        self.assertEqual(checkpoint["training_io"]["train"]["read_delta"]["volume_preparations"], 1)
        self.assertEqual(checkpoint["training_io"]["val"]["read_delta"]["volume_preparations"], 1)

    def test_invalid_cache_group_rejected_before_data_access(self):
        self.manifest.unlink()
        for extra in (("--volume-group-size", "3", "--cache-volumes", "2"),
                      ("--cache-mib", "nan"), ("--cache-mib", "0"), ("--cache-volumes", "0"),
                      ("--seed", "-1")):
            output = self.root / "invalid_sampling"
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", self.args(output, *extra)), \
                    self.assertRaises(SystemExit):
                train.main()
            self.assertFalse(output.exists())

    def test_gan_that_would_never_start_rejected_before_data_access(self):
        output = self.root / "never_gan"
        self.manifest.unlink()
        with mock.patch.object(sys, "argv", self.args(output, "--epochs", "1", "--warmup-epochs", "1", "--adv-weight", ".01")), \
                self.assertRaises(SystemExit):
            train.main()
        self.assertFalse(output.exists())

    def test_source_independence_and_phase_checks_do_not_infer_missing_history(self):
        expected = {"status": "verified", "manifest_identity_sha256": "same_split"}
        assert_source_independence(expected, expected, name="Synthetic")
        for source in (None, {"status": "unverified"}, {"status": "verified", "manifest_identity_sha256": "other_split"}):
            with self.subTest(source=source), self.assertRaises(ValueError):
                assert_source_independence(expected, source, name="Synthetic")
        for state in (None, {"current_run_phase": "warmup"}, {"current_run_phase": "gan", "adversarial_generator_updates": 0}):
            with self.subTest(state=state), self.assertRaises(ValueError):
                assert_prediction_phase({"TrainingState": state}, "gan")
        with self.assertRaises(ValueError):
            assert_prediction_phase({"TrainingState": {"current_run_phase": "gan", "adversarial_generator_updates": 1},
                                     "PredictionTrainingState": {"current_run_phase": "warmup"}}, "gan")


if __name__ == "__main__":
    unittest.main()

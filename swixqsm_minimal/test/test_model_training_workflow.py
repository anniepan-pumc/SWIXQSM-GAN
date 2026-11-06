"""Real synthetic training, checkpoint, inference and epoch-boundary resume."""
import csv
import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest import mock

import nibabel as nib
import numpy as np
import torch

from .. import train_models
from ..test import infer
from ..util.model_loading import generator_from_checkpoint


class ModelTrainingWorkflowTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for number, (sid, gid, split) in enumerate([
                ("train_a", "train_global", "train"),
                ("train_b", "train_global", "train"),
                ("val_a", "val_global", "val"),
                ("val_b", "val_global", "val"),
                ("test", "test_global", "test")]):
            shape = (32, 32, 1 if number == 2 else 2)
            signal = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + number
            values = {"swi_path": signal, "qsm_path": .03 * np.sin(signal / 100) - .01,
                      "swi_mask_path": np.ones(shape, np.float32), "mask_path": np.ones(shape, np.float32)}
            row = dict(subject_id=sid, global_subject_id=gid, split=split, qsm_units="ppm")
            for key, array in values.items():
                path = self.root / f"{sid}_{key}.nii.gz"
                image = nib.Nifti1Image(array, np.diag([1., 1., 2., 1.]))
                image.header.set_xyzt_units("mm")
                nib.save(image, path)
                row[key] = str(path)
            rows.append(row)
        # No test bytes or even headers may be decoded by this training entry.
        for key in values:
            Path(rows[-1][key]).write_bytes(b"not a NIfTI; held-out metadata only")
        self.rows = rows
        self.manifest = self.root / "data.csv"
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def args(self, output, model, *extra):
        return ["--model", model, "--manifest", str(self.manifest), "--output", str(output),
                "--size", "32", "--units", "ppm", "--qsm-scale", ".1", "--batch-size", "2",
                "--device", "cpu", "--num-threads", "1", "--diagnostic-batches", "1",
                "--allow-unverified-data", "Only synthetic numerical validation",
                "--allow-unverified-independence", "Only synthetic identities", *extra]

    def test_four_real_models_train_save_and_restore_signed_inference(self):
        initial_states, traces = {}, {}
        for name in train_models.FOUR_MODELS:
            output = self.root / name
            train_models.main(self.args(output, name, "--epochs", "1"))
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["config"]["recipe"], name)
            self.assertEqual(checkpoint["model_spec"]["model_name"], name)
            self.assertEqual(checkpoint["training_state"]["generator_updates"], 2)
            self.assertEqual(checkpoint["training_state"]["discriminator_updates"], 0 if name == "plain_unet" else 2)
            self.assertEqual(set(checkpoint["history"][0]["patient_mae"]), {"val_global"})
            self.assertTrue((output / ("best_l1.pt" if name == "plain_unet" else "best_gan.pt")).is_file())
            initial_states[name] = checkpoint["model_spec"]["initialization"]["generator_initial_state_sha256"]
            traces[name] = checkpoint["history"][0]["sampling_index_sha256"]
            generator = generator_from_checkpoint(checkpoint).eval()
            with torch.no_grad():
                generated = generator(torch.zeros(1, 1, 32, 32))
            self.assertEqual(tuple(generated.shape), (1, 1, 32, 32))
            self.assertTrue(torch.isfinite(generated).all())
            # Exercise the exact SWI-only saving path used on the server, with
            # the model selected from checkpoint metadata rather than a flag.
            prediction = self.root / f"{name}_prediction.nii.gz"
            cli = ["infer", "--checkpoint", str(output / "best.pt"), "--swi", self.rows[2]["swi_path"],
                   "--swi-mask", self.rows[2]["swi_mask_path"], "--output", str(prediction),
                   "--allow-unverified-data", "Synthetic inference", "--device", "cpu"]
            with mock.patch("sys.argv", cli):
                infer.main()
            image = nib.load(prediction)
            self.assertEqual(image.shape, (32, 32, 1))
            metadata = json.loads(prediction.with_suffix("").with_suffix(".json").read_text())
            self.assertEqual(metadata["ModelSpec"], checkpoint["model_spec"])
            self.assertEqual(metadata["ExperimentRecipe"], name)
            self.assertEqual(metadata["Units"], "ppm")
            del generator, checkpoint
            gc.collect()
        self.assertEqual(initial_states["swixqsm_gan"], initial_states["plain_unet"])
        self.assertEqual(len(set(traces.values())), 1)

    def test_resume_matches_uninterrupted_dropout_optimizer_and_selection(self):
        continuous, segmented = self.root / "continuous", self.root / "segmented"
        train_models.main(self.args(continuous, "swixqsm_gan", "--epochs", "2"))
        train_models.main(self.args(segmented, "swixqsm_gan", "--epochs", "2", "--max-epochs-this-run", "1"))
        train_models.main(["--resume", str(segmented / "last.pt"), "--device", "cpu"])
        a = torch.load(continuous / "last.pt", map_location="cpu", weights_only=True)
        b = torch.load(segmented / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(a["training_state"], b["training_state"])
        self.assertEqual(a["val_mae"], b["val_mae"])
        for key in ("generator", "discriminator"):
            for name in a[key]:
                self.assertTrue(torch.equal(a[key][name], b[key][name]), (key, name))
        for h1, h2 in zip(a["history"], b["history"]):
            self.assertEqual(h1["sampling_index_sha256"], h2["sampling_index_sha256"])
            self.assertEqual(h1["learning_rates"], h2["learning_rates"])
        self.assertEqual(a["scheduler_g"], b["scheduler_g"])
        self.assertEqual(a["scheduler_d"], b["scheduler_d"])
        with self.assertRaises(SystemExit):
            train_models.main(["--resume", str(segmented / "last.pt"), "--epochs", "3", "--device", "cpu"])

    def test_strict_ablation_has_same_initial_generator_and_no_adversarial_updates(self):
        path = self.root / "ablation"
        train_models.main(self.args(path, "swixqsm_l1_ablation", "--epochs", "1"))
        ckpt = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(ckpt["model_spec"]["model_name"], "swixqsm_gan")
        self.assertEqual(ckpt["training_objective"]["lambda_l1"], .2)
        self.assertEqual(ckpt["training_state"]["discriminator_updates"], 0)
        self.assertIsNone(ckpt["training_diagnostics"]["metrics"]["adv"]["mean"])
        self.assertEqual(ckpt["optimizer_d"]["state"], {})
        with self.assertRaises(ValueError):
            train_models.main(self.args(self.root / "wrong_ablation", "swixqsm_l1_ablation", "--lambda-l1", "1"))

    def test_warmup_schedules_only_real_d_updates_and_cannot_bypass_early_stop(self):
        output = self.root / "warmup"
        with mock.patch.object(train_models, "validate", return_value=(1., {"val_global": 1.})):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                train_models.main(self.args(output, "swixqsm_gan", "--epochs", "5", "--warmup-epochs", "2",
                                           "--early-stopping-patience", "1", "--max-epochs-this-run", "2"))
            self.assertFalse(any("lr_scheduler.step()" in str(w.message) for w in caught))
            warm = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(warm["epoch"], 2)
            self.assertEqual(warm["training_state"]["discriminator_updates"], 0)
            self.assertEqual(warm["scheduler_d"]["last_epoch"], 0)
            self.assertEqual(warm["optimizer_d"]["param_groups"][0]["lr"], 1e-4)
            train_models.main(["--resume", str(output / "last.pt"), "--device", "cpu"])
        stopped = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(stopped["epoch"], 4)
        self.assertEqual(stopped["training_state"]["discriminator_updates"], 4)
        self.assertEqual(stopped["scheduler_d"]["last_epoch"], 2)
        self.assertAlmostEqual(stopped["optimizer_d"]["param_groups"][0]["lr"], 2.5e-5)
        self.assertTrue((output / "best_gan.pt").is_file())
        self.assertEqual(json.loads((output / "run_result.json").read_text())["stop_reason"], "validation_patience_exhausted")
        with self.assertRaisesRegex(ValueError, "early-stopping rule"):
            train_models.main(["--resume", str(output / "last.pt"), "--device", "cpu"])

    def test_resume_rejects_changed_runtime_inputs_and_committed_history(self):
        output = self.root / "guards"
        train_models.main(self.args(output, "plain_unet", "--epochs", "2", "--max-epochs-this-run", "1"))
        cli = ["--resume", str(output / "last.pt"), "--device", "cpu"]
        with mock.patch.object(train_models, "runtime_versions", return_value={"torch": "other"}):
            with self.assertRaisesRegex(ValueError, "runtime versions"):
                train_models.main(cli)
        path = Path(self.rows[0]["qsm_path"])
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(ValueError, "input file metadata changed"):
            train_models.main(cli)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        history_path = output / "history.jsonl"
        record = json.loads(history_path.read_text())
        record["val_mae"] += .1
        history_path.write_text(json.dumps(record) + "\n")
        # Even if the external history hash is changed, its parsed contents
        # must still equal the committed history embedded in the checkpoint.
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        checkpoint["history_sha256"] = train_models.sha256(history_path)
        torch.save(checkpoint, output / "last.pt")
        with self.assertRaisesRegex(ValueError, "Resume history"):
            train_models.main(cli)

    def test_plain_gan_coefficients_and_holdout_identity_cannot_be_mislabeled(self):
        with self.assertRaises(ValueError):
            train_models.main(self.args(self.root / "wrong_plain", "plain_unet", "--lambda-adv", "1"))
        rows = [dict(row) for row in self.rows]
        rows[-1]["global_subject_id"] = "train_global"
        with self.assertRaisesRegex(ValueError, "crosses"):
            train_models.identity_checks(rows)
        with self.assertRaisesRegex(ValueError, "train AND val"):
            train_models.identity_checks([r for r in rows if r["split"] == "train"])

    def test_model_metadata_tampering_is_not_silently_loaded_as_legacy(self):
        with self.assertRaisesRegex(ValueError, "cannot omit"):
            generator_from_checkpoint({"schema": "swixqsm.comparison_checkpoint", "config": {}})
        g, _, spec = train_models.create_model("fcn_gan_l1", image_size=32)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            generator_from_checkpoint({"generator": g.state_dict(), "model_spec": spec,
                                       "config": {"model_name": "plain_unet", "size": 32}})
        spec["image_size"] = 64
        with self.assertRaisesRegex(ValueError, "image_size"):
            generator_from_checkpoint({"generator": g.state_dict(), "model_spec": spec,
                                       "config": {"model_name": "fcn_gan_l1", "size": 32}})


if __name__ == "__main__":
    unittest.main()

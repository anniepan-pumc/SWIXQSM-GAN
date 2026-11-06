"""Synthetic end-to-end scale contracts; no real cohort images or weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import torch

from ..fit_normalization import fit_training_normalization, save_fitted_normalization
from ..model import UNetSmall
from ..restore_scale import main as restore_main
from ..test.infer import main as infer_main
from ..util.evaluation import evaluate_rows
from ..util.nifti import sidecar_path
from ..util.model_loading import load_checkpoint_snapshot
from ..util.normalization import from_checkpoint, normalization_metadata
from ..util.normalization_workflow import bind_training_units, resolve_training_normalization
from ..validate import main as validate_main


class NormalizationWorkflowTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        self.targets = {}
        raw = np.arange(32 * 32 * 2, dtype=np.float32).reshape(32, 32, 2)
        self.brain = np.ones_like(raw)
        self.brain[0] = 0
        for i, split in enumerate(("train", "val", "test")):
            qsm = raw / 4000 - .2 + .03 * i
            units = "ppb" if split == "test" else "ppm"
            row = dict(subject_id=f"s{i}", global_subject_id=f"g{i}", case_id=f"c{i}",
                       split=split, qsm_units=units)
            self.targets[split] = qsm
            for key, values in dict(swi_path=raw + i, qsm_path=qsm * (1000 if units == "ppb" else 1),
                                    swi_mask_path=self.brain, mask_path=self.brain).items():
                path = self.root / f"{split}_{key}.nii.gz"
                image = nib.Nifti1Image(values, np.diag([.8, .9, 1.5, 1]))
                image.header.set_xyzt_units("mm")
                nib.save(image, path)
                row[key] = str(path)
            self.rows.append(row)
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.fit = self.root / "normalization.json"
        artifact = fit_training_normalization(self.rows, units="ppm", manifest_path=self.manifest)
        save_fitted_normalization(artifact, self.fit)
        self.args = argparse.Namespace(normalization_file=str(self.fit), qsm_scale=None,
                                       qsm_offset=None, units="ppm")
        self.norm, self.source = resolve_training_normalization(
            self.args, self.rows, {"status": "unverified"}, {"status": "unverified"})
        gen = UNetSmall(normalize=False)
        for parameter in gen.parameters():
            parameter.data.zero_()
        gen.final[0].bias.data.fill_(-2.)
        self.checkpoint = self.root / "oracle.pt"
        self.spec = normalization_metadata(self.norm)
        torch.save({"generator": gen.state_dict(), "normalization": self.spec,
                    "config": {"size": 32, "generator_output": "linear", "normalization": self.spec,
                               "qsm_scale": self.norm.scale, "qsm_offset": self.norm.offset,
                               "units": "ppm", "normalization_source": self.source}}, self.checkpoint)

    def test_bound_training_parameters_and_conflicts(self):
        self.assertEqual(self.source["kind"], "train_fitted")
        self.assertEqual(self.source["sha256"], hashlib.sha256(self.fit.read_bytes()).hexdigest())
        for key, value in (("qsm_scale", .1), ("qsm_offset", 0.), ("units", "ppb")):
            args = argparse.Namespace(**vars(self.args))
            setattr(args, key, value)
            with self.assertRaises(ValueError):
                resolve_training_normalization(args, self.rows, {}, {})
        changed = [dict(row) for row in self.rows]
        changed[0]["qsm_units"] = "ppb"
        with self.assertRaisesRegex(ValueError, "identities/paths/units"):
            resolve_training_normalization(self.args, changed, {}, {})

    def test_fit_source_units_inherited_without_reinterpreting_as_canonical(self):
        rows = [{k: v for k, v in row.items() if k != "qsm_units"} for row in self.rows]
        # Here the source values are explicitly interpreted as ppb; output is ppm.
        artifact = fit_training_normalization(rows, units="ppm", default_qsm_units="ppb")
        path = self.root / "source_ppb.json"
        save_fitted_normalization(artifact, path)
        args = argparse.Namespace(normalization_file=str(path), qsm_scale=None, qsm_offset=None,
                                  qsm_units=None, units="ppm")
        bound = bind_training_units(rows, args, {"status": "unverified"})
        self.assertEqual({r["qsm_units"] for r in bound}, {"ppb"})
        norm, _ = resolve_training_normalization(args, bound, {}, {})
        self.assertAlmostEqual(norm.offset, self.norm.offset / 1000)
        self.assertAlmostEqual(norm.scale, self.norm.scale / 1000)
        args.qsm_units = "ppm"
        with self.assertRaisesRegex(ValueError, "Source --qsm-units conflicts"):
            bind_training_units(rows, args, {"status": "unverified"})

    def test_infer_and_validate_test_inverse_in_ppb(self):
        output = self.root / "prediction.nii.gz"
        row = self.rows[2]
        infer_main(["--checkpoint", str(self.checkpoint), "--swi", row["swi_path"],
                    "--swi-mask", row["swi_mask_path"], "--output", str(output),
                    "--output-units", "ppb", "--allow-unverified-data", "Synthetic test"])
        expected = self.norm.denormalize(-2., "ppb")
        result = nib.load(output).get_fdata()
        np.testing.assert_allclose(result[self.brain > 0], expected, atol=3e-5)
        np.testing.assert_array_equal(result[self.brain <= 0], 0)
        meta = json.loads(sidecar_path(output).read_text())
        self.assertEqual(meta["ValueSpace"], "physical_qsm")
        self.assertTrue(meta["InverseNormalization"]["applied"])
        self.assertEqual(meta["Units"], "ppb")
        for split in ("val", "test"):
            report = self.root / f"{split}.json"
            validate_main(["--checkpoint", str(self.checkpoint), "--manifest", str(self.manifest),
                           "--split", split, "--output", str(report), "--output-units", "ppb",
                           "--allow-unverified-data", "Synthetic test"])
            data = json.loads(report.read_text())
            wanted = np.abs(self.targets[split][self.brain > 0].astype(np.float64) * 1000 - expected).mean()
            self.assertAlmostEqual(data["mae"], wanted, places=4)
            self.assertEqual(data["n_subjects"], 1)
            self.assertEqual(data["identity"], "global_subject_id")
            self.assertEqual(data["Units"], "ppb")
            self.assertFalse(data["normalization_fitted_here"])

    def test_restore_signed_float_and_refuse_double_inverse_or_wrong_checkpoint(self):
        raw_path = self.root / "network.nii.gz"
        image = nib.Nifti1Image(np.full(self.brain.shape, -2., dtype=np.float32),
                               nib.load(self.rows[0]["swi_path"]).affine)
        nib.save(image, raw_path)
        meta = {"ValueSpace": "network_normalized", "Units": "dimensionless",
                "Normalization": self.spec,
                "CheckpointSHA256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()}
        sidecar_path(raw_path).write_text(json.dumps(meta))
        out = self.root / "restored.nii.gz"
        options = ["--checkpoint", str(self.checkpoint), "--input-value-space", "network_normalized",
                   "--swi-mask", self.rows[0]["swi_mask_path"]]
        restore_main(options + ["--input", str(raw_path), "--output", str(out), "--output-units", "ppb"])
        np.testing.assert_allclose(nib.load(out).get_fdata()[self.brain > 0],
                                   self.norm.denormalize(-2., "ppb"), atol=3e-5)
        with self.assertRaisesRegex(ValueError, "already physical"):
            restore_main(options + ["--input", str(out), "--output", str(self.root / "double.nii.gz")])
        meta["CheckpointSHA256"] = "0" * 64
        sidecar_path(raw_path).write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            restore_main(options + ["--input", str(raw_path), "--output", str(self.root / "wrong.nii.gz")])

    def test_evaluation_refuses_network_normalized_values(self):
        row = dict(self.rows[2], prediction_path=self.rows[2]["qsm_path"])
        sidecar_path(row["prediction_path"]).write_text(json.dumps({
            "ValueSpace": "network_normalized", "Units": "ppb", "ReferenceID": "test_ref"}))
        with self.assertRaisesRegex(ValueError, "physical QSM"):
            evaluate_rows([row], units="ppb", reference_id="test_ref", hfen_sigma_mm=.1)

    def test_checkpoint_sha_binds_loaded_snapshot_when_path_changes(self):
        payload = self.checkpoint.read_bytes()
        original_reader = Path.read_bytes

        def read_then_replace(path):
            content = original_reader(path)
            path.write_bytes(b"replaced after read; no longer the loaded snapshot")
            return content

        with mock.patch("swixqsm_minimal.util.model_loading.Path.read_bytes", read_then_replace):
            checkpoint, digest = load_checkpoint_snapshot(self.checkpoint)
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
        self.assertEqual(from_checkpoint(checkpoint)[0], self.norm)
        self.assertNotEqual(digest, hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())

    def test_cli_fitted_training_checkpoint_keeps_exact_transform(self):
        for recipe in (None, "plain_unet"):
            experiment = self.root / (recipe or "legacy")
            cmd = [sys.executable, "-m", "swixqsm_minimal.train", "--manifest", str(self.manifest),
                   "--output", str(experiment), "--normalization-file", str(self.fit), "--units", "ppm",
                   "--size", "32", "--epochs", "1", "--warmup-epochs", "0", "--batch-size", "2",
                   "--device", "cpu", "--allow-unverified-data", "Synthetic integration fixture",
                   "--allow-unverified-independence", "Synthetic integration fixture"]
            if recipe:
                cmd += ["--model", recipe, "--num-threads", "1"]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            checkpoint = torch.load(experiment / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(from_checkpoint(checkpoint)[0], self.norm)
            self.assertEqual(checkpoint["config"]["normalization_source"], self.source)
            self.assertEqual(checkpoint["config"]["qsm_offset"], self.norm.offset)

    def test_fitted_resume_inherits_parameters_and_rejects_replaced_artifact(self):
        experiment = self.root / "resume"
        command = [sys.executable, "-m", "swixqsm_minimal.train", "--model", "plain_unet",
                   "--manifest", str(self.manifest), "--output", str(experiment),
                   "--normalization-file", str(self.fit), "--units", "ppm", "--size", "32",
                   "--epochs", "2", "--max-epochs-this-run", "1", "--batch-size", "2",
                   "--device", "cpu", "--num-threads", "1", "--allow-unverified-data", "Synthetic fixture",
                   "--allow-unverified-independence", "Synthetic fixture"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        resume = [sys.executable, "-m", "swixqsm_minimal.train", "--resume", str(experiment / "last.pt"),
                  "--device", "cpu"]
        fitted_bytes = self.fit.read_bytes()
        # Even a reserialized file with identical parameters is a changed bound artifact.
        self.fit.write_bytes(fitted_bytes + b"\n")
        failed = subprocess.run(resume, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("Resume normalization parameters or fitting artifact changed", failed.stderr)
        self.fit.write_bytes(fitted_bytes)
        subprocess.run(resume, check=True, capture_output=True, text=True)
        checkpoint = torch.load(experiment / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(from_checkpoint(checkpoint)[0], self.norm)
        self.assertEqual(checkpoint["config"]["normalization_source"], self.source)


if __name__ == "__main__":
    unittest.main()

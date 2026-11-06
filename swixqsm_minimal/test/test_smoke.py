import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..util.data import PairedSlices, read_manifest
from ..util.engine import PatchDiscriminator, train_step
from ..model import UNetSmall


class SmokeTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.affine = np.diag([0.8, 0.9, 1.5, 1.])
        self.rows = []
        for i, split in enumerate(("train", "val")):
            swi = np.arange(32*40*2, dtype=np.float32).reshape(32, 40, 2)+i
            arrays = dict(swi_path=swi, qsm_path=swi/10000-0.1,
                          swi_mask_path=np.ones_like(swi), mask_path=np.ones_like(swi))
            row = dict(subject_id=f"synthetic_{i}", split=split)
            for key, array in arrays.items():
                path = self.root/f"{i}_{key}.nii.gz"
                nib.save(nib.Nifti1Image(array, self.affine), path)
                row[key] = str(path)
            self.rows.append(row)
        self.manifest = self.root/"manifest.csv"
        self.write_manifest(self.rows)

    def write_manifest(self, rows):
        with self.manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_float_geometry_and_leakage(self):
        rows = read_manifest(self.manifest)
        ds = PairedSlices(rows, "train", 40, 0.1)
        _, y, mask, _ = ds[0]
        self.assertLess(float(y.min()), 0)
        original = nib.load(rows[0]["qsm_path"]).get_fdata()
        np.testing.assert_allclose(y[0, :40, :32].numpy().T*0.1, original[:, :, 0], atol=1e-7)
        self.assertEqual(float(mask.sum()), 32*40)
        self.rows[1]["subject_id"] = self.rows[0]["subject_id"]
        self.write_manifest(self.rows)
        with self.assertRaisesRegex(ValueError, "leakage"):
            read_manifest(self.manifest)
        qsm = Path(rows[0]["qsm_path"])
        nib.save(nib.Nifti1Image(original, np.eye(4)), qsm)
        with self.assertRaisesRegex(ValueError, "registered"):
            PairedSlices(rows, "train", 40, 0.1)

    def test_gradient_isolation(self):
        gen, disc = UNetSmall(normalize=False), PatchDiscriminator()
        og = torch.optim.Adam(gen.parameters(), lr=1e-4)
        od = torch.optim.Adam(disc.parameters(), lr=1e-4)
        x, y, mask = torch.rand(2, 1, 32, 32), torch.randn(2, 1, 32, 32), torch.ones(2, 1, 32, 32)
        before = gen.final[0].weight.detach().clone()
        losses = train_step(gen, disc, og, od, x, y, mask, 0.01)
        self.assertTrue(np.isfinite(losses).all())
        self.assertFalse(torch.equal(before, gen.final[0].weight))
        self.assertTrue(all(p.grad is None for p in disc.parameters()))

    def test_cli_train_infer(self):
        experiment = self.root/"experiment"
        subprocess.run([sys.executable, "-m", "swixqsm_minimal.train",
                        "--manifest", str(self.manifest), "--output", str(experiment),
                        "--size", "40", "--qsm-scale", "0.1", "--units", "ppm",
                        "--qsm-offset", "0.02",
                        "--allow-unverified-data", "Legacy synthetic fixture; geometry/reference QC tested separately",
                        "--allow-unverified-independence", "Synthetic legacy fixture; identity history tested separately",
                        "--epochs", "2", "--warmup-epochs", "1", "--batch-size", "2",
                        "--device", "cpu"], check=True, capture_output=True)
        checkpoint = torch.load(experiment/"last.pt", weights_only=True, map_location="cpu")
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["normalization"]["qsm"]["offset"], 0.02)
        out = self.root/"prediction.nii.gz"
        subprocess.run([sys.executable, "-m", "swixqsm_minimal.infer",
                        "--checkpoint", str(experiment/"best.pt"),
                        "--swi", self.rows[1]["swi_path"], "--swi-mask", self.rows[1]["swi_mask_path"],
                        "--output", str(out), "--allow-unverified-data", "Synthetic legacy checkpoint"], check=True, capture_output=True)
        image = nib.load(out)
        self.assertEqual(image.shape, (32, 40, 2))
        np.testing.assert_allclose(image.affine, self.affine)
        self.assertTrue(np.isfinite(image.get_fdata()).all())
        self.assertEqual(json.loads(out.with_suffix("").with_suffix(".json").read_text())["Units"], "ppm")
        # Known normalized output through checkpoint -> CLI -> disk must become -0.18 ppm.
        checkpoint["generator"]["final.0.weight"].zero_()
        checkpoint["generator"]["final.0.bias"].fill_(-2.)
        oracle = experiment/"oracle.pt"
        torch.save(checkpoint, oracle)
        oracle_out = self.root/"oracle_prediction.nii.gz"
        subprocess.run([sys.executable, "-m", "swixqsm_minimal.test.infer",
                        "--checkpoint", str(oracle), "--swi", self.rows[1]["swi_path"],
                        "--swi-mask", self.rows[1]["swi_mask_path"], "--output", str(oracle_out),
                        "--allow-unverified-data", "Synthetic legacy checkpoint"],
                       check=True, capture_output=True)
        np.testing.assert_allclose(nib.load(oracle_out).get_fdata(), -0.18, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

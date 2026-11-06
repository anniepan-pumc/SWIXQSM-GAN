import copy
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..util.normalization import QSMNormalization, from_checkpoint, normalization_metadata, normalize_swi
from ..util.nifti import save_qsm
from ..util.data import load_volume, PairedSlices
from ..util.engine import validate
from .infer import predict_volume


class NormalizationTest(unittest.TestCase):
    def test_roundtrip_signed_extremes_and_units(self):
        original = np.array([-3., -0.1, 0., 0.2, 5.], dtype=np.float32)
        norm = QSMNormalization(0.1, 0.025, "ppm")
        for x in (original, torch.from_numpy(original)):
            encoded = norm.normalize(x)
            np.testing.assert_allclose(norm.denormalize(encoded), original, atol=1e-7, rtol=1e-6)
            self.assertLess(float(encoded.min()), -1.)
            self.assertGreater(float(encoded.max()), 1.)
        ppb = original*1000
        np.testing.assert_allclose(norm.denormalize(norm.normalize(ppb, "ppb"), "ppb"), ppb, atol=1e-3)
        other = QSMNormalization(100, 25, "ppb")
        np.testing.assert_allclose(norm.normalize(original), other.normalize(original, "ppm"), atol=1e-6)

    def test_metadata_rejects_unknown_or_conflicting(self):
        norm = QSMNormalization(0.1, 0.02)
        checkpoint = {"normalization": normalization_metadata(norm), "config": {"size": 40}}
        loaded, _ = from_checkpoint(checkpoint)
        self.assertEqual(loaded, norm)
        for modified in (
            {"generator": {}},
            {**checkpoint, "config": {"size": 40, "qsm_scale": 100}},
            {**checkpoint, "config": {"size": 40, "generator_output": "sigmoid"}},
        ):
            with self.assertRaises(ValueError):
                from_checkpoint(modified)
        bad = copy.deepcopy(checkpoint)
        del bad["normalization"]["qsm"]["offset"]
        with self.assertRaises(ValueError):
            from_checkpoint(bad)
        legacy = {"config": {"size": 40, "qsm_scale": 0.1, "units": "ppm", "generator_output": "linear",
                             "normalization": "SWI brain p0.5-p99.5 -> [0,1]; QSM / fixed scale"}}
        self.assertEqual(from_checkpoint(legacy)[0], QSMNormalization(0.1))
        for scale in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                QSMNormalization(scale)

    def test_swi_statistics_never_define_qsm_scale(self):
        x = np.arange(32*40*2, dtype=np.float32).reshape(32, 40, 2)
        a = normalize_swi(x, np.ones_like(x, dtype=bool))
        b = normalize_swi(x*7+120, np.ones_like(x, dtype=bool))
        np.testing.assert_allclose(a, b, atol=1e-6)
        class Constant(torch.nn.Module):
            def forward(self, value):
                return torch.full_like(value, -2.)
        qsm = QSMNormalization(0.1, 0.03)
        for source in (x, x*7+120):
            result, _ = predict_volume(Constant(), source, np.ones_like(x), 40, qsm)
            np.testing.assert_allclose(result, -0.17, atol=1e-7)

    def test_nifti_scaling_and_numerical_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = np.arange(32*40*2, dtype=np.int16).reshape(32, 40, 2)
            affine = np.array([[.8, 0, 0, 12], [0, .9, 0, -7], [0, 0, 1.5, 3], [0, 0, 0, 1]])
            source = nib.Nifti1Image(raw, affine)
            source.header.set_slope_inter(2., 100.)
            source.header["cal_max"] = 32000
            nib.save(source, root/"source.nii.gz")
            ref, swi = load_volume(root/"source.nii.gz")
            np.testing.assert_allclose(swi, raw.astype(np.float32)*2+100)
            brain = np.ones_like(swi)
            brain[0] = 0
            norm = QSMNormalization(0.1, .02)
            # Spatially varying oracle: covers transpose, padding, both signs and de-normalization.
            class Oracle(torch.nn.Module):
                def forward(self, value):
                    return value*4-2
            result, fitted = predict_volume(Oracle(), swi, brain, 40, norm)
            expected = (normalize_swi(swi, brain > 0)*4-2)*.1+.02
            expected[brain <= 0] = 0
            out = root/"prediction.nii.gz"
            save_qsm(out, result, ref, {"Units": "ppm", "Normalization": normalization_metadata(norm), "SWINormalizationApplied": fitted})
            restored = nib.load(out)
            np.testing.assert_allclose(restored.get_fdata(), expected, atol=1e-7)
            np.testing.assert_allclose(restored.affine, affine)
            self.assertEqual(restored.get_data_dtype(), np.dtype("float32"))
            self.assertEqual(restored.dataobj.slope, 1.)
            self.assertEqual(restored.dataobj.inter, 0.)
            self.assertLess(float(restored.header["cal_max"]), 1.)
            self.assertEqual(json.loads((root/"prediction.json").read_text())["Units"], "ppm")
            with self.assertRaises(FileExistsError):
                save_qsm(out, result, ref, {"Units": "ppm"})

    def test_mixed_units_dataset_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = np.arange(32*40, dtype=np.float32).reshape(32, 40, 1)
            qsm = raw/1000-0.4
            rows = []
            for i, unit in enumerate(("ppm", "ppb")):
                row = {"subject_id": str(i), "split": "val", "qsm_units": unit}
                for key, data in dict(swi_path=raw, qsm_path=qsm*(1000 if unit == "ppb" else 1),
                                      swi_mask_path=np.ones_like(raw), mask_path=np.ones_like(raw)).items():
                    path = root/f"{i}_{key}.nii.gz"
                    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
                    row[key] = str(path)
                rows.append(row)
            norm = QSMNormalization(.1, .02)
            ds = PairedSlices(rows, "val", 40, norm)
            np.testing.assert_allclose(ds[0][1], ds[1][1], atol=1e-6)
            class Constant(torch.nn.Module):
                def forward(self, x):
                    return torch.zeros_like(x)
            score, patients = validate(Constant(), torch.utils.data.DataLoader(ds, batch_size=2), "cpu", norm)
            self.assertAlmostEqual(score, float(np.abs(qsm-.02).mean()), places=6)
            self.assertEqual(len(patients), 2)


if __name__ == "__main__":
    unittest.main()

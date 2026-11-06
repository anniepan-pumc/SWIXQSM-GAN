import copy
import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from ..util.baselines import fit_baselines, load_baselines, save_baselines
from ..util.normalization import normalize_swi
from ..fit_baselines import main as fit_main


class BaselineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.affine = np.diag([.8, .9, 1.5, 1.])
        self.shape = (8, 9, 3)
        self.brain = np.ones(self.shape, dtype=bool)
        self.rng = np.random.default_rng(41)
        self.counter = 0

    def swi(self):
        value = self.rng.normal(100, 20, self.shape).astype(np.float32)
        value.flat[0], value.flat[-1] = -1000, 1000
        return value

    def row(self, sid, swi, qsm, *, split="train", qsm_units="ppb", mask=None,
            affine=None, spatial_units="mm"):
        self.counter += 1
        row = {"subject_id": sid, "split": split, "qsm_units": qsm_units,
               "qsm_reference": "CSF-v1", "space_id": "study-template-v1"}
        for key, array in {"swi_path": swi, "qsm_path": qsm,
                           "swi_mask_path": self.brain.astype(np.float32),
                           "mask_path": self.brain.astype(np.float32) if mask is None else mask}.items():
            path = self.root / f"{self.counter}_{key}.nii.gz"
            image = nib.Nifti1Image(np.asarray(array, dtype=np.float32),
                                    self.affine if affine is None else affine)
            image.header.set_xyzt_units(spatial_units)
            nib.save(image, path)
            row[key] = str(path)
        return row

    def fit(self, rows, **kwargs):
        return fit_baselines(rows, reference_id="CSF-v1", space_id="study-template-v1", **kwargs)

    def predict(self, bundle, method, swi, **kwargs):
        return bundle.predict(method, swi, self.brain, self.affine,
                              reference_id="CSF-v1", space_id="study-template-v1", **kwargs)

    def test_known_spatial_linear_exact(self):
        intercept = np.linspace(-55, 20, np.prod(self.shape)).reshape(self.shape)
        slope = np.linspace(10, 90, np.prod(self.shape)).reshape(self.shape)
        rows, targets = [], []
        for i in range(5):
            swi = self.swi()
            qsm = intercept + slope * normalize_swi(swi, self.brain)
            targets.append(qsm)
            rows.append(self.row(f"train_{i}", swi, qsm))
        bundle = self.fit(rows)
        test_swi = self.swi()
        expected = intercept + slope * normalize_swi(test_swi, self.brain)
        np.testing.assert_allclose(self.predict(bundle, "linear", test_swi), expected,
                                   atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(self.predict(bundle, "train_mean", test_swi),
                                   np.mean(targets, axis=0), atol=1e-5)
        self.assertEqual(bundle.units, "ppb")
        self.assertEqual(bundle.metadata["grid_spatial_units"], "mm")
        self.assertEqual(bundle.metadata["fit"]["n_subjects"], 5)
        self.assertEqual(len(bundle.metadata["fit_data_sha256"]), 64)

    def test_mixed_units_and_explicit_fallback(self):
        raw = self.swi()
        qsm = normalize_swi(raw, self.brain) * 80 - 30
        first = self.row("one", raw, qsm, qsm_units="ppb")
        second = self.row("two", raw, qsm / 1000, qsm_units="ppm")
        bundle = self.fit([first, second])
        np.testing.assert_allclose(bundle.train_mean, qsm, atol=2e-6)
        ppm = self.fit([first, second], units="ppm")
        np.testing.assert_allclose(ppm.train_mean * 1000, bundle.train_mean, atol=2e-6)
        missing = copy.deepcopy(first)
        del missing["qsm_units"]
        with self.assertRaisesRegex(ValueError, "units"):
            self.fit([missing])
        fallback = self.fit([missing], default_qsm_units="ppb")
        np.testing.assert_allclose(fallback.train_mean, qsm, atol=2e-6)
        missing["qsm_units"] = ""
        with self.assertRaisesRegex(ValueError, "units"):
            self.fit([missing], default_qsm_units="ppb")

    def test_val_test_targets_are_never_loaded_or_hashed(self):
        raw = self.swi()
        train = self.row("train", raw, raw / 2)
        val = self.row("val", raw, raw, split="val")
        test = self.row("test", raw, raw, split="test")
        before = self.fit([train, val, test])
        Path(test["qsm_path"]).write_bytes(b"not a NIfTI, must not be inspected by fitting")
        Path(val["swi_path"]).unlink()
        after = self.fit([train, val, test])
        np.testing.assert_array_equal(before.train_mean, after.train_mean)
        np.testing.assert_array_equal(before.slope, after.slope)
        self.assertEqual(before.metadata, after.metadata)

    def test_repeated_visits_subject_weights_and_constant_x(self):
        raw = self.swi()
        ten = np.full(self.shape, 10.)
        thirty = np.full(self.shape, 30.)
        rows = [self.row("repeat", raw, ten), self.row("repeat", raw, ten),
                self.row("single", raw, thirty)]
        bundle = self.fit(rows)
        np.testing.assert_allclose(bundle.train_mean, 20.)
        np.testing.assert_allclose(bundle.slope, 0.)
        np.testing.assert_allclose(bundle.intercept, 20.)
        self.assertEqual(bundle.metadata["fit"]["visits_per_subject"], {"repeat": 2, "single": 1})
        # A missing visit contributes no weight at that voxel; no hidden imputation.
        mask = self.brain.copy()
        mask[1, 1, 1] = False
        rows[1] = self.row("repeat", raw, ten, mask=mask)
        partial = self.fit(rows)
        self.assertAlmostEqual(partial.train_mean[1, 1, 1], (10*.5 + 30) / 1.5)

    def test_uncovered_evaluation_mask_is_rejected(self):
        raw = self.swi()
        mask = self.brain.copy()
        mask[1, 2, 0] = False
        bundle = self.fit([self.row("one", raw, raw, mask=mask)])
        self.assertFalse(bundle.support[1, 2, 0])
        for method in ("train_mean", "linear"):
            with self.assertRaisesRegex(ValueError, "outside training coverage"):
                self.predict(bundle, method, raw)
            result = self.predict(bundle, method, raw, mask=mask)
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_array_equal(mask, self.brain & bundle.support)

    def test_geometry_reference_and_space_rejections(self):
        raw = self.swi()
        row = self.row("one", raw, raw)
        bundle = self.fit([row])
        shifted = self.affine.copy()
        shifted[0, 3] = 2
        with self.assertRaisesRegex(ValueError, "affine"):
            bundle.predict("linear", raw, self.brain, shifted,
                           reference_id="CSF-v1", space_id="study-template-v1")
        for reference, space in (("other", "study-template-v1"), ("CSF-v1", "other"), ("", "study-template-v1")):
            with self.assertRaises(ValueError):
                bundle.predict("linear", raw, self.brain, self.affine,
                               reference_id=reference, space_id=space)
        for key in ("qsm_reference", "space_id"):
            for value in ("", "different"):
                invalid = dict(row, **{key: value})
                with self.assertRaises(ValueError):
                    self.fit([invalid])
        different = self.row("two", raw, raw, affine=shifted)
        with self.assertRaisesRegex(ValueError, "registered"):
            self.fit([row, different])
        image = nib.load(row["qsm_path"])
        wrong = nib.Nifti1Image(image.get_fdata(), shifted)
        wrong.header.set_xyzt_units("mm")
        nib.save(wrong, row["qsm_path"])
        with self.assertRaisesRegex(ValueError, "registered"):
            self.fit([row])

    def test_spatial_units_are_checked_within_and_between_visits(self):
        raw = self.swi()
        mm = self.row("one", raw, raw)
        meter = self.row("two", raw, raw, spatial_units="meter")
        with self.assertRaisesRegex(ValueError, "spatial units"):
            self.fit([mm, meter])
        image = nib.load(mm["qsm_path"])
        image.header.set_xyzt_units("meter")
        nib.save(image, mm["qsm_path"])
        with self.assertRaisesRegex(ValueError, "spatial units"):
            self.fit([mm])
        unknown = self.row("unknown", raw, raw, spatial_units="unknown")
        bundle = self.fit([unknown])
        self.assertEqual(bundle.metadata["grid_spatial_units"], "unknown")

    def test_selected_evaluation_independence(self):
        raw = self.swi()
        train = self.row("train_person", raw, raw)
        bundle = self.fit([train])
        separate = self.row("held_out", raw, raw, split="test")
        bundle.assert_independent([separate])
        with self.assertRaisesRegex(ValueError, "leakage"):
            bundle.assert_independent([dict(separate, subject_id="train_person")])
        with self.assertRaisesRegex(ValueError, "leakage"):
            bundle.assert_independent([dict(separate, swi_path=train["swi_path"])])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.fit([train, dict(train, subject_id="alias")])

    def test_save_reload_and_schema_validation(self):
        raw = self.swi()
        bundle = self.fit([self.row("one", raw, raw / 10)])
        path = self.root / "baseline.npz"
        save_baselines(bundle, path)
        restored = load_baselines(path)
        self.assertEqual(restored.metadata, bundle.metadata)
        np.testing.assert_array_equal(restored.affine, bundle.affine)
        np.testing.assert_array_equal(restored.support, bundle.support)
        np.testing.assert_array_equal(self.predict(restored, "linear", raw), self.predict(bundle, "linear", raw))
        with self.assertRaises(FileExistsError):
            save_baselines(bundle, path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        metadata = json.loads(arrays["metadata_json"].item())
        metadata["schema_version"] = 999
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        invalid = self.root / "invalid.npz"
        np.savez(invalid, **arrays)
        with self.assertRaisesRegex(ValueError, "schema"):
            load_baselines(invalid)
        arrays["metadata_json"] = np.asarray(json.dumps(bundle.metadata))
        arrays["slope"] = arrays["slope"].copy()
        arrays["slope"].flat[0] = np.nan
        np.savez(invalid, **arrays)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_baselines(invalid)

    def test_cli_saves_train_only_provenance(self):
        raw = self.swi()
        rows = [self.row("train", raw, raw / 10), self.row("test", raw, raw, split="test")]
        # read_manifest checks existence only, and fitting must never read this target.
        Path(rows[1]["qsm_path"]).write_bytes(b"not a valid target")
        manifest = self.root / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        output = self.root / "artifact.npz"
        argv = ["fit_baselines", "--manifest", str(manifest), "--output", str(output),
                "--reference-id", "CSF-v1", "--space-id", "study-template-v1",
                "--allow-unverified-data", "First-stage synthetic fixture"]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            fit_main()
        restored = load_baselines(output)
        self.assertEqual(restored.metadata["training_subject_ids"], ["train"])
        self.assertEqual(restored.metadata["fit"]["n_visits"], 1)
        self.assertEqual(len(restored.metadata["provenance"]["manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

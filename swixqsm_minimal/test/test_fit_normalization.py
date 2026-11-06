"""Synthetic-only checks for train-fitted, reversible QSM normalization."""
import copy
import csv
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

from .. import fit_normalization as fitting
from ..util.normalization import normalization_metadata


class FitNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def case(self, subject, visit, values, *, units="ppm", mask=None, split="train", global_id=None):
        values = np.asarray(values, dtype=np.float32).reshape(-1, 1, 1)
        row = dict(subject_id=subject, case_id=visit, split=split, qsm_units=units)
        if global_id is not None:
            row["global_subject_id"] = global_id
        arrays = dict(swi_path=np.arange(values.size, dtype=np.float32).reshape(values.shape) + 1,
                      qsm_path=values, swi_mask_path=np.ones_like(values),
                      mask_path=np.ones_like(values) if mask is None else np.asarray(mask, dtype=np.float32).reshape(values.shape))
        for role, data in arrays.items():
            path = self.root / f"{visit}_{role}.nii.gz"
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            row[role] = str(path)
        return row

    def test_subject_equal_mixed_units_multiple_visits_and_mask(self):
        rows = [
            self.case("a", "a1", [0, 2], global_id="g-a"),
            self.case("a", "a2", [4000, 6000, 1e8], units="ppb", mask=[1, 1, 0], global_id="g-a"),
            self.case("b", "b1", [10, 14], global_id="g-b"),
        ]
        artifact = fitting.fit_training_normalization(rows, units="ppm")
        qsm = artifact["normalization"]["qsm"]
        # Person A: mean=3, var=5; B: mean=12, var=4. Both people weight 1/2.
        self.assertAlmostEqual(qsm["offset"], 7.5)
        self.assertAlmostEqual(qsm["scale"], np.sqrt(24.75))
        self.assertNotAlmostEqual(qsm["offset"], 6.)  # Pooled visits/voxels would over-weight person A.
        selection = artifact["fit_selection"]
        self.assertEqual((selection["n_subjects"], selection["n_visits"], selection["n_valid_voxels"]), (2, 3, 6))
        self.assertEqual(selection["split"], "train")
        self.assertEqual(selection["train_identity_sha256"], fitting.training_identity(list(reversed(rows))))
        self.assertEqual(artifact["data_qc"]["status"], "unverified")
        self.assertIn("does not recover", artifact["limitations"][0])
        rms = fitting.fit_training_normalization(rows, units="ppm", method="zero_rms")
        self.assertEqual(rms["normalization"]["qsm"]["offset"], 0.)
        self.assertAlmostEqual(rms["normalization"]["qsm"]["scale"], 9.)

    def test_global_subject_alias_visits_are_one_person(self):
        rows = [self.case("alias-a", "a", [0, 2], global_id="person"),
                self.case("alias-b", "b", [4, 6], global_id="person"),
                self.case("other", "c", [10, 14], global_id="other")]
        artifact = fitting.fit_training_normalization(rows, units="ppm")
        self.assertEqual(artifact["fit_selection"]["n_subjects"], 2)
        self.assertEqual(artifact["normalization"]["qsm"]["offset"], 7.5)

    def test_cli_never_opens_missing_or_corrupt_holdout_inputs(self):
        train = self.case("a", "a", [-2, 0, 2, 4])
        holdouts = []
        forbidden = set()
        for split in ("val", "test"):
            row = dict(subject_id=split, case_id=split, split=split, qsm_units="not-a-unit")
            for role in fitting.IMAGE_KEYS:
                path = self.root / f"{split}_{role}.nii.gz"
                if split == "val":
                    path.write_bytes(b"not a NIfTI")
                row[role] = str(path)
                forbidden.add(str(path))
            holdouts.append(row)
        manifest = self.root / "manifest.csv"
        fields = list(train)
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows([train] + holdouts)
        output = self.root / "fitted.json"
        original_hash, original_load = fitting._file_hash, fitting.load_volume
        touched = []

        def checked_hash(path):
            self.assertNotIn(str(path), forbidden)
            touched.append(str(path))
            return original_hash(path)

        def checked_load(path):
            self.assertNotIn(str(path), forbidden)
            return original_load(path)

        argv = ["fit_normalization", "--manifest", str(manifest), "--output", str(output),
                "--units", "ppm", "--method", "mean_std", "--allow-unverified-data", "synthetic test"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(fitting, "_file_hash", side_effect=checked_hash), \
                mock.patch.object(fitting, "load_volume", side_effect=checked_load):
            fitting.main()
            norm, artifact = fitting.load_fitted_normalization(output, rows=[train] + holdouts)
        self.assertEqual(norm.offset, 1.)
        self.assertAlmostEqual(norm.scale, np.sqrt(5))
        self.assertEqual(len(artifact["inputs"]), 1)
        self.assertEqual(artifact["provenance"]["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertTrue(touched)

    def test_traceable_save_load_and_changed_train_bytes_rejected(self):
        row = self.case("a", "a", [-4, -1, 2, 5])
        artifact = fitting.fit_training_normalization([row], units="ppb")
        output = self.root / "normalization.json"
        fitting.save_fitted_normalization(artifact, output)
        norm, restored = fitting.load_fitted_normalization(output, units="ppb", rows=[row])
        self.assertEqual(restored, artifact)
        self.assertEqual(restored["normalization"], normalization_metadata(norm))
        for entry in restored["inputs"][0]["files"].values():
            self.assertEqual(entry["sha256"], hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest())
        values = np.array([-10000., 0., 10000.], dtype=np.float32)
        np.testing.assert_allclose(norm.denormalize(norm.normalize(values)), values, atol=1e-3)
        nib.save(nib.Nifti1Image(np.array([7, 8, 9, 10], dtype=np.float32).reshape(4, 1, 1), np.eye(4)), row["qsm_path"])
        with self.assertRaisesRegex(ValueError, "input changed"):
            fitting.load_fitted_normalization(output)
        # Explicitly skipping contents does not claim fresh source verification.
        fitting.load_fitted_normalization(output, verify_inputs=False)

    def test_metadata_units_membership_and_safe_no_overwrite(self):
        row = self.case("a", "a", [-2, 2])
        artifact = fitting.fit_training_normalization([row], units="ppm")
        output = self.root / "normalization.json"
        fitting.save_fitted_normalization(artifact, output)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            fitting.save_fitted_normalization(artifact, output)
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(list(self.root.glob(".normalization-*")), [])
        with self.assertRaisesRegex(ValueError, "units conflict"):
            fitting.load_fitted_normalization(output, units="ppb")
        changed = {**row, "subject_id": "different"}
        with self.assertRaisesRegex(ValueError, "different training"):
            fitting.load_fitted_normalization(output, rows=[changed])
        for mutate in (
            lambda value: value["fit_selection"].update(split="test"),
            lambda value: value["normalization"]["qsm"].update(clip=True),
            lambda value: value["normalization"]["qsm"].update(offset=55.),
            lambda value: value["inputs"][0]["files"]["qsm_path"].update(sha256="x" * 64),
        ):
            invalid = copy.deepcopy(artifact)
            mutate(invalid)
            with self.assertRaises(ValueError):
                fitting.save_fitted_normalization(invalid, self.root / "bad.json")
            self.assertFalse((self.root / "bad.json").exists())
        link = self.root / "dangling.json"
        link.symlink_to(self.root / "nonexistent")
        with self.assertRaises(FileExistsError):
            fitting.save_fitted_normalization(artifact, link)
        self.assertTrue(link.is_symlink())

    def test_degenerate_missing_units_duplicate_and_invalid_masks_rejected(self):
        constant = self.case("a", "constant", [3, 3])
        with self.assertRaisesRegex(ValueError, "Zero or nonfinite fitted scale"):
            fitting.fit_training_normalization([constant], units="ppm")
        # With offset explicitly fixed at zero, a nonzero constant has positive RMS.
        rms = fitting.fit_training_normalization([constant], units="ppm", method="zero_rms")
        self.assertEqual(rms["normalization"]["qsm"]["scale"], 3.)
        zero = self.case("a", "zero", [0, 0])
        with self.assertRaises(ValueError):
            fitting.fit_training_normalization([zero], units="ppm", method="zero_rms")
        missing_units = self.case("b", "units", [-1, 1])
        del missing_units["qsm_units"]
        with self.assertRaises(ValueError):
            fitting.fit_training_normalization([missing_units], units="ppm")
        fitted = fitting.fit_training_normalization([missing_units], units="ppm", default_qsm_units="ppm")
        self.assertEqual(fitted["normalization"]["qsm"]["scale"], 1.)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            fitting.fit_training_normalization([missing_units, missing_units], units="ppm", default_qsm_units="ppm")
        empty = self.case("a", "empty", [-1, 1], mask=[0, 0])
        negative = self.case("a", "negative", [-1, 1], mask=[1, -1])
        for row in (empty, negative):
            with self.assertRaises(ValueError):
                fitting.fit_training_normalization([row], units="ppm")
        with self.assertRaisesRegex(ValueError, "No training rows"):
            fitting.fit_training_normalization([{**zero, "split": "test"}], units="ppm")

    def test_input_change_during_fit_is_not_silently_bound(self):
        row = self.case("a", "a", [-1, 1])
        original = fitting.load_volume

        def load_then_modify(path):
            image, array = original(path)
            if Path(path).resolve() == Path(row["qsm_path"]).resolve():
                nib.save(nib.Nifti1Image(np.ones((2, 1, 1), dtype=np.float32), np.eye(4)), path)
            return image, array

        with mock.patch.object(fitting, "load_volume", side_effect=load_then_modify):
            with self.assertRaisesRegex(RuntimeError, "changed during normalization fit"):
                fitting.fit_training_normalization([row], units="ppm")

    def test_input_brain_support_and_prepared_grids_are_checked(self):
        row = self.case("a", "support", [-1, 1])

        def replace(role, values, affine=None):
            nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32).reshape(2, 1, 1),
                                    np.eye(4) if affine is None else affine), row[role])

        replace("swi_mask_path", [0, 0])
        with self.assertRaisesRegex(ValueError, "brain mask.*nonempty"):
            fitting.fit_training_normalization([row], units="ppm")
        replace("swi_mask_path", [1, 0])
        with self.assertRaisesRegex(ValueError, "contained"):
            fitting.fit_training_normalization([row], units="ppm")
        replace("mask_path", [1, 0])
        fitting.fit_training_normalization([row], units="ppm", method="zero_rms")
        shifted = np.eye(4)
        shifted[0, 3] = 3.
        replace("swi_mask_path", [1, 1], affine=shifted)
        with self.assertRaisesRegex(ValueError, "same grid"):
            fitting.fit_training_normalization([row], units="ppm", method="zero_rms")
        replace("swi_mask_path", [1, 1])
        replace("swi_path", [1, 2], affine=shifted)
        with self.assertRaisesRegex(ValueError, "same grid"):
            fitting.fit_training_normalization([row], units="ppm", method="zero_rms")

    def test_swi_header_check_does_not_decode_intensities(self):
        row = self.case("a", "header_only", [-1, 1])
        actual_load = fitting.nib.load
        touched = []

        def header_only(path, *args, **kwargs):
            image = actual_load(path, *args, **kwargs)
            if Path(path).resolve() == Path(row["swi_path"]).resolve():
                touched.append(str(path))
                image.get_fdata = mock.Mock(side_effect=AssertionError("SWI intensities must not be decoded"))
            return image

        with mock.patch.object(fitting.nib, "load", side_effect=header_only):
            fitting.fit_training_normalization([row], units="ppm")
        self.assertEqual(len(touched), 1)

    def test_earlier_visit_changed_during_later_visit_is_rejected(self):
        first = self.case("a", "first", [-1, 1])
        second = self.case("b", "second", [2, 4])
        rows = sorted([first, second], key=lambda row: fitting._canonical(fitting._training_rows([row])[0]))
        original = fitting.load_volume

        def load_then_modify_earlier(path):
            image, array = original(path)
            if Path(path).resolve() == Path(rows[1]["qsm_path"]).resolve():
                earlier = rows[0]["qsm_path"]
                nib.save(nib.Nifti1Image(np.ones((2, 1, 1), dtype=np.float32), np.eye(4)), earlier)
            return image, array

        with mock.patch.object(fitting, "load_volume", side_effect=load_then_modify_earlier):
            with self.assertRaisesRegex(RuntimeError, "changed during normalization fit"):
                fitting.fit_training_normalization(rows, units="ppm")

    def test_stable_population_variance_at_large_offset(self):
        row = self.case("a", "large", [1e6, 1e6 + 2])
        artifact = fitting.fit_training_normalization([row], units="ppm")
        self.assertEqual(artifact["normalization"]["qsm"]["offset"], 1e6 + 1)
        self.assertEqual(artifact["normalization"]["qsm"]["scale"], 1.)


if __name__ == "__main__":
    unittest.main()

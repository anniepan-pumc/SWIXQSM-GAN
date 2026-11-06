import copy
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from ..util.reference import (assert_reference_identity, inspect_reference,
                              reference_identity, validate_reference_spec)


class ReferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shape = (4, 5, 3)
        self.affine = np.diag([.8, .9, 1.5, 1.])
        self.brain = np.zeros(self.shape, dtype=bool)
        self.brain[1:3, 1:4, :] = True
        self.reliable = self.brain.copy()
        self.reference = np.zeros(self.shape, dtype=np.float32)
        self.reference[1, 1, :] = 1
        self.spec = {"id": "ventricular-CSF-v1", "method": "mean", "region": "eroded-ventricular-CSF",
                     "target_ppb": 0, "tolerance_ppb": 1, "require_mask": True}
        self.row = {"qsm_reference": self.spec["id"], "qsm_reference_method": "mean",
                    "qsm_reference_region": self.spec["region"],
                    "qsm_reference_source": "STI-iLSQR preparation v1; explicit ventricular CSF mean subtraction",
                    "qsm_reference_mask_path": str(self.root / "reference.nii.gz")}
        self.save_mask(self.reference)

    def image(self, values, *, affine=None, units="mm"):
        image = nib.Nifti1Image(np.asarray(values, dtype=np.float64),
                               self.affine if affine is None else affine)
        image.header.set_xyzt_units(units)
        return image

    def save_mask(self, values, *, affine=None, units="mm"):
        nib.save(self.image(values, affine=affine, units=units), self.row["qsm_reference_mask_path"])

    def inspect(self, values=None, *, row=None, spec=None, qsm_units="ppb", brain=None, reliable=None, image=None):
        values = np.zeros(self.shape) if values is None else values
        return inspect_reference(self.row if row is None else row,
                                 self.image(values) if image is None else image, values,
                                 spec=self.spec if spec is None else spec, qsm_units=qsm_units,
                                 brain_mask=self.brain if brain is None else brain,
                                 reliable_mask=self.reliable if reliable is None else reliable)

    def test_spec_normalization_identity_and_invalid_specs(self):
        spec = dict(self.spec, id="  ventricular-CSF-v1  ", target_ppb=0)
        normalized = validate_reference_spec(spec)
        self.assertEqual(normalized["id"], self.spec["id"])
        self.assertIsInstance(normalized["target_ppb"], float)
        identity = reference_identity(spec)
        self.assertEqual(set(identity), {"id", "method", "region", "target_ppb"})
        self.assertEqual(assert_reference_identity(spec, identity), identity)
        policy = dict(spec, tolerance_ppb=20, require_mask=False)
        assert_reference_identity(spec, policy)
        with self.assertRaisesRegex(ValueError, "disagree"):
            assert_reference_identity(spec, dict(identity, target_ppb=1))
        invalid = [dict(self.spec, extra=True), dict(self.spec, id=" "), dict(self.spec, method="mode"),
                   dict(self.spec, region=""), dict(self.spec, target_ppb=float("nan")),
                   dict(self.spec, target_ppb=True), dict(self.spec, tolerance_ppb=-1),
                   dict(self.spec, tolerance_ppb=float("inf")), dict(self.spec, require_mask=1)]
        missing = dict(self.spec)
        del missing["region"]
        invalid.append(missing)
        for bad in invalid:
            with self.subTest(spec=bad), self.assertRaises(ValueError):
                validate_reference_spec(bad)
        with self.assertRaises(ValueError):
            self.inspect(spec=invalid[0])

    def test_ppm_conversion_and_inclusive_tolerance(self):
        values = np.zeros(self.shape)
        values[self.reference > 0] = .0005
        result = self.inspect(values, qsm_units="ppm")
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "verified")
        self.assertAlmostEqual(result["numerical"]["observed_ppb"], .5)
        self.assertEqual(result["numerical"]["n_voxels"], 3)
        values[self.reference > 0] = 1
        self.assertTrue(self.inspect(values)["passed"])
        values[self.reference > 0] = 1.001
        result = self.inspect(values)
        self.assertFalse(result["passed"])
        self.assertFalse(result["numerical"]["within_tolerance"])

    def test_mean_and_median_are_distinct_without_referencing(self):
        values = np.zeros(self.shape)
        values[1, 1, :] = [0, 0, 30]
        original = values.copy()
        mean = self.inspect(values)
        self.assertFalse(mean["passed"])
        self.assertEqual(mean["numerical"]["observed_ppb"], 10)
        spec = dict(self.spec, method="median", tolerance_ppb=0)
        row = dict(self.row, qsm_reference_method="median")
        median = self.inspect(values, row=row, spec=spec)
        self.assertTrue(median["passed"])
        self.assertEqual(median["numerical"]["observed_ppb"], 0)
        np.testing.assert_array_equal(values, original)

    def test_declared_only_does_not_claim_numerical_verification(self):
        row = dict(self.row)
        del row["qsm_reference_mask_path"]
        result = self.inspect(row=row, spec=dict(self.spec, require_mask=False))
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "declared_only")
        self.assertFalse(result["numerical"]["performed"])
        self.assertIn("not numerically verified", result["numerical"]["reason"])
        required = self.inspect(row=row)
        self.assertFalse(required["passed"])
        self.assertEqual(required["status"], "failed")
        self.assertTrue(any("required" in item for item in required["errors"]))

    def test_required_row_declarations_source_and_conflicts(self):
        for key in ("qsm_reference", "qsm_reference_method", "qsm_reference_region", "qsm_reference_source"):
            for value in (None, ""):
                with self.subTest(key=key, value=value):
                    result = self.inspect(row=dict(self.row, **{key: value}))
                    self.assertFalse(result["passed"])
                    self.assertTrue(any(key in item for item in result["errors"]))
        for key in ("qsm_reference", "qsm_reference_method", "qsm_reference_region"):
            result = self.inspect(row=dict(self.row, **{key: "different"}))
            self.assertFalse(result["passed"])
        result = self.inspect()
        self.assertEqual(result["provenance"]["source"], self.row["qsm_reference_source"])
        self.assertTrue(any("does not prove" in text for text in result["limitations"]))

    def test_zero_background_or_padding_cannot_be_a_reference(self):
        mask = np.zeros(self.shape)
        mask[0, 0, 0] = 1
        self.save_mask(mask)
        result = self.inspect()
        self.assertFalse(result["passed"])
        self.assertFalse(result["numerical"]["performed"])
        self.assertTrue(any("background/padding" in item for item in result["errors"]))

    def test_reconstruction_support_is_separate_from_analysis_support(self):
        restrictive = self.reliable.copy()
        restrictive[1, 1, :] = False
        result = self.inspect(reliable=restrictive)
        self.assertFalse(result["passed"])
        self.assertTrue(any("reconstruction-reliable" in item for item in result["errors"]))
        # Broader reconstruction support can include CSF outside an analysis ROI.
        row = dict(self.row, qsm_reconstruction_mask_path="/declared/reconstruction_mask.nii.gz",
                   mask_path="/declared/analysis_mask.nii.gz")
        result = self.inspect(row=row, reliable=self.reliable)
        self.assertTrue(result["passed"])
        self.assertEqual(result["provenance"]["reliable_mask_declared_path"], row["qsm_reconstruction_mask_path"])

    def test_expected_bad_masks_are_reported_without_crashing(self):
        for values in (np.zeros(self.shape), -self.reference, self.reference * np.nan,
                       np.ones((3, 3, 3)), np.ones(self.shape + (1,))):
            with self.subTest(shape=values.shape):
                self.save_mask(values)
                result = self.inspect()
                self.assertFalse(result["passed"])
                self.assertEqual(result["status"], "failed")
                self.assertTrue(result["errors"])
        self.save_mask(self.reference)
        shifted = self.affine.copy()
        shifted[0, 3] = 1
        self.save_mask(self.reference, affine=shifted)
        self.assertFalse(self.inspect()["passed"])
        self.save_mask(self.reference, units="meter")
        self.assertTrue(any("spatial units" in error for error in self.inspect()["errors"]))
        Path(self.row["qsm_reference_mask_path"]).write_bytes(b"corrupted NIfTI")
        self.assertFalse(self.inspect()["passed"])
        Path(self.row["qsm_reference_mask_path"]).unlink()
        self.assertFalse(self.inspect()["passed"])
        self.assertFalse(self.inspect(row=dict(self.row, qsm_reference_mask_path="relative.nii.gz"))["passed"])

    def test_input_errors_and_all_inputs_remain_unchanged(self):
        values = np.arange(np.prod(self.shape), dtype=np.float64).reshape(self.shape)
        values[self.reference > 0] = 0
        row, spec = copy.deepcopy(self.row), copy.deepcopy(self.spec)
        original, brain, reliable = values.copy(), self.brain.copy(), self.reliable.copy()
        self.assertTrue(self.inspect(values, row=row, spec=spec)["passed"])
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(self.brain, brain)
        np.testing.assert_array_equal(self.reliable, reliable)
        self.assertEqual(row, self.row)
        self.assertEqual(spec, self.spec)
        for units in ("", "unknown", None, []):
            result = self.inspect(values, qsm_units=units)
            self.assertFalse(result["passed"])
            self.assertTrue(any("units" in error.lower() for error in result["errors"]))
        bad = values.copy()
        bad[0, 0, 0] = np.nan
        self.assertFalse(self.inspect(bad)["passed"])
        self.assertFalse(self.inspect(values, brain=np.zeros(self.shape))["passed"])
        self.assertFalse(self.inspect(values, reliable=np.ones((2, 2, 2)))["passed"])
        self.assertFalse(self.inspect(values, image=object())["passed"])
        self.assertFalse(self.inspect(values, row="not a row")["passed"])


if __name__ == "__main__":
    unittest.main()

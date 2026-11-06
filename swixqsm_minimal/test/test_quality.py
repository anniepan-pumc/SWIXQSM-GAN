import copy
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from ..util.quality import (build_report, inspect_case, inspect_geometry, load_contract,
                            resolve_qc_paths, validate_contract, verify_report)


class DataQualityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.affine = np.diag([.8, .9, 2., 1.])
        self.shape = (9, 10, 7)
        self.brain = np.zeros(self.shape, dtype=np.float32)
        self.brain[1:-1, 1:-1, 1:-1] = 1
        self.contract = {
            "schema": "swixqsm.data_contract", "version": 1,
            "geometry": {"axis_codes": ["R", "A", "S"], "slice_axis": 2,
                         "spacing_mm_min": [.7, .8, 1.9], "spacing_mm_max": [.9, 1., 2.1],
                         "max_obliquity_deg": 5.},
            "reference": {"id": "brain-mean-v1", "method": "mean", "region": "brain",
                          "target_ppb": 0., "tolerance_ppb": .01, "require_mask": True},
            "registration": {"require_review": True},
        }
        self.row = self.make_case("train-subject", "train")

    def tearDown(self):
        self.temporary.cleanup()

    def write_image(self, path, values, affine=None, units="mm"):
        image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), self.affine if affine is None else affine)
        image.header.set_xyzt_units(units)
        nib.save(image, path)
        return str(path)

    def make_case(self, subject, split):
        directory = self.root / subject
        directory.mkdir()
        row = {"subject_id": subject, "split": split, "qsm_units": "ppb",
               "qsm_reference": "brain-mean-v1", "qsm_reference_method": "mean",
               "qsm_reference_region": "brain", "qsm_reference_source": "synthetic fixture, not real evidence",
               "registration_review": "accepted", "registration_reviewer": "Synthetic reviewer",
               "registration_review_date": "2026-10-17"}
        values = {"swi_path": np.arange(np.prod(self.shape)).reshape(self.shape),
                  "qsm_path": np.zeros(self.shape), "swi_mask_path": self.brain,
                  "mask_path": self.brain, "qsm_reference_mask_path": self.brain}
        for key, value in values.items():
            row[key] = self.write_image(directory/f"{key}.nii.gz", value)
        return row

    def save_report(self, report):
        path = self.root / "report.json"
        path.write_text(json.dumps(report, allow_nan=False))
        return path

    def test_contract_canonical_roundtrip_and_invalid_policies(self):
        normalized = validate_contract(self.contract)
        path = self.root / "contract.json"
        path.write_text(json.dumps(self.contract))
        self.assertEqual(load_contract(path), normalized)
        for field, value in (("slice_axis", 0), ("max_obliquity_deg", 90),
                             ("spacing_mm_min", [0., 1., 1.]), ("axis_codes", ["R", "A", "A"])):
            bad = copy.deepcopy(self.contract)
            bad["geometry"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_contract(bad)
        bad = copy.deepcopy(self.contract)
        bad["registration"]["require_review"] = False
        with self.assertRaises(ValueError):
            validate_contract(bad)

    def test_happy_case_report_and_compact_verified_record(self):
        before = Path(self.row["qsm_path"]).read_bytes()
        result = inspect_case(self.row, self.contract)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["reference"]["status"], "verified")
        self.assertEqual(len(result["file_sha256"]), 5)
        report = build_report([self.row], self.contract)
        report["provenance"] = {"manifest_sha256": "additional provenance allowed"}
        report["cases"][0]["preview"] = "extra UI field"
        path = self.save_report(report)
        verified = verify_report([self.row], path, reference_id="brain-mean-v1")
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(verified["n_cases"], 1)
        self.assertEqual(verified["report_path"], str(path.resolve()))
        self.assertNotIn("cases", verified)
        self.assertEqual(Path(self.row["qsm_path"]).read_bytes(), before)

    def test_unknown_units_require_explicit_assumption_and_meter_conversion(self):
        image = nib.Nifti1Image(np.ones(self.shape), self.affine)
        result = inspect_geometry(image, self.contract["geometry"])
        self.assertFalse(result["passed"])
        spec = {**self.contract["geometry"], "spatial_units_if_unknown": "mm"}
        result = inspect_geometry(image, spec)
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["spatial_units_assumed"])
        affine_m = self.affine.copy()
        affine_m[:3] /= 1000.
        image = nib.Nifti1Image(np.ones(self.shape), affine_m)
        image.header.set_xyzt_units("meter")
        result = inspect_geometry(image, self.contract["geometry"])
        self.assertTrue(result["passed"], result["errors"])
        np.testing.assert_allclose(result["spacing_mm"], [.8, .9, 2.], rtol=1e-6)

    def test_obliquity_is_acquisition_plane_angle_and_axis_codes_are_exact(self):
        angle = np.deg2rad(12.)
        rotation = np.array([[1., 0, 0], [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
        affine = self.affine.copy()
        affine[:3, :3] = rotation @ self.affine[:3, :3]
        image = nib.Nifti1Image(np.ones(self.shape), affine)
        image.header.set_xyzt_units("mm")
        result = inspect_geometry(image, self.contract["geometry"])
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["slice_normal_obliquity_deg"], 12.)
        self.assertNotIn("b0", result["obliquity_definition"].lower())
        spec = {**self.contract["geometry"], "max_obliquity_deg": 15.}
        self.assertTrue(inspect_geometry(image, spec)["passed"])
        flipped = self.affine.copy()
        flipped[0, 0] *= -1
        image = nib.Nifti1Image(np.ones(self.shape), flipped)
        image.header.set_xyzt_units("mm")
        self.assertFalse(inspect_geometry(image, spec)["passed"])

    def test_conflicting_forms_header_zooms_and_shear_rejected(self):
        invalid_units = nib.Nifti1Image(np.ones(self.shape), self.affine)
        invalid_units.header["xyzt_units"] = 7
        invalid = inspect_geometry(invalid_units, self.contract["geometry"])
        self.assertFalse(invalid["passed"])
        self.assertTrue(any("spatial-unit code" in error for error in invalid["errors"]))
        image = nib.Nifti1Image(np.ones(self.shape), self.affine)
        image.header.set_xyzt_units("mm")
        qform = self.affine.copy()
        qform[0, 3] = 1.
        image.set_qform(qform, code=1)
        image.set_sform(self.affine, code=1)
        result = inspect_geometry(image, self.contract["geometry"])
        self.assertTrue(any("qform" in error for error in result["errors"]))
        image = nib.Nifti1Image(np.ones(self.shape), self.affine)
        image.header.set_xyzt_units("mm")
        image.header.set_zooms((1., 1., 1.))
        result = inspect_geometry(image, self.contract["geometry"])
        self.assertTrue(any("Header voxel" in error for error in result["errors"]))
        shear = self.affine.copy()
        shear[0, 1] = .1
        image = nib.Nifti1Image(np.ones(self.shape), shear)
        image.header.set_xyzt_units("mm")
        result = inspect_geometry(image, self.contract["geometry"])
        self.assertTrue(any("orthogonal" in error for error in result["errors"]))

    def test_nonfinite_and_misaligned_case_images_fail(self):
        values = np.ones(self.shape)
        values[2, 2, 2] = np.nan
        self.write_image(self.row["swi_path"], values)
        case = inspect_case(self.row, self.contract)
        self.assertFalse(case["passed"])
        self.assertTrue(any("nonfinite" in error for error in case["errors"]))
        self.write_image(self.row["swi_path"], np.ones(self.shape))
        shifted = self.affine.copy()
        shifted[0, 3] = .2
        self.write_image(self.row["qsm_path"], np.zeros(self.shape), shifted)
        case = inspect_case(self.row, self.contract)
        self.assertTrue(any("same physical grid" in error for error in case["errors"]))

    def test_human_review_cannot_be_replaced_by_geometry(self):
        for changes in ({"registration_review": "pending"}, {"registration_review": "rejected"},
                        {"registration_reviewer": ""}, {"registration_review_date": "2026-02-30"},
                        {"registration_review_date": "20261017"}):
            with self.subTest(changes=changes):
                case = inspect_case({**self.row, **changes}, self.contract)
                self.assertFalse(case["passed"])
                self.assertTrue(case["geometry"]["swi_path"]["passed"])

    def test_configured_auxiliary_thresholds_require_evidence(self):
        contract = copy.deepcopy(self.contract)
        contract["registration"].update(min_mask_dice=.9, max_landmark_error_mm=1.)
        case = inspect_case(self.row, contract)
        self.assertFalse(case["passed"])
        self.assertTrue(any("requires independent" in error for error in case["errors"]))
        self.assertTrue(any("requires matched" in error for error in case["errors"]))

    def test_independent_mask_and_landmark_auxiliary_evidence_in_mm(self):
        row = self.row.copy()
        row["qsm_brain_mask_path"] = self.write_image(self.root/"independent.nii.gz", self.brain)
        row["qsm_brain_mask_source"] = "Independently segmented GRE magnitude (synthetic fixture)"
        landmarks = self.root/"landmarks.json"
        landmarks.write_text(json.dumps({"coordinate_system": "ras_mm", "fixed": [[0, 0, 0], [1, 2, 3]],
                                        "moving": [[.3, .4, 0], [1.3, 2.4, 3]]}))
        row["registration_landmarks_path"] = str(landmarks)
        contract = copy.deepcopy(self.contract)
        contract["registration"].update(min_mask_dice=.99, max_landmark_error_mm=.6)
        case = inspect_case(row, contract)
        self.assertTrue(case["passed"], case["errors"])
        registration = case["registration"]
        self.assertEqual(registration["mask_alignment"]["dice"], 1.)
        self.assertEqual(registration["mask_alignment"]["symmetric_mean_surface_distance_mm"], 0.)
        self.assertAlmostEqual(registration["landmarks"]["rms_mm"], .5)
        self.assertAlmostEqual(registration["landmarks"]["max_mm"], .5)
        self.assertEqual(registration["landmarks"]["n"], 2)
        contract["registration"]["max_landmark_error_mm"] = .4
        self.assertFalse(inspect_case(row, contract)["passed"])
        row["qsm_brain_mask_path"] = row["mask_path"]
        self.assertFalse(inspect_case(row, self.contract)["passed"])

    def test_reconstruction_reliable_mask_can_cover_reference_outside_analysis(self):
        valid = self.brain.copy()
        valid[2:4, 2:4, 2:4] = 0
        reference = np.zeros(self.shape)
        reference[2:4, 2:4, 2:4] = 1
        self.write_image(self.row["mask_path"], valid)
        self.write_image(self.row["qsm_reference_mask_path"], reference)
        self.assertFalse(inspect_case(self.row, self.contract)["passed"])
        row = {**self.row, "qsm_reconstruction_mask_path": self.write_image(self.root/"reliable.nii.gz", self.brain)}
        case = inspect_case(row, self.contract)
        self.assertTrue(case["passed"], case["errors"])

    def test_mask_surface_distance_uses_voxel_spacing_and_dice_threshold_fails(self):
        row = {**self.row, "qsm_brain_mask_source": "Independent synthetic mask"}
        shifted = np.roll(self.brain, 1, axis=0)
        row["qsm_brain_mask_path"] = self.write_image(self.root/"shifted_mask.nii.gz", shifted)
        case = inspect_case(row, self.contract)
        self.assertTrue(case["passed"], case["errors"])
        alignment = case["registration"]["mask_alignment"]
        self.assertAlmostEqual(alignment["symmetric_max_surface_distance_mm"], .8, places=6)
        self.assertAlmostEqual(alignment["dice"], 6/7)
        contract = copy.deepcopy(self.contract)
        contract["registration"]["min_mask_dice"] = .99
        self.assertFalse(inspect_case(row, contract)["passed"])

    def test_optional_registration_image_must_share_grid_and_singular_affine_fails(self):
        shifted = self.affine.copy()
        shifted[2, 3] = 1.
        row = {**self.row, "registration_image_path": self.write_image(self.root/"shifted_magnitude.nii.gz",
                                                                       np.ones(self.shape), shifted)}
        self.assertFalse(inspect_case(row, self.contract)["passed"])
        image = nib.Nifti1Image(np.ones(self.shape), self.affine)
        image.header.set_xyzt_units("mm")
        image._affine[:3, 0] = 0
        result = inspect_geometry(image, self.contract["geometry"])
        self.assertFalse(result["passed"])
        self.assertTrue(any("singular" in error for error in result["errors"]))

    def test_evidence_and_metadata_hashes_bind_verification(self):
        row = self.row.copy()
        transform = self.root/"transform.txt"
        transform.write_text("synthetic transform, never executed")
        row["registration_transform_path"] = str(transform)
        row["registration_image_path"] = self.write_image(self.root/"registered_magnitude.nii.gz", np.ones(self.shape))
        report = build_report([row], self.contract)
        self.assertEqual(len(report["cases"][0]["file_sha256"]), 7)
        path = self.save_report(report)
        verify_report([row], path)
        with self.assertRaisesRegex(ValueError, "metadata_sha256"):
            verify_report([{**row, "registration_reviewer": "Different reviewer"}], path)
        transform.write_text("changed transform evidence")
        with self.assertRaisesRegex(ValueError, "file_sha256"):
            verify_report([row], path)

    def test_verify_only_accesses_selected_splits(self):
        other = self.make_case("test-subject", "test")
        report = build_report([self.row, other], self.contract)
        path = self.save_report(report)
        for key, value in other.items():
            if key.endswith("_path"):
                Path(value).unlink()
        self.assertEqual(verify_report([self.row], path)["n_cases"], 1)
        with self.assertRaises(ValueError):
            verify_report([other], path)

    def test_contract_or_reference_tampering_and_failed_records_rejected(self):
        report = build_report([self.row], self.contract)
        report["contract"]["geometry"]["max_obliquity_deg"] = 10.
        path = self.save_report(report)
        with self.assertRaisesRegex(ValueError, "contract hash"):
            verify_report([self.row], path)
        report = build_report([self.row], self.contract)
        path = self.save_report(report)
        with self.assertRaisesRegex(ValueError, "reference ID"):
            verify_report([self.row], path, reference_id="different")
        report["cases"][0]["passed"] = False
        path = self.save_report(report)
        with self.assertRaisesRegex(ValueError, "passing QC"):
            verify_report([self.row], path)

    def test_path_resolution_and_unit_fallback_replay(self):
        relative = self.row.copy()
        relative.pop("qsm_units")
        for key in relative:
            if key.endswith("_path"):
                relative[key] = str(Path(relative[key]).relative_to(self.root))
        rows = resolve_qc_paths([relative], self.root/"manifest.csv")
        self.assertTrue(Path(rows[0]["qsm_reference_mask_path"]).is_absolute())
        report = build_report(rows, self.contract, default_qsm_units="ppb")
        self.assertEqual(report["status"], "passed")
        verified = verify_report(rows, self.save_report(report))
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(verified["default_qsm_units"], "ppb")
        self.assertFalse(inspect_case(rows[0], self.contract)["passed"])

    def test_prediction_columns_are_not_data_qc_inputs(self):
        report = build_report([self.row], self.contract)
        path = self.save_report(report)
        augmented = {**self.row, "prediction_path": "/does/not/exist.nii.gz",
                     "prediction_units": "ppm", "prediction_reference": "model metadata checked elsewhere"}
        self.assertEqual(verify_report([augmented], path)["status"], "verified")

    def test_file_change_detected_even_when_geometry_and_reference_still_pass(self):
        path = self.save_report(build_report([self.row], self.contract))
        image = nib.load(self.row["swi_path"])
        self.write_image(self.row["swi_path"], image.get_fdata()+1)
        self.assertTrue(inspect_case(self.row, self.contract)["passed"])
        with self.assertRaisesRegex(ValueError, "file_sha256"):
            verify_report([self.row], path)


if __name__ == "__main__":
    unittest.main()

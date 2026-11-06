import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from ..build_manifest import build_manifest
from ..prepare_swi_masks import prepare_swi_masks


class PrepareSWIMasksTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.manifest = self.root/"original_train_test.draft.csv"
        self.affine = np.diag((1., 2., 3., 1.))
        self.affine[:3, 3] = (10., 20., 30.)
        self.rows = [self.case("100", "train"), self.case("200", "train"), self.case("300", "test")]
        self.write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, path, values, *, affine=None, slope=None, intercept=None):
        image = nib.Nifti1Image(values, self.affine if affine is None else affine)
        image.header.set_xyzt_units("mm")
        image.set_qform(image.affine, 1)
        image.set_sform(image.affine, 2)
        if slope is not None:
            image.header.set_slope_inter(slope, intercept)
        nib.save(image, path)

    def case(self, case_id, split):
        directory = self.root/"primary_cohort"/("train" if split == "train" else "validate")/case_id
        directory.mkdir(parents=True)
        # Stored background 2 becomes exactly zero after slope=2/intercept=-4.
        # All retained brain values are negative; positive-only masking is wrong.
        raw = np.full((5, 5, 5), 2, dtype=np.int16)
        raw[1:4, 1:4, 1:4] = 1
        raw[2, 2, 2] = 2  # One enclosed hole; filling must not dilate its boundary.
        self.save(directory/"SWI.nii.gz", raw, slope=2., intercept=-4.)
        valid = np.zeros(raw.shape, dtype=np.float32)
        valid[1:4, 1:4, 1:4] = 7.
        valid[0, 0, 0] = 1.  # Known outside-SWI voxel; exclusion must be reported.
        self.save(directory/"QSM_mask.nii.gz", valid)
        (directory/"QSM.nii.gz").write_bytes(b"This is deliberately not a NIfTI; QSM labels must not be read")
        return dict(case_id=case_id, participant_id=case_id, original_split=split, split=split,
                    subject_id=case_id, global_subject_id="cohort:"+case_id, cohort="primary",
                    swi_path=str(directory/"SWI.nii.gz"), qsm_path=str(directory/"QSM.nii.gz"),
                    swi_mask_path="", mask_path=str(directory/"QSM_mask.nii.gz"), qsm_units="ppb",
                    input_mask_source="", source_case_path=str(directory),
                    registration_review="pending", registration_reviewer="real reviewer declaration",
                    registration_review_date="2026-10-17", registration_notes="Real record, not generated acceptance",
                    qsm_reference="recorded-zero", qsm_reference_method="recorded-method",
                    qsm_reference_region="recorded-region", qsm_reference_source="actual reconstruction note",
                    qsm_reference_target_ppb="0", registration_landmarks_path="evidence/landmarks.json",
                    evidence_paths=json.dumps(["evidence/slice.png"]))

    def write_manifest(self, rows=None):
        rows = self.rows if rows is None else rows
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def call(self, name="masks", **kwargs):
        return prepare_swi_masks(self.manifest, self.root/name, skull_stripped=True, **kwargs)

    def metadata(self, report):
        with Path(report["metadata_csv"]).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_scaled_negative_swi_only_masks_holes_intersection_and_source_integrity(self):
        source_paths = {row[key] for row in self.rows for key in ("swi_path", "qsm_path", "mask_path")}
        before = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_paths}
        original_load = nib.load
        loaded = []

        def checked_load(path, *args, **kwargs):
            self.assertNotIn(str(path), {row["qsm_path"] for row in self.rows})
            loaded.append(str(path))
            return original_load(path, *args, **kwargs)

        with patch("nibabel.load", checked_load):
            report = self.call()
        self.assertEqual(len(loaded), 6)
        self.assertFalse(report["qsm_label_intensities_read"])
        self.assertEqual(report["case_counts"], {"train": 2, "test": 1})
        for case in report["cases"]:
            self.assertEqual((case["raw_nonzero_voxels"], case["input_mask_voxels"], case["holes_filled_voxels"]), (26, 27, 1))
            self.assertEqual((case["original_valid_voxels"], case["final_valid_voxels"], case["excluded_valid_voxels"]), (28, 27, 1))
            image = nib.load(case["input_mask_path"])
            values = image.get_fdata()
            self.assertEqual(image.get_data_dtype(), np.dtype(np.uint8))
            self.assertEqual((image.dataobj.slope, image.dataobj.inter), (1., 0.))
            np.testing.assert_array_equal(image.affine, self.affine)
            self.assertEqual((int(image.header["qform_code"]), int(image.header["sform_code"])), (1, 2))
            self.assertEqual(image.header.get_xyzt_units()[0], "mm")
            self.assertEqual(values[2, 2, 2], 1)
            self.assertEqual(values[0, 0, 0], 0)
            self.assertEqual(int(values.sum()), 27)
            np.testing.assert_array_equal(nib.load(case["final_valid_mask_path"]).get_fdata(), values)
            self.assertEqual(case["swi_sha256"], before[case["swi_path"]])
            sidecar = json.loads(Path(case["provenance_path"]).read_text())
            self.assertTrue(sidecar["skull_stripped_declaration"])
            self.assertFalse(sidecar["anatomical_segmentation"])
            self.assertFalse(sidecar["qsm_reliability_validated"])
            self.assertTrue(Path(case["valid_mask_provenance_path"]).is_file())
        self.assertEqual({path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_paths}, before)
        self.assertEqual(set(report["source_sha256"]), {row[key] for row in self.rows for key in ("swi_path", "mask_path")})

    def test_no_fill_holes_and_preserve_never_reads_or_hashes_valid_mask(self):
        for row in self.rows:
            Path(row["mask_path"]).unlink()
        original_open = Path.open
        blocked = {row[key] for row in self.rows for key in ("qsm_path", "mask_path")}

        def checked_open(path, *args, **kwargs):
            if str(path) in blocked:
                raise AssertionError("preserve must not open QSM or original validity mask")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", checked_open):
            report = self.call(fill_holes=False, valid_mask_policy="preserve")
        rows = self.metadata(report)
        for case, row, original in zip(report["cases"], rows, self.rows):
            self.assertEqual(case["input_mask_voxels"], 26)
            self.assertEqual(case["holes_filled_voxels"], 0)
            self.assertEqual(row["mask_path"], original["mask_path"])
            self.assertIsNone(case["original_valid_voxels"])
            self.assertIsNone(case["original_valid_mask_sha256"])
        self.assertEqual(set(report["source_sha256"]), {row["swi_path"] for row in self.rows})

    def test_metadata_provenance_fields_preserved_and_final_paths_usable_by_builder(self):
        report = self.call()
        rows = self.metadata(report)
        self.assertEqual(len(rows), len(self.rows))
        for row, original in zip(rows, self.rows):
            self.assertNotIn("split", row)
            self.assertNotIn("participant_id", row)
            self.assertNotIn("source_case_path", row)
            for key in ("case_id", "original_split", "subject_id", "global_subject_id", "cohort", "qsm_units",
                        "registration_review", "registration_reviewer", "registration_review_date", "registration_notes",
                        "qsm_reference", "qsm_reference_method", "qsm_reference_region", "qsm_reference_source", "qsm_reference_target_ppb"):
                self.assertEqual(row[key], original[key])
            self.assertEqual(row["registration_landmarks_path"], str(self.root/"evidence/landmarks.json"))
            self.assertEqual(json.loads(row["evidence_paths"]), [str(self.root/"evidence/slice.png")])
            self.assertTrue(Path(row["swi_mask_path"]).is_file())
            self.assertTrue(Path(row["mask_path"]).is_file())
            self.assertNotIn(".swixqsm-swi-masks-", json.dumps(row))
        builder = build_manifest(self.root/"primary_cohort", self.root/"prepared_manifest", metadata_csv=report["metadata_csv"],
                                 expected_train=2, expected_test=1)
        self.assertTrue(builder["ready"])
        self.assertEqual(builder["test_global_subject_ids"], ["cohort:300"])

    def test_missing_required_file_rejected_before_any_nifti_read_or_staging(self):
        Path(self.rows[-1]["mask_path"]).unlink()
        with patch("nibabel.load", side_effect=AssertionError("All-table existence preflight must run first")), \
                patch("tempfile.TemporaryDirectory", side_effect=AssertionError("No writes before preflight")):
            with self.assertRaises(FileNotFoundError):
                self.call()
        self.assertFalse((self.root/"masks").exists())

    def test_late_nonfinite_swi_failure_publishes_no_partial_output(self):
        values = np.zeros((5, 5, 5), dtype=np.float32)
        values[1, 1, 1] = np.nan
        self.save(self.rows[-1]["swi_path"], values)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.call()
        self.assertFalse((self.root/"masks").exists())
        self.assertEqual(list(self.root.glob(".swixqsm-swi-masks-*")), [])

    def test_empty_swi_empty_intersection_negative_valid_and_wrong_grid_rejected(self):
        scenarios = ("empty_swi", "empty_valid", "empty_intersection", "negative_valid", "wrong_grid", "nonfinite_valid")
        for name in scenarios:
            with self.subTest(name=name):
                # Use only the first complete case for this independent failure.
                self.write_manifest([self.rows[0]])
                raw = np.zeros((5, 5, 5), dtype=np.float32)
                raw[1:4, 1:4, 1:4] = -1
                valid = (raw != 0).astype(np.float32)
                affine = self.affine.copy()
                if name == "empty_swi":
                    raw.fill(0)
                elif name == "empty_valid":
                    valid.fill(0)
                elif name == "empty_intersection":
                    valid.fill(0)
                    valid[0, 0, 0] = 1
                elif name == "negative_valid":
                    valid[2, 2, 2] = -1
                elif name == "wrong_grid":
                    affine[0, 3] += 1
                else:
                    valid[2, 2, 2] = np.inf
                self.save(self.rows[0]["swi_path"], raw)
                self.save(self.rows[0]["mask_path"], valid, affine=affine)
                with self.assertRaises(ValueError):
                    self.call(name)
                self.assertFalse((self.root/name).exists())

    def test_explicit_skull_stripped_required_invalid_policy_and_no_overwrite(self):
        for declaration in (False, "yes", 1, None):
            with self.subTest(declaration=declaration), self.assertRaisesRegex(ValueError, "skull_stripped"):
                prepare_swi_masks(self.manifest, self.root/"masks", skull_stripped=declaration)
        with self.assertRaises(ValueError):
            self.call(valid_mask_policy="infer")
        report = self.call()
        with self.assertRaises(FileExistsError):
            self.call()
        self.assertEqual(report["n_cases"], 3)

    def test_duplicate_cases_cross_split_identities_paths_and_source_conflict_rejected(self):
        for field, value in (("case_id", "100"), ("global_subject_id", "cohort:100"),
                             ("subject_id", "100"), ("swi_path", self.rows[0]["swi_path"])):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[-1][field] = value
                self.write_manifest(rows)
                with self.assertRaises(ValueError):
                    self.call()
                self.assertFalse((self.root/"masks").exists())
        rows = copy.deepcopy(self.rows)
        rows[0]["qsm_path"] = str(self.root/"masks/future-source.nii.gz")
        self.write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "source path"):
            self.call()


if __name__ == "__main__":
    unittest.main()

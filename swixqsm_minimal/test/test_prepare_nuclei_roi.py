"""Real tiny synthetic NIfTI binding; no cohort images, models or training."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from .. import prepare_nuclei_roi
from ..evaluate_native_nuclei import sha


class PrepareNucleiROITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="swix-prepare-nuclei-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.case_id = "synthetic__case1"
        self.gre_shape, self.swi_shape = (8, 6, 4), (6, 5, 3)
        self.gre_affine = np.diag([.8, .9, 2., 1.])
        self.swi_affine = np.diag([1.2, 1.1, 3., 1.])
        self.qsm = self.save("qsm.nii.gz", np.arange(np.prod(self.gre_shape)).reshape(self.gre_shape) / 100 - 1,
                             self.gre_affine)
        self.swi = self.save("swi.nii.gz", np.arange(np.prod(self.swi_shape)).reshape(self.swi_shape),
                             self.swi_affine)
        self.original_labels = {}
        targets = {}
        for target, shape, affine in (("gre", self.gre_shape, self.gre_affine),
                                      ("swi", self.swi_shape, self.swi_affine)):
            labels = {}
            for side, y in (("left", 0), ("right", 3)):
                array = np.zeros(shape, dtype=np.float32)
                for x, label in enumerate((385, 310, 352, 721)):
                    array[x, y, 0] = label
                array[4, y, 0] = 999  # unrelated atlas labels must be omitted.
                path = self.save(f"{target}_{side}.nii.gz", array, affine)
                self.original_labels[(target, side)] = array
                labels[side] = self.record(path)
            targets[target] = {"labels": labels}
        self.report = dict(case_id=self.case_id, subject_id="subject1", cohort="synthetic_external",
            registration_status="candidate_completed_pending_QC", protocol_revision="scanner_identity_initialization_v2",
            inputs={"qsm_grid": self.record(self.qsm), "swi": self.record(self.swi)}, targets=targets)
        self.report_path = self.root / "registration.json"
        self.write_report()
        self.cases = [dict(case_id=self.case_id, cohort="synthetic_external", subject_id="subject1",
                           registration_report=str(self.report_path))]
        self.cases_path = self.root / "cases.json"
        self.cases_path.write_text(json.dumps(self.cases))
        self.rows = [dict(case_id=self.case_id, subject_id="subject1", global_subject_id="person1",
                         cohort="synthetic_external", split="test", qsm_path=str(self.qsm), swi_path=str(self.swi)),
                     dict(case_id="synthetic__missing", subject_id="subject2", global_subject_id="person2",
                         cohort="synthetic_external", split="test", qsm_path="missing_qsm.nii.gz", swi_path="missing_swi.nii.gz"),
                     dict(case_id="unselected_train", subject_id="train_only", global_subject_id="train_person",
                         cohort="synthetic_external", split="train", qsm_path="must_not_read.nii.gz", swi_path="must_not_read_swi.nii.gz")]
        self.manifest = self.root / "core.csv"
        self.write_manifest()

    def save(self, name, values, affine, *, units="mm"):
        path = self.root / name
        image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), affine)
        image.header.set_xyzt_units(units)
        image.set_qform(affine, 1)
        image.set_sform(affine, 2)
        nib.save(image, path)
        return path

    @staticmethod
    def record(path):
        return {"path": str(path), "sha256": sha(path)}

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report))

    def write_manifest(self):
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def build(self, output, *, target="gre", grid_column="qsm_path"):
        return prepare_nuclei_roi.build(self.cases_path, self.manifest, output, split="test", target=target,
                                        source_root=self.root, grid_column=grid_column)

    def overlay(self, output):
        with (output / "roi_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))

    def replace_label(self, target, side, array, affine=None, *, units="mm", update_hash=True):
        record = self.report["targets"][target]["labels"][side]
        affine = affine if affine is not None else (self.gre_affine if target == "gre" else self.swi_affine)
        path = self.save(Path(record["path"]).name, array, affine, units=units)
        if update_hash:
            record["sha256"] = sha(path)
            self.write_report()

    def assert_not_published(self, output):
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".nuclei-labels-*")), [])

    def test_left_right_reencoding_whole_sn_bilateral_and_spatial_header(self):
        output = self.root / "labels"
        before_manifest, before_cases = self.manifest.read_bytes(), self.cases_path.read_bytes()
        original_hashes = {path: sha(path) for path in [self.qsm, self.swi, self.report_path,
            *(Path(record["path"]) for target in self.report["targets"].values() for record in target["labels"].values())]}
        report = self.build(output)
        rows = self.overlay(output)
        known = next(row for row in rows if row["case_id"] == self.case_id)
        image = nib.load(output / known["roi_labels_path"])
        values = image.get_fdata()
        expected = np.zeros(self.gre_shape, dtype=np.int32)
        offset = prepare_nuclei_roi.RIGHT_OFFSET
        for side, added in (("left", 0), ("right", offset)):
            original = self.original_labels[("gre", side)]
            wanted = np.isin(original, (385, 310, 352, 721))
            expected[wanted] = original[wanted].astype(np.int32) + added
        np.testing.assert_array_equal(values, expected)
        self.assertEqual(image.get_data_dtype(), np.dtype("int32"))
        self.assertEqual(image.header.get_xyzt_units()[0], "mm")
        np.testing.assert_allclose(image.affine, nib.load(self.qsm).affine, rtol=0, atol=0)
        self.assertEqual((int(image.header["qform_code"]), int(image.header["sform_code"])), (1, 2))
        spec = json.loads((output / "roi_spec.json").read_text())
        regions = {row["name"]: row["labels"] for row in spec["regions"]}
        self.assertEqual(len(regions), 15)
        self.assertEqual(regions["SN_L"], [310, 352])
        self.assertEqual(regions["SN_R"], [offset + 310, offset + 352])
        self.assertEqual(regions["SN_bilateral"], [310, 352, offset + 310, offset + 352])
        self.assertEqual(regions["RN_bilateral"], [385, offset + 385])
        self.assertEqual(int(np.isin(values, regions["SN_bilateral"]).sum()), 4)
        self.assertEqual(int(np.isin(values, regions["RN_bilateral"]).sum()), 2)
        self.assertFalse(np.any(values == 999))
        self.assertEqual(self.manifest.read_bytes(), before_manifest)
        self.assertEqual(self.cases_path.read_bytes(), before_cases)
        self.assertEqual(report["input_manifest_sha256"], sha(self.manifest))
        self.assertFalse(report["input_manifest_modified"])
        self.assertFalse(report["registration_or_intensity_resampling"])
        for path, digest in original_hashes.items():
            self.assertEqual(sha(path), digest)

    def test_missing_cases_remain_explicit_pending_without_reading_their_images(self):
        output = self.root / "missing_retained"
        report = self.build(output)
        rows = self.overlay(output)
        self.assertEqual((report["n_cases"], report["n_with_labels"], report["n_missing_labels_retained"]), (2, 1, 1))
        self.assertEqual({row["case_id"] for row in rows}, {self.case_id, "synthetic__missing"})
        missing = next(row for row in rows if row["case_id"] == "synthetic__missing")
        self.assertEqual((missing["roi_labels_path"], missing["roi_label_source"]), ("", ""))
        self.assertTrue(all(row["roi_review"] == "pending" and not row["roi_reviewer"] and not row["roi_review_date"] for row in rows))

    def test_swi_target_uses_its_native_grid_and_never_silently_fits_gre(self):
        output = self.root / "swi_labels"
        report = self.build(output, target="swi", grid_column="swi_path")
        known = next(row for row in self.overlay(output) if row["case_id"] == self.case_id)
        image = nib.load(output / known["roi_labels_path"])
        self.assertEqual(image.shape, self.swi_shape)
        np.testing.assert_allclose(image.affine, nib.load(self.swi).affine, rtol=0, atol=0)
        self.assertEqual((report["target_grid"], report["grid_column"]), ("swi", "swi_path"))
        rejected = self.root / "wrong_target_grid"
        with self.assertRaisesRegex(ValueError, "hash-bound registration target"):
            self.build(rejected, target="swi", grid_column="qsm_path")
        self.assert_not_published(rejected)

    def test_same_geometry_different_image_bytes_fail_target_hash_binding(self):
        image = nib.load(self.qsm)
        replacement = self.save("other_same_grid.nii.gz", image.get_fdata() + 7, image.affine)
        self.rows[0]["qsm_path"] = str(replacement)
        self.write_manifest()
        output = self.root / "same_grid_wrong_data"
        with self.assertRaisesRegex(ValueError, "hash-bound registration target"):
            self.build(output)
        self.assert_not_published(output)

    def test_report_case_subject_and_cohort_misbindings_are_rejected(self):
        original = dict(self.report)
        for field, value in (("case_id", "wrong_case"), ("subject_id", "wrong_person"), ("cohort", "wrong_cohort")):
            self.report[field] = value
            self.write_report()
            output = self.root / ("wrong_" + field)
            with self.assertRaises(ValueError):
                self.build(output)
            self.assert_not_published(output)
            self.report[field] = original[field]
        self.write_report()
        self.rows[0]["cohort"] = "manifest_other_cohort"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "participant/cohort"):
            self.build(self.root / "manifest_wrong_cohort")

    def test_known_spatial_unit_conflict_is_rejected_despite_identical_affine(self):
        array = self.original_labels[("gre", "left")]
        self.replace_label("gre", "left", array, units="meter")
        output = self.root / "known_units_conflict"
        with self.assertRaisesRegex(ValueError, "spatial units disagree"):
            self.build(output)
        self.assert_not_published(output)

    def test_changed_label_bytes_fractional_labels_and_hemisphere_overlap_are_rejected(self):
        original = self.original_labels[("gre", "left")]
        changed = original.copy(); changed[0, 0, 0] = 352
        self.replace_label("gre", "left", changed, update_hash=False)
        with self.assertRaisesRegex(ValueError, "label hash changed"):
            self.build(self.root / "stale_label")
        fractional = original.copy(); fractional[0, 0, 0] = 385.5
        self.replace_label("gre", "left", fractional)
        with self.assertRaisesRegex(ValueError, "integer labels"):
            self.build(self.root / "fractional")
        self.replace_label("gre", "left", original)
        right = self.original_labels[("gre", "right")].copy(); right[0, 0, 0] = 385
        self.replace_label("gre", "right", right)
        output = self.root / "hemisphere_overlap"
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.build(output)
        self.assert_not_published(output)

    def test_mismatched_affine_and_existing_output_cannot_be_overwritten(self):
        affine = self.gre_affine.copy(); affine[0, 3] += 1
        self.replace_label("gre", "left", self.original_labels[("gre", "left")], affine)
        output = self.root / "misregistered"
        with self.assertRaisesRegex(ValueError, "grids differ"):
            self.build(output)
        self.assert_not_published(output)
        output.mkdir()
        marker = output / "preserve.txt"; marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.build(output)
        self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()

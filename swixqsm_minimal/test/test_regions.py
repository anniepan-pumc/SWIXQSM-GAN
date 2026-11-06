"""Synthetic-only ROI evaluation contracts, with no model or registration."""
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from ..util.regions import RegionEvaluator


class RegionsTest(unittest.TestCase):
    MEASUREMENTS = ("ref_mean", "pred_mean", "difference", "pair_mean", "mae", "rmse", "bias", "nmse")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.shape = (6, 7, 4)
        self.affine = np.diag([.8, .9, 1.5, 1.])
        self.mask = np.ones(self.shape, dtype=bool)
        self.reference_image = self.image(np.zeros(self.shape))
        self.counter = 0
        self.spec = {
            "schema": "swixqsm.roi_spec", "version": 1, "id": "synthetic-fixed-regions-v1",
            "definition": "Synthetic left and bilateral unions for software tests; no clinical interpretation",
            "label_source": "Synthetic noninterpolated integer labelmaps on each case's evaluation grid",
            "regions": [{"name": "left", "labels": [1]}, {"name": "bilateral", "labels": [1, 2]}],
            "min_coverage": 1., "min_voxels": 1,
        }

    def image(self, array, *, affine=None, units="mm"):
        image = nib.Nifti1Image(np.asarray(array), self.affine if affine is None else affine)
        image.header.set_xyzt_units(units)
        return image

    def labels(self, left_width=2):
        labels = np.zeros(self.shape, dtype=np.int16)
        labels[:left_width] = 1
        labels[left_width:4] = 2
        return labels

    def case(self, case_id, subject_id="participant-a", *, labels=None, cohort="synthetic",
             review="accepted", source="Synthetic independently defined anatomy labels", split="test"):
        labels = self.labels() if labels is None else labels
        path = self.root / f"{case_id}_labels.nii.gz"
        nib.save(self.image(labels), path)
        row = {"case_id": case_id, "subject_id": subject_id, "cohort": cohort, "split": split,
               "roi_labels_path": str(path), "roi_label_source": source, "roi_review": review,
               "roi_reviewer": "Synthetic reader", "roi_review_date": "2026-10-17"}
        return row

    def evaluator(self, rows, spec=None):
        self.counter += 1
        path = self.root / f"roi-spec-{self.counter}.json"
        path.write_text(json.dumps(self.spec if spec is None else spec))
        return RegionEvaluator(rows, path, self.root / "manifest.csv", units="ppb",
                               reference_id="synthetic-CSF", bootstrap_reps=20, seed=42)

    def collect(self, evaluator, row, target=None, prediction=None, *, method="prediction", mask=None,
                units="ppb", reference_image=None, spatial_units=None):
        target = np.full(self.shape, 10.) if target is None else target
        prediction = target + 2 if prediction is None else prediction
        return evaluator.collect(row["case_id"], method, target, prediction,
                                 self.mask if mask is None else mask, units,
                                 self.reference_image if reference_image is None else reference_image,
                                 spatial_units=spatial_units)

    def assert_null_measurements(self, record):
        for name in self.MEASUREMENTS:
            self.assertIsNone(record[name], f"Unavailable ROI unexpectedly has {name}")

    @staticmethod
    def record(records, region, *, case_id=None, subject_id=None, method="prediction"):
        selected = [record for record in records if record["region"] == region and record["method"] == method
                    and (case_id is None or record.get("case_id") == case_id)
                    and (subject_id is None or record.get("subject_id") == subject_id)]
        if len(selected) != 1:
            raise AssertionError(f"Expected exactly one record, got {len(selected)}")
        return selected[0]

    def test_common_valid_mask_and_physical_means_are_identical_for_methods(self):
        row = self.case("single")
        evaluator = self.evaluator([row])
        target = np.full(self.shape, 10.)
        prediction = target + 2
        # Values outside all defined ROIs must not influence regional measures.
        target[4:] = 1e6
        prediction[4:] = -1e6
        self.collect(evaluator, row, target, np.zeros_like(target), method="zero")
        self.collect(evaluator, row, target, prediction)
        result = evaluator.finish()
        self.assertEqual(len(result["cases"]), 4)
        for region, count in (("left", 2 * 7 * 4), ("bilateral", 4 * 7 * 4)):
            measured = self.record(result["cases"], region, case_id="single")
            zero = self.record(result["cases"], region, case_id="single", method="zero")
            self.assertEqual(measured["status"], "ok")
            self.assertEqual(measured["n_label_voxels"], count)
            self.assertEqual(measured["n_valid_voxels"], count)
            self.assertEqual(measured["coverage"], 1.)
            self.assertEqual(measured["ref_mean"], 10.)
            self.assertEqual(measured["pred_mean"], 12.)
            self.assertEqual(measured["difference"], 2.)
            self.assertEqual(measured["pair_mean"], 11.)
            for field in ("mae", "rmse", "bias"):
                self.assertEqual(measured[field], 2.)
            self.assertAlmostEqual(measured["nmse"], .04)
            self.assertEqual(zero["n_valid_voxels"], measured["n_valid_voxels"])
            self.assertEqual(zero["ref_mean"], measured["ref_mean"])
            self.assertEqual(zero["pred_mean"], 0.)
            self.assertEqual(zero["nmse"], 1.)
        json.dumps(result, allow_nan=False)

    def test_incomplete_coverage_empty_and_missing_maps_never_disappear(self):
        spec = dict(self.spec, min_coverage=.8)
        spec["regions"] = [*copy.deepcopy(self.spec["regions"]), {"name": "absent", "labels": [99]}]
        partial = self.case("partial")
        missing = self.case("missing", "participant-b")
        missing["roi_labels_path"] = ""
        evaluator = self.evaluator([partial, missing], spec)
        mask = self.mask.copy()
        mask[0] = False
        self.collect(evaluator, partial, mask=mask)
        self.collect(evaluator, missing)
        result = evaluator.finish()
        self.assertEqual(len(result["cases"]), 6)
        partial_left = self.record(result["cases"], "left", case_id="partial")
        self.assertEqual(partial_left["status"], "insufficient_coverage")
        self.assertEqual(partial_left["n_label_voxels"], 56)
        self.assertEqual(partial_left["n_valid_voxels"], 28)
        self.assertEqual(partial_left["coverage"], .5)
        self.assert_null_measurements(partial_left)
        absent = self.record(result["cases"], "absent", case_id="partial")
        self.assertEqual(absent["status"], "empty_label")
        self.assertEqual(absent["n_label_voxels"], 0)
        self.assertIsNone(absent["coverage"])
        self.assert_null_measurements(absent)
        for region in ("left", "bilateral", "absent"):
            record = self.record(result["cases"], region, case_id="missing")
            self.assertEqual(record["status"], "missing_labelmap")
            self.assert_null_measurements(record)
        self.assertEqual(len(result["subjects"]), 6)
        for record in result["subjects"]:
            self.assertEqual(record["status"], "incomplete_visits")
            self.assertEqual(record["n_cases_expected"], 1)
            self.assertEqual(record["n_cases_included"], 0)
            self.assert_null_measurements(record)

    def test_min_voxels_is_explicit_not_a_silent_mask_reduction(self):
        row = self.case("small")
        spec = dict(self.spec, min_coverage=.001, min_voxels=2)
        evaluator = self.evaluator([row], spec)
        mask = np.zeros(self.shape, dtype=bool)
        mask[0, 0, 0] = True
        self.collect(evaluator, row, mask=mask)
        result = evaluator.finish()
        record = self.record(result["cases"], "left", case_id="small")
        self.assertEqual(record["status"], "insufficient_voxels")
        self.assertEqual(record["n_valid_voxels"], 1)
        self.assertAlmostEqual(record["coverage"], 1 / 56)
        self.assert_null_measurements(record)

    def test_bilateral_missing_component_is_not_accepted_as_a_single_side(self):
        labels = self.labels()
        labels[labels == 2] = 0
        row = self.case("missing_right", labels=labels)
        evaluator = self.evaluator([row])
        self.collect(evaluator, row)
        result = evaluator.finish()
        left = self.record(result["cases"], "left", case_id="missing_right")
        bilateral = self.record(result["cases"], "bilateral", case_id="missing_right")
        self.assertEqual(left["status"], "ok")
        self.assertEqual(bilateral["status"], "missing_component_label")
        self.assertEqual(bilateral["coverage"], 1.)
        self.assertEqual(bilateral["label_voxel_counts"], {"1": 56, "2": 0})
        self.assertEqual(bilateral["missing_labels"], [2])
        self.assert_null_measurements(bilateral)

    def test_multiple_visits_are_one_participant_scalar_not_independent_pairs(self):
        first = self.case("a1", labels=self.labels(left_width=1))
        second_labels = self.labels()
        second_labels[4] = 2  # Larger bilateral ROI on this visit.
        second = self.case("a2", labels=second_labels)
        third = self.case("b1", "participant-b")
        evaluator = self.evaluator([first, second, third])
        for row, value, bias in ((first, 10., 2.), (second, 30., 4.), (third, 100., 6.)):
            target = np.full(self.shape, value)
            self.collect(evaluator, row, target, np.zeros_like(target), method="zero")
            self.collect(evaluator, row, target, target + bias)
        result = evaluator.finish()
        self.assertEqual(len(result["cases"]), 12)
        self.assertEqual(len(result["subjects"]), 8)  # 2 participants x 2 regions x 2 methods.
        person = self.record(result["subjects"], "bilateral", subject_id="participant-a")
        n1, n2 = (int(np.count_nonzero(np.isin(labels, [1, 2])))
                  for labels in (self.labels(left_width=1), second_labels))
        self.assertEqual(person["status"], "ok")
        self.assertEqual(person["n_cases_expected"], 2)
        self.assertEqual(person["n_cases_included"], 2)
        self.assertEqual(person["n_valid_voxels"], n1 + n2)
        self.assertAlmostEqual(person["ref_mean"], (10 * n1 + 30 * n2) / (n1 + n2))
        self.assertAlmostEqual(person["pred_mean"], (12 * n1 + 34 * n2) / (n1 + n2))
        self.assertAlmostEqual(person["difference"], (2 * n1 + 4 * n2) / (n1 + n2))
        selected = [r for r in result["subjects"] if r["region"] == "bilateral" and r["method"] == "prediction"]
        self.assertEqual({r["subject_id"] for r in selected}, {"participant-a", "participant-b"})

    def test_one_bad_visit_excludes_participant_roi_instead_of_selecting_good_visit(self):
        first, second = self.case("one"), self.case("two", review="pending")
        evaluator = self.evaluator([first, second])
        self.collect(evaluator, first)
        self.collect(evaluator, second)
        result = evaluator.finish()
        second_record = self.record(result["cases"], "left", case_id="two")
        self.assertEqual(second_record["status"], "review_not_accepted")
        self.assert_null_measurements(second_record)
        participant = self.record(result["subjects"], "left", subject_id="participant-a")
        self.assertEqual(participant["status"], "incomplete_visits")
        self.assertEqual(participant["n_cases_expected"], 2)
        self.assertLess(participant["n_cases_included"], 2)
        self.assertEqual(participant["n_valid_voxels"], 0)
        self.assert_null_measurements(participant)

    def test_missing_source_and_nonaccepted_reviews_have_no_measurements(self):
        rows = [self.case("source", source=""), self.case("pending", "participant-b", review="pending"),
                self.case("rejected", "participant-c", review="rejected"),
                self.case("not_recorded", "participant-d")]
        del rows[-1]["roi_review"]
        evaluator = self.evaluator(rows)
        for row in rows:
            self.collect(evaluator, row)
        result = evaluator.finish()
        self.assertEqual(len(result["cases"]), 8)
        for record in result["cases"]:
            self.assertNotEqual(record["status"], "ok")
            self.assert_null_measurements(record)
        self.assertEqual(self.record(result["cases"], "left", case_id="source")["status"], "missing_label_source")

    def test_constructor_is_lazy_and_mixed_splits_and_ambiguous_people_reject_before_read(self):
        first = self.case("one")
        second = self.case("two", "participant-b")
        with patch("nibabel.load", side_effect=AssertionError("Constructor must not open label images")):
            self.evaluator([first, second])
            invalid_sets = []
            invalid_sets.append([first, dict(second, split="train")])
            invalid_sets.append([dict(first, global_subject_id="global-a"), second])
            invalid_sets.append([dict(first, global_subject_id="global-a"), dict(second, global_subject_id="global-a")])
            invalid_sets.append([dict(first, global_subject_id="global-a"),
                                 dict(second, subject_id=first["subject_id"], global_subject_id="global-b")])
            invalid_sets.append([first, dict(second, subject_id=first["subject_id"], cohort="other-cohort")])
            invalid_sets.append([first, dict(second, case_id=first["case_id"])])
            for rows in invalid_sets:
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    self.evaluator(rows)
        # A declared but not yet collected case's map is not decoded early.
        second["roi_labels_path"] = str(self.root / "unread_missing_labels.nii.gz")
        evaluator = self.evaluator([first, second])
        opened = []
        original_load = nib.load

        def load(path, *args, **kwargs):
            opened.append(str(Path(path).resolve()))
            return original_load(path, *args, **kwargs)

        with patch("nibabel.load", side_effect=load):
            self.collect(evaluator, first)
        self.assertTrue(opened)
        self.assertNotIn(second["roi_labels_path"], opened)

    def test_invalid_label_values_grid_or_units_reject_without_resampling(self):
        invalid_values = [self.labels().astype(float) + .25, -self.labels(),
                          np.full(self.shape, np.nan), np.zeros((2, 3, 4, 1), dtype=np.int16),
                          self.labels().astype(np.complex64) + 1j]
        for index, labels in enumerate(invalid_values):
            row = self.case(f"invalid_{index}", labels=labels)
            evaluator = self.evaluator([row])
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.collect(evaluator, row)
        for kind in ("affine", "units"):
            row = self.case(kind)
            affine = self.affine.copy()
            affine[0, 3] += 1
            image = self.image(self.labels(), affine=affine if kind == "affine" else self.affine,
                               units="meter" if kind == "units" else "mm")
            nib.save(image, row["roi_labels_path"])
            evaluator = self.evaluator([row])
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.collect(evaluator, row)
        row = self.case("susceptibility_units")
        evaluator = self.evaluator([row])
        with self.assertRaises(ValueError):
            self.collect(evaluator, row, units="ppm")

    def test_complex_physical_arrays_and_different_method_support_reject(self):
        row = self.case("physical_and_support")
        for field in ("target", "prediction"):
            evaluator = self.evaluator([row])
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.collect(evaluator, row, **{field: np.full(self.shape, 10. + 1j)})
        evaluator = self.evaluator([row])
        self.collect(evaluator, row, method="zero")
        changed_mask = self.mask.copy()
        changed_mask[0, 0, 0] = False
        with self.assertRaises(ValueError):
            self.collect(evaluator, row, mask=changed_mask)

    def test_duplicate_collect_and_incomplete_method_coverage_cannot_doublecount_or_drop(self):
        first, second = self.case("one"), self.case("two", "participant-b")
        evaluator = self.evaluator([first, second])
        self.collect(evaluator, first, method="zero")
        self.collect(evaluator, first)
        with self.assertRaises(ValueError):
            self.collect(evaluator, first)
        self.collect(evaluator, second, method="zero")
        with self.assertRaises(ValueError):
            evaluator.finish()

    def test_spec_validation_and_provenance_hashes_do_not_establish_segmentation_accuracy(self):
        row = self.case("provenance")
        invalid = [dict(self.spec, version=True), dict(self.spec, id=""), dict(self.spec, label_source=""),
                   dict(self.spec, definition=""), dict(self.spec, min_coverage=1.1), dict(self.spec, min_voxels=0),
                   dict(self.spec, regions=[{"name": "bad", "labels": [0]}]),
                   dict(self.spec, regions=[{"name": "bad", "labels": [True]}]),
                   dict(self.spec, regions=[{"name": "bad", "labels": [1.5]}]),
                   dict(self.spec, regions=[{"name": "bad", "labels": [1, 1]}]),
                   dict(self.spec, regions=[{"name": "same", "labels": [1]}, {"name": "same", "labels": [2]}])]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                self.evaluator([row], spec)
        for fields in ({"roi_reviewer": ""}, {"roi_review_date": "2026-02-31"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.evaluator([dict(row, **fields)])
        evaluator = self.evaluator([row])
        self.collect(evaluator, row)
        result = evaluator.finish()
        provenance = result["provenance"]
        encoded = json.dumps(provenance, sort_keys=True, allow_nan=False)
        self.assertIn(hashlib.sha256(Path(row["roi_labels_path"]).read_bytes()).hexdigest(), encoded)
        self.assertIn(row["roi_label_source"], encoded)
        self.assertNotEqual(provenance.get("status"), "verified")
        self.assertFalse(provenance.get("whole_image_qc_upgraded", False))

    def test_nmse_overflow_from_finite_sums_is_null_and_strict_json_safe(self):
        row = self.case("nmse_overflow")
        evaluator = self.evaluator([row])
        self.collect(evaluator, row, np.full(self.shape, 1e-150), np.full(self.shape, 1e100))
        with np.errstate(over="ignore", invalid="ignore"):
            result = evaluator.finish()
        json.dumps(result, allow_nan=False)
        for record in result["cases"]:
            self.assertIsNone(record["nmse"])
            self.assertNotEqual(record.get("nmse_status"), "ok")

    def test_large_finite_participant_errors_do_not_overflow_summary_json(self):
        rows = [self.case("huge_a"), self.case("huge_b", "participant-b")]
        evaluator = self.evaluator(rows)
        for row in rows:
            self.collect(evaluator, row, np.full(self.shape, 1e-154), np.ones(self.shape))
        with np.errstate(over="ignore", invalid="ignore"):
            result = evaluator.finish()
        for record in result["cases"]:
            self.assertTrue(math.isfinite(record["nmse"]))
        json.dumps(result, allow_nan=False)
        for group in result["agreement"]:
            descriptive = group["voxel_error_participant_summary"]["nmse"]
            for name in ("mean", "sd"):
                self.assertTrue(descriptive[name] is None or math.isfinite(descriptive[name]))


if __name__ == "__main__":
    unittest.main()

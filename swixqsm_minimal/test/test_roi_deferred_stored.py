"""Deferred ROI tests use synthetic maps only; no models or cohort data."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

from .. import evaluate
from ..util.data import read_manifest
from ..util.evaluation import evaluate_rows
from ..util.independence import build_independence_report
from ..util.nifti import sidecar_path
from ..util.quality_gate import prepare_quality_rows
from ..util.regions import RegionEvaluator


class DeferredStoredROITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="swix-deferred-roi-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shape = (8, 9, 3)
        self.affine = np.diag([.8, .9, 2., 1.])
        self.core, self.overlay, self.targets = [], [], {}
        raw = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
        self.labels = np.zeros(self.shape, dtype=np.float32)
        self.labels[:3] = 1
        self.labels[3:7] = 2
        for index in range(4):
            sid = 0 if index == 3 else index
            target = raw / 10 - 5 + index * 2
            error = np.ones(self.shape, dtype=np.float32)
            # Left ROI mean error cancels; voxel MAE remains two.
            error[:3, :, 0] = -3
            error[:3, :, 1:] = 1.5
            row = dict(case_id=f"case{index}", subject_id=f"p{sid}", global_subject_id=f"g{sid}",
                       split="test", cohort="synthetic", qsm_units="stored",
                       qsm_reference="unverified_original_reference", prediction_units="stored",
                       prediction_reference="unverified_original_reference")
            for role, array in dict(swi_path=raw + index, qsm_path=target, mask_path=np.ones(self.shape),
                                    swi_mask_path=np.ones(self.shape), prediction_path=target + error).items():
                row[role] = str(self.save(f"{index}_{role}.nii.gz", array))
            self.core.append(row)
            self.targets[row["case_id"]] = target
            label_path = self.save(f"{index}_labels.nii.gz", self.labels)
            self.overlay.append(dict(case_id=row["case_id"], roi_labels_path=label_path.name,
                                     roi_label_source="Synthetic integer nuclear labels; no actual atlas IDs",
                                     roi_review=("pending", "not_recorded", "unreviewed", "rejected")[index],
                                     roi_reviewer="", roi_review_date=""))
        self.manifest, self.roi_manifest = self.root / "core.csv", self.root / "roi.csv"
        self.write_csv(self.manifest, self.core)
        self.write_csv(self.roi_manifest, self.overlay)
        self.spec = self.root / "spec.json"
        self.spec.write_text(json.dumps(dict(schema="swixqsm.roi_spec", version=1, id="synthetic_nuclei",
            definition="Known synthetic labels, no anatomical claims", label_source="Synthetic fixture only",
            regions=[dict(name="RN_L", labels=[1]), dict(name="RN_R", labels=[2]),
                     dict(name="RN_bilateral", labels=[1, 2]), dict(name="absent", labels=[99])],
            min_coverage=1., min_voxels=1)))
        # Independently audit the unchanged core manifest, then bind predictions.
        report = build_independence_report(prepare_quality_rows(read_manifest(self.manifest), self.manifest))
        self.assertEqual(report["status"], "passed", report)
        self.independence = self.root / "independence.json"
        self.independence.write_text(json.dumps(report, allow_nan=False))
        for row in self.core:
            sidecar_path(row["prediction_path"]).write_text(json.dumps(dict(
                Units="stored", ValueSpace="restored_stored_values", UnitStatus="unverified_original_storage",
                ReferenceID="unverified_original_reference", DataQC=dict(status="unverified"),
                DataIndependence=dict(status="verified", manifest_identity_sha256=report["manifest_identity_sha256"]))))

    def save(self, name, values, affine=None):
        path = self.root / name
        image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), self.affine if affine is None else affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        return path

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def csv_rows(path):
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def command(self, output, *, units="stored", policy="deferred", extra=()):
        return ["evaluate", "--manifest", str(self.manifest), "--output", str(output), "--split", "test",
                "--units", units, "--reference-id", "unverified_original_reference",
                "--allow-unverified-data", "Synthetic units/reference/ROI QC deferred",
                "--independence-report", str(self.independence), "--roi-spec", str(self.spec),
                "--roi-manifest", str(self.roi_manifest), "--roi-review-policy", policy,
                "--roi-bootstrap-reps", "20", *extra]

    def run_evaluate(self, output, **kwargs):
        with mock.patch.object(sys, "argv", self.command(output, **kwargs)):
            evaluate.main()

    def evaluator(self, rows=None, **kwargs):
        merged, source = evaluate.roi_rows_with_overlay(self.core if rows is None else rows, self.roi_manifest)
        return RegionEvaluator(merged, self.spec, self.manifest, units="stored", reference_id="unverified_original_reference",
            review_policy="deferred", allow_unverified_data="Synthetic pending review", data_qc=dict(status="unverified"),
            roi_manifest_source=source, bootstrap_reps=20, **kwargs)

    def test_explicit_opt_in_and_no_verified_qc_or_fake_acceptance(self):
        with self.assertRaisesRegex(ValueError, "without explicit deferred"):
            RegionEvaluator(self.core, self.spec, self.manifest, units="stored", reference_id="unknown")
        for options in (dict(), dict(allow_unverified_data=""),
                        dict(allow_unverified_data="reason", data_qc=dict(status="verified"))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                RegionEvaluator(self.core, self.spec, self.manifest, units="stored", reference_id="unknown",
                                review_policy="deferred", **options)
        before = copy.deepcopy(self.overlay)
        self.run_evaluate(self.root / "candidate")
        self.assertEqual(self.overlay, before)
        records = self.csv_rows(self.root / "candidate" / "roi_per_case.csv")
        self.assertEqual({r["roi_review_status"] for r in records}, {"pending", "not_recorded", "unreviewed", "rejected"})
        self.assertTrue(all(r["research_only"] == "True" and r["formal_accuracy_claim"] == "False" for r in records))
        self.assertTrue(all(r["UnitStatus"] == "unverified_original_storage" and r["QCStatus"] == "unverified" for r in records))
        provenance = json.loads((self.root / "candidate" / "roi_agreement.json").read_text())["provenance"]
        self.assertTrue(all(not row["roi_reviewer"] and not row["roi_review_date"] for row in provenance["cases"]))
        with self.assertRaises(SystemExit):
            self.run_evaluate(self.root / "strict_stored", policy="accepted")
        argv = self.command(self.root / "cannot_verify")
        first = argv.index("--allow-unverified-data")
        argv[first:first + 2] = ["--qc-report", "not-read.json"]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            evaluate.main()

    def test_overlay_preserves_frozen_core_identity_and_reports_same_domain_errors(self):
        before = self.manifest.read_bytes()
        source_before = self.independence.read_bytes()
        output = self.root / "results"
        self.run_evaluate(output)
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertEqual(self.independence.read_bytes(), source_before)
        self.assertNotIn("roi_labels_path", read_manifest(self.manifest)[0])
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["config"]["data_independence"]["status"], "verified")
        self.assertEqual(summary["config"]["manifest_sha256"], hashlib.sha256(before).hexdigest())
        roi_source = summary["config"]["roi_manifest_source"]
        self.assertEqual(roi_source["sha256"], hashlib.sha256(self.roi_manifest.read_bytes()).hexdigest())
        records = self.csv_rows(output / "roi_per_case.csv")
        left = next(row for row in records if row["case_id"] == "case0" and row["region"] == "RN_L" and row["method"] == "prediction")
        self.assertEqual(left["status"], "ok")
        self.assertAlmostEqual(float(left["difference"]), 0., places=5)
        self.assertAlmostEqual(float(left["absolute_mean_difference"]), 0., places=5)
        self.assertAlmostEqual(float(left["mae"]), 2., places=5)
        self.assertAlmostEqual(float(left["rmse"]), np.sqrt(4.5), places=5)
        rejected = next(row for row in records if row["case_id"] == "case3" and row["region"] == "RN_L")
        self.assertEqual(rejected["status"], "review_rejected")
        self.assertEqual(rejected["mae"], "")
        for case in self.core:
            method_rows = [r for r in records if r["case_id"] == case["case_id"] and r["region"] == "RN_L"]
            for field in ("roi_support_sha256", "evaluation_support_sha256", "reference_array_sha256", "evaluation_grid_sha256"):
                self.assertEqual(len({r[field] for r in method_rows}), 1)
                self.assertEqual(len(method_rows[0][field]), 64)
        report = json.loads((output / "roi_agreement.json").read_text())
        overall = next(r for r in report["agreement"] if r["method"] == "prediction" and r["region"] == "RN_L" and r["group_type"] == "overall")
        self.assertEqual(overall["n_participants_expected"], 3)
        self.assertEqual(overall["n_participants_included"], 2)  # p0 has a rejected expected visit.
        self.assertIsNone(overall["agreement"]["estimates"]["icc_a1"])
        self.assertIsNone(overall["agreement"]["bootstrap"]["metrics"]["bias"]["lower"])
        self.assertEqual(overall["agreement"]["definitions"]["dimensional_metrics_units"], "stored")
        self.assertIn("icc_a1", overall["agreement"]["definitions"]["dimensionless_metrics"])
        self.assertFalse(any("same physical units" in value for value in overall["agreement"]["assumptions"]))
        self.assertTrue(overall["research_only"])
        self.assertIn("roi_mean_participant_summary", overall)
        self.assertIn("voxel_error_participant_summary", overall)

    def test_more_participants_can_define_dimensionless_candidate_icc(self):
        self.overlay[-1]["roi_review"] = "pending"
        self.write_csv(self.roi_manifest, self.overlay)
        output = self.root / "three_people"
        self.run_evaluate(output)
        report = json.loads((output / "roi_agreement.json").read_text())
        row = next(r for r in report["agreement"] if r["method"] == "prediction" and r["region"] == "RN_R" and r["group_type"] == "overall")
        self.assertEqual(row["n_participants_included"], 3)
        self.assertIsNotNone(row["agreement"]["estimates"]["icc_a1"])
        self.assertEqual(row["QCStatus"], "unverified")
        self.assertFalse(row["formal_accuracy_claim"])

    def test_overlay_whitelist_completeness_duplicates_and_alias_conflicts(self):
        original = copy.deepcopy(self.core)
        for modified in (self.overlay[:-1], [*self.overlay, dict(self.overlay[0])],
                         [dict(row, qsm_units="ppb") for row in self.overlay],
                         [dict(row, roi_review_status="accepted") for row in self.overlay]):
            self.write_csv(self.roi_manifest, modified)
            with self.subTest(modified=modified), self.assertRaises(ValueError):
                evaluate.roi_rows_with_overlay(self.core, self.roi_manifest)
        self.assertEqual(self.core, original)
        alias_rows = [{("roi_review_status" if key == "roi_review" else key): value for key, value in row.items()}
                      for row in self.overlay]
        self.write_csv(self.roi_manifest, alias_rows)
        merged, source = evaluate.roi_rows_with_overlay(self.core, self.roi_manifest)
        self.assertEqual(merged[0]["roi_review"], "pending")
        self.assertEqual(merged[0]["roi_labels_path"], str((self.root / self.overlay[0]["roi_labels_path"]).resolve()))
        self.assertEqual(source["n_rows"], 4)

    def test_invalid_integer_grid_or_stored_physical_conversion_publishes_nothing(self):
        for kind in ("integer", "grid", "units"):
            path = self.root / self.overlay[0]["roi_labels_path"]
            values = self.labels.copy()
            affine = self.affine.copy()
            if kind == "integer":
                values[0, 0, 0] = 1.5
            elif kind == "grid":
                affine[0, 3] += 2
            self.save(path.name, values, affine)
            if kind == "units":
                self.core[0]["qsm_units"] = "ppb"
                self.write_csv(self.manifest, self.core)
            output = self.root / f"bad_{kind}"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.run_evaluate(output)
            self.assertFalse(output.exists())
            self.assertEqual(list(self.root.glob(".swixqsm-evaluation-*")), [])

    def test_roi_manifest_or_spec_change_is_rejected_at_finish(self):
        for role in ("manifest", "spec"):
            evaluator = self.evaluator()
            for case in self.core:
                image = nib.load(case["qsm_path"])
                target = image.get_fdata()
                evaluator.collect(case["case_id"], "prediction", target, target, np.ones(self.shape), "stored", image)
            source = self.roi_manifest if role == "manifest" else self.spec
            original = source.read_bytes()
            source.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                evaluator.finish()
            source.write_bytes(original)

    def test_default_acceptance_still_excludes_pending_and_requires_review_records(self):
        merged, _ = evaluate.roi_rows_with_overlay(self.core, self.roi_manifest)
        # Original accepted-only physical workflow is unchanged.
        for row in merged:
            row["roi_review"] = "pending"
        evaluator = RegionEvaluator(merged, self.spec, self.manifest, units="ppb", reference_id="declared", bootstrap_reps=0)
        for case in self.core:
            image = nib.load(case["qsm_path"])
            target = image.get_fdata()
            evaluator.collect(case["case_id"], "prediction", target, target, np.ones(self.shape), "ppb", image)
        result = evaluator.finish()
        self.assertTrue(all(row["status"] == "review_not_accepted" for row in result["cases"]))
        merged[0]["roi_review"] = "accepted"
        with self.assertRaisesRegex(ValueError, "roi_reviewer"):
            RegionEvaluator(merged, self.spec, self.manifest, units="stored", reference_id="unknown", review_policy="deferred",
                            allow_unverified_data="Explicit research candidate")

    def test_engine_guard_does_not_allow_arbitrary_stored_roi_callback(self):
        with self.assertRaisesRegex(ValueError, "ROI physical-unit"):
            evaluate_rows(self.core, units="stored", reference_id="unverified_original_reference",
                          region_callback=lambda *args: None, roi_review_policy="deferred")


if __name__ == "__main__":
    unittest.main()

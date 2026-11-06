"""Saved CSV/JSON comparisons only; fixtures have no image files or training."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

from .. import compare_models


class CompareModelsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a = self.fixture("A", [1., 3., 9.], with_ci=True)
        self.b = self.fixture("B", [2., 4., 11.])

    def fixture(self, name, errors, *, with_ci=False, units="stored"):
        directory = self.root / name
        directory.mkdir()
        parameters = dict(data_range=500., ssim_window=7, ssim_axis=2, ssim_covariance="sample",
                          hfen_sigma_mm=1.5, hfen_truncate=4., hfen_zero_dc_correction=True,
                          hfen_operator="zero_dc_corrected_gaussian_second_derivatives")
        manifest, cases, subjects = [], [], []
        case_info = [("c1", "s1", "g1", "cohortA", 10), ("c2", "s1", "g1", "cohortA", 30),
                     ("c3", "s2", "g2", "cohortB", 20)]
        for case_id, sid, gid, cohort, count in case_info:
            row = dict(case_id=case_id, subject_id=sid, global_subject_id=gid, cohort=cohort, split="test",
                       qsm_units=units, qsm_reference="unverified_original_reference")
            for field in ("swi_path", "qsm_path", "swi_mask_path", "mask_path"):
                row[field] = f"/synthetic_nonexistent/{case_id}_{field}.nii.gz"
            row.update(prediction_path=f"/{name}/predictions/{case_id}.nii.gz", prediction_units=units,
                       prediction_reference="unverified_original_reference")
            manifest.append(row)
        metrics = ("mae", "rmse", "hfen_rmse")
        for method, values in (("zero", [4., 8., 6.]), ("prediction", errors)):
            for (case_id, sid, gid, cohort, count), error in zip(case_info, values):
                cases.append(dict(method=method, subject_id=sid, case_id=case_id, cohort=cohort,
                    units=units, reference_id="unverified_original_reference", n_voxels=count,
                    ssim_centers=0, hfen_centers=0, ref_mean=5., ref_std=1., mae=error, rmse=error,
                    hfen_rmse=None, metric_status={"mae": "ok", "rmse": "ok", "hfen_rmse": "insufficient_support"},
                    parameters=dict(parameters, spacing_mm=[1., 1., 1.], hfen_radius_voxels=[6, 6, 6]),
                    metric_units={"mae": units, "rmse": units, "hfen_rmse": units + "/mm^2"}))
            for sid in ("s1", "s2"):
                visits = [case for case in cases if case["subject_id"] == sid and case["method"] == method]
                n_voxels = sum(case["n_voxels"] for case in visits)
                mae = sum(case["mae"] * case["n_voxels"] for case in visits) / n_voxels
                rmse = math.sqrt(sum(case["mae"] ** 2 * case["n_voxels"] for case in visits) / n_voxels)
                subjects.append(dict(method=method, subject_id=sid, cohort=visits[0]["cohort"],
                    units=units, reference_id="unverified_original_reference", n_voxels=n_voxels, n_cases=len(visits),
                    ssim_centers=0, hfen_centers=0, mae=mae, rmse=rmse, hfen_rmse=None,
                    metric_status={"mae": "ok", "rmse": "ok", "hfen_rmse": "insufficient_support"}))
        summary = {}
        for method in ("prediction", "zero"):
            summary[method] = {}
            for group in ("all", "cohort:cohortA", "cohort:cohortB"):
                selected = [row for row in subjects if row["method"] == method
                            and (group == "all" or "cohort:" + row["cohort"] == group)]
                result = dict(n_subjects=len(selected), n_cases=sum(row["n_cases"] for row in selected), metrics={})
                for metric in metrics:
                    defined = [row[metric] for row in selected if row[metric] is not None]
                    result["metrics"][metric] = dict(mean=statistics.mean(defined) if defined else None,
                        sd=statistics.stdev(defined) if len(defined) > 1 else None,
                        n_defined=len(defined), n_undefined=len(selected) - len(defined),
                        undefined_reasons={} if defined else {"insufficient_support": len(selected)})
                if with_ci and method == "prediction" and group == "all":
                    result["metrics"]["mae"].update(ci_low=4., ci_high=8., ci_level=.95)
                summary[method][group] = result
        checkpoint_sha = hashlib.sha256(name.encode()).hexdigest()
        provenance = {row["prediction_path"]: dict(SourceCheckpointSHA256=checkpoint_sha,
            PredictionDataQC={"status": "unverified"}, PredictionDataIndependence={"status": "unverified"},
            PredictionTrainingState={"current_run_phase": "gan" if name == "A" else "l1"}) for row in manifest}
        config = dict(split="test", units=units, reference_id="unverified_original_reference", data_range=500.,
                      ssim_window=7, hfen_sigma_mm=1.5, prediction_name="prediction", space_id=None, spatial_units=None,
                      aggregation="voxel-pooled visits per participant; equal-participant mean and sample SD",
                      zero_baseline="physical zero, independent of neural target offset",
                      calibration="no clipping, target-based scaling, re-referencing or post-hoc fitting",
                      metrics_parameters=parameters, data_qc={"status": "unverified", "reason": "QC deferred"},
                      data_independence={"status": "unverified", "reason": "Identity evidence deferred"},
                      data_reference=None, model_selection={"status": "not_recorded"}, prediction_provenance=provenance)
        (directory / "summary.json").write_text(json.dumps(dict(config=config, summary=summary), allow_nan=False))
        (directory / "manifest_resolved.json").write_text(json.dumps(manifest))
        compare_models.write_csv(directory / "per_case.csv", cases)
        compare_models.write_csv(directory / "per_subject.csv", subjects)
        return directory

    def cli(self, output, *extra):
        return ["--expected-model", "A", "--expected-model", "B", "--evaluation", f"A={self.a}",
                "--evaluation", f"B={self.b}", "--output", str(output), *extra]

    def read(self, directory, filename):
        with (directory / filename).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def change_json(self, path, mutate):
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload))

    def rewrite_csv(self, path, mutate):
        rows = self.read(path.parent, path.name)
        mutate(rows)
        path.unlink()
        compare_models.write_csv(path, rows)

    def test_separate_case_and_subject_tables_keep_unverified_stored_and_original_summary(self):
        output = self.root / "comparison"
        report = compare_models.main(self.cli(output))
        cases = self.read(output, "all_models_per_case.csv")
        subjects = self.read(output, "all_models_per_subject.csv")
        long = self.read(output, "model_metrics_long.csv")
        wide = self.read(output, "model_metrics_wide.csv")
        self.assertEqual((len(cases), len(subjects)), (12, 8))
        self.assertTrue(all(row["units"] == "stored" and row["qc_status"] == "unverified" for row in cases))
        zero = [row for row in cases if row["method_role"] == "zero"]
        self.assertTrue(all(row["comparison_series"] == "zero" and row["baseline_repeated_context"] == "True" for row in zero))
        self.assertEqual({row["evaluation_model"] for row in zero}, {"A", "B"})
        self.assertTrue(all(row["reference_provenance_status"] == "unverified" for row in wide))
        self.assertEqual({row["global_subject_id"] for row in cases}, {"g1", "g2"})
        overall = next(row for row in long if row["model"] == "A" and row["method_role"] == "prediction"
                       and row["group"] == "all" and row["metric"] == "mae")
        self.assertEqual(float(overall["mean"]), 5.75)  # (.25*1+.75*3 + 9) / 2
        self.assertNotAlmostEqual(float(overall["mean"]), (1 + 3 + 9) / 3)
        self.assertEqual((overall["n"], overall["n_undefined"]), ("2", "0"))
        self.assertEqual((overall["ci_low"], overall["ci_high"], overall["ci_level"]), ("4.0", "8.0", "0.95"))
        b_overall = next(row for row in wide if row["model"] == "B" and row["method_role"] == "prediction" and row["group"] == "all")
        self.assertEqual(b_overall["mae__ci_low"], "")
        self.assertEqual(b_overall["hfen_rmse__mean"], "")
        self.assertEqual((b_overall["hfen_rmse__n"], b_overall["hfen_rmse__n_undefined"]), ("0", "2"))
        self.assertEqual(json.loads(b_overall["hfen_rmse__undefined_reasons"]), {"insufficient_support": 2})
        self.assertEqual(report["available_models"], ["A", "B"])
        self.assertEqual(report["missing_models"], {})
        self.assertEqual(len(report["sources"]), 2)
        self.assertEqual(report["sources"][0]["config"]["prediction_provenance"]["/A/predictions/c1.nii.gz"]["PredictionTrainingState"]["current_run_phase"], "gan")

    def test_incomparable_units_split_reference_and_metric_parameters_reject(self):
        original = (self.b / "summary.json").read_text()
        for key, value in (("units", "ppm"), ("split", "val"), ("reference_id", "another_reference"),
                           ("data_range", 600.), ("ssim_window", 9), ("hfen_sigma_mm", 2.)):
            self.change_json(self.b / "summary.json", lambda payload: payload["config"].update({key: value}))
            output = self.root / ("bad_" + key)
            with self.assertRaises(ValueError):
                compare_models.main(self.cli(output))
            self.assertFalse(output.exists())
            (self.b / "summary.json").write_text(original)

    def test_case_identity_and_mask_path_support_must_match(self):
        for field in ("global_subject_id", "mask_path"):
            path = self.b / "manifest_resolved.json"
            original = path.read_text()
            self.change_json(path, lambda rows: rows[0].update({field: "different_source"}))
            output = self.root / field
            with self.assertRaisesRegex(ValueError, "identity metadata"):
                compare_models.main(self.cli(output))
            self.assertFalse(output.exists())
            path.write_text(original)

    def test_incomplete_method_and_tampered_summary_are_not_silently_averaged(self):
        path = self.b / "per_case.csv"
        original = path.read_text()
        self.rewrite_csv(path, lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "missing evaluation cases"):
            compare_models.main(self.cli(self.root / "truncated"))
        path.write_text(original)
        self.change_json(self.b / "summary.json", lambda payload:
                         payload["summary"]["prediction"]["all"]["metrics"]["mae"].update(mean=0.))
        with self.assertRaisesRegex(ValueError, "Summary/per-subject inconsistency"):
            compare_models.main(self.cli(self.root / "tampered"))

    def test_missing_models_default_refuses_and_explicit_mode_lists_without_zero_fill(self):
        self.b.rename(self.root / "B_not_available")
        output = self.root / "missing"
        with self.assertRaisesRegex(ValueError, "Missing expected"):
            compare_models.main(self.cli(output))
        self.assertFalse(output.exists())
        report = compare_models.main(self.cli(output, "--allow-missing"))
        self.assertEqual(list(report["missing_models"]), ["B"])
        row = next(row for row in self.read(output, "model_metrics_long.csv") if row["model"] == "B")
        self.assertEqual(row["result_status"], "missing")
        self.assertEqual((row["n"], row["mean"], row["std"]), ("", "", ""))

    def test_duplicate_names_paths_and_source_checkpoints_are_rejected(self):
        with self.assertRaises(SystemExit):
            compare_models.main(self.cli(self.root / "duplicate_name", "--evaluation", f"A={self.a}"))
        args = self.cli(self.root / "duplicate_path")
        args[args.index(f"B={self.b}")] = f"B={self.a}"
        with self.assertRaises(SystemExit):
            compare_models.main(args)
        self.change_json(self.b / "summary.json", lambda payload: [value.update(SourceCheckpointSHA256=hashlib.sha256(b"A").hexdigest())
            for value in payload["config"]["prediction_provenance"].values()])
        with self.assertRaisesRegex(ValueError, "same source checkpoint"):
            compare_models.main(self.cli(self.root / "same_checkpoint"))

    def test_source_change_during_export_leaves_no_published_or_staging_directory(self):
        actual = compare_models.sha256
        calls = {}
        watched = (self.a / "summary.json").resolve()
        def changed(path):
            path = Path(path).resolve()
            calls[path] = calls.get(path, 0) + 1
            return "changed" if path == watched and calls[path] > 1 else actual(path)
        output = self.root / "changed_midrun"
        with mock.patch.object(compare_models, "sha256", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "changed while comparing"):
                compare_models.main(self.cli(output))
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".swixqsm-compare-models-*")), [])

    def test_existing_output_never_overwritten_and_default_expects_eight(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(SystemExit):
            compare_models.main(self.cli(output))
        self.assertEqual(marker.read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "Missing expected"):
            compare_models.main(["--evaluation", "swixqsm_gan=" + str(self.a), "--output", str(self.root / "eight")])
        self.assertEqual(len(compare_models.REVIEW_MODELS), 8)


if __name__ == "__main__":
    unittest.main()

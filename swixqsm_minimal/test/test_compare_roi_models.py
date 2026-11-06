"""Native ROI CSV/JSON fixtures only; no image files, inference or training."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

from .. import compare_models, compare_roi_models
from ..util.agreement import paired_agreement
from . import test_compare_models


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


class CompareROIModelsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a = test_compare_models.CompareModelsTest.fixture(self, "A", [10., 12., 14.])
        self.b = test_compare_models.CompareModelsTest.fixture(self, "B", [15., 16., 17.])
        for name, directory in (("A", self.a), ("B", self.b)):
            self.make_three_subjects(directory)
            self.add_roi(name, directory)

    def read_csv(self, directory, filename):
        with (directory / filename).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def write_csv(self, directory, filename, rows):
        path = directory / filename
        if path.exists():
            path.unlink()
        compare_models.write_csv(path, rows)

    def make_three_subjects(self, directory):
        path = directory / "manifest_resolved.json"
        manifest = json.loads(path.read_text())
        manifest[1].update(subject_id="s3", global_subject_id="g3")
        for row in manifest:
            row.update(roi_labels_path=f"/synthetic_nonexistent/{row['case_id']}_roi.nii.gz",
                       roi_review="pending", roi_label_source="Synthetic fixed labels only")
        path.write_text(json.dumps(manifest))
        cases = self.read_csv(directory, "per_case.csv")
        for row in cases:
            if row["case_id"] == "c2":
                row["subject_id"] = "s3"
        subjects = [{key: value for key, value in row.items() if key not in {"case_id", "parameters", "metric_units"}}
                    for row in cases]
        for row in subjects:
            row["n_cases"] = 1
        self.write_csv(directory, "per_case.csv", cases)
        self.write_csv(directory, "per_subject.csv", subjects)
        summary_path = directory / "summary.json"
        payload = json.loads(summary_path.read_text())
        for method, groups in payload["summary"].items():
            for group, result in groups.items():
                rows = [row for row in subjects if row["method"] == method and
                        (group == "all" or "cohort:" + row["cohort"] == group)]
                result.update(n_subjects=len(rows), n_cases=len(rows))
                for metric, values in result["metrics"].items():
                    numbers = [float(row[metric]) for row in rows if row[metric] != ""]
                    values.update(mean=statistics.mean(numbers) if numbers else None,
                                  sd=statistics.stdev(numbers) if len(numbers) > 1 else None,
                                  n_defined=len(numbers), n_undefined=len(rows) - len(numbers),
                                  undefined_reasons={} if numbers else {"insufficient_support": len(rows)})
        payload["config"]["roi_review_policy"] = "deferred"
        summary_path.write_text(json.dumps(payload))

    def add_roi(self, name, directory):
        whole = json.loads((directory / "summary.json").read_text())
        config = whole["config"]
        manifest = json.loads((directory / "manifest_resolved.json").read_text())
        spec = dict(schema="swixqsm.roi_spec", version=1, id="synthetic_nuclei",
                    definition="Synthetic protocol, no real segmentation", label_source="synthetic",
                    min_coverage=1., min_voxels=1,
                    regions=[{"name": "RN_L", "labels": [1]}, {"name": "RN_R", "labels": [2]}])
        spec_sha = digest(compare_models.canonical(spec))
        sources, cases, subjects, agreements, summaries = [], [], [], [], []
        flags = dict(roi_review_status="pending", roi_review_policy="deferred", roi_qc_status="unverified",
                     QCStatus="unverified", UnitStatus="unverified_stored", value_space="original_stored_scale",
                     research_only=True, formal_accuracy_claim=False)
        for original in manifest:
            source = {key: original[key] for key in ("case_id", "subject_id", "global_subject_id", "cohort", "roi_labels_path", "roi_label_source", "roi_review")}
            source.update(label_sha256=digest(original["case_id"] + "labels"),
                          reference_array_sha256=digest(original["case_id"] + "target_float64"),
                          evaluation_support_sha256=digest(original["case_id"] + "common_mask"),
                          evaluation_grid_sha256=digest(original["case_id"] + "grid"),
                          input_paths={key: original[key] for key in ("swi_path", "qsm_path", "swi_mask_path", "mask_path")})
            sources.append(source)
        for method in ("zero", "prediction"):
            for index, original in enumerate(manifest):
                for region_index, region in enumerate(spec["regions"]):
                    eligible = not (index == 1 and region_index == 1)
                    status = "ok" if eligible else "insufficient_coverage"
                    ref_mean = [10., 20., 40.][index] + region_index
                    bias = -ref_mean if method == "zero" else [1., 2., -1.][index] * (1 if name == "A" else 2)
                    mae = ref_mean if method == "zero" else (abs(bias + 3) + abs(bias - 3)) / 2
                    rmse = ref_mean if method == "zero" else math.sqrt(bias * bias + 9.)
                    values = dict(ref_mean=ref_mean, pred_mean=ref_mean + bias, difference=bias,
                                  absolute_mean_difference=abs(bias), pair_mean=ref_mean + bias / 2, mae=mae, rmse=rmse, bias=bias,
                                  nmse=rmse ** 2 / ref_mean ** 2)
                    if not eligible:
                        values = {key: None for key in values}
                    source = sources[index]
                    row = {key: original[key] for key in ("case_id", "subject_id", "global_subject_id", "cohort")}
                    row.update(method=method, region=region["name"], units="stored", reference_id=config["reference_id"],
                        n_label_voxels=2, n_valid_voxels=2 if eligible else 1, coverage=1. if eligible else .5,
                        status=status, roi_review="pending", label_voxel_counts={str(region_index + 1): 2}, missing_labels=[],
                        roi_spec_sha256=spec_sha, roi_spec_id=spec["id"], region_labels=region["labels"],
                        roi_support_sha256=digest(original["case_id"] + region["name"] + "ROI_mask"),
                        **{key: source[key] for key in compare_roi_models.HASH_FIELDS[1:]},
                        **flags, **values, nmse_status=status)
                    cases.append(row)
                    subject = {key: row[key] for key in ("method", "region", "subject_id", "global_subject_id", "cohort", "units", "reference_id")}
                    subject.update(n_cases_expected=1, n_cases_included=int(eligible), n_valid_voxels=2 if eligible else 0,
                                   status="ok" if eligible else "incomplete_visits", visit_status_counts={status: 1},
                                   **flags, **values, nmse_status="ok" if eligible else "incomplete_visits")
                    subjects.append(subject)
        for method in ("zero", "prediction"):
            for region in spec["regions"]:
                for group_type, group in (("overall", "all"), ("cohort", "cohortA"), ("cohort", "cohortB")):
                    selected = [row for row in subjects if row["method"] == method and row["region"] == region["name"]
                                and (group_type == "overall" or row["cohort"] == group)]
                    usable = sorted((row for row in selected if row["status"] == "ok"), key=lambda row: row["subject_id"])
                    seed = int(digest(method + region["name"] + group)[:8], 16)
                    stats = paired_agreement([row["ref_mean"] for row in usable], [row["pred_mean"] for row in usable], bootstrap_reps=40, seed=seed)
                    header = dict(method=method, region=region["name"], group_type=group_type, group=group, units="stored",
                        reference_id=config["reference_id"], n_participants_expected=len(selected),
                        n_participants_included=len(usable), n_participants_excluded=len(selected) - len(usable),
                        excluded_status_counts=dict(Counter(row["status"] for row in selected if row["status"] != "ok")),
                        roi_review_status_counts={"pending": len(selected)}, **flags)
                    errors = {}
                    for metric in ("mae", "rmse", "bias", "nmse"):
                        numbers = [row[metric] for row in usable if row[metric] is not None]
                        errors[metric] = dict(mean=statistics.mean(numbers) if numbers else None,
                            sd=statistics.stdev(numbers) if len(numbers) > 1 else None, n_defined=len(numbers),
                            n_undefined=len(usable) - len(numbers), mean_status="defined" if numbers else "no_defined_participants",
                            sd_status="defined" if len(numbers) > 1 else "insufficient_n")
                    agreements.append(dict(header, voxel_error_participant_summary=errors, agreement=stats))
                    scalar_summaries = {}
                    for metric in ("ref_mean", "pred_mean", "difference", "absolute_mean_difference"):
                        numbers = [row[metric] for row in usable if row[metric] is not None]
                        scalar_summaries[metric] = dict(mean=statistics.mean(numbers) if numbers else None,
                            sd=statistics.stdev(numbers) if len(numbers) > 1 else None, n_defined=len(numbers),
                            n_undefined=len(usable) - len(numbers), mean_status="defined" if numbers else "no_defined_participants",
                            sd_status="defined" if len(numbers) > 1 else "insufficient_n")
                    agreements[-1]["roi_mean_participant_summary"] = scalar_summaries
                    flat = dict(header, **stats["estimates"], metric_status=stats["metric_status"])
                    for metric, interval in stats["bootstrap"]["metrics"].items():
                        for field in ("lower", "upper", "status", "valid_replicates", "undefined_replicates"):
                            flat[f"{metric}_ci_{field}"] = interval[field]
                    for metric, values in errors.items():
                        flat[f"voxel_{metric}_participant_mean"] = values["mean"]
                        flat[f"voxel_{metric}_participant_sd"] = values["sd"]
                    for metric, values in scalar_summaries.items():
                        flat[f"roi_mean_{metric}_participant_mean"] = values["mean"]
                        flat[f"roi_mean_{metric}_participant_sd"] = values["sd"]
                    summaries.append(flat)
        payload = dict(schema="swixqsm.frozen_roi_agreement", version=1, units="stored", reference_id=config["reference_id"],
            analysis_status="unverified_deferred_roi_candidate", **{key: config.get(key) for key in
                ("data_qc", "data_reference", "data_independence", "prediction_provenance")},
            provenance=dict(specification={"path": "/nonexistent/fixed_spec.json", "sha256": spec_sha, "content": spec},
                            cases=sources, identity="declared_one_to_one"),
            configuration=dict(bootstrap_reps=40, bootstrap_seed=42, review_policy="deferred",
                               visit_policy="all expected visits eligible, then voxel-pooled participant ROI mean", model_forward_or_training=False),
            agreement=agreements)
        for filename, rows in (("roi_per_case.csv", cases), ("roi_per_subject.csv", subjects), ("roi_summary.csv", summaries)):
            self.write_csv(directory, filename, rows)
        (directory / "roi_agreement.json").write_text(json.dumps(payload, allow_nan=False))

    def cli(self, output, *extra):
        return ["--expected-model", "A", "--expected-model", "B", "--evaluation", f"A={self.a}",
                "--evaluation", f"B={self.b}", "--output", str(output), *extra]

    def mutate_json(self, directory, filename, mutate):
        path = directory / filename
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))

    def test_separate_cases_subjects_mean_agreement_and_voxel_error_with_existing_ci(self):
        output = self.root / "comparison"
        manifest = compare_roi_models.main(self.cli(output))
        cases = self.read_csv(output, "all_models_roi_per_case.csv")
        subjects = self.read_csv(output, "all_models_roi_per_subject.csv")
        long = self.read_csv(output, "roi_model_metrics_long.csv")
        wide = self.read_csv(output, "roi_model_metrics_wide.csv")
        self.assertEqual((len(cases), len(subjects)), (24, 24))
        self.assertTrue(all(row["units"] == "stored" and row["qc_status"] == "unverified" for row in cases))
        self.assertTrue(all(row["roi_review_status"] == "pending" and row["formal_accuracy_claim"] == "False" for row in cases))
        point = next(row for row in long if row["model"] == "A" and row["method"] == "prediction"
                     and row["region"] == "RN_L" and row["group_type"] == "overall"
                     and row["statistic_family"] == "roi_mean_agreement" and row["metric"] == "bias")
        self.assertAlmostEqual(float(point["estimate"]), 2 / 3)
        self.assertEqual(point["n"], "3")
        self.assertNotEqual(point["ci_lower"], "")
        voxel = next(row for row in long if row["model"] == "A" and row["method"] == "prediction"
                     and row["region"] == "RN_L" and row["group_type"] == "overall"
                     and row["statistic_family"] == "voxel_error_participant_summary" and row["metric"] == "mae")
        self.assertEqual(float(voxel["mean"]), 3.)
        self.assertEqual(voxel["estimate"], "")
        self.assertEqual(voxel["ci_lower"], "")
        scalar = next(row for row in long if row["model"] == "A" and row["method"] == "prediction"
                      and row["region"] == "RN_L" and row["group_type"] == "overall"
                      and row["statistic_family"] == "roi_mean_participant_summary" and row["metric"] == "absolute_mean_difference")
        self.assertAlmostEqual(float(scalar["mean"]), 4 / 3)
        self.assertTrue(any(row["comparison_series"] == "zero" and row["baseline_repeated_context"] == "True" for row in cases))
        self.assertIn("voxel_mae_participant_mean", wide[0])
        self.assertEqual(manifest["available_models"], ["A", "B"])

    def test_exclusions_and_insufficient_n_stay_null_not_zero_or_accepted(self):
        output = self.root / "missing_roi"
        compare_roi_models.main(self.cli(output))
        cases = self.read_csv(output, "all_models_roi_per_case.csv")
        excluded = next(row for row in cases if row["case_id"] == "c2" and row["region"] == "RN_R")
        self.assertEqual((excluded["status"], excluded["coverage"], excluded["mae"]), ("insufficient_coverage", "0.5", ""))
        long = self.read_csv(output, "roi_model_metrics_long.csv")
        icc = next(row for row in long if row["region"] == "RN_R" and row["group_type"] == "overall"
                   and row["metric"] == "icc_a1")
        self.assertEqual((icc["n"], icc["estimate"]), ("2", ""))
        self.assertIn("insufficient", icc["metric_status"])

    def test_support_and_input_hashes_are_strict_not_just_matching_voxel_counts(self):
        rows = self.read_csv(self.b, "roi_per_case.csv")
        for row in rows:
            if row["case_id"] == "c1" and row["region"] == "RN_L":
                row["roi_support_sha256"] = digest("different same-size support")
        self.write_csv(self.b, "roi_per_case.csv", rows)
        with self.assertRaisesRegex(ValueError, "support or eligibility|Zero ROI baseline"):
            compare_roi_models.main(self.cli(self.root / "support_mismatch"))
        self.assertFalse((self.root / "support_mismatch").exists())

    def test_missing_required_hash_and_specification_mismatch_reject(self):
        rows = self.read_csv(self.b, "roi_per_case.csv")
        original = [dict(row) for row in rows]
        rows[0]["reference_array_sha256"] = ""
        self.write_csv(self.b, "roi_per_case.csv", rows)
        with self.assertRaisesRegex(ValueError, "hash"):
            compare_roi_models.main(self.cli(self.root / "missing_hash"))
        self.write_csv(self.b, "roi_per_case.csv", original)
        new_hash = digest("different specification")
        for row in original:
            row["roi_spec_sha256"] = new_hash
        self.write_csv(self.b, "roi_per_case.csv", original)
        self.mutate_json(self.b, "roi_agreement.json", lambda value: value["provenance"]["specification"].update(sha256=new_hash))
        with self.assertRaisesRegex(ValueError, "specification hash"):
            compare_roi_models.main(self.cli(self.root / "different_spec"))

    def test_case_region_side_and_cross_grid_native_outputs_cannot_be_mixed(self):
        self.mutate_json(self.b, "roi_agreement.json", lambda value: value.update(schema="swixqsm.dual_native_roi_means"))
        with self.assertRaisesRegex(ValueError, "schema"):
            compare_roi_models.main(self.cli(self.root / "dual_native"))
        self.add_roi("B", self.b)
        rows = self.read_csv(self.b, "roi_per_case.csv")
        rows.pop()
        self.write_csv(self.b, "roi_per_case.csv", rows)
        with self.assertRaisesRegex(ValueError, "membership"):
            compare_roi_models.main(self.cli(self.root / "truncated"))

    def test_summary_ci_and_point_estimate_must_match_json_and_subjects(self):
        rows = self.read_csv(self.b, "roi_summary.csv")
        rows[0]["bias_ci_lower"] = "123.0"
        self.write_csv(self.b, "roi_summary.csv", rows)
        with self.assertRaisesRegex(ValueError, "ROI CI"):
            compare_roi_models.main(self.cli(self.root / "tampered_ci"))

    def test_mixed_checkpoint_and_default_missing_models_are_rejected(self):
        self.mutate_json(self.b, "summary.json", lambda value:
            next(iter(value["config"]["prediction_provenance"].values())).update(SourceCheckpointSHA256=digest("other")))
        with self.assertRaisesRegex(ValueError, "multiple source checkpoints"):
            compare_roi_models.main(self.cli(self.root / "mixed_checkpoint"))
        with self.assertRaisesRegex(ValueError, "Missing expected"):
            compare_roi_models.main(["--evaluation", f"swixqsm_gan={self.a}", "--output", str(self.root / "default_eight")])

    def test_explicit_missing_model_is_not_filled_and_no_overwrite(self):
        self.b.rename(self.root / "B_unavailable")
        output = self.root / "missing"
        with self.assertRaisesRegex(ValueError, "Missing expected"):
            compare_roi_models.main(self.cli(output))
        report = compare_roi_models.main(self.cli(output, "--allow-missing"))
        missing = next(row for row in self.read_csv(output, "roi_model_metrics_long.csv") if row["model"] == "B")
        self.assertEqual((missing["n"], missing["estimate"], missing["mean"]), ("", "", ""))
        self.assertEqual(list(report["missing_models"]), ["B"])
        with self.assertRaises(SystemExit):
            compare_roi_models.main(self.cli(output, "--allow-missing"))

    def test_source_change_cleans_staging_without_publishing(self):
        original = compare_roi_models.sha256
        watched = (self.a / "roi_agreement.json").resolve()
        count = 0
        def changed(path):
            nonlocal count
            if Path(path).resolve() == watched:
                count += 1
                if count > 1:
                    return "changed"
            return original(path)
        output = self.root / "changed"
        with mock.patch.object(compare_roi_models, "sha256", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "changed during export"):
                compare_roi_models.main(self.cli(output))
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".swixqsm-compare-roi-*")), [])


if __name__ == "__main__":
    unittest.main()

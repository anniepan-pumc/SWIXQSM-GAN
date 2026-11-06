"""Compare native evaluate ROI tables without images, fitting or re-selection.

Participant ROI-mean agreement and pointwise voxel errors have separate
statistic_family values. Supplied confidence intervals are copied, never fitted.
Stored-scale/deferred review results retain their original unverified status.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import tempfile

from .compare_models import (FILES as IMAGE_FILES, REVIEW_MODELS, canonical, close,
                             compare_evaluations, integer, load_evaluation, number,
                             read_csv, read_json, role, sha256, write_csv)
from .util.agreement import paired_agreement


ROI_FILES = ("roi_per_case.csv", "roi_per_subject.csv", "roi_summary.csv", "roi_agreement.json")
HASH_FIELDS = ("roi_support_sha256", "evaluation_support_sha256", "reference_array_sha256", "evaluation_grid_sha256")
CASE_SUPPORT = ("subject_id", "global_subject_id", "cohort", "region", "side", "units", "reference_id",
                "n_label_voxels", "n_valid_voxels", "coverage", "status", "roi_review", "roi_review_status",
                "roi_review_policy", "roi_qc_status", "QCStatus", "UnitStatus", "value_space",
                "label_voxel_counts", "missing_labels", "roi_spec_sha256", "roi_spec_id", "region_labels", *HASH_FIELDS)
SUBJECT_SUPPORT = ("global_subject_id", "cohort", "units", "reference_id", "n_cases_expected",
                   "n_cases_included", "n_valid_voxels", "status", "visit_status_counts")
ROI_METRICS = ("ref_mean", "pred_mean", "difference", "absolute_mean_difference", "pair_mean", "mae", "rmse", "bias", "nmse")


def region_key(row):
    # Current specs encode RN_L/R/whole in region.name. Preserve an explicit
    # side column if a future native table supplies one; never infer it.
    return row["region"], row.get("side", "")


def result_key(row):
    return row["method"], *region_key(row), row["group_type"], row["group"]


def require_hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Missing/invalid required ROI hash: " + name)


def load_roi_evaluation(name, directory):
    evaluation = load_evaluation(name, directory)
    evaluation["hashes"].update({filename: sha256(directory / filename) for filename in ROI_FILES})
    payload = read_json(directory / "roi_agreement.json")
    if (not isinstance(payload, dict) or payload.get("schema") != "swixqsm.frozen_roi_agreement"
            or type(payload.get("version")) is not int or payload["version"] != 1):
        raise ValueError("Unsupported ROI agreement schema/version")
    config = evaluation["config"]
    for key in ("units", "reference_id"):
        if payload.get(key) != config[key]:
            raise ValueError("ROI agreement differs from whole-image " + key)
    for key in ("data_qc", "data_reference", "data_independence", "prediction_provenance"):
        if canonical(payload.get(key)) != canonical(config.get(key)):
            raise ValueError("ROI/whole-image provenance differs: " + key)
    provenance = payload.get("provenance")
    specification = (provenance or {}).get("specification")
    if not isinstance(provenance, dict) or not isinstance(specification, dict):
        raise ValueError("ROI provenance/specification is missing")
    overlay = payload.get("roi_manifest_source")
    if canonical(overlay) != canonical(provenance.get("roi_manifest")):
        raise ValueError("ROI overlay manifest provenance is inconsistent")
    if overlay is not None:
        require_hash(overlay.get("sha256"), "ROI overlay manifest sha256")
        if canonical(config.get("roi_manifest_source")) != canonical(overlay):
            raise ValueError("ROI overlay manifest differs from whole-image configuration")
    require_hash(specification.get("sha256"), "specification.sha256")
    spec = specification.get("content")
    if not isinstance(spec, dict) or not isinstance(spec.get("regions"), list) or not spec["regions"]:
        raise ValueError("Bound ROI specification content is missing")
    regions = {entry["name"]: entry["labels"] for entry in spec["regions"]}
    if len(regions) != len(spec["regions"]):
        raise ValueError("Duplicate region names in ROI specification")
    source_by_case = {}
    for source in provenance.get("cases", []):
        case_id = source.get("case_id")
        if case_id not in evaluation["manifest"] or case_id in source_by_case:
            raise ValueError("ROI source provenance has unknown/duplicate cases")
        for field in ("subject_id", "global_subject_id", "cohort"):
            if source.get(field, "") != evaluation["manifest"][case_id].get(field, ""):
                raise ValueError("ROI source participant identity differs from resolved manifest")
        if source.get("label_sha256") is not None:
            require_hash(source["label_sha256"], "label_sha256")
        for field in HASH_FIELDS[1:]:
            require_hash(source.get(field), "source " + field)
        source_by_case[case_id] = source
    if set(source_by_case) != set(evaluation["manifest"]):
        raise ValueError("ROI source provenance does not cover every expected case")
    cases, subjects = read_csv(directory / "roi_per_case.csv"), read_csv(directory / "roi_per_subject.csv")
    summaries = read_csv(directory / "roi_summary.csv")
    methods = set(evaluation["summary"])
    case_index, subject_index, case_regions = {}, {}, set()
    for row in cases:
        for field in ("label_voxel_counts", "missing_labels", "region_labels"):
            if field not in row:
                raise ValueError("ROI table lacks required bound support field: " + field)
            row[field] = json.loads(row[field])
        case_id, method = row.get("case_id"), row.get("method")
        identity = region_key(row)
        key = method, case_id, *identity
        if case_id not in evaluation["manifest"] or method not in methods or key in case_index:
            raise ValueError("Unknown/duplicate ROI case/method/region/side")
        if row["region"] not in regions or row["region_labels"] != regions[row["region"]]:
            raise ValueError("ROI region labels differ from the bound specification")
        if row.get("roi_spec_sha256") != specification["sha256"] or row.get("roi_spec_id") != spec.get("id"):
            raise ValueError("ROI case specification hash/ID differs from provenance")
        for field in HASH_FIELDS:
            require_hash(row.get(field), field)
            if field != "roi_support_sha256" and row[field] != source_by_case[case_id][field]:
                raise ValueError("ROI case/source input or grid hash differs")
        source = evaluation["manifest"][case_id]
        for field in ("subject_id", "global_subject_id", "cohort"):
            if row.get(field, "") != source.get(field, ""):
                raise ValueError("ROI case participant identity differs from the manifest")
        if row.get("units") != config["units"] or row.get("reference_id") != config["reference_id"]:
            raise ValueError("ROI case units/reference differ from configuration")
        labels = integer(row.get("n_label_voxels"), "n_label_voxels")
        valid = integer(row.get("n_valid_voxels"), "n_valid_voxels")
        coverage = number(row.get("coverage"))
        if valid > labels or (coverage is None) != (labels == 0):
            raise ValueError("Invalid ROI coverage/support counts")
        if labels:
            close(coverage, valid / labels, "ROI coverage")
        for metric in ROI_METRICS:
            if metric not in row:
                raise ValueError("ROI case metric column is absent: " + metric)
            value = number(row[metric])
            if row["status"] != "ok" and value is not None:
                raise ValueError("Ineligible ROI case contains reported numeric metrics")
        if row["status"] == "ok":
            if not valid or any(number(row[key]) is None for key in ROI_METRICS if key != "nmse"):
                raise ValueError("Eligible ROI case has missing numeric measurements")
            close(row["difference"], number(row["pred_mean"]) - number(row["ref_mean"]), "ROI mean difference")
            close(row["absolute_mean_difference"], abs(number(row["difference"])), "absolute ROI mean difference")
        case_index[key] = row
        case_regions.add(identity)
    expected = {(method, case_id, *identity) for method in methods for case_id in evaluation["manifest"] for identity in case_regions}
    if set(case_index) != expected or {region for region, side in case_regions} != set(regions):
        raise ValueError("ROI methods must cover identical expected case/region/side membership")
    for method, case_id, region, side in case_index:
        zero, current = case_index[("zero", case_id, region, side)], case_index[(method, case_id, region, side)]
        if any(canonical(current.get(key)) != canonical(zero.get(key)) for key in CASE_SUPPORT):
            raise ValueError("ROI methods use different source/support/review eligibility")
        close(current["ref_mean"], number(zero["ref_mean"]), "ROI reference mean")
    for row in subjects:
        method, sid = row.get("method"), row.get("subject_id")
        identity = region_key(row)
        key = method, sid, *identity
        if method not in methods or key in subject_index:
            raise ValueError("Unknown/duplicate ROI subject/method/region/side")
        visits = [case for case in cases if case["method"] == method and case["subject_id"] == sid and region_key(case) == identity]
        usable = [visit for visit in visits if visit["status"] == "ok"]
        if not visits:
            raise ValueError("ROI subject has no corresponding expected cases")
        for field in ("global_subject_id", "cohort", "units", "reference_id"):
            if row.get(field, "") != visits[0].get(field, ""):
                raise ValueError("ROI subject identity/units/reference differ from visits")
        row["visit_status_counts"] = json.loads(row["visit_status_counts"])
        if (integer(row["n_cases_expected"], "n_cases_expected") != len(visits)
                or integer(row["n_cases_included"], "n_cases_included") != len(usable)
                or row["visit_status_counts"] != dict(Counter(visit["status"] for visit in visits))):
            raise ValueError("ROI subject completeness counts differ from visits")
        for metric in ROI_METRICS:
            value = number(row.get(metric))
            if row["status"] != "ok" and value is not None:
                raise ValueError("Ineligible ROI participant has reported numeric metrics")
        if row["status"] == "ok":
            total = sum(int(visit["n_valid_voxels"]) for visit in usable)
            if len(usable) != len(visits) or integer(row["n_valid_voxels"], "participant support", minimum=1) != total:
                raise ValueError("Eligible ROI participant does not have complete visit support")
            for metric in ("ref_mean", "pred_mean", "difference", "mae", "bias"):
                close(row[metric], sum(number(visit[metric]) * int(visit["n_valid_voxels"]) for visit in usable) / total,
                      "voxel-pooled participant " + metric)
            close(row["absolute_mean_difference"], abs(number(row["difference"])), "participant absolute ROI mean difference")
        subject_index[key] = row
    expected_subjects = {(method, row["subject_id"], *identity) for method in methods
                         for row in evaluation["manifest"].values() for identity in case_regions}
    if set(subject_index) != expected_subjects:
        raise ValueError("ROI participant table is missing expected method/region/side membership")
    summary_index = {}
    for row in summaries:
        key = result_key(row)
        if key in summary_index:
            raise ValueError("Duplicate ROI summary group")
        summary_index[key] = row
    agreement_index = {}
    for row in payload.get("agreement", []):
        key = result_key(row)
        if key in agreement_index or key not in summary_index:
            raise ValueError("Unknown/duplicate ROI agreement group")
        flat = summary_index[key]
        selected = [subject for subject in subjects if subject["method"] == row["method"]
                    and region_key(subject) == region_key(row)
                    and (row["group_type"] == "overall" and row["group"] == "all"
                         or row["group_type"] == "cohort" and subject["cohort"] == row["group"])]
        usable = sorted((subject for subject in selected if subject["status"] == "ok"), key=lambda value: value["subject_id"])
        counts = dict(n_participants_expected=len(selected), n_participants_included=len(usable),
                      n_participants_excluded=len(selected) - len(usable))
        for field, value in counts.items():
            if integer(row[field], field) != value or integer(flat[field], field) != value:
                raise ValueError("ROI agreement/summary participant counts differ from subject table")
        excluded = dict(Counter(subject["status"] for subject in selected if subject["status"] != "ok"))
        if row["excluded_status_counts"] != excluded or json.loads(flat["excluded_status_counts"]) != excluded:
            raise ValueError("ROI agreement exclusions differ from subject table")
        agreement = row["agreement"]
        points = paired_agreement([number(subject["ref_mean"]) for subject in usable],
                                  [number(subject["pred_mean"]) for subject in usable], bootstrap_reps=0)
        if integer(agreement["n_participants"], "agreement n") != len(usable):
            raise ValueError("ROI agreement n differs from eligible participant count")
        flat_status = json.loads(flat["metric_status"])
        for metric, value in agreement["estimates"].items():
            close(value, points["estimates"][metric], "ROI-mean agreement " + metric)
            close(flat.get(metric), number(value), "ROI summary agreement " + metric)
            if agreement["metric_status"][metric] != points["metric_status"][metric] or flat_status[metric] != agreement["metric_status"][metric]:
                raise ValueError("ROI agreement status differs from participant values")
        for metric, interval in agreement.get("bootstrap", {}).get("metrics", {}).items():
            for field in ("lower", "upper"):
                close(flat.get(f"{metric}_ci_{field}"), number(interval[field]), "ROI CI " + metric + " " + field)
            for field in ("status", "valid_replicates", "undefined_replicates"):
                value = flat.get(f"{metric}_ci_{field}")
                if (str(value) != str(interval[field])):
                    raise ValueError("ROI summary CI status/count differs from agreement JSON")
        for family, prefix in (("voxel_error_participant_summary", "voxel"), ("roi_mean_participant_summary", "roi_mean")):
            for metric, values in row.get(family, {}).items():
                defined = [number(subject[metric]) for subject in usable if number(subject[metric]) is not None]
                if (integer(values["n_defined"], "descriptive metric n") != len(defined)
                        or integer(values["n_undefined"], "descriptive metric missing") != len(usable) - len(defined)):
                    raise ValueError("ROI descriptive metric counts differ from participants")
                close(values["mean"], statistics.mean(defined) if defined else None, family + " mean " + metric)
                close(values["sd"], statistics.stdev(defined) if len(defined) > 1 else None, family + " SD " + metric)
                close(flat.get(f"{prefix}_{metric}_participant_mean"), number(values["mean"]), "ROI summary descriptive mean")
                close(flat.get(f"{prefix}_{metric}_participant_sd"), number(values["sd"]), "ROI summary descriptive SD")
        agreement_index[key] = row
    expected_groups = {(method, region, side, "overall", "all") for method in methods for region, side in case_regions}
    expected_groups |= {(method, region, side, "cohort", row.get("cohort", "")) for method in methods
                        for region, side in case_regions for row in evaluation["manifest"].values()}
    if set(agreement_index) != expected_groups or set(summary_index) != expected_groups:
        raise ValueError("ROI agreement/summary is missing expected region/group/side membership")
    evaluation.update(roi_payload=payload, roi_cases=cases, roi_subjects=subjects, roi_summaries=summaries,
                      roi_case_index=case_index, roi_subject_index=subject_index, roi_agreement_index=agreement_index,
                      roi_summary_index=summary_index, roi_spec=specification, roi_source_cases=source_by_case)
    return evaluation


def compare_roi_evaluations(evaluations):
    compare_evaluations(evaluations)
    first = evaluations[0]
    for current in evaluations:
        for field in ("sha256", "content"):
            if canonical(current["roi_spec"][field]) != canonical(first["roi_spec"][field]):
                raise ValueError("Incomparable ROI specification hash/content")
        if canonical(current["roi_payload"].get("roi_manifest_source")) != canonical(first["roi_payload"].get("roi_manifest_source")):
            raise ValueError("Incomparable ROI overlay manifest hash/source")
        if canonical(current["roi_source_cases"]) != canonical(first["roi_source_cases"]):
            raise ValueError("Incomparable ROI label/input/reference/support provenance hashes")
        if canonical(current["roi_payload"].get("configuration")) != canonical(first["roi_payload"].get("configuration")):
            raise ValueError("Incomparable ROI aggregation/bootstrap/review configuration")
        if set(current["roi_case_index"]) != set(first["roi_case_index"]):
            raise ValueError("Incomparable ROI case/region/side membership")
        for key, reference in first["roi_case_index"].items():
            row = current["roi_case_index"][key]
            if any(canonical(row.get(field)) != canonical(reference.get(field)) for field in CASE_SUPPORT):
                raise ValueError("Incomparable ROI case region/side support or eligibility")
            close(row["ref_mean"], number(reference["ref_mean"]), "cross-model ROI reference mean")
            if key[0] == "zero" and canonical(row) != canonical(reference):
                raise ValueError("Zero ROI baseline differs across evaluation models")
        for key, reference in first["roi_subject_index"].items():
            row = current["roi_subject_index"].get(key)
            if row is None or any(canonical(row.get(field)) != canonical(reference.get(field)) for field in SUBJECT_SUPPORT):
                raise ValueError("Incomparable ROI participant eligibility/support")


def tables(evaluations, missing):
    cases, subjects, long_rows, wide_rows = [], [], [], []
    for evaluation in evaluations:
        config, payload, name = evaluation["config"], evaluation["roi_payload"], evaluation["name"]
        common = dict(model=name, evaluation_model=name, result_status="available", split=config["split"],
                      **evaluation["flags"], roi_analysis_status=payload.get("analysis_status", "not_recorded"),
                      roi_review_policy=payload.get("roi_review_policy", payload.get("configuration", {}).get("roi_review_policy",
                          payload.get("configuration", {}).get("review_policy", config.get("roi_review_policy", "accepted")))),
                      source_evaluation_dir=str(evaluation["directory"]), source_roi_agreement_sha256=evaluation["hashes"]["roi_agreement.json"],
                      roi_spec_sha256=evaluation["roi_spec"]["sha256"], roi_spec_id=evaluation["roi_spec"]["content"]["id"])
        def context(source):
            method_role = role(source["method"], evaluation["prediction_name"])
            return dict(common, method_role=method_role,
                        comparison_series=name if method_role == "prediction" else method_role,
                        baseline_repeated_context=method_role != "prediction")
        for source in evaluation["roi_cases"]:
            cases.append({**context(source), **source,
                "measurement_scope": "ref_mean/pred_mean/difference compare ROI means; mae/rmse/bias/nmse are pointwise voxel errors on ROI support"})
        for source in evaluation["roi_subjects"]:
            subjects.append({**context(source), **source,
                "measurement_scope": "Complete visits voxel-pooled within participant/ROI; no hemisphere or ROI counted as an independent participant"})
        for key, source in evaluation["roi_agreement_index"].items():
            header = {key: value for key, value in source.items()
                      if key not in {"agreement", "voxel_error_participant_summary", "roi_mean_participant_summary"}}
            base = {**context(source), **header, "data_range": config["data_range"]}
            agreement = source["agreement"]
            bootstrap = agreement.get("bootstrap", {})
            for metric, value in agreement["estimates"].items():
                row = dict(base, statistic_family="roi_mean_agreement", metric=metric, n=agreement["n_participants"],
                           estimate=value, metric_status=agreement["metric_status"][metric],
                           metric_units="dimensionless" if metric in {"pearson_r", "icc_a1"} else config["units"])
                interval = bootstrap.get("metrics", {}).get(metric)
                if interval is not None:
                    row.update({"ci_" + field: value for field, value in interval.items()})
                    row.update(ci_method=bootstrap.get("method"), ci_confidence=bootstrap.get("confidence"))
                long_rows.append(row)
            for family in ("voxel_error_participant_summary", "roi_mean_participant_summary"):
                for metric, values in source.get(family, {}).items():
                    long_rows.append(dict(base, statistic_family=family, metric=metric,
                        n=values["n_defined"], n_undefined=values["n_undefined"], mean=values["mean"], std=values["sd"],
                        mean_status=values.get("mean_status"), std_status=values.get("sd_status"),
                        metric_units="dimensionless" if metric == "nmse" else config["units"]))
            wide_rows.append({**context(source), **evaluation["roi_summary_index"][key],
                "data_range": config["data_range"],
                "measurement_scope": "bias/LoA/ICC/r compare participant ROI means; voxel_* columns summarize pointwise voxel errors"})
    for name, reason in missing.items():
        row = dict(model=name, evaluation_model=name, result_status="missing", missing_reason=reason,
                   method_role="prediction", group_type="overall", group="all")
        long_rows.append(dict(row, statistic_family="missing_evaluation", metric="", n=None, estimate=None, mean=None, std=None))
        wide_rows.append(row)
    return cases, subjects, long_rows, wide_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", default=[], metavar="NAME=DIRECTORY")
    parser.add_argument("--expected-model", action="append", help="Repeat to override the default eight reviewer recipes")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output", required=True, help="New atomically published directory")
    args = parser.parse_args(argv)
    expected = args.expected_model or list(REVIEW_MODELS)
    if len(expected) != len(set(expected)):
        parser.error("Expected model labels must be unique")
    supplied, directories = {}, set()
    for value in args.evaluation:
        name, separator, path = value.partition("=")
        if not separator or not path.strip() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name):
            parser.error("Use --evaluation NAME=DIRECTORY with a safe model label")
        directory = Path(path).expanduser().resolve()
        if name in supplied or directory in directories:
            parser.error("Duplicate model label or evaluation directory")
        supplied[name] = directory
        directories.add(directory)
    if set(supplied) - set(expected):
        parser.error("Unexpected model label(s): " + ", ".join(sorted(set(supplied) - set(expected))))
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error("ROI comparison output already exists")
    evaluations, missing = [], {}
    for name in expected:
        directory = supplied.get(name)
        absent = list(IMAGE_FILES + ROI_FILES) if directory is None else [filename for filename in IMAGE_FILES + ROI_FILES if not (directory / filename).is_file()]
        if absent:
            missing[name] = "not supplied" if directory is None else "missing files: " + ", ".join(absent)
        else:
            evaluations.append(load_roi_evaluation(name, directory))
    if missing and not args.allow_missing:
        raise ValueError("Missing expected ROI model evaluations: " + canonical(missing))
    if not evaluations:
        raise ValueError("At least one complete ROI evaluation is required")
    compare_roi_evaluations(evaluations)
    results = tables(evaluations, missing)
    filenames = ("all_models_roi_per_case.csv", "all_models_roi_per_subject.csv", "roi_model_metrics_long.csv", "roi_model_metrics_wide.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".swixqsm-compare-roi-", dir=output.parent) as temporary:
        staging = Path(temporary) / "results"
        staging.mkdir()
        for filename, rows in zip(filenames, results):
            write_csv(staging / filename, rows)
        for evaluation in evaluations:
            for filename, digest in evaluation["hashes"].items():
                if sha256(evaluation["directory"] / filename) != digest:
                    raise RuntimeError("ROI evaluation changed during export: " + str(evaluation["directory"] / filename))
        manifest = dict(schema="swixqsm.roi_model_comparison", version=1, expected_models=expected,
            available_models=[evaluation["name"] for evaluation in evaluations], missing_models=missing,
            sources=[dict(model=evaluation["name"], directory=str(evaluation["directory"]),
                file_sha256=evaluation["hashes"], config=evaluation["config"], flags=evaluation["flags"],
                roi_configuration=evaluation["roi_payload"].get("configuration"), roi_provenance=evaluation["roi_payload"]["provenance"],
                roi_analysis_status=evaluation["roi_payload"].get("analysis_status")) for evaluation in evaluations],
            row_counts={filename: len(rows) for filename, rows in zip(filenames, results)},
            checks="Exact case/participant/region/side, unit/reference/data-range, specification/label/target/grid/common/ROI-support hash and eligibility matching; native CSV/JSON consistency",
            agreement="Participant ROI-mean bias, LoA, Pearson r and ICC(A,1); retain source estimates, statuses and existing paired-participant CIs",
            voxel_errors="MAE/RMSE/bias/NMSE are pointwise ROI voxel errors, pooled by visit then summarized equally by participant; separate statistic_family",
            repeated_zero_baseline="The same fixed zero baseline remains in evaluation_model contexts for provenance, not independent estimators",
            limitations=["No images, model inference, training, new confidence intervals or checkpoint selection occur here.",
                "Hash-bound supports and numeric comparisons do not establish anatomical correctness, expert acceptance, clinical validity, units or reference truth.",
                "Deferred or stored-scale results retain source unverified flags; null/undefined/excluded measurements are never replaced with zero.",
                "Region sides are defined by specification names unless an explicit side column is provided; hemispheres/subnuclei are dependent."])
        (staging / "roi_comparison_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError("ROI comparison output appeared; refusing to replace it")
        staging.rename(output)
    print(f"Compared ROI results for {len(evaluations)} models; {len(missing)} explicitly missing; {output}")
    return manifest


if __name__ == "__main__":
    main()

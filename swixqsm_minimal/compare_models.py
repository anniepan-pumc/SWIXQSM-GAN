"""Compare saved evaluate outputs without reading images or choosing checkpoints.

The main summaries retain evaluate's equal-subject mean and sample SD. No
confidence intervals or missing results are invented. Unverified/stored-scale
analyses remain explicitly unverified, even when their settings match.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tempfile


REVIEW_MODELS = ("swixqsm_gan", "fcn_gan_l1", "plain_unet", "pix2pix", "swixqsm_l1_ablation",
                 "swixqsm_lambda1", "swixqsm_lambda10", "swixqsm_lambda100")
FILES = ("summary.json", "per_case.csv", "per_subject.csv", "manifest_resolved.json")
SETTINGS = ("split", "units", "reference_id", "data_range", "ssim_window", "hfen_sigma_mm",
            "space_id", "spatial_units", "aggregation", "zero_baseline", "calibration", "metrics_parameters")
REQUIRED_SETTINGS = {"split", "units", "reference_id", "data_range", "ssim_window", "hfen_sigma_mm",
                     "aggregation", "zero_baseline", "calibration", "metrics_parameters"}
SUPPORT_FIELDS = ("subject_id", "cohort", "units", "reference_id", "n_voxels", "ssim_centers",
                  "hfen_centers", "ref_mean", "ref_std", "parameters", "metric_units")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def bad_constant(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(), parse_constant=bad_constant)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV has missing/duplicate column names: " + str(path))
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV must contain complete rows: " + str(path))
    return rows


def number(value):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean metric is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite metric value")
    return result


def integer(value, label, *, minimum=0):
    numeric = number(value)
    if numeric is None or numeric < minimum or not numeric.is_integer():
        raise ValueError(label + " must be a nonnegative integer")
    return int(numeric)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def role(method, prediction_name):
    return "prediction" if method == prediction_name else method


def close(actual, expected, label):
    actual = number(actual)
    if ((actual is None) != (expected is None)
            or actual is not None and not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError("Summary/per-subject inconsistency: " + label)


def load_evaluation(name, directory):
    hashes = {filename: sha256(directory / filename) for filename in FILES}
    payload = read_json(directory / "summary.json")
    config, summary = payload.get("config"), payload.get("summary")
    if not isinstance(config, dict) or not isinstance(summary, dict) or not summary:
        raise ValueError("Expected evaluate summary.json config/summary objects")
    if not REQUIRED_SETTINGS.issubset(config):
        raise ValueError("Evaluation lacks comparison settings: " + ", ".join(sorted(REQUIRED_SETTINGS - set(config))))
    if config["units"] not in {"ppm", "ppb", "stored"} or not config["reference_id"]:
        raise ValueError("Explicit stored/ppm/ppb units and reference ID are required")
    for key in ("data_range", "ssim_window", "hfen_sigma_mm"):
        if config["metrics_parameters"].get(key) != config[key]:
            raise ValueError("Internal metric setting disagrees with metrics_parameters: " + key)
    prediction_name = config.get("prediction_name", "prediction")
    if prediction_name not in summary or "zero" not in summary:
        raise ValueError("Each evaluation requires its declared prediction and zero baseline")
    cases, subjects = read_csv(directory / "per_case.csv"), read_csv(directory / "per_subject.csv")
    manifest = read_json(directory / "manifest_resolved.json")
    if not isinstance(manifest, list) or not manifest or any(not isinstance(row, dict) for row in manifest):
        raise ValueError("Expected nonempty manifest_resolved.json row array")
    manifest_by_case = {}
    for row in manifest:
        case_id = row.get("case_id")
        if not case_id or case_id in manifest_by_case or row.get("split") != config["split"]:
            raise ValueError("Resolved manifest needs unique case IDs in the selected split")
        for field in ("subject_id", "swi_path", "qsm_path", "swi_mask_path", "mask_path"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError("Resolved manifest lacks " + field)
        manifest_by_case[case_id] = row
    methods = set(summary)
    by_method, case_index = defaultdict(list), {}
    metric_units = {}
    for row in cases:
        method, case_id = row.get("method"), row.get("case_id")
        if method not in methods or case_id not in manifest_by_case or (method, case_id) in case_index:
            raise ValueError("Unknown/duplicate per-case method or identity")
        source = manifest_by_case[case_id]
        if row.get("subject_id") != source["subject_id"] or row.get("cohort", "") != source.get("cohort", ""):
            raise ValueError("Per-case identity differs from the resolved manifest")
        if row.get("units") != config["units"] or row.get("reference_id") != config["reference_id"]:
            raise ValueError("Per-case units/reference differ from summary configuration")
        for field in ("n_voxels", "ssim_centers", "hfen_centers"):
            integer(row.get(field), field, minimum=int(field == "n_voxels"))
        row["parameters"] = json.loads(row["parameters"])
        row["metric_units"] = json.loads(row["metric_units"])
        row["metric_status"] = json.loads(row["metric_status"])
        parameters = {key: value for key, value in row["parameters"].items()
                      if key not in {"spacing_mm", "hfen_radius_voxels"}}
        if parameters != config["metrics_parameters"]:
            raise ValueError("Per-case metric parameters differ from configuration")
        metric_units.update(row["metric_units"])
        case_index[(method, case_id)] = row
        by_method[method].append(row)
    expected_case_ids = set(manifest_by_case)
    for method in methods:
        if {row["case_id"] for row in by_method[method]} != expected_case_ids:
            raise ValueError("A method is missing evaluation cases; no intersection/subsampling is allowed")
        for case_id in expected_case_ids:
            reference_row = case_index[("zero", case_id)]
            compared = case_index[(method, case_id)]
            if any(canonical(compared.get(key)) != canonical(reference_row.get(key)) for key in SUPPORT_FIELDS):
                raise ValueError("Methods use different reference/metric support within one evaluation")
    subject_index = {}
    for row in subjects:
        method, sid = row.get("method"), row.get("subject_id")
        if method not in methods or (method, sid) in subject_index:
            raise ValueError("Unknown/duplicate per-subject method or identity")
        visits = [case for case in by_method[method] if case["subject_id"] == sid]
        if (not visits or integer(row.get("n_cases"), "n_cases", minimum=1) != len(visits)
                or integer(row.get("n_voxels"), "n_voxels", minimum=1) != sum(int(case["n_voxels"]) for case in visits)):
            raise ValueError("Per-subject visits/voxel support do not match per-case rows")
        if row.get("cohort", "") != ";".join(sorted({case["cohort"] for case in visits})):
            raise ValueError("Per-subject cohort membership does not match cases")
        if row.get("units") != config["units"] or row.get("reference_id") != config["reference_id"]:
            raise ValueError("Per-subject units/reference differ from configuration")
        row["metric_status"] = json.loads(row["metric_status"])
        subject_index[(method, sid)] = row
    groups_by_role, metrics_by_role = {}, {}
    for method, groups in summary.items():
        selected = [row for row in subjects if row["method"] == method]
        if {row["subject_id"] for row in selected} != {row["subject_id"] for row in by_method[method]}:
            raise ValueError("Per-subject table is missing participants")
        expected_groups = {"all"} | {"cohort:" + row["cohort"] for row in selected if row["cohort"]}
        if set(groups) != expected_groups:
            raise ValueError("Summary groups differ from per-subject membership")
        groups_by_role[role(method, prediction_name)] = sorted(groups)
        for group, result in groups.items():
            people = selected if group == "all" else [row for row in selected if "cohort:" + row["cohort"] == group]
            if (integer(result.get("n_subjects"), "n_subjects") != len(people)
                    or integer(result.get("n_cases"), "n_cases") != sum(int(row["n_cases"]) for row in people)):
                raise ValueError("Summary sample counts differ from per-subject rows")
            metrics = result.get("metrics")
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError("Summary metrics are missing")
            metrics_by_role[(role(method, prediction_name), group)] = sorted(metrics)
            for metric, values in metrics.items():
                if not isinstance(values, dict) or not {"mean", "sd", "n_defined", "n_undefined", "undefined_reasons"}.issubset(values):
                    raise ValueError("Incomplete summary statistics: " + metric)
                if any(metric not in row or metric not in row["metric_status"] for row in people):
                    raise ValueError("Per-subject metric or missingness status is absent: " + metric)
                defined = [number(row[metric]) for row in people if number(row[metric]) is not None]
                missing = Counter(row["metric_status"][metric] for row in people if number(row[metric]) is None)
                if (integer(values["n_defined"], "n_defined") != len(defined)
                        or integer(values["n_undefined"], "n_undefined") != len(people) - len(defined)
                        or values["undefined_reasons"] != dict(missing)):
                    raise ValueError("Summary metric counts/missingness differ from per-subject rows")
                close(values["mean"], statistics.mean(defined) if defined else None, metric + " mean")
                close(values["sd"], statistics.stdev(defined) if len(defined) > 1 else None, metric + " sample SD")
    metadata_rows = [{key: value for key, value in row.items() if not key.startswith("prediction_")}
                     for row in manifest]
    identity = sorted(metadata_rows, key=lambda row: row["case_id"])
    provenance = config.get("prediction_provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Prediction provenance must be an object")
    checkpoint_shas = sorted({value.get("SourceCheckpointSHA256") for value in provenance.values()
                              if isinstance(value, dict) and value.get("SourceCheckpointSHA256")})
    if len(checkpoint_shas) > 1:
        raise ValueError("One model evaluation contains multiple source checkpoints")
    qc = config.get("data_qc") or {}
    independence = config.get("data_independence") or {}
    flags = dict(data_qc_status=qc.get("status", "not_recorded"), data_qc_reason=qc.get("reason", ""),
                 qc_status=qc.get("status", "not_recorded"),
                 data_independence_status=independence.get("status", "not_recorded"),
                 reference_provenance_status="verified" if qc.get("status") == "verified" and config.get("data_reference") else "unverified",
                 analysis_status=config.get("analysis_status") or ("unverified_data" if qc.get("status") != "verified" else "QC_verified"),
                 model_selection_status=(config.get("model_selection") or {}).get("status", "not_recorded"),
                 source_checkpoint_sha256=checkpoint_shas[0] if checkpoint_shas else "not_recorded",
                 prediction_qc_statuses=sorted({(value.get("PredictionDataQC") or {}).get("status", "not_recorded")
                     for value in provenance.values() if isinstance(value, dict)}),
                 prediction_independence_statuses=sorted({(value.get("PredictionDataIndependence") or {}).get("status", "not_recorded")
                     for value in provenance.values() if isinstance(value, dict)}),
                 prediction_phases=sorted({(value.get("PredictionTrainingState") or {}).get("current_run_phase", "not_recorded")
                     for value in provenance.values() if isinstance(value, dict)}))
    return dict(name=name, directory=directory, hashes=hashes, payload=payload, config=config, summary=summary,
                cases=cases, subjects=subjects, manifest=manifest_by_case, identity=identity,
                prediction_name=prediction_name, groups=groups_by_role, metrics=metrics_by_role,
                case_index=case_index, flags=flags, metric_units=metric_units)


def compare_evaluations(evaluations):
    first = evaluations[0]
    used_checkpoints = {}
    known_references = set()
    for current in evaluations:
        for key in SETTINGS:
            if canonical(current["config"].get(key)) != canonical(first["config"].get(key)):
                raise ValueError("Incomparable evaluation setting: " + key)
        if canonical(current["identity"]) != canonical(first["identity"]):
            raise ValueError("Incomparable selected case/participant/source-mask identity metadata")
        if current["groups"] != first["groups"] or current["metrics"] != first["metrics"]:
            raise ValueError("Incomparable method, cohort group or metric sets")
        for case_id in first["manifest"]:
            if canonical(current["case_index"][("zero", case_id)]) != canonical(first["case_index"][("zero", case_id)]):
                raise ValueError("Incomparable zero-baseline reference/support statistics: " + case_id)
        known = current["flags"]["source_checkpoint_sha256"]
        if known != "not_recorded":
            if known in used_checkpoints:
                raise ValueError("The same source checkpoint is labelled as multiple models: " + used_checkpoints[known] + ", " + current["name"])
            used_checkpoints[known] = current["name"]
        reference = current["config"].get("data_reference")
        if reference is not None:
            known_references.add(canonical(reference))
    if len(known_references) > 1:
        raise ValueError("Incomparable full QSM reference provenance")


def model_tables(evaluations, missing):
    case_rows, subject_rows, long_rows, wide_rows = [], [], [], []
    for evaluation in evaluations:
        config, name = evaluation["config"], evaluation["name"]
        common = dict(model=name, evaluation_model=name, result_status="available", split=config["split"],
                      **evaluation["flags"], source_evaluation_dir=str(evaluation["directory"]),
                      source_summary_sha256=evaluation["hashes"]["summary.json"])
        for source in evaluation["cases"]:
            row = dict(common, **source, method_role=role(source["method"], evaluation["prediction_name"]))
            row.update(comparison_series=name if row["method_role"] == "prediction" else row["method_role"],
                       baseline_repeated_context=row["method_role"] != "prediction")
            row["global_subject_id"] = evaluation["manifest"][source["case_id"]].get("global_subject_id", "")
            case_rows.append(row)
        for source in evaluation["subjects"]:
            row = dict(common, **source, method_role=role(source["method"], evaluation["prediction_name"]))
            row.update(comparison_series=name if row["method_role"] == "prediction" else row["method_role"],
                       baseline_repeated_context=row["method_role"] != "prediction")
            globals_ = sorted({entry.get("global_subject_id", "") for entry in evaluation["manifest"].values()
                               if entry["subject_id"] == source["subject_id"]})
            row["global_subject_ids"] = globals_
            row["aggregation_identity"] = "subject_id; preserved from evaluate, no regrouping here"
            subject_rows.append(row)
        for method, groups in evaluation["summary"].items():
            for group, result in groups.items():
                base = dict(common, method=method, method_role=role(method, evaluation["prediction_name"]),
                            group=group, units=config["units"], reference_id=config["reference_id"],
                            data_range=config["data_range"], n_subjects=result["n_subjects"], n_cases=result["n_cases"])
                base.update(comparison_series=name if base["method_role"] == "prediction" else base["method_role"],
                            baseline_repeated_context=base["method_role"] != "prediction")
                wide = dict(base)
                for metric, values in result["metrics"].items():
                    statistics_ = dict(n=values["n_defined"], n_undefined=values["n_undefined"],
                                       mean=values["mean"], std=values["sd"], undefined_reasons=values["undefined_reasons"])
                    # Preserve any genuinely supplied CI/extra statistic verbatim;
                    # current evaluate supplies no global confidence intervals.
                    statistics_.update({key: value for key, value in values.items()
                                        if key not in {"n_defined", "n_undefined", "mean", "sd", "undefined_reasons"}})
                    fallback_units = ("dB" if metric == "psnr_db" else "dimensionless"
                        if metric in {"ssim", "nmse", "hfen_relative"} else config["units"]
                        if metric in {"ref_mean", "pred_mean", "ref_std", "pred_std"} else "not_recorded")
                    long_rows.append(dict(base, metric=metric,
                        metric_units=evaluation["metric_units"].get(metric, fallback_units), **statistics_))
                    wide.update({metric + "__" + key: value for key, value in statistics_.items()})
                wide_rows.append(wide)
    for name, reason in missing.items():
        row = dict(model=name, result_status="missing", missing_reason=reason, method_role="prediction", group="all")
        long_rows.append(dict(row, metric="", n=None, mean=None, std=None))
        wide_rows.append(row)
    return case_rows, subject_rows, long_rows, wide_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", default=[], metavar="NAME=DIRECTORY")
    parser.add_argument("--expected-model", action="append", help="Expected label; repeat to override the default eight reviewer recipes")
    parser.add_argument("--allow-missing", action="store_true", help="Explicitly list missing evaluations; never fill metrics with zero")
    parser.add_argument("--output", required=True, help="New comparison directory")
    args = parser.parse_args(argv)
    expected = args.expected_model or list(REVIEW_MODELS)
    if len(expected) != len(set(expected)):
        parser.error("Expected model labels must be unique")
    supplied, directories = {}, set()
    for value in args.evaluation:
        name, separator, raw_path = value.partition("=")
        if not separator or not raw_path.strip() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name):
            parser.error("Use --evaluation NAME=DIRECTORY with a safe, nonempty model label")
        path = Path(raw_path).expanduser().resolve()
        if name in supplied or path in directories:
            parser.error("Duplicate model label or evaluation directory; do not select among repeat runs")
        supplied[name] = path
        directories.add(path)
    if set(supplied) - set(expected):
        parser.error("Unexpected model label(s): " + ", ".join(sorted(set(supplied) - set(expected))))
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error("Comparison output already exists")
    missing, evaluations = {}, []
    for name in expected:
        directory = supplied.get(name)
        absent = list(FILES) if directory is None else [filename for filename in FILES if not (directory / filename).is_file()]
        if absent:
            missing[name] = "not supplied" if directory is None else "missing files: " + ", ".join(absent)
        else:
            evaluations.append(load_evaluation(name, directory))
    if missing and not args.allow_missing:
        raise ValueError("Missing expected model evaluations: " + canonical(missing))
    if not evaluations:
        raise ValueError("At least one complete evaluation is required")
    compare_evaluations(evaluations)
    tables = model_tables(evaluations, missing)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".swixqsm-compare-models-", dir=output.parent) as temporary:
        staging = Path(temporary) / "results"
        staging.mkdir()
        for filename, rows in zip(("all_models_per_case.csv", "all_models_per_subject.csv", "model_metrics_long.csv", "model_metrics_wide.csv"), tables):
            write_csv(staging / filename, rows)
        for evaluation in evaluations:
            for filename, digest in evaluation["hashes"].items():
                if sha256(evaluation["directory"] / filename) != digest:
                    raise RuntimeError("Evaluation changed while comparing: " + str(evaluation["directory"] / filename))
        provenance = dict(schema="swixqsm.model_comparison", version=1, expected_models=expected,
            available_models=[evaluation["name"] for evaluation in evaluations], missing_models=missing,
            settings={key: evaluations[0]["config"].get(key) for key in SETTINGS},
            sources=[dict(model=evaluation["name"], directory=str(evaluation["directory"]),
                          file_sha256=evaluation["hashes"], config=evaluation["config"], flags=evaluation["flags"])
                     for evaluation in evaluations],
            aggregation="Retain evaluate's per-subject equal-weight mean and sample SD; no case averaging or re-selection",
            confidence_intervals="Only copy statistics present in source summary; no CI fitted here",
            case_identity="Exact selected case/subject/global-ID and non-prediction manifest metadata match",
            support_check="Exact source-mask paths, per-case support counts/parameters/reference statistics and zero-baseline rows match",
            repeated_zero_baseline="One identical zero baseline is retained in each evaluation_model context for provenance; these rows are not independent baseline estimators",
            limitations=["Matching metadata/statistics cannot prove unchanged image bytes or anatomical alignment when QC is unverified.",
                         "Model labels are explicit caller declarations; missing checkpoint provenance remains not_recorded.",
                         "subject_id aggregation is preserved from evaluate; this script does not merge aliases using global_subject_id.",
                         "No image is read, no model is trained, no checkpoint is selected, and no units are converted."],
            row_counts={filename: len(rows) for filename, rows in zip(
                ("all_models_per_case.csv", "all_models_per_subject.csv", "model_metrics_long.csv", "model_metrics_wide.csv"), tables)})
        (staging / "comparison_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Comparison output appeared; refusing to replace it")
        staging.rename(output)
    print(f"Compared {len(evaluations)} models; {len(missing)} explicitly missing; {output}")
    return provenance


if __name__ == "__main__":
    main()

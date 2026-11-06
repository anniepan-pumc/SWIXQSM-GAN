"""Apply the preregistered internal criteria for replacing the primary model.

Inputs are two predict_split outputs for the same split. Reference QSM,
evaluation masks and both predictions are read once; participant-level
endpoints are compared by paired bootstrap against thresholds taken only from
the hash-pinned preregistration file. No checkpoint is selected and no
threshold can be supplied on the command line.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from .util.data import load_volume, read_manifest, require_grid
from .util.evaluation import resolve_prediction_paths, spacing_mm


SCHEMA = "swixqsm.primary_model_preregistration"
MODELS = ("candidate", "comparator")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_preregistration(path, expected_sha256):
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"Preregistration hash mismatch: expected {expected_sha256}, found {actual}")
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA or data.get("version") != 1:
        raise ValueError("Unsupported preregistration schema/version")
    internal = data["internal"]
    criteria = internal["criteria"]
    return data, actual, dict(
        split=internal["split"],
        replicates=int(internal["bootstrap"]["replicates"]),
        seed=int(internal["bootstrap"]["seed"]),
        confidence=float(internal["bootstrap"]["confidence"]),
        sigma_mm=float(internal["high_frequency"]["sigma_mm"]),
        interior_weight=float(internal["high_frequency"]["interior_weight"]),
        mae_margin_fraction=float(criteria["A_mae_noninferiority"]["mae_margin_fraction_of_gain"]),
        hf_correlation_margin=float(criteria["C_no_loss_of_correct_detail"]["hf_correlation_margin"]),
    )


def split_rows(manifest, split):
    rows = [row for row in read_manifest(manifest, validate_inputs=False) if row["split"] == split]
    if not rows:
        raise ValueError(f"No {split} rows in {manifest}")
    rows = resolve_prediction_paths(rows, manifest)
    if not all(row.get("prediction_path") for row in rows):
        raise ValueError(f"{manifest} lacks predictions for every {split} row")
    for row in rows:
        if not row.get("qsm_units") or row.get("prediction_units") != row["qsm_units"]:
            raise ValueError("Prediction and reference storage units must be declared and identical: " + row["subject_id"])
    return rows


def paired_rows(candidate_manifest, comparator_manifest, split):
    candidate = split_rows(candidate_manifest, split)
    comparator = split_rows(comparator_manifest, split)
    keys = ("subject_id", "swi_path", "qsm_path", "mask_path", "qsm_units")

    def index(rows):
        result = {}
        for position, row in enumerate(rows):
            case_id = row.get("case_id") or f"{row['subject_id']}_row{position + 1:06d}"
            if case_id in result:
                raise ValueError("Duplicate case_id: " + case_id)
            result[case_id] = row
        return result

    candidate, comparator = index(candidate), index(comparator)
    if set(candidate) != set(comparator):
        raise ValueError("Candidate and comparator cover different cases")
    pairs = []
    for case_id in sorted(candidate):
        a, b = candidate[case_id], comparator[case_id]
        if any(a[key] != b[key] for key in keys):
            raise ValueError("Candidate and comparator disagree on case inputs: " + case_id)
        if a.get("prediction_reference", "") != b.get("prediction_reference", ""):
            raise ValueError("Candidate and comparator declare different QSM references: " + case_id)
        pairs.append((case_id, a, b))
    return pairs


def high_pass(images, mask, sigma_voxels, interior_weight):
    """In-plane residual after mask-normalized Gaussian smoothing, on interior voxels."""
    weights = mask.astype(np.float64)
    smoothed_mask = ndimage.gaussian_filter(weights, sigma_voxels, mode="constant", cval=0., truncate=4.)
    support = mask & (smoothed_mask >= interior_weight)
    residuals = []
    for image in images:
        smoothed = ndimage.gaussian_filter(image * weights, sigma_voxels, mode="constant", cval=0., truncate=4.)
        residuals.append(image[support] - smoothed[support] / smoothed_mask[support])
    return residuals


def case_sums(row_a, row_b, sigma_mm, interior_weight, slice_axis, spatial_units):
    ref_image, reference = load_volume(row_a["qsm_path"])
    mask_image, mask = load_volume(row_a["mask_path"])
    require_grid(ref_image, mask_image)
    predictions = []
    for row in (row_a, row_b):
        image, prediction = load_volume(row["prediction_path"])
        require_grid(ref_image, image)
        predictions.append(prediction.astype(np.float64))
    mask = mask > 0
    if not mask.any():
        raise ValueError("Empty evaluation mask: " + row_a["mask_path"])
    reference = reference.astype(np.float64)
    spacing = spacing_mm(ref_image, spatial_units)
    sigma_voxels = [sigma_mm / step for step in spacing]
    sigma_voxels[slice_axis] = 0.
    sums = defaultdict(float)
    ref_values = reference[mask]
    sums["n_mask"] = float(ref_values.size)
    sums["abs_ref"] = float(np.abs(ref_values).sum())
    sums["ref_sum"], sums["ref_sq"] = float(ref_values.sum()), float(np.square(ref_values).sum())
    for name, prediction in zip(MODELS, predictions):
        values = prediction[mask]
        sums[f"abs_err_{name}"] = float(np.abs(values - ref_values).sum())
        sums[f"{name}_sum"], sums[f"{name}_sq"] = float(values.sum()), float(np.square(values).sum())
    hf_ref, *hf_predictions = high_pass([reference, *predictions], mask, sigma_voxels, interior_weight)
    sums["n_hf"] = float(hf_ref.size)
    sums["hf_ref_sum"], sums["hf_ref_sq"] = float(hf_ref.sum()), float(np.square(hf_ref).sum())
    for name, values in zip(MODELS, hf_predictions):
        sums[f"hf_{name}_sum"], sums[f"hf_{name}_sq"] = float(values.sum()), float(np.square(values).sum())
        sums[f"hf_ref_{name}"] = float((hf_ref * values).sum())
    return sums


def participant_endpoints(subject_id, n_cases, sums):
    n, n_hf = sums["n_mask"], sums["n_hf"]
    if n_hf < 2 or sums["hf_ref_sq"] <= 0:
        raise ValueError("No usable high-frequency support for participant " + subject_id)

    def std(total, squares, count):
        return math.sqrt(max(0., squares / count - (total / count) ** 2))

    ref_std = std(sums["ref_sum"], sums["ref_sq"], n)
    if ref_std <= 0:
        raise ValueError("Constant reference QSM for participant " + subject_id)
    hf_ref_mean = sums["hf_ref_sum"] / n_hf
    hf_ref_var = sums["hf_ref_sq"] / n_hf - hf_ref_mean ** 2
    row = dict(subject_id=subject_id, n_cases=n_cases, n_mask_voxels=int(n), n_hf_voxels=int(n_hf),
               mae_zero=sums["abs_ref"] / n, ref_std=ref_std)
    for name in MODELS:
        mean = sums[f"hf_{name}_sum"] / n_hf
        variance = sums[f"hf_{name}_sq"] / n_hf - mean ** 2
        covariance = sums[f"hf_ref_{name}"] / n_hf - hf_ref_mean * mean
        power_ratio = sums[f"hf_{name}_sq"] / sums["hf_ref_sq"]
        if power_ratio <= 0 or variance <= 0 or hf_ref_var <= 0:
            raise ValueError(f"Degenerate high-frequency content for {name} in participant {subject_id}")
        row.update({
            f"mae_{name}": sums[f"abs_err_{name}"] / n,
            f"sd_ratio_{name}": std(sums[f"{name}_sum"], sums[f"{name}_sq"], n) / ref_std,
            f"hf_power_ratio_{name}": power_ratio,
            f"abs_log_hf_power_ratio_{name}": abs(math.log(power_ratio)),
            f"hf_correlation_{name}": covariance / math.sqrt(variance * hf_ref_var),
        })
    return row


def paired_interval(differences, replicates, seed, confidence):
    differences = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 1000):
        count = min(1000, replicates - start)
        means[start:start + count] = differences[rng.integers(0, differences.size, size=(count, differences.size))].mean(axis=1)
    alpha = (1. - confidence) / 2.
    lower, upper = np.quantile(means, [alpha, 1. - alpha])
    return dict(mean=float(differences.mean()), lower=float(lower), upper=float(upper))


def apply_criteria(subjects, settings):
    column = lambda key: np.array([row[key] for row in subjects], dtype=np.float64)
    boot = dict(replicates=settings["replicates"], seed=settings["seed"], confidence=settings["confidence"])
    delta_mae = paired_interval(column("mae_candidate") - column("mae_comparator"), **boot)
    gain = float(np.mean(column("mae_zero") - column("mae_comparator")))
    margin = settings["mae_margin_fraction"] * gain
    delta_hf_power = paired_interval(column("abs_log_hf_power_ratio_candidate")
                                     - column("abs_log_hf_power_ratio_comparator"), **boot)
    delta_hf_corr = paired_interval(column("hf_correlation_candidate") - column("hf_correlation_comparator"), **boot)
    criteria = {
        "A_mae_noninferiority": dict(estimate=delta_mae, comparator_gain_over_zero=gain, margin=margin,
                                     passed=bool(delta_mae["upper"] < margin)),
        "B_less_oversmoothing": dict(estimate=delta_hf_power, passed=bool(delta_hf_power["upper"] < 0.)),
        "C_no_loss_of_correct_detail": dict(estimate=delta_hf_corr, margin=-settings["hf_correlation_margin"],
                                            passed=bool(delta_hf_corr["lower"] > -settings["hf_correlation_margin"])),
    }
    met = all(item["passed"] for item in criteria.values())
    decision = ("internal_criteria_met_external_criterion_D_pending" if met
                else "internal_criteria_not_met_comparator_remains_primary")
    descriptive = {key: {name: float(np.mean(column(f"{key}_{name}"))) for name in MODELS}
                   for key in ("mae", "sd_ratio", "hf_power_ratio", "hf_correlation")}
    descriptive["mae"]["zero"] = float(np.mean(column("mae_zero")))
    return criteria, decision, descriptive


def prediction_provenance(manifest):
    run = Path(manifest).resolve().parent / "prediction_run.json"
    if not run.is_file():
        return {"prediction_run": None}
    data = json.loads(run.read_text())
    return {"prediction_run": str(run), "checkpoint_path": data.get("checkpoint_path"),
            "checkpoint_sha256": data.get("checkpoint_sha256"), "training_state": data.get("training_state")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True, help="Hash recorded when the rule was registered")
    parser.add_argument("--candidate", required=True, help="evaluation_manifest.csv from predict_split for the candidate")
    parser.add_argument("--comparator", required=True, help="evaluation_manifest.csv from predict_split for the comparator")
    parser.add_argument("--output", required=True, help="New directory")
    parser.add_argument("--slice-axis", type=int, choices=[0, 1, 2], default=2, help="Array axis of the 2D generator's slices")
    parser.add_argument("--spatial-units", choices=["mm", "meter", "micron"], help="Only when NIfTI spatial units are unknown")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().absolute()
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    registration, registration_sha, settings = load_preregistration(args.preregistration, args.preregistration_sha256)
    inputs = {name: (path, sha256(path)) for name, path in (("candidate", args.candidate), ("comparator", args.comparator))}
    pairs = paired_rows(args.candidate, args.comparator, settings["split"])

    totals, cases = defaultdict(lambda: defaultdict(float)), defaultdict(int)
    for case_id, row_a, row_b in pairs:
        sums = case_sums(row_a, row_b, settings["sigma_mm"], settings["interior_weight"],
                         args.slice_axis, args.spatial_units)
        for key, value in sums.items():
            totals[row_a["subject_id"]][key] += value
        cases[row_a["subject_id"]] += 1
    subjects = [participant_endpoints(sid, cases[sid], totals[sid]) for sid in sorted(totals)]
    criteria, decision, descriptive = apply_criteria(subjects, settings)

    for name, (path, digest) in inputs.items():
        if sha256(path) != digest:
            raise RuntimeError(f"{name} manifest changed during the comparison")
    if sha256(args.preregistration) != registration_sha:
        raise RuntimeError("Preregistration changed during the comparison")
    output.mkdir(parents=True)
    with (output / "per_subject.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(subjects[0]))
        writer.writeheader()
        writer.writerows(subjects)
    summary = dict(
        schema="swixqsm.primary_model_comparison", version=1,
        preregistration=str(Path(args.preregistration).resolve()), preregistration_sha256=registration_sha,
        candidate_label=registration["candidate"]["label"], comparator_label=registration["comparator"]["label"],
        split=settings["split"], n_subjects=len(subjects), n_cases=len(pairs),
        settings=dict(settings, slice_axis=args.slice_axis),
        inputs={name: dict(manifest=str(Path(path).resolve()), manifest_sha256=digest, **prediction_provenance(path))
                for name, (path, digest) in inputs.items()},
        descriptive_means=descriptive, criteria=criteria, internal_decision=decision,
        external_criterion="D_external_roi_noninferiority is evaluated separately on the frozen external cohort",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{decision}; " + "; ".join(f"{key}={'pass' if value['passed'] else 'fail'}" for key, value in criteria.items()))
    return summary


if __name__ == "__main__":
    main()

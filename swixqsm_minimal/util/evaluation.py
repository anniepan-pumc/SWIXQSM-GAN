"""Physical-unit evaluation of saved predictions and train-only baselines."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import load_volume, require_grid
from .metrics import physical_metrics
from .nifti import sidecar_path
from .normalization import unit_factor


def declaration(sources, name):
    """Require one explicit value, and reject conflicting declarations."""
    values = []
    for source, value in sources:
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Empty or invalid {name} in {source}")
        values.append((source, value.strip()))
    if not values:
        raise ValueError(f"Explicit {name} is required")
    if len({value for _, value in values}) != 1:
        raise ValueError(f"Conflicting {name}: {values}")
    return values[0][1]


def spatial_unit(image, fallback=None):
    unit = image.header.get_xyzt_units()[0]
    if unit == "unknown":
        if fallback is None:
            raise ValueError("Unknown NIfTI spatial units; supply --spatial-units explicitly")
        unit = fallback
    elif fallback is not None and fallback != unit:
        raise ValueError("Declared spatial units conflict with the NIfTI header")
    if unit not in {"mm", "meter", "micron"}:
        raise ValueError(f"Unsupported spatial units: {unit}")
    return unit


def spacing_mm(image, fallback=None):
    linear = np.asarray(image.affine[:3, :3], dtype=np.float64)
    spacing = np.linalg.norm(linear, axis=0)
    if not np.isfinite(image.affine).all() or np.any(spacing <= 0):
        raise ValueError("Nonfinite or singular image affine")
    directions = linear / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("Sheared/nonorthogonal grids require preparation before physical HFEN")
    factor = {"mm": 1., "meter": 1000., "micron": .001}[spatial_unit(image, fallback)]
    return tuple(float(x) for x in spacing * factor)


def require_evaluation_grid(reference, image, fallback=None):
    require_grid(reference, image)
    if spatial_unit(reference, fallback) != spatial_unit(image, fallback):
        raise ValueError("Images on the evaluation grid have different spatial units")


def resolve_prediction_paths(rows, manifest_path):
    """Prediction paths are optional, but must cover every selected case if used."""
    present = [bool(row.get("prediction_path", "").strip()) for row in rows]
    if any(present) and not all(present):
        raise ValueError("prediction_path must cover every case in the selected split")
    if all(present):
        parent = Path(manifest_path).resolve().parent
        for row in rows:
            path = Path(row["prediction_path"])
            path = (parent / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            row["prediction_path"] = str(path)
    return rows


def prediction_metadata(path):
    sidecar = sidecar_path(path)
    if not sidecar.exists():
        return {}
    data = json.loads(sidecar.read_text())
    if not isinstance(data, dict):
        raise ValueError("Prediction sidecar must be a JSON object")
    return data


def _record(method, row, case_id, metrics, units, reference_id):
    status = ({"value_space": "restored_stored_values", "unit_status": "unverified_original_storage"}
              if units == "stored" else {})
    return {"method": method, "subject_id": row["subject_id"], "case_id": case_id,
            "cohort": row.get("cohort", ""), "units": units,
            "reference_id": reference_id, **status, **metrics}


def aggregate_subjects(cases, data_range):
    """Pool visits within a participant before giving participants equal weight."""
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case["method"], case["subject_id"])].append(case)
    subjects = []
    for (method, sid), visits in grouped.items():
        totals = defaultdict(float)
        for visit in visits:
            for key, value in visit["sufficient_statistics"].items():
                if value is None or not math.isfinite(value):
                    raise ValueError(f"Nonrepresentable metric statistic {key}; refusing to aggregate")
                totals[key] += value
        n = int(totals["n"])
        mse = totals["squared_error_sum"] / n
        ref_mean, pred_mean = totals["reference_sum"] / n, totals["prediction_sum"] / n
        values = {
            "mae": totals["abs_error_sum"] / n, "mse": mse, "rmse": math.sqrt(mse),
            "bias": totals["signed_error_sum"] / n,
            "nmse": totals["squared_error_sum"] / totals["reference_energy"] if totals["reference_energy"] > 0 else None,
            "ref_mean": ref_mean, "pred_mean": pred_mean,
            "ref_std": math.sqrt(max(0., totals["reference_energy"] / n - ref_mean ** 2)),
            "pred_std": math.sqrt(max(0., totals["prediction_energy"] / n - pred_mean ** 2)),
            "ssim": totals["ssim_sum"] / totals["ssim_count"] if totals["ssim_count"] else None,
            "hfen_rmse": math.sqrt(totals["hfen_error_energy"] / totals["hfen_count"]) if totals["hfen_count"] else None,
            "hfen_relative": math.sqrt(totals["hfen_error_energy"] / totals["hfen_reference_energy"]) if totals["hfen_reference_energy"] > 0 else None,
            "psnr_db": 20 * math.log10(data_range) - 10 * math.log10(mse) if data_range is not None and mse > 0 else None,
        }
        statuses = {key: "ok" if value is not None else "undefined" for key, value in values.items()}
        statuses["nmse"] = "ok" if totals["reference_energy"] > 0 else "zero_reference_energy"
        statuses["psnr_db"] = ("missing_data_range" if data_range is None else "infinite" if mse == 0 else "ok")
        for key, count in (("ssim", "ssim_count"), ("hfen_rmse", "hfen_count")):
            if not totals[count]:
                statuses[key] = "insufficient_support"
        if values["hfen_relative"] is None:
            statuses["hfen_relative"] = "insufficient_support" if not totals["hfen_count"] else "zero_reference_high_frequency_energy"
        if data_range is None:
            statuses["ssim"] = "missing_data_range"
        subjects.append({"method": method, "subject_id": sid,
                         "cohort": ";".join(sorted({v["cohort"] for v in visits})),
                         "units": visits[0]["units"], "reference_id": visits[0]["reference_id"],
                         **({"value_space": "restored_stored_values", "unit_status": "unverified_original_storage"}
                            if visits[0]["units"] == "stored" else {}),
                         "n_cases": len(visits), "n_voxels": n, **values,
                         "ssim_centers": int(totals["ssim_count"]), "hfen_centers": int(totals["hfen_count"]),
                         "metric_status": statuses})
    return subjects


def summarize(subjects):
    """Equal-participant descriptive summaries, with every undefined case counted."""
    output = {}
    metric_names = ("mae", "rmse", "bias", "mse", "nmse", "psnr_db", "ssim",
                    "hfen_rmse", "hfen_relative", "ref_mean", "pred_mean", "ref_std", "pred_std")
    for method in sorted({r["method"] for r in subjects}):
        rows = [r for r in subjects if r["method"] == method]
        groups = {"all": rows}
        for cohort in sorted({r["cohort"] for r in rows if r["cohort"]}):
            groups[f"cohort:{cohort}"] = [r for r in rows if r["cohort"] == cohort]
        output[method] = {}
        for group, selected in groups.items():
            result = {"n_subjects": len(selected), "n_cases": sum(r["n_cases"] for r in selected), "metrics": {}}
            for name in metric_names:
                values = [r[name] for r in selected if r[name] is not None]
                reasons = defaultdict(int)
                for row in selected:
                    if row[name] is None:
                        reasons[row["metric_status"].get(name, "undefined")] += 1
                result["metrics"][name] = {
                    "mean": float(np.mean(values)) if values else None,
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "n_defined": len(values), "n_undefined": len(selected) - len(values),
                    "undefined_reasons": dict(reasons),
                }
            output[method][group] = result
    return output


def evaluate_rows(rows, *, units="ppb", qsm_units=None, reference_id,
                  spatial_units=None, space_id=None, baseline_bundle=None,
                  prediction_units=None, prediction_reference_id=None,
                  prediction_name="prediction", data_range=None, ssim_window=7,
                  hfen_sigma_mm=1.5, save_prediction=None, geometry_contract=None,
                  diagnostic_callback=None, region_callback=None, roi_review_policy="accepted"):
    if not rows:
        raise ValueError("No cases in the selected split")
    unit_factor(units, units)
    if roi_review_policy not in {"accepted", "deferred"}:
        raise ValueError("ROI review policy must be accepted or deferred")
    if units == "stored" and (baseline_bundle is not None or region_callback is not None and
            (roi_review_policy != "deferred" or getattr(getattr(region_callback, "__self__", None), "review_policy", None) != "deferred")):
        raise ValueError("stored evaluation does not support anatomical baselines or ROI physical-unit analysis")
    reference_id = declaration([("evaluation", reference_id)], "QSM reference ID")
    if data_range is not None and (not math.isfinite(data_range) or data_range <= 0):
        raise ValueError("data_range must be finite and positive, in output units")
    if prediction_name in {"zero", "train_mean", "linear"} or not prediction_name.strip():
        raise ValueError("Prediction name must be nonempty and distinct from baseline names")
    if baseline_bundle is not None:
        baseline_bundle.assert_independent(rows)
        declaration([("evaluation", reference_id), ("baseline", baseline_bundle.metadata["reference_id"])], "QSM reference ID")
        if space_id is None:
            space_id = declaration([("manifest", row.get("space_id")) for row in rows], "evaluation space ID")
        space_id = declaration([("evaluation", space_id), ("baseline", baseline_bundle.metadata["space_id"])], "space ID")
    cases, case_ids = [], set()
    for index, row in enumerate(rows):
        case_id = row.get("case_id") or f"{row['subject_id']}_row{index + 1:06d}"
        if case_id in case_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        case_ids.add(case_id)
        declaration([("evaluation", reference_id), ("manifest", row.get("qsm_reference"))], "QSM reference ID")
        if space_id is not None:
            declaration([("evaluation", space_id), ("manifest", row.get("space_id"))], "space ID")
        ref, swi = load_volume(row["swi_path"])
        assumed_units = geometry_contract.get("spatial_units_if_unknown") if geometry_contract else None
        case_units = spatial_units
        if case_units is None and assumed_units is not None and ref.header.get_xyzt_units()[0] in {"unknown", assumed_units}:
            case_units = assumed_units
        if ref.header.get_xyzt_units()[0] == "unknown" and assumed_units is not None and case_units != assumed_units:
            raise ValueError("Runtime spatial-unit assumption conflicts with the QC contract")
        spacing = spacing_mm(ref, case_units)
        arrays = []
        for name in ("qsm_path", "swi_mask_path", "mask_path"):
            image, array = load_volume(row[name])
            if image.header.get_xyzt_units()[0] == "unknown" and assumed_units is not None and case_units != assumed_units:
                raise ValueError("Runtime spatial-unit assumption conflicts with the QC contract")
            require_evaluation_grid(ref, image, case_units)
            arrays.append(array)
        target, brain, mask = arrays
        brain, mask = brain > 0, mask > 0
        if not brain.any() or not mask.any() or np.any(mask & ~brain):
            raise ValueError("Evaluation support must be nonempty and contained in the input brain mask")
        source_units = row.get("qsm_units") or qsm_units
        if source_units is None:
            raise ValueError("qsm_units column or --qsm-units is required; units are never guessed")
        target = target.astype(np.float64) * unit_factor(source_units, units)
        if not np.isfinite(target).all():
            raise ValueError("Nonfinite physical reference after unit conversion")

        def score(method, prediction):
            result = physical_metrics(target, prediction, mask, data_range=data_range,
                                      spacing=spacing, ssim_window=ssim_window,
                                      hfen_sigma_mm=hfen_sigma_mm)
            result["metric_units"] = {key: value.replace("input_susceptibility", units)
                                      for key, value in result["metric_units"].items()}
            cases.append(_record(method, row, case_id, result, units, reference_id))
            if diagnostic_callback is not None:
                diagnostic_callback(case_id, method, target, prediction, mask, units)
            if region_callback is not None:
                region_callback(case_id, method, target, prediction, mask, units, ref, case_units)
            if save_prediction is not None:
                save_prediction(case_id, method, prediction, ref, units, reference_id, space_id)

        # This is zero in physical units, including when a neural model used nonzero offset.
        score("zero", np.zeros_like(target))
        if baseline_bundle is not None:
            native_unit = ref.header.get_xyzt_units()[0]
            artifact_unit = baseline_bundle.metadata.get("grid_spatial_units")
            if artifact_unit != native_unit:
                raise ValueError("Baseline and evaluation NIfTI spatial-unit declarations differ")
            for method in ("train_mean", "linear"):
                prediction = baseline_bundle.predict(method, swi, brain, ref.affine, mask=mask,
                                                     reference_id=reference_id, space_id=space_id)
                prediction = prediction.astype(np.float64) * unit_factor(baseline_bundle.metadata["units"], units)
                score(method, prediction)
        if row.get("prediction_path"):
            path = row["prediction_path"]
            image, prediction = load_volume(path)
            if image.header.get_xyzt_units()[0] == "unknown" and assumed_units is not None and case_units != assumed_units:
                raise ValueError("Prediction spatial-unit assumption conflicts with the QC contract")
            require_evaluation_grid(ref, image, case_units)
            sidecar = prediction_metadata(path)
            if units == "stored":
                if sidecar.get("ValueSpace") != "restored_stored_values":
                    raise ValueError("stored evaluation requires explicit restored_stored_values prediction metadata")
                if sidecar.get("UnitStatus") not in {None, "unverified_original_storage"}:
                    raise ValueError("stored predictions must retain unverified original-storage unit status")
                if sidecar.get("DataQC", {}).get("status") in {"verified", "geometry_verified", "passed"}:
                    raise ValueError("stored predictions cannot claim verified physical-unit QC")
            elif sidecar.get("ValueSpace") not in {None, "physical_qsm"}:
                raise ValueError("Evaluation requires physical QSM predictions; restore normalized network outputs first")
            pred_units = declaration([("manifest", row.get("prediction_units")),
                                      ("sidecar", sidecar.get("Units")),
                                      ("CLI", prediction_units)], "prediction units")
            pred_reference = declaration([("manifest", row.get("prediction_reference")),
                                          ("sidecar", sidecar.get("ReferenceID")),
                                          ("CLI", prediction_reference_id)], "prediction reference ID")
            declaration([("reference", reference_id), ("prediction", pred_reference)], "QSM reference ID")
            if sidecar.get("SpaceID") is not None:
                expected_space = declaration([("evaluation", space_id),
                                              ("manifest", row.get("space_id"))], "evaluation space ID")
                declaration([("reference", expected_space), ("prediction sidecar", sidecar["SpaceID"])], "space ID")
            prediction = prediction.astype(np.float64) * unit_factor(pred_units, units)
            score(prediction_name, prediction)
    subjects = aggregate_subjects(cases, data_range)
    return {"cases": cases, "subjects": subjects, "summary": summarize(subjects)}


def write_csv(path, records):
    records = [{key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items() if key != "sufficient_statistics"}
               for row in records]
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

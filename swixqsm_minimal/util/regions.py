"""Prespecified native-grid ROI measurements on frozen physical predictions.

No segmentation, registration, calibration, model inference or training occurs.
An ROI audit is separate from image QC. Hashes bind declared sources, never
establish anatomical correctness or prospective specification by themselves.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
import hashlib
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np

from .agreement import paired_agreement
from .evaluation import require_evaluation_grid
from .normalization import unit_factor


METRICS = ("ref_mean", "pred_mean", "difference", "absolute_mean_difference", "pair_mean", "mae", "rmse", "bias", "nmse")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Nonempty {name} is required")
    return value.strip()


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_roi_spec(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    spec = json.loads(payload.decode("utf-8"))
    required = {"schema", "version", "id", "definition", "label_source", "regions"}
    if (not isinstance(spec, dict) or not required.issubset(spec)
            or set(spec) - required - {"min_coverage", "min_voxels"}
            or spec["schema"] != "swixqsm.roi_spec" or type(spec["version"]) is not int or spec["version"] != 1):
        raise ValueError("Unsupported or incomplete ROI specification")
    spec = deepcopy(spec)
    for key in ("id", "definition", "label_source"):
        spec[key] = _text(spec[key], f"ROI specification {key}")
    if not isinstance(spec["regions"], list) or not spec["regions"]:
        raise ValueError("ROI specification requires a nonempty regions list")
    names = set()
    for region in spec["regions"]:
        if not isinstance(region, dict) or set(region) != {"name", "labels"}:
            raise ValueError("Each ROI needs exactly name and labels")
        region["name"] = _text(region["name"], "region name")
        labels = region["labels"]
        if (region["name"] in names or not isinstance(labels, list) or not labels
                or any(type(v) is not int or v < 1 or v > np.iinfo(np.int32).max for v in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError("Region names and within-region positive integer labels must be unique")
        names.add(region["name"])
    coverage = spec.setdefault("min_coverage", 1.)
    voxels = spec.setdefault("min_voxels", 1)
    if (isinstance(coverage, bool) or not isinstance(coverage, (int, float))
            or not math.isfinite(coverage) or not 0 < coverage <= 1):
        raise ValueError("ROI min_coverage must be finite in (0,1]")
    if type(voxels) is not int or voxels < 1:
        raise ValueError("ROI min_voxels must be a positive integer")
    return spec, {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def _measure(target, prediction):
    """Pointwise statistics only; spatial SSIM/HFEN retain their existing scope."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = prediction - target
        sums = {"n": len(target), "reference_sum": float(target.sum(dtype=np.float64)),
                "prediction_sum": float(prediction.sum(dtype=np.float64)),
                "reference_energy": float(np.sum(target ** 2, dtype=np.float64)),
                "signed_error_sum": float(error.sum(dtype=np.float64)),
                "abs_error_sum": float(np.abs(error).sum(dtype=np.float64)),
                "squared_error_sum": float(np.sum(error ** 2, dtype=np.float64))}
    if not all(math.isfinite(value) for value in sums.values()):
        return {key: None for key in METRICS}, None, "numerical_overflow"
    try:
        values = _from_sums(sums)
    except (ValueError, OverflowError):
        return {key: None for key in METRICS}, None, "numerical_overflow"
    return values, sums, "ok"


def _from_sums(sums):
    n = sums["n"]
    ref_mean = sums["reference_sum"] / n
    pred_mean = sums["prediction_sum"] / n
    values = {"ref_mean": ref_mean, "pred_mean": pred_mean, "difference": pred_mean - ref_mean,
            "absolute_mean_difference": abs(pred_mean - ref_mean),
            "pair_mean": ref_mean / 2 + pred_mean / 2,
            "mae": sums["abs_error_sum"] / n,
            "rmse": math.sqrt(sums["squared_error_sum"] / n),
            "bias": sums["signed_error_sum"] / n,
            "nmse": sums["squared_error_sum"] / sums["reference_energy"] if sums["reference_energy"] > 0 else None}
    if not all(value is None or math.isfinite(value) for value in values.values()):
        raise ValueError("Nonrepresentable ROI metric")
    return values


def _participant_summary(values, total):
    """Stable equal-participant arithmetic, with explicit unavailable SD."""
    result = {"mean": None, "sd": None, "n_defined": len(values), "n_undefined": total-len(values),
              "mean_status": "no_defined_participants", "sd_status": "insufficient_n"}
    if values:
        scale = max(abs(value) for value in values) or 1.
        scaled = np.asarray(values, dtype=np.float64) / scale
        mean = float(np.mean(scaled)) * scale
        result.update(mean=mean if math.isfinite(mean) else None,
                      mean_status="defined" if math.isfinite(mean) else "numerical_overflow")
        if len(values) > 1:
            sd = float(np.std(scaled, ddof=1)) * scale
            result.update(sd=sd if math.isfinite(sd) else None,
                          sd_status="defined" if math.isfinite(sd) else "numerical_overflow")
    return result


class RegionEvaluator:
    """Collect a single selected split using the existing evaluation score hook.

    Rows must have unique case_id and explicit subject_id. Optional global IDs
    must be complete and map one-to-one; cohort changes within a participant are
    rejected. All expected visits must be eligible for a participant/ROI to enter
    agreement. Visits are voxel-pooled; n for agreement remains participants.
    Only the current case label image is retained, never all cohort volumes.
    """
    def __init__(self, rows, spec_path, manifest_path, *, units, reference_id,
                 bootstrap_reps=2000, seed=42, review_policy="accepted",
                 allow_unverified_data=None, data_qc=None, roi_manifest_source=None):
        unit_factor(units, units)
        if review_policy not in {"accepted", "deferred"}:
            raise ValueError("ROI review policy must be accepted or deferred")
        if review_policy == "deferred":
            if not isinstance(allow_unverified_data, str) or not allow_unverified_data.strip():
                raise ValueError("Deferred ROI requires an explicit nonempty allow_unverified_data reason")
            if data_qc is not None and data_qc.get("status") != "unverified":
                raise ValueError("Deferred ROI requires unverified data; a verified physical QC report cannot be used")
        elif units == "stored":
            raise ValueError("ROI physical-unit evaluation does not support unverified stored units without explicit deferred review policy")
        self.review_policy = review_policy
        self.unverified_reason = allow_unverified_data.strip() if review_policy == "deferred" else None
        self.data_qc_status = ("unverified" if review_policy == "deferred" else
                               (data_qc or {}).get("status", "not_recorded"))
        self.units, self.reference_id = units, _text(reference_id, "reference_id")
        self.roi_manifest_source = deepcopy(roi_manifest_source)
        # Validate statistical options before loading any labels.
        paired_agreement([], [], bootstrap_reps=bootstrap_reps, seed=seed)
        self.bootstrap_reps, self.seed = bootstrap_reps, seed
        self.spec, self.spec_source = read_roi_spec(spec_path)
        self.rows = deepcopy(rows)
        if (not self.rows or len({r.get("split") for r in self.rows}) != 1
                or self.rows[0].get("split") not in {"train", "val", "test"}):
            raise ValueError("RegionEvaluator requires rows from exactly one selected split")
        self.by_case, subjects, globals_to_subjects = {}, {}, {}
        any_global = any(str(r.get("global_subject_id", "")).strip() for r in self.rows)
        parent = Path(manifest_path).resolve().parent
        for row in self.rows:
            sid = _text(row.get("subject_id"), "subject_id")
            case = _text(row.get("case_id"), "case_id")
            if case in self.by_case:
                raise ValueError("Duplicate ROI case_id")
            cohort = str(row.get("cohort", "")).strip()
            gid = _text(row.get("global_subject_id"), "global_subject_id") if any_global else ""
            identity = (cohort, gid)
            if sid in subjects and subjects[sid] != identity:
                raise ValueError("ROI participant must have one cohort and one global_subject_id")
            if gid and gid in globals_to_subjects and globals_to_subjects[gid] != sid:
                raise ValueError("Global participant has multiple subject aliases; resolve identity before ROI analysis")
            subjects[sid] = identity
            if gid:
                globals_to_subjects[gid] = sid
            row.update(subject_id=sid, case_id=case, cohort=cohort, global_subject_id=gid)
            path = row.get("roi_labels_path", "").strip()
            if path:
                path = Path(path).expanduser()
                row["roi_labels_path"] = str((parent / path if not path.is_absolute() else path).resolve())
            else:
                row["roi_labels_path"] = ""
            review = row.get("roi_review", "").strip() or "not_recorded"
            valid_reviews = {"accepted", "pending", "rejected", "not_recorded"}
            if self.review_policy == "deferred":
                valid_reviews.add("unreviewed")
            if review not in valid_reviews:
                raise ValueError("ROI review must be accepted, pending, rejected or not_recorded")
            row["roi_review"] = review
            if review == "accepted":
                _text(row.get("roi_reviewer"), "roi_reviewer")
                date_text = _text(row.get("roi_review_date"), "roi_review_date")
                try:
                    valid_date = date.fromisoformat(date_text).isoformat() == date_text
                except ValueError:
                    valid_date = False
                if not valid_date:
                    raise ValueError("ROI review date must be YYYY-MM-DD")
            self.by_case[case] = row
        self.global_identity = "declared_one_to_one" if any_global else "not_recorded; subject_id is declared participant"
        self.cases, self.provenance, self._seen = [], [], set()
        self._current_case, self._current = None, None
        self._cached_result = None

    def _interpretation(self, reviews):
        counts = Counter(reviews)
        return {"roi_review_policy": self.review_policy,
                "roi_review_status": ";".join(sorted(counts)), "roi_review_status_counts": dict(counts),
                "roi_qc_status": ("research_candidate_unverified" if self.review_policy == "deferred"
                                  else "author_review_declarations_only"),
                "QCStatus": self.data_qc_status,
                "UnitStatus": "unverified_original_storage" if self.units == "stored" else "declared_physical_units",
                "ReferenceStatus": "unverified" if self.review_policy == "deferred" or self.data_qc_status != "verified" else "verified_by_data_qc",
                "icc_a1_units": "dimensionless", "pearson_r_units": "dimensionless",
                "value_space": "restored_stored_values" if self.units == "stored" else "physical_qsm",
                "research_only": self.review_policy == "deferred", "formal_accuracy_claim": False,
                "roi_spec_sha256": self.spec_source["sha256"], "roi_spec_id": self.spec["id"]}

    def _prepare(self, case_id, reference_image, mask, spatial_units):
        row = self.by_case[case_id]
        path = row["roi_labels_path"]
        provenance = {key: row.get(key, "") for key in
                      ("case_id", "subject_id", "global_subject_id", "cohort", "roi_labels_path",
                       "roi_label_source", "roi_review", "roi_reviewer", "roi_review_date")}
        provenance.update(label_sha256=None, review_evidence="author declaration only; hash is not anatomical QC",
                          **self._interpretation([row["roi_review"]]))
        labels = None
        if path:
            before = _digest(path)
            image = nib.load(path)
            require_evaluation_grid(reference_image, image, spatial_units)
            if image.get_data_dtype().kind not in "iuf":
                raise ValueError("ROI label storage must be real numeric, never complex or RGB")
            values = image.get_fdata(dtype=np.float64)
            if (values.ndim != 3 or values.shape != mask.shape or not np.isfinite(values).all()
                    or np.any(values < 0) or np.any(values > np.iinfo(np.int32).max)
                    or not np.array_equal(values, np.rint(values))):
                raise ValueError("ROI labels must be finite, nonnegative integer 3D labels on the evaluation grid")
            if _digest(path) != before:
                raise ValueError("ROI label image changed while reading")
            provenance["label_sha256"] = before
            labels = values.astype(np.int32)
        self.provenance.append(provenance)
        prepared = []
        for region in self.spec["regions"]:
            region_mask = np.isin(labels, region["labels"]) if labels is not None else np.zeros(mask.shape, dtype=bool)
            label_counts = {str(label): int(np.count_nonzero(labels == label)) if labels is not None else 0
                            for label in region["labels"]}
            missing_labels = [label for label in region["labels"] if label_counts[str(label)] == 0]
            n_label = int(region_mask.sum())
            indices = np.flatnonzero(region_mask & mask)
            n_valid = len(indices)
            coverage = n_valid / n_label if n_label else None
            if labels is None:
                status = "missing_labelmap"
            elif not row.get("roi_label_source", "").strip():
                status = "missing_label_source"
            elif row["roi_review"] == "rejected" and self.review_policy == "deferred":
                status = "review_rejected"
            elif row["roi_review"] != "accepted" and self.review_policy == "accepted":
                status = "review_not_accepted"
            elif not n_label:
                status = "empty_label"
            elif missing_labels:
                status = "missing_component_label"
            elif n_valid < self.spec["min_voxels"]:
                status = "insufficient_voxels"
            elif coverage < self.spec["min_coverage"]:
                status = "insufficient_coverage"
            else:
                status = "ok"
            prepared.append({"region": region["name"], "n_label_voxels": n_label,
                             "n_valid_voxels": n_valid, "coverage": coverage, "status": status,
                             "region_labels": list(region["labels"]),
                             "roi_support_sha256": hashlib.sha256(
                                 np.ascontiguousarray(region_mask & mask, dtype=np.bool_).tobytes()).hexdigest(),
                             "label_voxel_counts": label_counts, "missing_labels": missing_labels,
                             "indices": indices})
        # Retain only sparse support, not raw label/target/prediction volumes.
        return prepared

    def collect(self, case_id, method, target, prediction, mask, units, reference_image, spatial_units=None):
        if self._cached_result is not None:
            raise ValueError("Cannot collect after ROI finish")
        if case_id not in self.by_case or not isinstance(method, str) or not method.strip():
            raise ValueError("Unknown ROI case or invalid method")
        if (case_id, method) in self._seen:
            raise ValueError("Duplicate ROI case/method")
        if units != self.units:
            raise ValueError("ROI callback units differ from evaluation units")
        target, prediction, mask = np.asarray(target), np.asarray(prediction), np.asarray(mask)
        if (target.ndim != 3 or target.shape != prediction.shape or target.shape != mask.shape
                or target.shape != reference_image.shape or target.dtype.kind not in "iuf"
                or prediction.dtype.kind not in "iuf" or mask.dtype.kind not in "biuf"
                or not np.isfinite(mask).all() or np.any(mask < 0)):
            raise ValueError("ROI callback requires matching 3D arrays and finite nonnegative support")
        mask = mask > 0
        if not np.isfinite(target[mask]).all() or not np.isfinite(prediction[mask]).all():
            raise ValueError("ROI values must be finite on common evaluation support")
        reference_sha = hashlib.sha256(np.ascontiguousarray(target, dtype="<f8").tobytes()).hexdigest()
        grid = {"shape": list(reference_image.shape), "affine": reference_image.affine.tolist(),
                "header_spatial_units": reference_image.header.get_xyzt_units()[0],
                "explicit_spatial_units": spatial_units}
        grid_sha = hashlib.sha256(json.dumps(grid, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if self._current_case != case_id:
            if any(case == case_id for case, _ in self._seen):
                raise ValueError("ROI methods must be consecutive per case to keep one label working set")
            self._current = self._prepare(case_id, reference_image, mask, spatial_units)
            self._current_case = case_id
            self._mask_sha = hashlib.sha256(mask.tobytes()).hexdigest()
            self._reference_sha = reference_sha
            self._grid_sha = grid_sha
            self.provenance[-1].update(reference_array_sha256=reference_sha,
                evaluation_support_sha256=self._mask_sha, evaluation_grid_sha256=self._grid_sha,
                evaluation_grid=grid,
                input_paths={key: self.by_case[case_id].get(key) for key in
                             ("swi_path", "qsm_path", "swi_mask_path", "mask_path")})
        elif hashlib.sha256(mask.tobytes()).hexdigest() != self._mask_sha:
            raise ValueError("ROI methods must share identical common evaluation support")
        elif reference_sha != self._reference_sha:
            raise ValueError("ROI methods must share identical reference values")
        elif grid_sha != self._grid_sha:
            raise ValueError("ROI methods must share identical evaluation grid")
        self._seen.add((case_id, method))
        row = self.by_case[case_id]
        for region in self._current:
            record = {key: row[key] for key in ("case_id", "subject_id", "global_subject_id", "cohort")}
            record.update(method=method, region=region["region"], units=units, reference_id=self.reference_id,
                          n_label_voxels=region["n_label_voxels"], n_valid_voxels=region["n_valid_voxels"],
                          coverage=region["coverage"], status=region["status"], roi_review=row["roi_review"],
                          **self._interpretation([row["roi_review"]]),
                          region_labels=region["region_labels"], roi_support_sha256=region["roi_support_sha256"],
                          evaluation_support_sha256=self._mask_sha, reference_array_sha256=self._reference_sha,
                          evaluation_grid_sha256=self._grid_sha,
                          label_voxel_counts=region["label_voxel_counts"], missing_labels=region["missing_labels"],
                          **{key: None for key in METRICS}, sufficient_statistics=None)
            if record["status"] == "ok":
                positions = region["indices"]
                coordinates = np.unravel_index(positions, target.shape)
                values, sums, status = _measure(np.asarray(target[coordinates], dtype=np.float64),
                                                np.asarray(prediction[coordinates], dtype=np.float64))
                record.update(values)
                record.update(status=status, sufficient_statistics=sums)
            record["nmse_status"] = ("zero_reference_energy" if record["status"] == "ok" and record["nmse"] is None
                                     else record["status"])
            self.cases.append(record)

    def finish(self):
        if self._cached_result is not None:
            return deepcopy(self._cached_result)
        methods = sorted({method for _, method in self._seen})
        if not methods or self._seen != {(case, method) for case in self.by_case for method in methods}:
            raise ValueError("ROI methods must cover all expected cases before finish")
        grouped = defaultdict(list)
        for case in self.cases:
            grouped[(case["method"], case["region"], case["subject_id"])].append(case)
        subjects = []
        for (method, region, sid), visits in sorted(grouped.items()):
            expected = sum(row["subject_id"] == sid for row in self.rows)
            included = [row for row in visits if row["status"] == "ok"]
            complete = len(included) == expected
            row = {"method": method, "region": region, "subject_id": sid,
                   "global_subject_id": visits[0]["global_subject_id"], "cohort": visits[0]["cohort"],
                   "units": self.units, "reference_id": self.reference_id,
                   "n_cases_expected": expected, "n_cases_included": len(included),
                   "n_valid_voxels": 0, "status": "ok" if complete else "incomplete_visits",
                   "visit_status_counts": dict(Counter(v["status"] for v in visits)),
                   **self._interpretation(v["roi_review"] for v in visits),
                   "region_labels": list(visits[0]["region_labels"]),
                   **{key: None for key in METRICS}}
            if complete:
                totals = {key: sum(v["sufficient_statistics"][key] for v in visits)
                          for key in visits[0]["sufficient_statistics"]}
                if not all(math.isfinite(value) for value in totals.values()):
                    row["status"] = "numerical_overflow"
                else:
                    try:
                        row.update(_from_sums(totals))
                        row["n_valid_voxels"] = int(totals["n"])
                    except (ValueError, OverflowError):
                        row["status"] = "numerical_overflow"
            row["nmse_status"] = ("zero_reference_energy" if row["status"] == "ok" and row["nmse"] is None
                                  else row["status"])
            subjects.append(row)
        agreement, summary = [], []
        for method in methods:
            for region in self.spec["regions"]:
                selected = [r for r in subjects if r["method"] == method and r["region"] == region["name"]]
                groups = [("overall", "all", selected)]
                groups += [("cohort", cohort, [r for r in selected if r["cohort"] == cohort])
                           for cohort in sorted({r["cohort"] for r in selected})]
                for group_type, group, records in groups:
                    usable = sorted((r for r in records if r["status"] == "ok"), key=lambda r: r["subject_id"])
                    # Each named analysis has stable RNG independent of loop order.
                    token = json.dumps([self.seed, method, region["name"], group_type, group], separators=(",", ":"))
                    group_seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
                    statistics = paired_agreement([r["ref_mean"] for r in usable], [r["pred_mean"] for r in usable],
                                                  bootstrap_reps=self.bootstrap_reps, seed=group_seed)
                    if self.review_policy == "deferred":
                        statistics["assumptions"] = [
                            ("Both vectors use the same declared ROI and reference convention; spatial QC and reference consistency remain unverified."
                             if "same physical units" in value else value) for value in statistics["assumptions"]]
                        statistics["limitations"].append("Research candidate numerical agreement only; no formal anatomical accuracy or physical calibration claim is supported.")
                    if self.units == "stored":
                        statistics["definitions"]["dimensional_metrics_units"] = "stored"
                        statistics["definitions"]["dimensionless_metrics"] = ["icc_a1", "pearson_r"]
                        statistics["assumptions"].append("Values remain on one unverified original-storage scale; no ppm/ppb conversion or physical calibration was performed.")
                    statistics["metric_units"] = {key: "dimensionless" if key in {"icc_a1", "pearson_r"} else self.units
                                                  for key in statistics["estimates"]}
                    statistics.update(QCStatus=self.data_qc_status,
                                      UnitStatus="unverified_original_storage" if self.units == "stored" else "declared_physical_units",
                                      research_only=self.review_policy == "deferred", formal_accuracy_claim=False)
                    header = {"method": method, "region": region["name"], "group_type": group_type,
                              "group": group, "units": self.units, "reference_id": self.reference_id,
                              "n_participants_expected": len(records), "n_participants_included": len(usable),
                              "n_participants_excluded": len(records)-len(usable),
                              **self._interpretation(v["roi_review"] for r in records for v in
                                  grouped[(method, region["name"], r["subject_id"])]),
                              "region_labels": list(region["labels"]),
                              "excluded_status_counts": dict(Counter(r["status"] for r in records if r["status"] != "ok"))}
                    errors = {}
                    for key in ("mae", "rmse", "bias", "nmse"):
                        values = [r[key] for r in usable if r[key] is not None]
                        errors[key] = _participant_summary(values, len(usable))
                    scalar_summaries = {key: _participant_summary([r[key] for r in usable if r[key] is not None], len(usable))
                                        for key in ("ref_mean", "pred_mean", "difference", "absolute_mean_difference")}
                    agreement.append({**header, "voxel_error_participant_summary": errors,
                                      "roi_mean_participant_summary": scalar_summaries, "agreement": statistics})
                    flat = dict(header)
                    flat.update(statistics["estimates"])
                    flat["metric_status"] = statistics["metric_status"]
                    for key, interval in statistics["bootstrap"]["metrics"].items():
                        for field in ("lower", "upper", "status", "valid_replicates", "undefined_replicates"):
                            flat[f"{key}_ci_{field}"] = interval[field]
                    for key, values in errors.items():
                        flat[f"voxel_{key}_participant_mean"] = values["mean"]
                        flat[f"voxel_{key}_participant_sd"] = values["sd"]
                    for key, values in scalar_summaries.items():
                        flat[f"roi_mean_{key}_participant_mean"] = values["mean"]
                        flat[f"roi_mean_{key}_participant_sd"] = values["sd"]
                    summary.append(flat)
        self._current = None
        if _digest(self.spec_source["path"]) != self.spec_source["sha256"]:
            raise ValueError("ROI specification changed during evaluation")
        if self.roi_manifest_source is not None and _digest(self.roi_manifest_source["path"]) != self.roi_manifest_source["sha256"]:
            raise ValueError("ROI manifest changed during evaluation")
        self._cached_result = {"cases": self.cases, "subjects": subjects, "agreement": agreement,
                               "summary": summary, "provenance": {
            "specification": {**self.spec_source, "content": self.spec}, "cases": self.provenance,
            "roi_manifest": self.roi_manifest_source,
            "identity": self.global_identity,
            **self._interpretation(row["roi_review"] for row in self.rows),
            "unverified_reason": self.unverified_reason,
            "array_hash_method": {"reference_array_sha256": "full actual evaluation-reference array, little-endian float64 C-order bytes; shape/grid bound separately",
                                  "evaluation_support_sha256": "full common evaluation mask, bool C-order bytes",
                                  "roi_support_sha256": "region label union intersect common evaluation mask, bool C-order bytes",
                                  "evaluation_grid_sha256": "sorted-key JSON shape, affine, header and explicit spatial units"},
            "status": "source_bound_author_review_declarations; anatomical correctness not independently established",
            "aggregation": "All expected visits eligible, then voxel-pool within participant/ROI; participant once per method/region/group",
            "support": "Configured component labels must all appear; union intersect common evaluation mask, no resampling",
            "interpretation": "Agreement concerns aggregated participant ROI means; not single-scan repeatability, lesion accuracy or clinical validity",
            "ci_scope": "Pointwise paired-participant percentile CIs, no simultaneous multiple-ROI coverage or hypothesis tests",
            "limitations": ["Within-image label coverage cannot establish anatomy outside the acquired field of view.",
                            "Integer labels, source hashes and author review declarations do not prove segmentation or registration accuracy.",
                            "Overlapping or bilateral regions are dependent; their estimates are not independent replications.",
                            "Prespecification is declared, not proven by a newly created specification file or hash.",
                            "Participant identifiers remain declarations unless separately verified by the independence audit."],
        }}
        return deepcopy(self._cached_result)

"""Audit declared participant membership and exact decoded SWI duplication.

An audit records inspectable evidence and declarations. It cannot establish the
truth of global identities or the completeness of a pretrained model's history.
"""

import copy
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np


MAIN_PATHS = ("swi_path", "qsm_path", "swi_mask_path", "mask_path")
SPLITS = {"train", "val", "test"}
ROLES = {"train", "selection", "evaluation", "unknown"}
FINGERPRINT_METHOD = "sha256_decoded_float32_little_endian_C_order_positive_zero_v1"
EXPECTED_FILE_ERRORS = (OSError, ValueError, TypeError, EOFError,
                        nib.filebasedimages.ImageFileError)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _normalize_row(original):
    row = {str(key): str(value) if isinstance(value, Path) else value
           for key, value in dict(original).items() if not str(key).startswith("prediction_")}
    for key in ("subject_id", "global_subject_id", "split"):
        if isinstance(row.get(key), str):
            row[key] = row[key].strip()
    for key in MAIN_PATHS:
        if row.get(key) is not None:
            row[key] = str(Path(str(row[key])).expanduser().resolve()) if str(row[key]).strip() else ""
    return row


def _case_key(row):
    return _digest({key: row.get(key) for key in
                    ("subject_id", "global_subject_id", "split", "case_id", "swi_path")})


def _manifest_identity(rows):
    # Reordering CSV rows does not change participant membership or visit metadata.
    return _digest(sorted(rows, key=lambda row: _canonical(row)))


def _swi_fingerprint(path):
    image = nib.load(str(path))
    # get_fdata applies NIfTI scaling before canonical float32 serialization.
    values = image.get_fdata(dtype=np.float32)
    if values.ndim != 3 or any(extent == 0 for extent in values.shape) or not np.isfinite(values).all():
        raise ValueError("SWI fingerprint requires finite nonempty 3D decoded data")
    values = np.array(values, dtype="<f4", order="C", copy=True)
    values[values == 0] = 0.  # Canonicalize -0.0 without changing nonzero values.
    digest = hashlib.sha256()
    digest.update(FINGERPRINT_METHOD.encode("ascii") + b"\0")
    digest.update(np.asarray(values.shape, dtype="<u8").tobytes())
    digest.update(values.tobytes(order="C"))
    return {"sha256": digest.hexdigest(), "shape": list(values.shape),
            "method": FINGERPRINT_METHOD}


def _identity_errors(rows):
    errors = []
    subject_to_global, global_to_subject, global_to_split, seen_paths = {}, {}, {}, {}
    for index, row in enumerate(rows):
        sid, gid, split = row.get("subject_id"), row.get("global_subject_id"), row.get("split")
        if not _text(sid):
            errors.append(f"Row {index+1} has no nonempty subject_id")
        if not _text(gid):
            errors.append(f"Row {index+1} has no nonempty global_subject_id")
        if not isinstance(split, str) or split not in SPLITS:
            errors.append(f"Row {index+1} split must be train, val, or test")
        if _text(sid) and _text(gid):
            subject_to_global.setdefault(sid, set()).add(gid)
            global_to_subject.setdefault(gid, set()).add(sid)
            if isinstance(split, str):
                global_to_split.setdefault(gid, set()).add(split)
        path = row.get("swi_path")
        if path:
            if path in seen_paths:
                errors.append(f"Repeated SWI path in rows {seen_paths[path]+1} and {index+1}: {path}")
            seen_paths[path] = index
    for sid, gids in subject_to_global.items():
        if len(gids) != 1:
            errors.append(f"subject_id {sid!r} maps to multiple global_subject_id values")
    for gid, sids in global_to_subject.items():
        if len(sids) != 1:
            errors.append(f"global_subject_id {gid!r} has multiple subject_id aliases; unify them before analysis")
    for gid, splits in global_to_split.items():
        if len(splits) != 1:
            errors.append(f"global_subject_id {gid!r} crosses splits: {sorted(splits)}")
    return errors


def _inspect_inputs(row):
    errors = []
    result = {"key": _case_key(row), "subject_id": row.get("subject_id"),
              "global_subject_id": row.get("global_subject_id"), "split": row.get("split"),
              "metadata_sha256": _digest(row), "file_sha256": {},
              "swi_fingerprint": None, "errors": errors, "status": "failed"}
    for key in MAIN_PATHS:
        path = row.get(key)
        if not path:
            errors.append(f"Missing required primary path: {key}")
            continue
        try:
            result["file_sha256"][key] = _file_hash(path)
        except EXPECTED_FILE_ERRORS as exc:
            errors.append(f"Cannot hash {key}: {exc}")
    if row.get("swi_path"):
        try:
            result["swi_fingerprint"] = _swi_fingerprint(row["swi_path"])
        except EXPECTED_FILE_ERRORS as exc:
            errors.append(f"Cannot fingerprint SWI: {exc}")
    result["status"] = "failed" if errors else "passed"
    return result


def _validate_history(history):
    required = {"schema", "version", "checkpoint_sha256", "coverage", "source", "exposures"}
    if not isinstance(history, dict) or set(history) != required:
        raise ValueError(f"Initialization history requires exactly {sorted(required)}")
    if (history["schema"] != "swixqsm.initialization_history"
            or type(history["version"]) is not int or history["version"] != 1):
        raise ValueError("Unsupported initialization history schema/version")
    checkpoint_sha = history["checkpoint_sha256"]
    if not isinstance(checkpoint_sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha) is None:
        raise ValueError("History checkpoint_sha256 must be a 64-digit SHA256")
    if history["coverage"] not in {"complete", "unknown"}:
        raise ValueError("History coverage must be complete or unknown")
    source = _text(history["source"])
    if not source:
        raise ValueError("History source must be nonempty")
    if not isinstance(history["exposures"], list):
        raise ValueError("History exposures must be a list")
    exposures, seen = [], set()
    for exposure in history["exposures"]:
        if not isinstance(exposure, dict) or set(exposure) != {"global_subject_id", "role"}:
            raise ValueError("Each history exposure requires exactly global_subject_id and role")
        gid = _text(exposure["global_subject_id"])
        role = exposure["role"]
        if not gid or not isinstance(role, str) or role not in ROLES:
            raise ValueError("Each exposure requires a nonempty global identity and a known role")
        if (gid, role) in seen:
            raise ValueError("History contains a duplicate global_subject_id/role exposure")
        seen.add((gid, role))
        exposures.append({"global_subject_id": gid, "role": role})
    return {"schema": "swixqsm.initialization_history", "version": 1,
            "checkpoint_sha256": checkpoint_sha.lower(), "coverage": history["coverage"],
            "source": source, "exposures": sorted(exposures, key=lambda item: (item["global_subject_id"], item["role"]))}


def _inspect_initialization(rows, *, checkpoint_path=None, history_path=None,
                            declared_checkpoint_sha=None, read_checkpoint=True):
    errors, reasons = [], []
    result = {"kind": "pretrained" if checkpoint_path is not None or declared_checkpoint_sha is not None else "random",
              "status": "unverified", "checkpoint_path": None, "checkpoint_sha256": None,
              "history_path": None, "history_sha256": None, "history": None,
              "coverage": "unknown", "source": None, "holdout_exposures": [],
              "training_exposures": [], "errors": errors, "reasons": reasons}
    if result["kind"] == "random":
        if history_path is not None:
            errors.append("History is not used with random initialization; do not supply --history without --checkpoint")
            result["status"] = "failed"
        else:
            result.update(status="passed", coverage="not_applicable", source="random_initialization")
        return result
    if checkpoint_path is not None:
        result["checkpoint_path"] = str(Path(checkpoint_path).expanduser().resolve())
    if read_checkpoint:
        try:
            result["checkpoint_sha256"] = _file_hash(result["checkpoint_path"])
        except EXPECTED_FILE_ERRORS as exc:
            errors.append(f"Cannot hash initialization checkpoint: {exc}")
    else:
        result["checkpoint_sha256"] = declared_checkpoint_sha
    if history_path is None:
        reasons.append("Pretrained checkpoint has no complete initialization history")
    else:
        result["history_path"] = str(Path(history_path).expanduser().resolve())
        try:
            result["history_sha256"] = _file_hash(result["history_path"])
            with Path(result["history_path"]).open(encoding="utf-8") as stream:
                history = _validate_history(json.load(stream))
            result.update(history=history, coverage=history["coverage"], source=history["source"])
            if history["checkpoint_sha256"] != result["checkpoint_sha256"]:
                errors.append("Initialization history is bound to a different checkpoint SHA256")
            if history["coverage"] != "complete":
                reasons.append("Initialization-history coverage is unknown; empty exposures do not establish independence")
            split_by_global = {}
            for row in rows:
                gid = _text(row.get("global_subject_id"))
                if gid and isinstance(row.get("split"), str):
                    split_by_global.setdefault(gid, set()).add(row.get("split"))
            for exposure in history["exposures"]:
                splits = split_by_global.get(exposure["global_subject_id"], set())
                if splits & {"val", "test"}:
                    record = {**exposure, "current_splits": sorted(splits)}
                    result["holdout_exposures"].append(record)
                    errors.append(f"Current holdout {exposure['global_subject_id']!r} was previously exposed as {exposure['role']}")
                elif "train" in splits:
                    result["training_exposures"].append(exposure)
        except EXPECTED_FILE_ERRORS as exc:
            errors.append(f"Cannot validate initialization history: {exc}")
    result["status"] = "failed" if errors else "unverified" if reasons else "passed"
    return result


def build_independence_report(rows, *, checkpoint_path=None, history_path=None):
    """Audit all supplied splits, including SWI content and primary file hashes."""
    rows = [_normalize_row(row) for row in rows]
    errors = _identity_errors(rows)
    if not rows:
        errors.append("The audit requires a nonempty full manifest")
    cases = [_inspect_inputs(row) for row in rows]
    seen = {}
    for case in cases:
        fingerprint = case["swi_fingerprint"]
        if fingerprint is not None:
            value = fingerprint["sha256"]
            if value in seen:
                previous = seen[value]
                error = f"Decoded SWI content duplicates another case ({previous['subject_id']!r}, {case['subject_id']!r})"
                errors.append(error)
                for item in (previous, case):
                    if error not in item["errors"]:
                        item["errors"].append(error)
                    item["status"] = "failed"
            else:
                seen[value] = case
    if any(case["status"] != "passed" for case in cases):
        errors.append("One or more primary inputs failed content/hash inspection")
    initialization = _inspect_initialization(rows, checkpoint_path=checkpoint_path, history_path=history_path)
    errors.extend(initialization["errors"])
    status = "failed" if errors else initialization["status"]
    return {"schema": "swixqsm.independence_report", "version": 1, "status": status,
            "errors": errors, "manifest_identity_sha256": _manifest_identity(rows),
            "manifest_rows": rows, "n_cases": len(rows),
            "n_subjects": len({_text(row.get('global_subject_id')) for row in rows if _text(row.get('global_subject_id'))}),
            "fingerprint_method": FINGERPRINT_METHOD, "cases": cases, "initialization": initialization,
            "limitations": [
                "Global participant identifiers and their cross-cohort identity mapping are author declarations, not automatically verified identities.",
                "Complete initialization-history coverage is a source declaration; a file cannot prove the absence of unrecorded training, selection, or evaluation exposure.",
                "Exact decoded float32 SWI duplicates are detected across paths; nonidentical images, transformed/resampled copies, and incorrect identity aliases can evade content matching.",
                "Equal QSM values are never used to infer participant identity.",
                "The audit CLI reads SWI and hashes primary files from all splits; consuming workflows recheck only explicitly selected content splits.",
            ]}


def verify_independence_report(rows, report_path, *, checkpoint_path=None,
                               content_splits=("train", "val"), check_initialization=True):
    """Verify full-manifest metadata, selected file contents, and history binding.

    Training uses check_initialization=True and must match the actual G
    initializer, including random-vs-pretrained. Baseline/evaluation may set it
    False: no checkpoint is opened, while complete unchanged history evidence,
    metadata, and the originally passed audit remain mandatory.
    """
    if (not isinstance(content_splits, (list, tuple, set)) or not content_splits
            or any(not isinstance(split, str) or split not in SPLITS for split in content_splits)):
        raise ValueError("content_splits must be nonempty and contain train/val/test only")
    if type(check_initialization) is not bool:
        raise ValueError("check_initialization must be boolean")
    report_path = Path(report_path).expanduser().resolve()
    with report_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if (not isinstance(report, dict) or report.get("schema") != "swixqsm.independence_report"
            or type(report.get("version")) is not int or report.get("version") != 1):
        raise ValueError("Unsupported independence report schema/version")
    if report.get("status") != "passed" or report.get("errors") != []:
        raise ValueError("Independence report must be completely passed; failed/unknown history cannot be bypassed")
    rows = [_normalize_row(row) for row in rows]
    identity = _manifest_identity(rows)
    if identity != report.get("manifest_identity_sha256"):
        raise ValueError("Full manifest identity/metadata differs from the independence audit")
    if _manifest_identity(report.get("manifest_rows", [])) != identity:
        raise ValueError("Independence report's stored manifest metadata binding is invalid")
    identity_errors = _identity_errors(rows)
    if identity_errors:
        raise ValueError("Invalid current participant identity mapping: " + "; ".join(identity_errors))
    if report.get("fingerprint_method") != FINGERPRINT_METHOD:
        raise ValueError("Unsupported SWI content fingerprint method")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != len(rows) or report.get("n_cases") != len(rows):
        raise ValueError("Independence report case membership is inconsistent")
    expected_subjects = len({row["global_subject_id"] for row in rows})
    if report.get("n_subjects") != expected_subjects:
        raise ValueError("Independence report subject count is inconsistent")
    old, known_fingerprints = {}, set()
    for case in cases:
        if (not isinstance(case, dict) or not case.get("key") or case["key"] in old
                or case.get("status") != "passed" or case.get("errors") != []):
            raise ValueError("Independence report contains invalid, failed, or duplicate cases")
        hashes = case.get("file_sha256")
        if (not isinstance(hashes, dict) or set(hashes) != set(MAIN_PATHS)
                or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                       for value in hashes.values())):
            raise ValueError("Independence report has invalid primary-file hashes")
        fingerprint = case.get("swi_fingerprint")
        if (not isinstance(fingerprint, dict) or fingerprint.get("method") != FINGERPRINT_METHOD
                or not isinstance(fingerprint.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", fingerprint["sha256"]) is None
                or not isinstance(fingerprint.get("shape"), list) or len(fingerprint["shape"]) != 3
                or any(type(n) is not int or n <= 0 for n in fingerprint["shape"])):
            raise ValueError("Independence report has an invalid decoded SWI fingerprint")
        if fingerprint["sha256"] in known_fingerprints:
            raise ValueError("Independence report retains duplicate decoded SWI contents")
        known_fingerprints.add(fingerprint["sha256"])
        old[case["key"]] = case
    selected_count = 0
    for row in rows:
        previous = old.get(_case_key(row))
        if previous is None or previous.get("metadata_sha256") != _digest(row):
            raise ValueError("Independence case metadata does not match the full manifest")
        if row.get("split") in content_splits:
            selected_count += 1
            current = _inspect_inputs(row)
            if current["status"] != "passed":
                raise ValueError("Selected independence inputs fail reinspection: " + "; ".join(current["errors"]))
            for key, value in current.items():
                if key not in previous or _canonical(value) != _canonical(previous[key]):
                    raise ValueError(f"Selected case independence evidence changed: {key}")
    if not selected_count:
        raise ValueError("No cases belong to the selected content_splits")
    original = report.get("initialization")
    if not isinstance(original, dict) or original.get("status") != "passed":
        raise ValueError("Initialization audit is not passed")
    kind = original.get("kind")
    if kind not in {"random", "pretrained"}:
        raise ValueError("Invalid initialization kind in independence report")
    if check_initialization and ((kind == "random") != (checkpoint_path is None)):
        raise ValueError("Actual random/pretrained initialization differs from the audited initialization")
    declared_sha = original.get("checkpoint_sha256") if kind == "pretrained" else None
    if kind == "pretrained" and (not isinstance(declared_sha, str) or re.fullmatch(r"[0-9a-f]{64}", declared_sha) is None):
        raise ValueError("Invalid audited checkpoint SHA256")
    current_initialization = _inspect_initialization(
        rows, checkpoint_path=checkpoint_path if check_initialization else None,
        history_path=original.get("history_path"), declared_checkpoint_sha=declared_sha,
        read_checkpoint=check_initialization)
    if current_initialization["status"] != "passed":
        raise ValueError("Current initialization/history verification failed: "
                         + "; ".join(current_initialization["errors"] + current_initialization["reasons"]))
    # An explicit initializer can be relocated if its bytes are unchanged.
    ignored = {"checkpoint_path"}
    for key, value in current_initialization.items():
        if key not in ignored and (key not in original or _canonical(value) != _canonical(original[key])):
            raise ValueError(f"Initialization history binding changed: {key}")
    compact_initialization = {key: original.get(key) for key in
                              ("kind", "status", "checkpoint_sha256", "history_sha256", "coverage", "source")}
    return {"status": "verified", "report_path": str(report_path),
            "report_sha256": _file_hash(report_path), "manifest_identity_sha256": identity,
            "n_cases": len(rows), "n_subjects": report.get("n_subjects"),
            "content_splits": sorted(set(content_splits)), "n_content_cases": selected_count,
            "initialization_checked_against_current_checkpoint": check_initialization,
            "checkpoint_sha256": declared_sha, "initialization": copy.deepcopy(compact_initialization),
            "limitations": copy.deepcopy(report.get("limitations", []))}

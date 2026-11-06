"""Inventory explicit case directories and prepare a protected new split.

No NIfTI/header/array is opened. By default, root directories are ``train``
(original training pool) and ``validate`` (permanent test). Explicit ``flat``
layout instead assigns root-level case directories from both original ID lists.
A ready inventory only establishes declared identities, files and metadata;
it never means data QC or independence has passed.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re

from .split_training import create_split


ROLES = ("swi_path", "qsm_path", "swi_mask_path", "mask_path")
DEFAULT_QSM_NAMES = ("QSM.nii.gz", "QSM/QSM.nii.gz")
DEFAULT_VALID_NAMES = ("QSM_mask.nii.gz", "QSM/QSM_mask.nii.gz")
DEFAULT_SUBJECT_PATTERN = r"^(?P<participant>[0-9]+)(?:_.*)?$"
BASE_FIELDS = ("case_id", "participant_id", "subject_id", "global_subject_id", "cohort",
               "original_split", "split", *ROLES, "qsm_units", "input_mask_source",
               "source_case_path")
TEMPLATE_FIELDS = ("case_id", "original_split", "subject_id", "global_subject_id", *ROLES,
                   "qsm_units", "input_mask_source", "registration_review",
                   "registration_reviewer", "registration_review_date", "registration_notes",
                   "qsm_reference", "qsm_reference_method", "qsm_reference_region",
                   "qsm_reference_source", "qsm_reference_target_ppb",
                   "qsm_reference_mask_path", "qsm_reconstruction_mask_path", "space_id")


def _ids_sha(ids):
    """SHA256 of sorted unique IDs, UTF-8, one ID plus newline per entry."""
    return hashlib.sha256("".join(identifier + "\n" for identifier in sorted(set(ids))).encode("utf-8")).hexdigest()


def _csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _read_csv(path):
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames
    if not fields or len(set(fields)) != len(fields) or any(not field for field in fields):
        raise ValueError(f"CSV needs unique nonempty headers: {path}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"CSV row has a mismatched number of fields: {path}")
    return path, data, fields, rows


def _name_candidates(value, role):
    values = [value] if isinstance(value, str) else list(value)
    if not values or any(not isinstance(name, str) or not name.strip() for name in values):
        raise ValueError(f"{role} requires nonempty literal case-relative filenames")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate filename candidate for {role}")
    for name in values:
        path = Path(name)
        if (name != name.strip() or path.is_absolute() or ".." in path.parts
                or name in (".", "") or any(char in name for char in "*?[]\\")):
            raise ValueError(f"Unsafe {role} filename: {name!r}; use literal paths inside each case")
    return values


def _resolve_metadata_paths(row, parent):
    result = dict(row)
    for key, value in row.items():
        if not value or not value.strip():
            continue
        if key.endswith("_path"):
            path = Path(value).expanduser()
            result[key] = str((parent/path if not path.is_absolute() else path).resolve())
        elif key.endswith("_paths"):
            try:
                values = json.loads(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{key} must be a JSON array of nonempty paths") from exc
            if not isinstance(values, list) or any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"{key} must be a JSON array of nonempty paths")
            result[key] = json.dumps([str((parent/Path(item).expanduser()
                                          if not Path(item).expanduser().is_absolute()
                                          else Path(item).expanduser()).resolve()) for item in values],
                                      ensure_ascii=False)
    return result


def _metadata(path):
    if path is None:
        return {}, None
    path, data, fields, rows = _read_csv(path)
    if not {"case_id", "original_split"}.issubset(fields):
        raise ValueError("metadata_csv requires case_id and original_split (train/test) columns")
    reserved = {"split", "source_case_path", "directory_present", "case_origin",
                "role_candidates_json", "issues_json", "complete_required_files"}
    if reserved.intersection(fields):
        raise ValueError("metadata_csv cannot override split, source_case_path or inventory bookkeeping fields")
    result = {}
    for row in rows:
        key = row["case_id"], row["original_split"]
        if not key[0] or key[0] != key[0].strip() or key[1] not in ("train", "test"):
            raise ValueError("Metadata rows need a nonempty case_id and original_split=train/test")
        if key in result:
            raise ValueError(f"Duplicate metadata key: {key}")
        result[key] = _resolve_metadata_paths(row, path.parent)
    return result, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "n_rows": len(rows)}


def _id_list(path, pattern, split):
    if path is None:
        return None, None
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    text = data.decode("utf-8-sig")
    if path.suffix.lower() == ".csv":
        _, _, fields, rows = _read_csv(path)
        names = {name.lower(): name for name in fields}
        candidates = [names[name] for name in ("participant_id", "eid", "subject_id", "global_subject_id", "id") if name in names]
        if len(candidates) != 1:
            raise ValueError("ID-list CSV needs exactly one ID column: participant_id/eid/subject_id/global_subject_id/id")
        raw = []
        for row in rows:
            declared_split = row.get("original_split", row.get("split", split))
            if declared_split != split:
                raise ValueError(f"{split} ID-list row declares a different split")
            raw.append(row[candidates[0]])
    else:
        raw = [line.strip() for line in text.splitlines() if line.strip()]
    ids = []
    for value in raw:
        if not value or value != value.strip():
            raise ValueError("Original ID lists require nonempty IDs without surrounding whitespace")
        match = pattern.fullmatch(value)
        if match is None or not match.group("participant"):
            raise ValueError(f"ID-list entry does not match subject_pattern: {value!r}")
        ids.append(match.group("participant"))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate participant in original {split} ID list")
    return set(ids), {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                      "participant_ids": sorted(ids), "sorted_ids_sha256": _ids_sha(ids), "n_participants": len(ids)}


def build_manifest(data_root, output, *, qsm_units=None, metadata_csv=None,
                   original_train_ids=None, original_test_ids=None,
                   expected_train=5083, expected_test=1773, inventory_only=False,
                   val_fraction=.1, seed=42, swi_name="SWI.nii.gz",
                   qsm_name=DEFAULT_QSM_NAMES, swi_mask_name="brain_mask.nii.gz",
                   valid_mask_name=DEFAULT_VALID_NAMES, input_mask_source=None,
                   subject_pattern=DEFAULT_SUBJECT_PATTERN, layout="split_dirs"):
    """Inventory metadata and, if ready, write original + new split manifests.

    ``qsm_name``/``valid_mask_name`` (also other role names) accept a string or
    explicit ordered list/tuple. Exactly one existing candidate may be used;
    multiple existing candidates block readiness unless metadata explicitly
    names the intended file. No recursive discovery or first-match selection.

    A direct case directory name is its ``case_id``. The regex's named group
    ``participant`` is its original EID; default subject/global IDs are EID and
    ``cohort:EID``. ID-list CSV/TXT entries are matched to these regex-extracted
    participant IDs, not guessed from overridden identity labels. Metadata is
    keyed by (case_id, original_split), may declare identities/roles/QC fields,
    and resolves all *_path / JSON *_paths against the metadata CSV parent.

    ``split_dirs`` preserves the train/validate directory convention. ``flat``
    requires both disjoint original participant lists, inventories every visible
    direct directory, and never guesses membership from ordering or metadata.
    Unmatched directories remain unassigned in the draft and block readiness.

    Ordinary incompleteness writes inventory + draft and returns ready=False.
    Unsafe configuration or malformed/duplicate metadata raises before output.
    inventory_only always returns ready=False and never creates a new split.
    A ready=True record is preparation readiness only, never passed image QC.
    """
    if layout not in ("split_dirs", "flat"):
        raise ValueError("layout must be explicitly split_dirs or flat")
    if layout == "flat" and (original_train_ids is None or original_test_ids is None):
        raise ValueError("flat layout requires both original_train_ids and original_test_ids")
    for name, value in (("expected_train", expected_train), ("expected_test", expected_test)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be an explicit nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    if isinstance(val_fraction, bool) or not isinstance(val_fraction, (int, float)) or not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be finite and strictly between zero and one")
    if qsm_units not in (None, "ppm", "ppb", "stored"):
        raise ValueError("qsm_units must be explicitly ppm, ppb or stored, or omitted for a draft")
    if input_mask_source is not None and (not isinstance(input_mask_source, str) or not input_mask_source.strip()):
        raise ValueError("input_mask_source must be a nonempty actual provenance declaration")
    try:
        pattern = re.compile(subject_pattern)
    except (TypeError, re.error) as exc:
        raise ValueError("subject_pattern must be a valid regex with named group participant") from exc
    if "participant" not in pattern.groupindex:
        raise ValueError("subject_pattern must contain named group participant")
    role_names = {role: _name_candidates(value, role) for role, value in zip(
        ROLES, (swi_name, qsm_name, swi_mask_name, valid_mask_name))}
    original_root = Path(os.path.abspath(Path(data_root).expanduser()))
    original_output = Path(os.path.abspath(Path(output).expanduser()))
    root = original_root.resolve()
    output = original_output.resolve()
    if layout == "flat" and (root == output or root in output.parents
                              or original_root == original_output or original_root in original_output.parents):
        raise ValueError("flat output must be outside the entire input data root; no descendant is allowed")
    if os.path.lexists(original_output) or output.exists():
        raise FileExistsError("Manifest output already exists; use a new directory")
    split_dirs = ({"train": root/"train", "test": root/"validate"} if layout == "split_dirs"
                  else {"train": root, "test": root})
    resolved_dirs = [path.resolve() for path in split_dirs.values()]
    if layout == "split_dirs" and (resolved_dirs[0] == resolved_dirs[1] or resolved_dirs[0] in resolved_dirs[1].parents
                                   or resolved_dirs[1] in resolved_dirs[0].parents):
        raise ValueError("Original train and validate/test directories overlap")
    if any(directory == output or directory in output.parents for directory in resolved_dirs):
        raise ValueError("Output must not create new case directories inside original train or validate/test")
    metadata, metadata_source = _metadata(metadata_csv)
    declared, id_sources = {}, {}
    for split, path in (("train", original_train_ids), ("test", original_test_ids)):
        declared[split], id_sources[split] = _id_list(path, pattern, split)
    if declared["train"] is not None and declared["test"] is not None and declared["train"] & declared["test"]:
        raise ValueError("Original train/test participant ID lists overlap")

    issues, cases, ignored = [], {}, []

    def issue(code, message, *, key=None, role=None, **details):
        entry = {"severity": "error", "code": code, "message": message, **details}
        if key is not None:
            entry.update(case_id=key[0], original_split=key[1])
        if role is not None:
            entry["role"] = role
        issues.append(entry)
        return entry

    discovery_roots = list(split_dirs.items()) if layout == "split_dirs" else [("unassigned", root)]
    for discovery_split, directory in discovery_roots:
        if not directory.is_dir():
            issue("missing_split_directory" if layout == "split_dirs" else "missing_data_root",
                  f"Missing original {discovery_split} directory" if layout == "split_dirs" else "Missing flat data root",
                  path=str(directory))
            continue
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.name.startswith("."):
                ignored.append({"path": str(child), "reason": "hidden_or_AppleDouble"})
                continue
            if not child.is_dir():
                if layout == "flat" and not child.is_file():
                    key = (child.name, "unassigned")
                    issue("unclassified_flat_entry", "Visible flat entry is neither a case directory nor an ordinary file; it was not silently ignored",
                          key=key, path=str(child))
                    cases[key] = {"directory": child, "present": False, "origin": "unclassified_flat_entry"}
                    continue
                ignored.append({"path": str(child), "reason": "not_a_direct_case_directory"})
                continue
            if any(key[0] == child.name for key in cases):
                raise ValueError(f"Same case_id appears in both original splits: {child.name}")
            split = discovery_split
            if layout == "flat":
                match = pattern.fullmatch(child.name)
                participant = match.group("participant") if match else ""
                memberships = [name for name in ("train", "test") if participant in declared[name]]
                if len(memberships) == 1:
                    split = memberships[0]
                elif participant:
                    issue("unassigned_flat_participant", "Flat case participant is absent from both original ID lists; membership was not guessed",
                          key=(child.name, "unassigned"), participant_id=participant, path=str(child))
            origin = ("directory" if layout == "split_dirs" else
                      "directory_original_id_list_membership" if split != "unassigned" else "unassigned_flat_directory")
            cases[(child.name, split)] = {"directory": child, "present": True, "origin": origin}
            if layout == "flat" and (child.resolve() == output or child.resolve() in output.parents):
                raise ValueError("Output must not be inside an original case directory, including symlink targets")
    actual_ids = {split: set() for split in split_dirs}
    for key, info in cases.items():
        match = pattern.fullmatch(key[0])
        participant = match.group("participant") if match else ""
        info["participant"] = participant
        if participant and key[1] in actual_ids:
            actual_ids[key[1]].add(participant)
        elif not participant:
            issue("invalid_case_identity", "Case directory does not match participant regex", key=key)
    overlap = actual_ids["train"] & actual_ids["test"]
    if overlap:
        issue("participant_identity_crosses_splits", "Original participant EIDs occur in both train and test; metadata aliases cannot change membership", participant_ids=sorted(overlap))
    for split in split_dirs:
        if declared[split] is not None:
            missing, unexpected = declared[split] - actual_ids[split], actual_ids[split] - declared[split]
            if missing:
                issue("missing_declared_participants", "Original membership includes participants without a case directory", original_split=split, participant_ids=sorted(missing))
            if unexpected:
                issue("unexpected_participants", "Case directories include participants absent from original membership", original_split=split, participant_ids=sorted(unexpected))
            for participant in sorted(missing):
                key = (participant, split)
                if any(existing[0] == participant for existing in cases):
                    # An invalid/custom-pattern directory must remain visible;
                    # never overwrite it with a synthetic missing-member row.
                    key = ("missing-participant:"+participant, split)
                cases[key] = {"directory": split_dirs[split]/participant, "present": False,
                              "origin": "original_id_list_missing_directory", "participant": participant}
    for key in metadata:
        if key not in cases:
            issue("unknown_metadata_case", "Metadata key has no discovered or original-list case", key=key)
            match = pattern.fullmatch(key[0])
            cases[key] = {"directory": split_dirs[key[1]]/key[0], "present": False,
                          "origin": "unknown_metadata_case", "participant": match.group("participant") if match else ""}

    manifest_rows, inventory_rows = [], []
    for key, info in sorted(cases.items(), key=lambda item: (item[0][1] == "test", item[0][0])):
        case_id, split = key
        participant, directory = info["participant"], info["directory"]
        row = dict(case_id=case_id, participant_id=participant, subject_id=participant,
                   global_subject_id="cohort:"+participant if participant else "", cohort="primary",
                   original_split=split, split=split, qsm_units=qsm_units or "",
                   input_mask_source=input_mask_source or "", source_case_path=str(directory.resolve()))
        override = metadata.get(key, {})
        if override.get("participant_id") and override["participant_id"] != participant:
            issue("participant_override_conflict", "Metadata participant_id cannot rewrite directory/list membership", key=key)
        for field, value in override.items():
            if field not in ("case_id", "original_split", "participant_id") and value.strip():
                row[field] = value
        candidate_records = {}
        for role in ROLES:
            explicit = override.get(role)
            explicit = explicit if explicit and explicit.strip() else None
            candidates = [Path(explicit)] if explicit and explicit.strip() else [directory/name for name in role_names[role]]
            observations, found = [], []
            for candidate in candidates:
                candidate = candidate.resolve()
                escaped = not explicit and directory.resolve() not in candidate.parents
                exists = candidate.is_file() if not escaped else False
                observations.append({"path": str(candidate), "is_file": exists, "explicit_metadata": bool(explicit), "escapes_case": escaped})
                if escaped:
                    issue("candidate_escapes_case", "A literal role candidate resolves outside its case directory", key=key, role=role, path=str(candidate))
                if exists:
                    found.append(str(candidate))
            candidate_records[role] = observations
            if len(found) == 1:
                row[role] = found[0]
            elif not found:
                row[role] = str(candidates[0].resolve()) if explicit else ""
                issue("missing_role_file", "Required role file is missing; case retained in draft", key=key, role=role, candidates=[item["path"] for item in observations])
            else:
                row[role] = ""
                issue("ambiguous_role_files", "Multiple explicit candidates exist; metadata must select the intended path", key=key, role=role, candidates=found)
        if not info["present"]:
            issue("missing_case_directory", "Case retained despite absent original case directory", key=key, path=str(directory))
        if row.get("qsm_units") not in ("ppm", "ppb", "stored"):
            issue("missing_or_invalid_qsm_units", "Declare ppm/ppb, or stored for unverified original numerical values", key=key)
        if not row.get("input_mask_source", "").strip():
            issue("missing_input_mask_source", "Declare actual SWI/deployment-available input brain-mask provenance", key=key)
        for field in ("subject_id", "global_subject_id"):
            if not row.get(field) or row[field] != row[field].strip():
                issue("invalid_identity", f"{field} must be nonempty without surrounding whitespace", key=key)
        used = {}
        for role in ROLES:
            path = row.get(role)
            if path:
                if path in used:
                    issue("roles_share_file", "Distinct image/mask roles must not reuse one file; QSM mask is not an input SWI mask", key=key, roles=[used[path], role], path=path)
                used[path] = role
        manifest_rows.append(row)
        inventory_rows.append({**row, "directory_present": info["present"], "case_origin": info["origin"],
                               "role_candidates_json": json.dumps(candidate_records, ensure_ascii=False, sort_keys=True)})

    group_splits, sid_globals, sid_splits, used_role_paths, used_directories, participant_globals = {}, {}, {}, {}, {}, {}
    for row, inventory in zip(manifest_rows, inventory_rows):
        key = row["case_id"], row["original_split"]
        gid, sid, split = row["global_subject_id"], row["subject_id"], row["split"]
        if layout == "flat" and row["participant_id"]:
            participant = row["participant_id"]
            if participant in participant_globals and participant_globals[participant] != gid:
                issue("participant_global_conflict", "Metadata cannot split visits of one original participant into different global groups",
                      key=key, participant_id=participant)
            participant_globals[participant] = gid
        for identifier, groups, code in ((gid, group_splits, "global_identity_crosses_splits"),
                                         (sid, sid_splits, "subject_identity_crosses_splits")):
            if identifier and identifier in groups and groups[identifier] != split:
                issue(code, "Participant identity crosses original train/test boundary", key=key, identity=identifier)
            groups[identifier] = split
        if sid in sid_globals and sid_globals[sid] != gid:
            issue("subject_global_conflict", "One subject_id maps to multiple global identities", key=key)
        sid_globals[sid] = gid
        if inventory["directory_present"]:
            path = row["source_case_path"]
            if path in used_directories:
                issue("duplicate_case_directory", "Different case names resolve to the same directory", key=key, other_case=list(used_directories[path]))
            used_directories[path] = key
        for role in ROLES:
            path = row.get(role)
            if path and (role, path) in used_role_paths:
                issue("duplicate_role_path", "Multiple cases reuse the same role file", key=key, role=role, other_case=list(used_role_paths[(role, path)]), path=path)
            if path:
                used_role_paths[(role, path)] = key
    counts = {}
    for split, expected in (("train", expected_train), ("test", expected_test)):
        present = [row for row in inventory_rows if row["original_split"] == split and row["directory_present"]]
        gids = {row["global_subject_id"] for row in present if row["global_subject_id"]}
        counts[split] = {"case_directories": len(present), "participants": len(actual_ids[split]),
                         "global_subjects": len(gids), "expected_global_subjects": expected,
                         "inventory_rows_including_missing": sum(row["original_split"] == split for row in inventory_rows)}
        if len(gids) != expected:
            issue("unexpected_global_subject_count", "Global participant count does not match explicit expected count", original_split=split, actual=len(gids), expected=expected)
    if not inventory_only and counts["train"]["global_subjects"] < 2:
        issue("insufficient_train_groups", "At least two original training participant groups are needed for train/val")
    data_complete = not issues
    ready = data_complete and not inventory_only
    case_issue_map = {}
    for item in issues:
        if "case_id" in item:
            case_issue_map.setdefault((item["case_id"], item["original_split"]), []).append(item)
    for row in inventory_rows:
        row["issues_json"] = json.dumps(case_issue_map.get((row["case_id"], row["original_split"]), []), ensure_ascii=False)
        row["complete_required_files"] = all(row.get(role) and Path(row[role]).is_file() for role in ROLES)
    manifest_fields = [*BASE_FIELDS, *sorted(set().union(*(row.keys() for row in manifest_rows)) - set(BASE_FIELDS))]
    inventory_fields = [*manifest_fields, "directory_present", "case_origin", "role_candidates_json", "issues_json", "complete_required_files"]
    test_ids = sorted(actual_ids["test"] | (declared["test"] or set()))
    test_global_ids = sorted({row["global_subject_id"] for row in manifest_rows if row["split"] == "test" and row["global_subject_id"]})
    membership_codes = {"missing_declared_participants", "unexpected_participants"}
    if layout == "flat":
        membership_codes.update(("unassigned_flat_participant", "unclassified_flat_entry", "invalid_case_identity", "unknown_metadata_case", "participant_global_conflict"))
    record = {
        "schema": "swixqsm.swixqsm_manifest_inventory", "version": 1,
        "data_root": str(root), "output_directory": str(output), "layout": layout, "ready": ready,
        "data_complete": data_complete, "inventory_only": bool(inventory_only),
        "qc_status": "not_evaluated", "independence_status": "metadata_only_not_content_audited",
        "split_directory_policy": {"train": str(split_dirs["train"]), "test": str(split_dirs["test"])},
        "validate_directory_is_test": layout == "split_dirs", "counts": counts, "issues": issues,
        "membership_policy": "explicit_original_id_lists" if layout == "flat" else "train_validate_directory_names_checked_against_supplied_lists",
        "unassigned_case_ids": sorted(row["case_id"] for row in inventory_rows if row["original_split"] == "unassigned"),
        "subject_pattern": subject_pattern, "default_identity_policy": "subject_id=participant; global_subject_id=cohort:participant; metadata may explicitly override",
        "qsm_units_argument": qsm_units, "input_mask_source_argument": input_mask_source,
        "role_name_candidates": role_names, "metadata_source": metadata_source,
        "original_id_list_sources": id_sources, "original_membership_verified": all(declared[split] is not None for split in split_dirs) and not any(item["code"] in membership_codes for item in issues),
        "test_participant_ids": test_ids, "test_participant_ids_sorted_sha256": _ids_sha(test_ids),
        "actual_test_participant_ids": sorted(actual_ids["test"]), "actual_test_participant_ids_sorted_sha256": _ids_sha(actual_ids["test"]),
        "test_global_subject_ids": test_global_ids, "test_global_subject_ids_sorted_sha256": _ids_sha(test_global_ids),
        "id_hash_definition": "SHA256 over sorted unique UTF-8 IDs, each followed by newline",
        "ignored_entries": ignored, "image_header_reads": 0, "image_array_reads": 0,
        "read_scope": "Directory/file existence metadata and explicitly supplied CSV/TXT only; no image content, headers or voxel arrays",
        "limitations": ["Ready means files/identity declarations available for preparation, never that image QC passed",
                        "Units and input-mask source are user declarations; no intensity-based inference or reference-zero verification",
                        "Flat membership comes only from both supplied original ID lists; unmatched directories are retained as unassigned" if layout == "flat" else "Default original membership is inferred from directory names; both original ID lists are needed to verify participant sets",
                        "Missing/ambiguous cases are retained and block split creation; no complete-case subset is silently selected",
                        "Run registration/unit/reference and content-independence audits before training"],
        "artifacts": {"inventory_cases": str(output/"inventory_cases.csv"),
                      "draft_manifest": str(output/"original_train_test.draft.csv"),
                      "case_metadata_template": str(output/"case_metadata.template.csv")},
        "split_configuration": {"val_fraction": val_fraction, "seed": seed},
    }
    output.mkdir(parents=True, exist_ok=False)
    (output/"inventory_cases.csv").write_bytes(_csv_bytes(inventory_rows, inventory_fields))
    (output/"original_train_test.draft.csv").write_bytes(_csv_bytes(manifest_rows, manifest_fields))
    template_rows = [{**{field: row.get(field, "") for field in TEMPLATE_FIELDS},
                      "registration_review": "pending"} for row in manifest_rows]
    (output/"case_metadata.template.csv").write_bytes(_csv_bytes(template_rows, TEMPLATE_FIELDS))
    if ready:
        original_path = output/"original_train_test.csv"
        original_bytes = _csv_bytes(manifest_rows, manifest_fields)
        original_path.write_bytes(original_bytes)
        split_record = create_split(original_path, output/"split", val_fraction=val_fraction, seed=seed)
        record["artifacts"].update(original_manifest=str(original_path), split_manifest=str(output/"split/manifest.csv"), split_record=str(output/"split/split_record.json"))
        record["original_manifest_sha256"] = hashlib.sha256(original_bytes).hexdigest()
        record["new_split_group_counts"] = split_record["group_counts"]
        record["test_unchanged_by_new_split"] = split_record["test_membership_unchanged"]
    else:
        record["split_not_created_reason"] = "inventory_only" if inventory_only else "blocking_inventory_issues"
    (output/"inventory.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="New directory")
    parser.add_argument("--layout", choices=["split_dirs", "flat"], default="split_dirs",
                        help="split_dirs: train/validate subdirectories; flat: root-level cases assigned only by both original ID lists")
    parser.add_argument("--qsm-units", choices=["ppm", "ppb", "stored"])
    parser.add_argument("--input-mask-source", help="Actual provenance of deployment-available SWI brain masks")
    parser.add_argument("--metadata-csv")
    parser.add_argument("--original-train-ids")
    parser.add_argument("--original-test-ids")
    parser.add_argument("--expected-train", type=int, default=5083)
    parser.add_argument("--expected-test", type=int, default=1773)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swi-name", default="SWI.nii.gz")
    parser.add_argument("--qsm-name", action="append", help="Repeat for explicit candidates; default QSM.nii.gz and QSM/QSM.nii.gz")
    parser.add_argument("--swi-mask-name", default="brain_mask.nii.gz")
    parser.add_argument("--valid-mask-name", action="append", help="Repeat for explicit candidates; default QSM_mask.nii.gz and QSM/QSM_mask.nii.gz")
    parser.add_argument("--subject-pattern", default=DEFAULT_SUBJECT_PATTERN)
    args = vars(parser.parse_args())
    args["qsm_name"] = args["qsm_name"] or DEFAULT_QSM_NAMES
    args["valid_mask_name"] = args["valid_mask_name"] or DEFAULT_VALID_NAMES
    try:
        record = build_manifest(**args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({"ready": record["ready"], "inventory_only": record["inventory_only"],
                      "counts": record["counts"], "n_issues": len(record["issues"]),
                      "inventory": str(Path(args["output"])/"inventory.json")}, ensure_ascii=False))
    if not record["ready"] and not record["inventory_only"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

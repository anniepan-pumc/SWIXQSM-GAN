"""Prepare a protected train/val/test split and run selected prospective comparisons.

Usage: python -m swixqsm_minimal.run_server --config server_training.json
       --model pix2pix [--prepare-only]

Child programs receive argument lists, never shell command strings.  The old
validate directory is a protected test source, not model-selection validation.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from .train_models import DEFAULTS, FOUR_MODELS, RECIPES
from .util.normalization import unit_factor


TOP_KEYS = {"schema", "version", "data_root", "work_dir", "model", "manifest", "quality", "input_masks",
            "independence", "training", "simple_baselines"}
MANIFEST_KEYS = {"layout", "qsm_units", "metadata_csv", "original_train_ids", "original_test_ids",
                 "expected_train", "expected_test", "val_fraction", "seed", "swi_name", "qsm_name",
                 "swi_mask_name", "valid_mask_name", "input_mask_source", "subject_pattern"}
TRAINING_KEYS = {"units", "qsm_units", "normalization_file", "qsm_scale", "qsm_offset", "reference_id", "size", "batch_size", "epochs",
                 "lr", "lr_disc", "lr_policy", "min_lr", "warmup_epochs", "early_stopping_patience",
                 "plateau_factor", "plateau_patience", "plateau_threshold",
                 "seed", "device", "sampling_mode", "cache_volumes", "cache_mib", "volume_group_size",
                 "diagnostic_batches", "adversarial_mode", "deterministic", "num_threads", "max_epochs_this_run"}
ROLES = ("swi_path", "qsm_path", "swi_mask_path", "mask_path")
MODEL_CHOICES = (*RECIPES, "all", "reviewer_suite")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _ids_sha(values):
    return hashlib.sha256("".join(value + "\n" for value in sorted(values)).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _mapping(value, allowed, name):
    if not isinstance(value, dict) or set(value) - allowed:
        extra = sorted(set(value) - allowed) if isinstance(value, dict) else []
        raise ValueError(f"{name} must be an object with supported keys only; unknown keys: {extra}")
    return dict(value)


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Configuration numbers must be finite")
    if isinstance(value, dict):
        for child in value.values():
            _finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _finite_json(child)


def _path(value, parent, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a nonempty path")
    path = Path(value).expanduser()
    return str((parent / path if not path.is_absolute() else path).resolve())


def _policy(value, allowed, name, parent, default=None):
    policy = _mapping(default if value is None else value, allowed, name)
    if len(policy) != 1:
        raise ValueError(name + " requires exactly one explicit policy")
    key, item = next(iter(policy.items()))
    if key in {"contract", "report"}:
        policy[key] = _path(item, parent, name + "." + key)
        if not Path(policy[key]).is_file():
            raise FileNotFoundError(policy[key])
    elif key == "run_audit":
        if item is not True:
            raise ValueError("independence.run_audit must be true; otherwise supply a report or explicit reason")
    elif not isinstance(item, str) or not item.strip():
        raise ValueError(name + " opt-out requires a nonempty reason")
    return policy


def load_config(config_path, model=None):
    """Validate the whitelist and resolve every configuration-owned path."""
    config_path = Path(config_path).expanduser().resolve()
    raw = _mapping(json.loads(config_path.read_text()), TOP_KEYS, "configuration")
    _finite_json(raw)
    if raw.get("schema") != "swixqsm.server_training" or type(raw.get("version")) is not int or raw["version"] != 1:
        raise ValueError("Expected swixqsm.server_training schema version 1")
    parent = config_path.parent
    config = dict(raw)
    for key in ("data_root", "work_dir"):
        config[key] = _path(raw.get(key), parent, key)
    data, work = Path(config["data_root"]), Path(config["work_dir"])
    protected = (data / "train", data / "validate")
    if work == data or any(work == directory or directory in work.parents for directory in protected):
        raise ValueError("work_dir must not be the data root or lie inside original train/validate case directories")
    selected = model if model is not None else raw.get("model", "all")
    if selected not in MODEL_CHOICES:
        raise ValueError("Unsupported model selection")
    config["model"] = selected
    manifest = _mapping(raw.get("manifest", {}), MANIFEST_KEYS, "manifest")
    layout = manifest.get("layout", "split_dirs")
    if not isinstance(layout, str) or layout not in {"split_dirs", "flat"}:
        raise ValueError("manifest.layout must be split_dirs or flat")
    if layout == "flat" and (work == data or data in work.parents):
        raise ValueError("Flat-layout work_dir must be outside the input data root")
    for key in ("metadata_csv", "original_train_ids", "original_test_ids"):
        if manifest.get(key) is None:
            manifest.pop(key, None)
            continue
        if key in manifest:
            manifest[key] = _path(manifest[key], parent, "manifest." + key)
            if not Path(manifest[key]).is_file():
                raise FileNotFoundError(manifest[key])
    if layout == "flat" and not all(manifest.get(key) for key in ("original_train_ids", "original_test_ids")):
        raise ValueError("Flat layout requires both original_train_ids and original_test_ids; directory order cannot define splits")
    manifest.setdefault("expected_train", 5083)
    manifest.setdefault("expected_test", 1773)
    for key in ("expected_train", "expected_test"):
        if type(manifest[key]) is not int or manifest[key] < 1:
            raise ValueError("manifest." + key + " must be a positive integer")
    for key in ("swi_name", "qsm_name", "swi_mask_name", "valid_mask_name"):
        if key in manifest:
            names = manifest[key] if isinstance(manifest[key], list) else [manifest[key]]
            if not names or any(not isinstance(name, str) or not name.strip() or Path(name).is_absolute()
                                or ".." in Path(name).parts for name in names):
                raise ValueError("Manifest role names must be safe relative filenames, not arbitrary arguments or parent paths")
    training = _mapping(raw.get("training", {}), TRAINING_KEYS, "training")
    if training.get("units") not in {"ppm", "ppb", "stored"}:
        raise ValueError("training.units must explicitly be ppm, ppb or stored")
    if "qsm_units" in training and (not isinstance(training["qsm_units"], str)
                                     or training["qsm_units"] not in {"ppm", "ppb", "stored"}):
        raise ValueError("training.qsm_units must be ppm, ppb or stored")
    fitted_fallback = None
    if "normalization_file" in training:
        if "qsm_scale" in training or "qsm_offset" in training:
            raise ValueError("training.normalization_file cannot accompany qsm_scale or qsm_offset")
        training["normalization_file"] = _path(training["normalization_file"], parent, "training.normalization_file")
        if not Path(training["normalization_file"]).is_file():
            raise FileNotFoundError(training["normalization_file"])
        from .fit_normalization import load_fitted_normalization
        # Configuration validation reads the JSON only. The training child
        # verifies actual train membership and source image hashes later.
        _, fitted = load_fitted_normalization(training["normalization_file"],
                                             units=training["units"], verify_inputs=False)
        fitted_fallback = fitted["fit_selection"].get("default_input_qsm_units")
        if fitted_fallback is not None and (not isinstance(fitted_fallback, str)
                                             or fitted_fallback not in {"ppm", "ppb", "stored"}):
            raise ValueError("Fitted normalization has invalid input-unit fallback")
        if training.get("qsm_units") and fitted_fallback and training["qsm_units"] != fitted_fallback:
            raise ValueError("training.qsm_units conflicts with fitted input-unit fallback")
    elif (isinstance(training.get("qsm_scale"), bool) or not isinstance(training.get("qsm_scale"), (int, float))
          or training["qsm_scale"] <= 0):
        raise ValueError("training.qsm_scale must be an explicit positive fixed scale or supply normalization_file")
    training.setdefault("seed", 42)
    if type(training["seed"]) is not int or not 0 <= training["seed"] < 2**32:
        raise ValueError("training.seed must be an integer in [0,2**32)")
    if "seed" in manifest and (type(manifest["seed"]) is not int or manifest["seed"] != training["seed"]):
        raise ValueError("Manifest split seed and training seed must agree")
    manifest["seed"] = training["seed"]
    input_fallback = training.get("qsm_units") or fitted_fallback
    if input_fallback is None and "normalization_file" not in training and manifest.get("qsm_units") is None:
        input_fallback = training["units"]  # Retain the manual-scale configuration contract.
    if manifest.get("qsm_units") is None:
        manifest["qsm_units"] = input_fallback
    if manifest["qsm_units"] not in {None, "ppm", "ppb", "stored"}:
        raise ValueError("manifest.qsm_units must be ppm, ppb or stored")
    if input_fallback and manifest["qsm_units"] != input_fallback:
        raise ValueError("Manifest and training input-unit fallback declarations conflict")
    if manifest["qsm_units"] is not None:
        unit_factor(manifest["qsm_units"], training["units"])
    for key, value in training.items():
        if key in {"units", "qsm_units", "normalization_file", "qsm_scale", "reference_id", "device", "lr_policy", "sampling_mode", "adversarial_mode"}:
            continue
        if key == "deterministic":
            if type(value) is not bool:
                raise ValueError("training.deterministic must be boolean")
        elif key in {"qsm_offset", "lr", "lr_disc", "min_lr", "cache_mib", "plateau_factor", "plateau_threshold"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("training." + key + " must be numeric")
        elif value is not None and type(value) is not int:
            raise ValueError("training." + key + " must be an integer")
    if training.get("reference_id") is not None and (not isinstance(training["reference_id"], str)
                                                       or not training["reference_id"].strip()):
        raise ValueError("training.reference_id must be a nonempty actual reference declaration")
    if training.get("device") is not None:
        import torch
        if not isinstance(training["device"], str) or not training["device"].strip():
            raise ValueError("training.device must name CPU or CUDA")
        if torch.device(training["device"]).type not in {"cpu", "cuda"}:
            raise ValueError("Comparison training supports CPU or CUDA only")
    quality = _policy(raw.get("quality"), {"contract", "report", "allow_unverified_reason"}, "quality", parent, default={})
    if training["units"] == "stored" and "allow_unverified_reason" not in quality:
        raise ValueError("stored training requires explicit unverified quality; physical QC contracts cannot certify unknown units")
    independence = _policy(raw.get("independence"), {"run_audit", "report", "allow_unverified_reason"},
                           "independence", parent, default={"run_audit": True})
    simple = _mapping(raw.get("simple_baselines", {}), {"enabled", "reference_id", "space_id", "units"}, "simple_baselines")
    simple.setdefault("enabled", False)
    simple.setdefault("units", "ppb")
    if type(simple["enabled"]) is not bool or simple["units"] not in {"ppm", "ppb"}:
        raise ValueError("simple_baselines needs boolean enabled and ppm/ppb units")
    if simple["enabled"]:
        if training["units"] == "stored" or manifest["qsm_units"] == "stored":
            raise ValueError("stored values do not support anatomical mean/linear baseline fitting in this workflow")
        for key in ("reference_id", "space_id"):
            if not isinstance(simple.get(key), str) or not simple[key].strip():
                raise ValueError("Fitting anatomical mean/linear baselines requires actual " + key)
        if training.get("reference_id") not in (None, simple["reference_id"]):
            raise ValueError("Simple-baseline and training reference declarations conflict")
    masks = _mapping(raw.get("input_masks", {}), {"enabled", "skull_stripped", "fill_holes", "valid_mask_policy"}, "input_masks")
    masks.setdefault("enabled", False)
    masks.setdefault("skull_stripped", False)
    masks.setdefault("fill_holes", True)
    masks.setdefault("valid_mask_policy", "intersect")
    if (any(type(masks[key]) is not bool for key in ("enabled", "skull_stripped", "fill_holes"))
            or masks["valid_mask_policy"] not in {"intersect", "preserve"}):
        raise ValueError("input_masks requires boolean flags and valid_mask_policy=intersect/preserve")
    if masks["enabled"] and not masks["skull_stripped"]:
        raise ValueError("Generating SWI masks requires the explicit input_masks.skull_stripped=true declaration")
    config.update(manifest=manifest, training=training, quality=quality, input_masks=masks,
                  independence=independence, simple_baselines=simple)
    # Reuse the training parser's complete numeric/range validation without
    # constructing a model, opening an image, or creating an experiment.
    from .train_models import parse_args, resolve_args
    numeric_config = {**config, "training": {**training, "device": "cpu"}}
    check = _training_command(numeric_config, "plain_unet", "/tmp/swixqsm-server-parser-only", "/tmp/manifest-not-read.csv",
                              ["--allow-unverified-data", "Parser validation only"],
                              ["--allow-unverified-independence", "Parser validation only"])[3:]
    parser, arguments = parse_args(check)
    resolve_args(parser, arguments)
    return config


def _read_csv(path):
    path = Path(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Protected split CSV needs unique column names")
        rows = list(reader)
    required = {"subject_id", "global_subject_id", "split", *ROLES}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Protected split is empty or lacks identity/image-role columns")
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed protected split CSV row")
        for key, value in row.items():
            if value and key.endswith("_path"):
                row[key] = _path(value, path.parent, key)
            elif value and key.endswith("_paths"):
                items = json.loads(value)
                if not isinstance(items, list):
                    raise ValueError("Path-list CSV fields must contain JSON arrays")
                row[key] = json.dumps([_path(item, path.parent, key) for item in items], ensure_ascii=False)
    return rows


def assert_protected_split(inventory, original_manifest, split_manifest, split_record):
    """Independently bind full cases, patient groups, and untouched test rows."""
    original, split = _read_csv(original_manifest), _read_csv(split_manifest)
    record = json.loads(Path(split_record).read_text())
    groups = {name: set() for name in ("train", "val", "test")}
    owners = {}
    for row in split:
        gid, name = row["global_subject_id"], row["split"]
        if not gid.strip() or name not in groups or gid in owners and owners[gid] != name:
            raise ValueError("Participant crosses the protected train/val/test split")
        owners[gid] = name
        groups[name].add(gid)
    if not groups["train"] or not groups["val"] or not groups["test"]:
        raise ValueError("Protected preparation requires nonempty train, val, and test groups")
    original_groups = {name: {row["global_subject_id"] for row in original if row["split"] == name}
                       for name in ("train", "test")}
    if (any(row["split"] not in original_groups for row in original)
            or original_groups["train"] & original_groups["test"]
            or groups["train"] | groups["val"] != original_groups["train"]
            or groups["test"] != original_groups["test"]):
        raise ValueError("Only original train may supply new val; original test membership must remain exact")
    def case(row):
        return row["subject_id"], row["global_subject_id"], row["swi_path"]
    old = {case(row): row for row in original}
    if len(old) != len(original) or len(split) != len(original):
        raise ValueError("Original case rows are duplicated, omitted, or added")
    seen = set()
    for row in split:
        key = case(row)
        if key not in old or key in seen:
            raise ValueError("Split case identity differs from original membership")
        seen.add(key)
        before, after = dict(old[key]), dict(row)
        old_split, new_split = before.pop("split"), after.pop("split")
        if before != after or old_split == "test" and new_split != "test":
            raise ValueError("Protected case metadata or test role changed")
    old_test = [row for row in original if row["split"] == "test"]
    new_test = [row for row in split if row["split"] == "test"]
    expected_groups = {key: sorted(value) for key, value in groups.items()}
    if (old_test != new_test or record.get("schema") != "swixqsm.new_training_split"
            or record.get("version") != 1 or record.get("group_ids") != expected_groups
            or record.get("original_train_group_ids") != sorted(original_groups["train"])
            or record.get("original_test_group_ids") != sorted(original_groups["test"])
            or record.get("source_manifest_sha256") != _sha(original_manifest)
            or record.get("output_manifest_sha256") != _sha(split_manifest)
            or record.get("test_membership_unchanged") is not True
            or record.get("test_rows_metadata_unchanged_after_path_resolution") is not True
            or record.get("test_resolved_rows_sha256_before") != _canonical_sha(old_test)
            or record.get("test_resolved_rows_sha256_after") != _canonical_sha(new_test)):
        raise ValueError("Protected split record, hashes, or test rows do not match the actual CSVs")
    participant_ids = sorted({row.get("participant_id", "") for row in old_test})
    actual_ids = inventory.get("actual_test_participant_ids")
    if (not participant_ids or "" in participant_ids or actual_ids != participant_ids
            or inventory.get("actual_test_participant_ids_sorted_sha256") != _ids_sha(actual_ids)
            or inventory.get("test_global_subject_ids") != sorted(groups["test"])
            or inventory.get("test_global_subject_ids_sorted_sha256") != _ids_sha(groups["test"])):
        raise ValueError("Inventory test-ID binding differs from the protected original and split manifests")
    return {"group_ids": expected_groups, "group_counts": {key: len(value) for key, value in groups.items()},
            "test_participant_ids_sha256": _ids_sha(actual_ids), "test_global_subject_ids_sha256": _ids_sha(groups["test"]),
            "test_rows_sha256": _canonical_sha(new_test), "test_membership_unchanged": True,
            "original_id_lists_membership_verified": inventory.get("original_membership_verified", False),
            "binding_scope": "Protected source-directory membership; historical-list identity verified only when both original lists were supplied and matched"}


def _code_hashes():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): _sha(path) for path in sorted(root.rglob("*.py"))
            if "test" not in path.relative_to(root).parts and "__pycache__" not in path.parts}


def _source_bindings(config):
    paths = [config["manifest"][key] for key in ("metadata_csv", "original_train_ids", "original_test_ids")
             if config["manifest"].get(key)]
    paths.extend(value for policy in (config["quality"], config["independence"])
                 for key, value in policy.items() if key in {"contract", "report"})
    if config["training"].get("normalization_file"):
        paths.append(config["training"]["normalization_file"])
    return {str(path): _sha(path) for path in sorted(set(paths))}


def _input_stats(manifest):
    """Train/val ordinary-change detector; test image bytes are not opened."""
    result = {}
    for row in _read_csv(manifest):
        if row["split"] == "test":
            continue
        for role in ROLES:
            path = Path(row[role])
            stat = path.stat()
            result[str(path)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def _path_stats(paths):
    """Ordinary-change detector only; no file contents are opened."""
    result = {}
    for filename in sorted(paths):
        stat = Path(filename).stat()
        result[str(filename)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def _default_runner(command, *, log_path, cwd):
    with Path(log_path).open("x") as log:
        return subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                              text=True, shell=False, check=False).returncode


def _execute(command, work, runner, label):
    history = work / "commands.jsonl"
    index = len(history.read_text().splitlines()) if history.exists() else 0
    log = work / "logs" / f"{index:03d}_{label}.log"
    log.parent.mkdir(exist_ok=True)
    record = {"argv": command, "shell": False, "label": label, "log_path": str(log),
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    _write_json(work / "active_command.json", record)
    try:
        result = runner(command, log_path=log, cwd=str(Path(__file__).resolve().parent.parent))
        returncode = result.returncode if hasattr(result, "returncode") else result
        if type(returncode) is not int:
            raise RuntimeError("Child runner must return an integer return code")
        record.update(returncode=returncode, status="completed" if returncode == 0 else "failed")
        if returncode:
            raise RuntimeError(f"{label} failed with exit code {returncode}; inspect {log}")
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with history.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        _write_json(work / "active_command.json", record)


def _training_command(config, model, output, manifest, quality_args, independence_args):
    command = [sys.executable, "-m", "swixqsm_minimal.train", "--model", model,
               "--manifest", str(manifest), "--output", str(output)]
    for key, value in sorted(config["training"].items()):
        if value is None:
            continue
        if key == "deterministic":
            command.append("--deterministic" if value else "--no-deterministic")
        else:
            command.extend(("--" + key.replace("_", "-"), str(value)))
    return command + list(quality_args) + list(independence_args)


def _passed_report(path, schema):
    report = json.loads(Path(path).read_text())
    if report.get("schema") != schema or report.get("version") != 1 or report.get("status") != "passed":
        raise ValueError("Audit must produce a completely passed report: " + str(path))
    return report


def _prepare_audits(config, manifest, work, runner):
    bound, policy_status = [], {}
    input_units_args = (["--qsm-units", config["manifest"]["qsm_units"]]
                        if config["manifest"].get("qsm_units") else [])
    quality = config["quality"]
    if "contract" in quality:
        output = work / "quality"
        _execute([sys.executable, "-m", "swixqsm_minimal.qc", "--manifest", str(manifest),
                  "--contract", quality["contract"], "--output", str(output), "--split", "train", "val",
                  *input_units_args], work, runner, "quality")
        report = output / "report.json"
        _passed_report(report, "swixqsm.data_qc_report")
        quality_args = ["--qc-report", str(report)]
        bound.append(report)
        policy_status["quality"] = "passed_train_val_report"
    elif "report" in quality:
        from .util.quality import verify_report
        rows = [row for row in _read_csv(manifest) if row["split"] in {"train", "val"}]
        verify_report(rows, quality["report"], reference_id=config["training"].get("reference_id"))
        quality_args = ["--qc-report", quality["report"]]
        bound.append(Path(quality["report"]))
        policy_status["quality"] = "verified_train_val_report"
    else:
        quality_args = ["--allow-unverified-data", quality["allow_unverified_reason"]]
        policy_status["quality"] = "explicitly_unverified"
    independence = config["independence"]
    if independence.get("run_audit"):
        output = work / "independence"
        _execute([sys.executable, "-m", "swixqsm_minimal.audit_splits", "--manifest", str(manifest),
                  "--output", str(output)], work, runner, "independence")
        report = output / "report.json"
        _passed_report(report, "swixqsm.independence_report")
        independence_args = ["--independence-report", str(report)]
        bound.append(report)
        policy_status["independence"] = "passed_full_identity_content_audit"
    elif "report" in independence:
        from .util.independence import verify_independence_report
        verify_independence_report(_read_csv(manifest), independence["report"], content_splits=("train", "val"))
        independence_args = ["--independence-report", independence["report"]]
        bound.append(Path(independence["report"]))
        policy_status["independence"] = "verified_full_metadata_train_val_content"
    else:
        independence_args = ["--allow-unverified-independence", independence["allow_unverified_reason"]]
        policy_status["independence"] = "explicitly_unverified"
    return quality_args, independence_args, bound, policy_status


def _mask_stage_allowed(inventory, config):
    """Only missing SWI input support/provenance can be repaired here."""
    for name in ("train", "test"):
        expected = config["manifest"]["expected_" + name]
        if inventory.get("counts", {}).get(name, {}).get("global_subjects") != expected:
            return False
    return all(item.get("code") == "missing_input_mask_source"
               or item.get("code") == "missing_role_file" and item.get("role") == "swi_mask_path"
               for item in inventory.get("issues", []))


def _assert_mask_derivation(before, after):
    old, new = _read_csv(before), _read_csv(after)
    key = lambda row: (row.get("case_id"), row.get("original_split"))
    mapping = {key(row): row for row in old}
    if len(mapping) != len(old) or len(new) != len(old) or {key(row) for row in new} != set(mapping):
        raise ValueError("Input-mask preparation changed original case membership")
    allowed = {"swi_mask_path", "mask_path", "input_mask_source"}
    for row in new:
        original = mapping[key(row)]
        if any(row.get(field) != value for field, value in original.items() if field not in allowed):
            raise ValueError("Input-mask preparation changed identity, image targets, or original QC/reference metadata")


def _generated_mask_bindings(stage, valid_mask_policy):
    """Verify every generated mask/sidecar against its declared report SHA."""
    cases = stage.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Input-mask report must enumerate its generated case files")
    report = json.loads(Path(stage["report_path"]).read_text())
    if report.get("cases") != cases:
        raise ValueError("Input-mask case declarations differ from their saved report")
    fields = [("input_mask_path", "input_mask_sha256"), ("provenance_path", "provenance_sha256")]
    if valid_mask_policy == "intersect":
        fields.extend((("final_valid_mask_path", "final_valid_mask_sha256"),
                       ("valid_mask_provenance_path", "valid_mask_provenance_sha256")))
    bindings = {}
    for case in cases:
        for path_key, hash_key in fields:
            filename, expected = case.get(path_key), case.get(hash_key)
            if (not isinstance(filename, str) or not Path(filename).is_absolute()
                    or not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
                raise ValueError("Generated input-mask files need absolute paths and explicit SHA256 declarations")
            if filename in bindings and bindings[filename] != expected:
                raise ValueError("Generated mask paths have inconsistent SHA256 declarations")
            if _sha(filename) != expected:
                raise ValueError("Generated input-mask artifact differs from its report SHA256: " + filename)
            bindings[filename] = expected
    return bindings


def _prepare(config, work, signature, runner, builder, mask_preparer):
    work.mkdir(parents=True, exist_ok=False)
    _write_json(work / "server_plan.json", {"schema": "swixqsm.server_plan", "version": 1,
                "config_prepare_signature": signature, "resolved_configuration": config,
                "test_policy": "Original test is fixed by source split directories or explicit original membership lists; new val comes only from original train",
                "test_evaluation": "No test inference/scoring; declared input-mask preparation and content-independence audit may read test input/source images, never to select a model",
                "model_execution": "serial; same protected split, seed and common training protocol"})
    fixed_extra, mask_stage = [], {"status": "disabled"}
    builder_options = dict(config["manifest"])
    if config["input_masks"]["enabled"]:
        before_path = work / "inventory_before_masks"
        before = builder(config["data_root"], before_path, inventory_only=True, **builder_options)
        if not _mask_stage_allowed(before, config):
            result = {"status": "blocked", "stage": "before_input_masks", "ready": False,
                      "issues": before.get("issues", []), "counts": before.get("counts"),
                      "inventory": str(before_path / "inventory.json"), "training_started": False}
            _write_json(work / "server_status.json", result)
            return result
        if mask_preparer is None:
            from .prepare_swi_masks import prepare_swi_masks
            mask_preparer = prepare_swi_masks
        draft = Path(before["artifacts"]["draft_manifest"])
        mask_stage = mask_preparer(draft, work / "input_masks", **{key: value for key, value in config["input_masks"].items() if key != "enabled"})
        generated = _generated_mask_bindings(mask_stage, config["input_masks"]["valid_mask_policy"])
        builder_options["metadata_csv"] = mask_stage["metadata_csv"]
        fixed_extra.extend([before_path / "inventory.json", before_path / "inventory_cases.csv", draft,
                            Path(mask_stage["metadata_csv"]), Path(mask_stage["report_path"])])
        fixed_extra.extend(Path(filename) for filename in generated)
    inventory_path = work / "manifest"
    inventory = builder(config["data_root"], inventory_path, inventory_only=False, **builder_options)
    if not inventory.get("ready"):
        result = {"status": "blocked", "stage": "manifest", "ready": False,
                  "issues": inventory.get("issues", []), "counts": inventory.get("counts"),
                  "inventory": str(inventory_path / "inventory.json"), "training_started": False}
        _write_json(work / "server_status.json", result)
        return result
    artifacts = inventory["artifacts"]
    original, manifest, split_record = (Path(artifacts[key]) for key in ("original_manifest", "split_manifest", "split_record"))
    if config["input_masks"]["enabled"]:
        _assert_mask_derivation(draft, original)
        if (before["actual_test_participant_ids_sorted_sha256"] != inventory["actual_test_participant_ids_sorted_sha256"]
                or before["test_global_subject_ids_sorted_sha256"] != inventory["test_global_subject_ids_sorted_sha256"]):
            raise ValueError("Input-mask preparation changed protected test membership")
    protection = assert_protected_split(inventory, original, manifest, split_record)
    quality_args, independence_args, audit_files, policy_status = _prepare_audits(config, manifest, work, runner)
    baseline = {"status": "pending_not_fitted", "reason": "Simple baselines disabled; zero/mean/linear test evaluation remains to be performed separately",
                "test_evaluated": False}
    if config["simple_baselines"]["enabled"]:
        simple = config["simple_baselines"]
        artifact = work / "simple_baselines.npz"
        input_units_args = (["--qsm-units", config["manifest"]["qsm_units"]]
                            if config["manifest"].get("qsm_units") else [])
        _execute([sys.executable, "-m", "swixqsm_minimal.fit_baselines", "--manifest", str(manifest),
                  "--output", str(artifact), "--units", simple["units"], *input_units_args,
                  "--reference-id", simple["reference_id"], "--space-id", simple["space_id"],
                  *quality_args, *independence_args], work, runner, "simple_baselines")
        if not artifact.is_file():
            raise RuntimeError("Simple-baseline child succeeded without creating its declared artifact")
        audit_files.append(artifact)
        baseline = {"status": "mean_linear_fitted_train_only", "artifact": str(artifact),
                    "zero_baseline": "No fitting; available in subsequent evaluate command", "test_evaluated": False}
    fixed = [inventory_path / "inventory.json", inventory_path / "inventory_cases.csv", original, manifest, split_record, *audit_files, *fixed_extra]
    prepared = {"schema": "swixqsm.server_prepared", "version": 1, "status": "prepared",
                "config_prepare_signature": signature, "artifacts": {"original_manifest": str(original), "manifest": str(manifest),
                    "split_record": str(split_record), "inventory": str(inventory_path / "inventory.json")},
                "protected_split": protection, "quality_args": quality_args, "independence_args": independence_args,
                "policy_status": policy_status, "simple_baselines": baseline, "input_masks": mask_stage,
                "fixed_file_sha256": {str(path): _sha(path) for path in fixed},
                "configuration_source_sha256": _source_bindings(config), "code_sha256": _code_hashes(),
                "train_val_input_stat_signature": _input_stats(manifest),
                "mask_source_stat_signature": _path_stats(mask_stage.get("source_sha256", {})),
                "input_signature_limitation": "Size/mtime detect ordinary changes only; passed reports and training gates verify content. Explicit opt-outs remain unverified.",
                "formal_test_evaluation_performed": False}
    _write_json(work / "prepared.json", prepared)
    return prepared


def _reuse(config, work, signature):
    path = work / "prepared.json"
    if not path.is_file():
        raise FileExistsError("Work directory has no successful prepared.json; preserve the partial run and choose a new work_dir")
    prepared = json.loads(path.read_text())
    if (prepared.get("schema") != "swixqsm.server_prepared" or prepared.get("version") != 1
            or prepared.get("status") != "prepared" or prepared.get("config_prepare_signature") != signature
            or prepared.get("configuration_source_sha256") != _source_bindings(config)
            or prepared.get("code_sha256") != _code_hashes()):
        raise ValueError("Prepared configuration, source data declarations, or implementation changed; choose a new work_dir")
    artifacts = prepared["artifacts"]
    minimum = {artifacts[key] for key in ("original_manifest", "manifest", "split_record", "inventory")}
    minimum.add(str(work / "manifest" / "inventory_cases.csv"))
    if not minimum.issubset(prepared.get("fixed_file_sha256", {})):
        raise ValueError("Prepared artifact hash inventory is incomplete")
    if config["input_masks"]["enabled"]:
        generated = _generated_mask_bindings(prepared["input_masks"], config["input_masks"]["valid_mask_policy"])
        if any(prepared["fixed_file_sha256"].get(filename) != expected for filename, expected in generated.items()):
            raise ValueError("Prepared generated-mask hash inventory is incomplete or differs from the report")
    for filename, expected in prepared["fixed_file_sha256"].items():
        if _sha(filename) != expected:
            raise ValueError("Prepared artifact hash changed: " + filename)
    artifacts = prepared["artifacts"]
    inventory = json.loads(Path(artifacts["inventory"]).read_text())
    protection = assert_protected_split(inventory, artifacts["original_manifest"], artifacts["manifest"], artifacts["split_record"])
    if protection != prepared["protected_split"] or _input_stats(artifacts["manifest"]) != prepared["train_val_input_stat_signature"]:
        raise ValueError("Protected split or training/validation input metadata changed")
    if _path_stats(prepared.get("input_masks", {}).get("source_sha256", {})) != prepared.get("mask_source_stat_signature"):
        raise ValueError("Input-mask source metadata changed, including original SWI or validity masks; prepare a new work_dir")
    return prepared


def _training_outcome(output, recipe, config):
    path = output / "run_result.json"
    if not path.is_file():
        raise RuntimeError("Training child returned success without its run_result.json")
    result = json.loads(path.read_text())
    requested = config["training"].get("epochs", DEFAULTS["epochs"])
    completed = result.get("epochs_completed")
    reason = result.get("stop_reason", "")
    if (result.get("recipe") != recipe or result.get("model_name") != RECIPES[recipe][0]
            or type(completed) is not int or not 1 <= completed <= requested):
        raise RuntimeError("Training child result does not match the requested recipe or epoch budget")
    if reason == "epoch_budget_completed" and completed == requested:
        return {"recipe": recipe, "status": "epoch_budget_completed", "epochs_completed": completed}
    if reason == "validation_patience_exhausted" and config["training"].get("early_stopping_patience", 0) > 0:
        return {"recipe": recipe, "status": "prespecified_early_stop", "epochs_completed": completed}
    if isinstance(reason, str) and reason.startswith("segment_epoch_budget") and config["training"].get("max_epochs_this_run"):
        return {"recipe": recipe, "status": "needs_explicit_resume", "epochs_completed": completed,
                "resume_argv": [sys.executable, "-m", "swixqsm_minimal.train", "--resume", str(output / "last.pt")]}
    raise RuntimeError("Training child stop reason is inconsistent with the requested completion/segmentation policy")


def run_server(config_path, model=None, *, prepare_only=False, runner=None, builder=None, mask_preparer=None):
    """Prepare/reuse once, then launch selected models; injectable children aid tests.

    Small declared test subsets are permitted only with prepare_only=True.
    The public CLI has no switch that relaxes the production 1773-person lock.
    Existing experiments are never overwritten or implicitly resumed.
    """
    config = load_config(config_path, model)
    if not prepare_only and config["manifest"]["expected_test"] != 1773:
        raise ValueError("Production training requires the protected expected_test=1773; small sample configurations are prepare-only")
    if not prepare_only:
        import torch
        desired = torch.device(config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if desired.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA training requested but CUDA is unavailable; --prepare-only remains available on a CPU machine")
        if desired.type == "cuda" and desired.index is not None and desired.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device index is unavailable")
    signature_config = {key: value for key, value in config.items() if key != "model"}
    signature = _canonical_sha(signature_config)
    work = Path(config["work_dir"])
    runner = _default_runner if runner is None else runner
    if builder is None:
        from .build_manifest import build_manifest
        builder = build_manifest
    try:
        prepared = _reuse(config, work, signature) if work.exists() else _prepare(config, work, signature, runner, builder, mask_preparer)
        if prepared.get("status") == "blocked":
            return prepared
        if prepare_only:
            result = {"status": "prepared", "training_started": False, "work_dir": str(work),
                      "protected_split": prepared["protected_split"], "simple_baselines": prepared["simple_baselines"],
                      "policy_status": prepared["policy_status"], "formal_test_evaluation_performed": False}
            _write_json(work / "server_status.json", result)
            return result
        if prepared["protected_split"]["group_counts"]["test"] != 1773:
            raise ValueError("Actual protected test membership is not exactly 1773; no training may start")
        selected = FOUR_MODELS if config["model"] == "all" else tuple(RECIPES) if config["model"] == "reviewer_suite" else (config["model"],)
        experiments = work / "experiments"
        commands = []
        for name in selected:
            output = experiments / name
            if output.exists():
                raise FileExistsError(f"Experiment already exists: {output}. Continue an interrupted individual run explicitly with: "
                                      f"python -m swixqsm_minimal.train --resume {output / 'last.pt'}")
            commands.append((name, _training_command(config, name, output, prepared["artifacts"]["manifest"],
                                                     prepared["quality_args"], prepared["independence_args"])))
        experiments.mkdir(exist_ok=True)
        completed, outcomes = [], []
        for name, command in commands:
            # Recheck membership and every prepared binding immediately before
            # each model; model outputs never determine a split or next recipe.
            _reuse(config, work, signature)
            _execute(command, work, runner, "train_" + name)
            outcome = _training_outcome(experiments / name, name, config)
            outcomes.append(outcome)
            if outcome["status"] != "needs_explicit_resume":
                completed.append(name)
            _write_json(work / "server_status.json", {"status": "training_in_progress", "selected": list(selected),
                        "completed": completed, "outcomes": outcomes, "training_started": True, "formal_test_evaluation_performed": False})
        result = {"status": "partial_training" if any(item["status"] == "needs_explicit_resume" for item in outcomes) else "training_completed",
                  "selected": list(selected), "completed": completed, "outcomes": outcomes,
                  "training_started": True, "work_dir": str(work), "policy_status": prepared["policy_status"],
                  "simple_baselines": prepared["simple_baselines"], "formal_test_evaluation_performed": False,
                  "scope": "Training only; no automatic held-out inference, evaluation, model ranking or manuscript claims"}
        _write_json(work / "server_status.json", result)
        return result
    except BaseException as error:
        if (work / "server_plan.json").is_file():
            _write_json(work / "server_failure.json", {"status": "failed", "error": str(error),
                        "no_implicit_overwrite_or_resume": True, "formal_test_evaluation_performed": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and audit; never launch neural training")
    arguments = parser.parse_args(argv)
    result = run_server(arguments.config, arguments.model, prepare_only=arguments.prepare_only)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())

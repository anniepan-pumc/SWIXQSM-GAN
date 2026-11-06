"""Print shell-safe commands from a prepared run; never read images or execute jobs.

The original preparation configuration stays immutable. Prospective training
uses its protected manifest/settings with the separately fitted normalization
file, replacing the manual preparation-only scale and offset.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys

from .run_server import MODEL_CHOICES, _canonical_sha, _sha, _training_command
from .train_models import FOUR_MODELS, RECIPES


def commands_from_prepared(work_dir, normalization_file, *, stage="fit", model="swixqsm_gan",
                           output_root=None, split="val", qc_report=None):
    if stage not in {"fit", "train", "validate"} or split not in {"val", "test"} or model not in MODEL_CHOICES:
        raise ValueError("Unsupported stage, split or model")
    if qc_report is not None and stage != "validate":
        raise ValueError("QC override is only supported for validation/test; fit/train inherit prepared gates")
    work = Path(work_dir).expanduser().resolve()
    prepared = json.loads((work / "prepared.json").read_text())
    plan = json.loads((work / "server_plan.json").read_text())
    if (prepared.get("schema") != "swixqsm.server_prepared" or prepared.get("version") != 1
            or prepared.get("status") != "prepared" or plan.get("schema") != "swixqsm.server_plan"):
        raise ValueError("Expected a successfully prepared server run")
    config = plan["resolved_configuration"]
    signature = _canonical_sha({key: value for key, value in config.items() if key != "model"})
    if (signature != prepared.get("config_prepare_signature") or signature != plan.get("config_prepare_signature")
            or Path(config["work_dir"]).resolve() != work):
        raise ValueError("Prepared run and saved configuration disagree")
    gates = []
    for key in ("quality_args", "independence_args"):
        args = prepared.get(key)
        if not isinstance(args, list) or not args or any(not isinstance(a, str) or not a for a in args):
            raise ValueError("Missing prepared gate arguments: " + key)
        gates.extend(args)
    manifest = prepared["artifacts"]["manifest"]
    expected = prepared.get("fixed_file_sha256", {}).get(manifest)
    if not expected or _sha(manifest) != expected:
        raise ValueError("Prepared manifest bytes changed or have no saved hash")
    normfile = str(Path(normalization_file).expanduser().resolve())
    root = (work.parent / "revision_models_v1" if output_root is None
            else Path(output_root).expanduser().resolve())
    data = Path(config["data_root"]).resolve()
    if any(path == data or data in path.parents for path in (root, Path(normfile))):
        raise ValueError("Normalization/model/evaluation output must be outside the input data root")
    if stage == "validate":
        if qc_report is not None:
            gates = ["--qc-report", str(Path(qc_report).expanduser().resolve()), *prepared["independence_args"]]
        elif split == "test" and "--qc-report" in prepared["quality_args"]:
            raise ValueError("Test requires --qc-report covering test; prepared QC may cover train/val only")
    original = config["training"]
    if stage == "fit":
        command = [sys.executable, "-m", "swixqsm_minimal.fit_normalization", "--manifest", manifest,
                   "--output", normfile, "--units", original["units"], "--method", "mean_std"]
        if config["manifest"].get("qsm_units"):
            command += ["--qsm-units", config["manifest"]["qsm_units"]]
        if original.get("reference_id"):
            command += ["--reference-id", original["reference_id"]]
        return [command + gates]
    selected = FOUR_MODELS if model == "all" else tuple(RECIPES) if model == "reviewer_suite" else (model,)
    training = {key: value for key, value in original.items() if key not in {"qsm_scale", "qsm_offset", "normalization_file"}}
    training["normalization_file"] = normfile
    revised = {**config, "training": training}
    result = []
    for recipe in selected:
        experiment = root / recipe
        if stage == "train":
            result.append(_training_command(revised, recipe, experiment, manifest,
                                            prepared["quality_args"], prepared["independence_args"]))
        else:
            phase = "l1" if RECIPES[recipe][2] == 0 else "gan"
            command = [sys.executable, "-m", "swixqsm_minimal.validate",
                           "--checkpoint", str(experiment / f"best_{phase}.pt"), "--manifest", manifest,
                           "--split", split, "--expected-phase", phase,
                           "--output", str(root / "evaluation" / f"{recipe}_{split}.json"),
                           "--device", original.get("device") or "cpu",
                           "--output-units", original["units"]]
            if original.get("reference_id"):
                command += ["--reference-id", original["reference_id"]]
            result.append(command + gates)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--normalization-file", required=True)
    p.add_argument("--stage", choices=["fit", "train", "validate"], default="fit")
    p.add_argument("--model", choices=MODEL_CHOICES, default="swixqsm_gan")
    p.add_argument("--output-root")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--qc-report", help="Validation/test QC override; test needs a report covering test, not just prepared train/val QC")
    args = p.parse_args(argv)
    for command in commands_from_prepared(**vars(args)):
        print(shlex.join(str(token) for token in command))


if __name__ == "__main__":
    main()

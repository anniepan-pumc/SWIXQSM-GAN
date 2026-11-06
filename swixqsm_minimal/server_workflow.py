"""Explicit stages for the upload bundle; no task runs without a stage command."""
import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys

import nibabel as nib
import numpy as np

from .build_manifest import TEMPLATE_FIELDS, _metadata, _read_csv
from .prepared_commands import commands_from_prepared
from .run_server import MODEL_CHOICES, _canonical_sha, _reuse, _sha, load_config
from .train_models import FOUR_MODELS, RECIPES
from .util.quality import load_contract, resolve_qc_paths


STAGES = ("environment", "inventory", "masks", "init-review", "headers", "review-images",
          "review-preview", "precheck", "prepare", "test-qc", "fit", "show-normalization",
          "train", "resume", "validate", "test", "predict-val", "predict-test",
          "evaluate-val", "evaluate-test", "range", "compare-val", "compare-test",
          "compare-roi-val", "compare-roi-test", "status")


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv_new(path, rows, fields=None):
    path = Path(path)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Workflow:
    def __init__(self, settings_path):
        self.settings_path = Path(settings_path).expanduser().resolve()
        self.bundle = self.settings_path.parent
        self.settings = json.loads(self.settings_path.read_text())
        if self.settings.get("schema") != "swixqsm.upload_settings" or self.settings.get("version") != 1:
            raise ValueError("Expected swixqsm.upload_settings version 1")
        self.root = Path(self.settings["run_root"]).expanduser().resolve()
        self.config_path = (self.bundle / self.settings["prepare_config"]).resolve()
        self.config = json.loads(self.config_path.read_text())
        self.deferred = self.settings.get("quality_mode", "reviewed") == "deferred"
        if self.settings.get("quality_mode", "reviewed") not in {"reviewed", "deferred"}:
            raise ValueError("quality_mode must be reviewed or deferred")
        self.data = Path(self.config["data_root"]).expanduser().resolve()
        if self.root == self.data or self.data in self.root.parents:
            raise ValueError("Run output must be outside the entire input data root")
        self.prepared = self.root / "prepared"
        self.review = self.root / "review"
        self.metadata = self.review / "reviewed_case_metadata.csv"
        self.contract = self.review / "data_contract.json"
        self.normfile = self.root / "qsm_normalization.json"
        self.rangefile = self.root / "metric_range.json"
        self.models = self.root / "models"
        if self.config.get("manifest", {}).get("layout") != "flat":
            raise ValueError("Bundle requires flat layout")
        if self.deferred:
            if (self.config.get("quality", {}).keys() != {"allow_unverified_reason"}
                    or not self.config["quality"]["allow_unverified_reason"].strip()
                    or self.config["manifest"].get("metadata_csv")):
                raise ValueError("Deferred QC requires an explicit reason and no fabricated reviewed metadata")
            paths = [(self.config["work_dir"], self.prepared)]
        else:
            if self.config.get("input_masks", {}).get("enabled") is not False:
                raise ValueError("Reviewed route requires standalone masks before final prepare")
            paths = [(self.config["work_dir"], self.prepared),
                     (self.config["manifest"]["metadata_csv"], self.metadata),
                     (self.config["quality"]["contract"], self.contract)]
        for actual, expected in paths:
            if Path(actual).expanduser().resolve() != expected:
                raise ValueError("Bundle settings and preparation paths disagree: " + str(expected))
        if self.config.get("independence") != {"run_audit": True}:
            raise ValueError("Upload workflow requires the full pretraining independence audit")
        self.units = self.config["training"]["units"]
        if self.units not in ({"ppm", "ppb", "stored"} if self.deferred else {"ppm", "ppb"}):
            raise ValueError("Explicit canonical units are required")
        self.roi = self.settings.get("roi_evaluation", {"enabled": False})
        if (not isinstance(self.roi, dict) or type(self.roi.get("enabled")) is not bool
                or set(self.roi) - {"enabled", "spec", "manifests", "review_policy", "bootstrap_reps", "bootstrap_seed"}):
            raise ValueError("roi_evaluation requires enabled and supported ROI settings only")
        if self.roi["enabled"]:
            if not self.roi.get("spec") or not isinstance(self.roi.get("manifests"), dict):
                raise ValueError("Enabled ROI evaluation requires spec and per-split manifests")
            if (not self.roi["manifests"] or set(self.roi["manifests"]) - {"val", "test"}
                    or any(not isinstance(v, str) or not v.strip() for v in self.roi["manifests"].values())):
                raise ValueError("ROI manifests must explicitly map val and/or test to complete ROI CSVs")
            policy = self.roi.get("review_policy", "accepted")
            if policy not in {"accepted", "deferred"} or (policy == "deferred" and not self.deferred):
                raise ValueError("Deferred ROI requires the deferred-QC workflow")
            if self.units == "stored" and policy != "deferred":
                raise ValueError("Stored-unit ROI requires explicit deferred review policy")

    def roi_arguments(self, split):
        if not self.roi["enabled"] or split not in self.roi["manifests"]:
            return []
        return ["--roi-spec", str((self.bundle / self.roi["spec"]).resolve()),
                "--roi-manifest", str((self.bundle / self.roi["manifests"][split]).resolve()),
                "--roi-review-policy", self.roi.get("review_policy", "accepted"),
                "--roi-bootstrap-reps", str(self.roi.get("bootstrap_reps", 2000)),
                "--roi-bootstrap-seed", str(self.roi.get("bootstrap_seed", 42))]

    def id_path(self, key):
        return str((self.config_path.parent / self.config["manifest"][key]).resolve())

    def source_rows(self):
        source = self.prepared / "input_masks/metadata.csv" if self.deferred else self.root / "masks/metadata.csv"
        _, _, _, rows = _read_csv(source)
        return rows

    def review_rows(self):
        metadata, _ = _metadata(self.metadata)
        rows = [dict(row, split=row["original_split"]) for row in metadata.values()]
        return resolve_qc_paths(rows, self.metadata)

    def prepared_record(self, *, planning=False):
        path = self.prepared / "prepared.json"
        if path.is_file():
            record = json.loads(path.read_text())
            if record.get("status") != "prepared":
                raise ValueError("Preparation is not complete")
            if planning:
                return record
            config = load_config(self.config_path)
            signature = _canonical_sha({key: value for key, value in config.items() if key != "model"})
            return _reuse(config, self.prepared, signature)
        if planning:
            return {"artifacts": {"manifest": str(self.prepared / "manifest/split/manifest.csv")},
                    "quality_args": ["--qc-report", str(self.prepared / "quality/report.json")],
                    "independence_args": ["--independence-report", str(self.prepared / "independence/report.json")]}
        raise FileNotFoundError("Run prepare successfully before fitting, training or evaluating")

    def test_report(self, label):
        return self.root / f"test_qc_{label}" / "report.json"

    def require_test_qc(self, label):
        record = self.prepared_record()
        path = self.test_report(label)
        report = json.loads(path.read_text())
        manifest = Path(record["artifacts"]["manifest"])
        if (report.get("status") != "passed" or report.get("schema") != "swixqsm.data_qc_report"
                or report.get("provenance", {}).get("manifest_sha256") != _sha(manifest)
                or report.get("provenance", {}).get("contract_file_sha256") != _sha(self.contract)
                or report.get("provenance", {}).get("selected_splits") != ["test"]):
            raise ValueError("test-qc must pass on this frozen manifest/contract before training or test scoring")
        return path

    def metric_range(self, *, planning=False):
        if self.rangefile.is_file():
            from .fit_normalization import training_identity
            from .util.data import read_manifest
            artifact = json.loads(self.rangefile.read_text())
            manifest = self.prepared_record(planning=planning)["artifacts"]["manifest"]
            if not planning and (artifact.get("schema") != "swixqsm.train_metric_range"
                    or artifact.get("version") != 1 or artifact.get("units") != self.units
                    or artifact.get("manifest_sha256") != _sha(manifest)
                    or artifact.get("train_identity_sha256") != training_identity(read_manifest(manifest))):
                raise ValueError("Metric range is not bound to this train split/units")
            return artifact["data_range"]
        if planning:
            return "TRAIN_ONLY_RANGE_FROM_METRIC_RANGE_JSON"
        raise FileNotFoundError("Run range before full metric evaluation; it uses train only")

    def precheck(self):
        contract = load_contract(self.contract)
        rows = self.review_rows()
        expected = {(r["case_id"], r["original_split"]) for r in self.source_rows()}
        if {(r["case_id"], r["original_split"]) for r in rows} != expected or len(rows) != len(expected):
            raise ValueError("Reviewed metadata must preserve every source train AND test case exactly once")
        errors = []
        for row in rows:
            prefix = row["case_id"] + ": "
            if row.get("registration_review") != "accepted":
                errors.append(prefix + "registration_review is not accepted; complete actual review")
            try:
                date.fromisoformat(row.get("registration_review_date", ""))
            except ValueError:
                errors.append(prefix + "a real YYYY-MM-DD review date is required")
            for key in ("registration_reviewer", "input_mask_source", "qsm_reference_source"):
                if not row.get(key, "").strip() or re.search(r"TODO|REPLACE|待填写|待核实", row.get(key, ""), re.I):
                    errors.append(prefix + "missing/placeholder " + key)
            for key, value in (("qsm_reference", contract["reference"]["id"]),
                               ("qsm_reference_method", contract["reference"]["method"]),
                               ("qsm_reference_region", contract["reference"]["region"])):
                if row.get(key) != value:
                    errors.append(prefix + key + " does not match the actual contract")
            if row.get("qsm_units") not in {"ppm", "ppb"}:
                errors.append(prefix + "declare actual qsm_units")
            keys = ["swi_path", "qsm_path", "swi_mask_path", "mask_path"]
            if contract["reference"]["require_mask"]:
                keys.append("qsm_reference_mask_path")
            for key in keys:
                if not row.get(key) or not Path(row[key]).is_file():
                    errors.append(prefix + "missing " + key)
        for key in ("id", "region"):
            if re.search(r"actual_reference|TODO|REPLACE|待填写|待核实", contract["reference"][key], re.I):
                errors.append("Contract still contains a placeholder reference " + key)
        if errors:
            raise ValueError(f"Review precheck failed ({len(errors)} findings).\n" + "\n".join(errors[:30]))
        return {"status": "metadata_prechecked_not_image_qc", "n_cases": len(rows),
                "scope": "All original train and test declarations frozen before final split; image/reference numerical QC follows"}

    def initialize_review(self):
        if self.review.exists():
            raise FileExistsError("Review directory exists; edit its existing files, do not initialize over them")
        rows = self.source_rows()
        for row in rows:
            for field in TEMPLATE_FIELDS:
                row.setdefault(field, "")
            row["registration_review"] = "pending"
            row["registration_reviewer"] = ""
            row["registration_review_date"] = ""
        template = json.loads((self.bundle / "templates/data_contract.TEMPLATE.json").read_text())
        self.review.mkdir(parents=True, exist_ok=False)
        write_csv_new(self.metadata, rows)
        write_json_new(self.contract, template)
        print(f"Edit the pending table: {self.metadata}\nFill the actual, initially incomplete contract: {self.contract}")

    def headers(self, label, limit):
        rows = self.review_rows()
        records, errors = [], []
        for row in rows[:limit or None]:
            for role in ("swi_path", "qsm_path", "swi_mask_path", "mask_path"):
                try:
                    image = nib.load(row[role])  # Header/proxy only; never get_fdata here.
                    basis = np.asarray(image.affine[:3, :3], dtype=float)
                    lengths = np.linalg.norm(basis, axis=0)
                    if len(image.shape) != 3 or not np.isfinite(basis).all() or np.any(lengths <= 0) or np.linalg.matrix_rank(basis) < 3:
                        raise ValueError("Non-3D or invalid/singular current image grid")
                    unit = image.header.get_xyzt_units()[0]
                    factor = {"mm": 1., "meter": 1000., "micron": .001}.get(unit)
                    normal = np.cross(basis[:, 0], basis[:, 1])
                    angle = float(np.degrees(np.arccos(np.clip(abs(normal[2]) / np.linalg.norm(normal), 0, 1))))
                    records.append({"case_id": row["case_id"], "role": role, "shape": list(image.shape),
                                    "axis_codes": list(nib.aff2axcodes(image.affine)), "header_spatial_units": unit,
                                    "spacing_native": lengths.tolist(),
                                    "spacing_mm": (lengths * factor).tolist() if factor is not None else None,
                                    "slice_normal_to_superior_deg": angle})
                except Exception as exc:
                    errors.append({"case_id": row["case_id"], "role": role, "error": str(exc)})
        payload = {"schema": "swixqsm.header_inventory", "version": 1, "voxel_arrays_read": False,
                   "policy_selected_automatically": False, "n_cases_examined": len(rows[:limit or None]),
                   "axis_counts": dict(Counter("/".join(r["axis_codes"]) for r in records)),
                   "records": records, "errors": errors,
                   "limitations": "Observed current headers do not establish QSM units, reference anatomy, registration or original B0 head angle; choose protocol limits independently"}
        output = self.root / f"headers_{label}.json"
        write_json_new(output, payload)
        print(output)

    def review_images(self, label, limit):
        from .util.qc_preview import write_preview
        output = self.root / f"review_images_{label}"
        if output.exists():
            raise FileExistsError(output)
        rows = self.review_rows()
        output.mkdir(parents=True)
        write_preview(output / "preview.html", rows, [{"key": r["case_id"]} for r in rows], limit=limit)
        path = output / "preview.html"
        page = path.read_text()
        page = page.replace("<h1>配准与数据质量预览</h1>", "<h1>仅供人工查看：未运行QC，未自动审核任何病例</h1>")
        path.write_text(page)
        write_json_new(output / "report.json", {"schema": "swixqsm.review_preview_only", "version": 1,
                                               "status": "not_qc", "registration_accepted_automatically": 0,
                                               "display_limit": limit, "n_source_cases": len(rows)})
        print(path)

    def recipes(self, model):
        return FOUR_MODELS if model == "all" else tuple(RECIPES) if model == "reviewer_suite" else (model,)

    def commands(self, stage, args, *, planning=False):
        base = [sys.executable, "-m"]
        manifest_cfg = self.config["manifest"]
        if stage == "inventory":
            command = base + ["swixqsm_minimal.build_manifest", "--data-root", str(self.data), "--layout", "flat",
                    "--original-train-ids", self.id_path("original_train_ids"), "--original-test-ids", self.id_path("original_test_ids"),
                    "--expected-train", str(manifest_cfg["expected_train"]), "--expected-test", str(manifest_cfg["expected_test"]),
                    "--qsm-units", manifest_cfg["qsm_units"], "--inventory-only", "--output", str(self.root / "inventory")]
            for key in ("swi_name", "swi_mask_name", "subject_pattern", "input_mask_source", "seed", "val_fraction"):
                if manifest_cfg.get(key) is not None:
                    command += ["--" + key.replace("_", "-"), str(manifest_cfg[key])]
            for key in ("qsm_name", "valid_mask_name"):
                values = manifest_cfg.get(key, [])
                for value in [values] if isinstance(values, str) else values:
                    command += ["--" + key.replace("_", "-"), value]
            return [command]
        if stage == "masks":
            command = base + ["swixqsm_minimal.prepare_swi_masks", "--manifest", str(self.root / "inventory/original_train_test.draft.csv"),
                    "--output", str(self.root / "masks"), "--skull-stripped", "--valid-mask-policy",
                    self.config["input_masks"].get("valid_mask_policy", "intersect")]
            command += ["--fill-holes" if self.config["input_masks"].get("fill_holes", True) else "--no-fill-holes"]
            return [command]
        if stage == "review-preview":
            return [base + ["swixqsm_minimal.qc", "--manifest", str(self.root / f"review_manifest_{args.label}.csv"),
                    "--contract", str(self.contract), "--output", str(self.root / f"review_qc_{args.label}"),
                    "--split", "train", "test", "--qsm-units", manifest_cfg["qsm_units"],
                    "--preview-limit", str(args.preview_limit)]]
        if stage == "prepare":
            return [base + ["swixqsm_minimal.run_server", "--config", str(self.config_path), "--prepare-only"]]
        prepared = self.prepared_record(planning=planning)
        manifest = prepared["artifacts"]["manifest"]
        test_report = str(self.test_report(args.label))
        if stage == "range":
            return [base + ["swixqsm_minimal.fit_metric_range", "--manifest", manifest, "--output", str(self.rangefile),
                    "--units", self.units, *prepared["quality_args"], *prepared["independence_args"]]]
        if stage.startswith("compare-"):
            roi_only = stage.startswith("compare-roi-")
            split = stage.removeprefix("compare-roi-" if roi_only else "compare-")
            roi_enabled = bool(self.roi_arguments(split))
            if roi_only and not roi_enabled:
                raise ValueError("ROI comparison requires enabled ROI specification/manifest for this split")
            command = base + ["swixqsm_minimal.compare_models", "--output",
                              str(self.root / "comparison" / f"{split}_{args.label}")]
            for recipe in self.recipes(args.model):
                command += ["--expected-model", recipe, "--evaluation",
                            f"{recipe}={self.root / 'evaluation' / f'{recipe}_{split}_{args.label}'}"]
            commands = [] if roi_only else [command]
            if roi_enabled:
                regional = base + ["swixqsm_minimal.compare_roi_models", "--output",
                                   str(self.root / "comparison_roi" / f"{split}_{args.label}")]
                for recipe in self.recipes(args.model):
                    regional += ["--expected-model", recipe, "--evaluation",
                                 f"{recipe}={self.root / 'evaluation' / f'{recipe}_{split}_{args.label}'}"]
                commands.append(regional)
            return commands
        if stage == "test-qc":
            return [base + ["swixqsm_minimal.qc", "--manifest", manifest, "--contract", str(self.contract),
                    "--output", str(self.test_report(args.label).parent), "--split", "test",
                    "--qsm-units", manifest_cfg["qsm_units"], "--preview-limit", str(args.preview_limit)]]
        if stage in {"fit", "train"}:
            if not planning:
                result = commands_from_prepared(self.prepared, self.normfile, stage=stage, model=args.model, output_root=self.models)
                if stage == "fit":
                    result[0][result[0].index("--method") + 1] = self.settings.get("normalization_method", "mean_std")
                return result
            if stage == "fit":
                return [base + ["swixqsm_minimal.fit_normalization", "--manifest", manifest, "--output", str(self.normfile),
                        "--units", self.units, "--method", self.settings.get("normalization_method", "mean_std"),
                        *prepared["quality_args"], *prepared["independence_args"]]]
            from .run_server import _training_command
            training = {k: v for k, v in self.config["training"].items() if k not in {"qsm_scale", "qsm_offset"}}
            training["normalization_file"] = str(self.normfile)
            return [_training_command({**self.config, "training": training}, recipe, self.models / recipe, manifest,
                                      prepared["quality_args"], prepared["independence_args"]) for recipe in self.recipes(args.model)]
        result = []
        split = "test" if stage in {"test", "predict-test", "evaluate-test"} else "val"
        quality = (["--qc-report", test_report] if split == "test" and not self.deferred else prepared["quality_args"])
        for recipe in self.recipes(args.model):
            phase = "l1" if RECIPES[recipe][2] == 0 else "gan"
            checkpoint = str(self.models / recipe / f"best_{phase}.pt")
            if stage == "resume":
                if args.model in {"all", "reviewer_suite"}:
                    raise ValueError("Resume one named recipe at a time")
                result.append(base + ["swixqsm_minimal.train", "--resume", str(self.models / recipe / "last.pt"),
                                      "--device", self.config["training"]["device"]])
            elif stage in {"validate", "test"}:
                result.append(base + ["swixqsm_minimal.validate", "--checkpoint", checkpoint, "--manifest", manifest,
                        "--split", split, "--output", str(self.root / "mae" / f"{recipe}_{split}_{args.label}.json"),
                        "--device", self.config["training"]["device"], "--output-units", self.units,
                        "--expected-phase", phase, *quality, *prepared["independence_args"]])
            elif stage.startswith("predict-"):
                result.append(base + ["swixqsm_minimal.predict_split", "--checkpoint", checkpoint, "--manifest", manifest,
                        "--split", split, "--output", str(self.root / "predictions" / f"{recipe}_{split}_{args.label}"),
                        "--device", self.config["training"]["device"], "--output-units", self.units,
                        "--expected-phase", phase, *quality, *prepared["independence_args"]])
                if self.deferred:
                    result[-1] += ["--reference-id", self.settings.get("reference_id", "unverified_original_reference")]
            elif stage.startswith("evaluate-"):
                if self.deferred:
                    reference_id = self.settings.get("reference_id", "unverified_original_reference")
                elif planning:
                    raw = json.loads(self.contract.read_text()) if self.contract.is_file() else {}
                    reference_id = raw.get("reference", {}).get("id") or "FILL_ACTUAL_REFERENCE_IN_CONTRACT"
                else:
                    reference_id = load_contract(self.contract)["reference"]["id"]
                command = base + ["swixqsm_minimal.evaluate", "--manifest",
                        str(self.root / "predictions" / f"{recipe}_{split}_{args.label}" / "evaluation_manifest.csv"),
                        "--output", str(self.root / "evaluation" / f"{recipe}_{split}_{args.label}"),
                        "--split", split, "--units", self.units, "--reference-id", reference_id,
                        "--prediction-name", recipe, "--expected-prediction-phase", phase,
                        "--distribution-diagnostics", *quality, *prepared["independence_args"],
                        *self.roi_arguments(split)]
                data_range = args.data_range if args.data_range is not None else self.settings.get("data_range")
                if self.deferred and data_range is None:
                    data_range = self.metric_range(planning=planning)
                if data_range is not None:
                    is_placeholder = planning and data_range == "TRAIN_ONLY_RANGE_FROM_METRIC_RANGE_JSON"
                    if not is_placeholder and (isinstance(data_range, bool) or not math.isfinite(data_range) or data_range <= 0):
                        raise ValueError("data_range must be a positive fixed range in " + self.units)
                    command += ["--data-range", str(data_range)]
                result.append(command)
        if not result:
            raise ValueError("No external command for stage " + stage)
        return result

    def execute(self, stage, args):
        if args.dry_run:
            if stage in {"environment", "init-review", "headers", "review-images", "precheck", "show-normalization", "status"}:
                print(f"# Built-in stage {stage}; no files are read beyond bundle settings/config and no task is executed")
            else:
                for command in self.commands(stage, args, planning=True):
                    print(shlex.join(str(a) for a in command))
            return
        if stage == "environment":
            import torch
            import scipy
            print(json.dumps({"python": sys.version, "torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
                              "cuda_runtime": torch.version.cuda, "nibabel": nib.__version__, "numpy": np.__version__,
                              "scipy": scipy.__version__}, indent=2))
            return
        if stage == "init-review":
            return self.initialize_review()
        if stage == "headers":
            return self.headers(args.label, args.header_limit)
        if stage == "review-images":
            return self.review_images(args.label, args.preview_limit)
        if stage == "precheck":
            print(json.dumps(self.precheck(), ensure_ascii=False, indent=2))
            return
        if stage == "show-normalization":
            data = json.loads(self.normfile.read_text())
            print(json.dumps({"qsm": data["normalization"]["qsm"], "method": data["method"],
                              "fit_selection": data["fit_selection"]}, ensure_ascii=False, indent=2))
            return
        if stage == "status":
            for path in [self.prepared / "server_status.json", self.normfile, *[self.models / r / "run_result.json" for r in self.recipes(args.model)]]:
                print(f"{'EXISTS' if path.is_file() else 'MISSING'} {path}")
            return
        if self.deferred and stage in {"masks", "headers", "review-images", "review-preview", "precheck", "test-qc"}:
            raise ValueError("Deferred route uses prepare for masks/split; manual QC belongs to the separate later review route")
        if stage == "masks":
            inventory = json.loads((self.root / "inventory/inventory.json").read_text())
            fatal = [i for i in inventory.get("issues", [])
                     if not (i["code"] == "missing_role_file" and i.get("role") == "swi_mask_path")
                     and i["code"] != "missing_input_mask_source"]
            if fatal or not inventory.get("original_membership_verified"):
                raise ValueError("Resolve inventory membership/required SWI/QSM/QSM-mask problems before masks; only missing input brain masks may be prepared here")
            if self.config["input_masks"].get("skull_stripped") is not True:
                raise ValueError("Standalone SWI support masks require the actual skull-stripped declaration")
        if stage == "review-preview":
            load_contract(self.contract)
            write_csv_new(self.root / f"review_manifest_{args.label}.csv", self.review_rows())
        if stage == "prepare" and not self.deferred:
            self.precheck()
        if not self.deferred and stage in {"fit", "train", "resume", "test", "predict-test", "evaluate-test"}:
            self.require_test_qc(args.label)
        for command in self.commands(stage, args):
            self.root.mkdir(parents=True, exist_ok=True)
            started = datetime.now(timezone.utc).isoformat()
            print(shlex.join(str(a) for a in command), flush=True)
            with (self.root / "workflow_commands.jsonl").open("a") as stream:
                stream.write(json.dumps({"started_utc": started, "stage": stage, "argv": [str(a) for a in command]}, ensure_ascii=False) + "\n")
            completed = subprocess.run([str(a) for a in command], cwd=self.bundle, check=False)
            if completed.returncode:
                raise SystemExit(completed.returncode)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings", required=True)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--model", choices=MODEL_CHOICES)
    p.add_argument("--label", default="final", help="New output label for previews/test QC/metrics; use the same label in dependent stages")
    p.add_argument("--preview-limit", type=int, default=12)
    p.add_argument("--header-limit", type=int, default=0, help="0 inspects all headers, positive value limits cases; no voxel arrays")
    p.add_argument("--data-range", type=float, help="Prespecified physical PSNR/SSIM range in configured output units")
    p.add_argument("--dry-run", action="store_true", help="Print commands only; never execute stages")
    args = p.parse_args(argv)
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.label)
            or args.preview_limit < 0 or args.header_limit < 0):
        p.error("Use a safe nonempty label and nonnegative preview/header limits")
    try:
        workflow = Workflow(args.settings)
        args.model = args.model or workflow.settings.get("default_model", "swixqsm_gan")
        if args.model not in MODEL_CHOICES:
            p.error("Unsupported default_model")
        workflow.execute(args.stage, args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        p.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()

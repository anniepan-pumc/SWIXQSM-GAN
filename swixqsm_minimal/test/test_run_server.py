"""Metadata/synthetic probes; child training and image decoding are mocked."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from .. import run_server as server
from ..split_training import create_split


def ids_sha(values):
    return hashlib.sha256("".join(value + "\n" for value in sorted(values)).encode()).hexdigest()


class RunServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="swix-server-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Other development agents may edit unrelated package files during
        # these orchestration probes. Pin the code inventory here; a separate
        # targeted test below verifies that changed inventories are rejected.
        self.code_inventory = server._code_hashes()
        code_patch = mock.patch.object(server, "_code_hashes", return_value=self.code_inventory)
        code_patch.start()
        self.addCleanup(code_patch.stop)
        self.config = {"schema": "swixqsm.server_training", "version": 1,
                       "data_root": "source", "work_dir": "work", "model": "pix2pix",
                       "manifest": {"expected_train": 2, "expected_test": 1773},
                       "quality": {"allow_unverified_reason": "Synthetic orchestration only"},
                       "independence": {"allow_unverified_reason": "Synthetic identities only"},
                       "training": {"units": "ppm", "qsm_scale": .1, "size": 32,
                                    "epochs": 1, "batch_size": 2, "device": "cpu", "seed": 42}}
        self.calls = []

    def save(self, config=None):
        path = self.root / "config.json"
        path.write_text(json.dumps(self.config if config is None else config))
        return path

    def fitted_config(self, default_units="ppb"):
        """Fit only tiny synthetic arrays; server tests never open these inputs."""
        import nibabel as nib
        import numpy as np
        from .. import fit_normalization as fitting
        row = dict(subject_id="synthetic", case_id="synthetic", split="train", qsm_units="ppb")
        for role in server.ROLES:
            path = self.root / ("fit_" + role + ".nii.gz")
            values = np.array([-100., 300.], dtype=np.float32).reshape(2, 1, 1) if role == "qsm_path" else np.ones((2, 1, 1), dtype=np.float32)
            nib.save(nib.Nifti1Image(values, np.eye(4)), path)
            row[role] = str(path)
        artifact = fitting.fit_training_normalization([row], units="ppm", default_qsm_units=default_units)
        path = self.root / "fitted config" / "normalization.json"
        fitting.save_fitted_normalization(artifact, path)
        config = copy.deepcopy(self.config)
        config["training"].pop("qsm_scale")
        config["training"]["normalization_file"] = str(path.relative_to(self.root))
        return config, path

    def test_fitted_json_is_resolved_and_dispatched_without_reading_images(self):
        from .. import fit_normalization as fitting
        config, path = self.fitted_config()
        with mock.patch("nibabel.load", side_effect=AssertionError("No image reads during config validation")), \
                mock.patch.object(fitting, "_file_snapshot", side_effect=AssertionError("No source-image rehash during config validation")):
            loaded = server.load_config(self.save(config))
        self.assertEqual(loaded["training"]["normalization_file"], str(path.resolve()))
        self.assertEqual(loaded["manifest"]["qsm_units"], "ppb")
        command = server._training_command(loaded, "plain_unet", "unused", "not-read.csv", [], [])
        self.assertEqual(command[command.index("--normalization-file") + 1], str(path.resolve()))
        self.assertEqual(command[command.index("--units") + 1], "ppm")
        self.assertNotIn("--qsm-scale", command)
        self.assertNotIn("--qsm-offset", command)

    def test_fitted_json_requires_units_and_rejects_manual_or_input_unit_conflicts(self):
        config, _ = self.fitted_config()
        for extra in ({"qsm_scale": .1}, {"qsm_offset": 0.}, {"qsm_units": "ppm"}, {"qsm_units": "unknown"}):
            bad = copy.deepcopy(config); bad["training"].update(extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                server.load_config(self.save(bad))
        missing = copy.deepcopy(config); missing["training"].pop("units")
        with self.assertRaisesRegex(ValueError, "training.units"):
            server.load_config(self.save(missing))
        mismatch = copy.deepcopy(config); mismatch["training"]["units"] = "ppb"
        with self.assertRaisesRegex(ValueError, "canonical units"):
            server.load_config(self.save(mismatch))
        missing_file = copy.deepcopy(config); missing_file["training"]["normalization_file"] = "missing.json"
        with self.assertRaises(FileNotFoundError):
            server.load_config(self.save(missing_file))

    def test_fitted_unknown_input_units_are_not_guessed_from_canonical_units(self):
        config, _ = self.fitted_config(default_units=None)
        loaded = server.load_config(self.save(config))
        self.assertIsNone(loaded["manifest"]["qsm_units"])
        config["training"]["qsm_units"] = "ppb"
        loaded = server.load_config(self.save(config))
        self.assertEqual(loaded["manifest"]["qsm_units"], "ppb")
        command = server._training_command(loaded, "plain_unet", "unused", "not-read.csv", [], [])
        self.assertEqual(command[command.index("--qsm-units") + 1], "ppb")
        config["manifest"]["qsm_units"] = "ppm"
        with self.assertRaisesRegex(ValueError, "input-unit fallback"):
            server.load_config(self.save(config))

    def test_fitted_json_sha_binds_preparation_and_reuse(self):
        config, path = self.fitted_config()
        config["manifest"]["expected_test"] = 2
        config_path = self.save(config)
        server.run_server(config_path, prepare_only=True, runner=self.runner, builder=self.builder)
        prepared = json.loads((self.root / "work" / "prepared.json").read_text())
        self.assertEqual(prepared["configuration_source_sha256"][str(path.resolve())], server._sha(path))
        self.assertEqual(self.calls, [])
        server.run_server(config_path, prepare_only=True, runner=self.runner,
                          builder=lambda *_a, **_k: self.fail("Must reuse prepared metadata"))
        # Whitespace changes preserve semantics but must invalidate byte-bound reuse.
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "configuration.*changed"):
            server.run_server(config_path, prepare_only=True, runner=self.runner, builder=self.builder)

    def test_input_fallback_reaches_quality_and_simple_baseline_without_canonical_override(self):
        config = copy.deepcopy(self.config)
        config["training"]["qsm_units"] = "ppb"
        config["quality"] = {"contract": "contract.json"}
        (self.root / "contract.json").write_text("{}")
        config["simple_baselines"] = dict(enabled=True, reference_id="synthetic-ref", space_id="synthetic-space", units="ppm")
        server.run_server(self.save(config), runner=self.runner, builder=self.builder)
        for command in self.calls:
            if command[2] in {"swixqsm_minimal.qc", "swixqsm_minimal.fit_baselines", "swixqsm_minimal.train"}:
                self.assertEqual(command[command.index("--qsm-units") + 1], "ppb")
        manual = copy.deepcopy(self.config); manual["manifest"]["qsm_units"] = "ppb"
        self.assertEqual(server.load_config(self.save(manual))["manifest"]["qsm_units"], "ppb")

    def builder(self, data_root, output, **kwargs):
        """Publish genuine split metadata with 1773 fake held-out participants."""
        output = Path(output)
        output.mkdir()
        rows = []
        for split, number in (("train", kwargs["expected_train"]), ("test", kwargs["expected_test"])):
            for index in range(number):
                eid = str(subjT000 + index if split == "train" else subjT002 + index)
                row = dict(case_id=eid, participant_id=eid, subject_id=eid,
                           global_subject_id="cohort:" + eid, original_split=split, split=split, qsm_units="ppm",
                           input_mask_source="Synthetic pre-existing input-only mask")
                for role in server.ROLES:
                    path = self.root / "source" / split / eid / (role + ".nii.gz")
                    if split == "train":
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"metadata-only fixture, not an image")
                    row[role] = str(path)
                rows.append(row)
        if kwargs.get("metadata_csv"):
            with Path(kwargs["metadata_csv"]).open(newline="") as stream:
                overrides = {(row["case_id"], row["original_split"]): row for row in csv.DictReader(stream)}
            rows = [{**row, **overrides.get((row["case_id"], row["original_split"]), {})} for row in rows]
        original = output / "original_train_test.csv"
        with original.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        create_split(original, output / "split", seed=kwargs["seed"], val_fraction=kwargs.get("val_fraction", .1))
        actual_ids = sorted({row["participant_id"] for row in rows if row["split"] == "test"})
        global_ids = ["cohort:" + value for value in actual_ids]
        draft = output / "original_train_test.draft.csv"
        draft.write_bytes(original.read_bytes())
        inventory = {"schema": "swixqsm.swixqsm_manifest_inventory", "version": 1, "ready": True,
                     "issues": [], "actual_test_participant_ids": actual_ids,
                     "actual_test_participant_ids_sorted_sha256": ids_sha(actual_ids),
                     "test_global_subject_ids": global_ids, "test_global_subject_ids_sorted_sha256": ids_sha(global_ids),
                     "counts": {"train": {"global_subjects": kwargs["expected_train"]},
                                "test": {"global_subjects": kwargs["expected_test"]}},
                     "artifacts": {"original_manifest": str(original),
                                   "draft_manifest": str(draft),
                                   "split_manifest": str(output / "split" / "manifest.csv"),
                                   "split_record": str(output / "split" / "split_record.json")}}
        (output / "inventory.json").write_text(json.dumps(inventory))
        (output / "inventory_cases.csv").write_text("case_id\nmetadata_only\n")
        return inventory

    def runner(self, argv, *, log_path, cwd):
        self.assertIsInstance(argv, list)
        self.assertEqual(argv[:2], [server.sys.executable, "-m"])
        self.calls.append(argv)
        Path(log_path).write_text("Synthetic child; no neural training or image reads\n")
        module = argv[2]
        if module == "swixqsm_minimal.train":
            path = Path(argv[argv.index("--output") + 1])
            path.mkdir(parents=True)
            (path / "last.pt").write_bytes(b"mock checkpoint")
            recipe = argv[argv.index("--model") + 1]
            epochs = int(argv[argv.index("--epochs") + 1]) if "--epochs" in argv else 100
            segmented = "--max-epochs-this-run" in argv
            completed = min(epochs, int(argv[argv.index("--max-epochs-this-run") + 1])) if segmented else epochs
            (path / "run_result.json").write_text(json.dumps({"recipe": recipe, "model_name": server.RECIPES[recipe][0],
                "epochs_completed": completed, "stop_reason": "segment_epoch_budget; resume last.pt with the original total epochs" if completed < epochs else "epoch_budget_completed"}))
        elif module == "swixqsm_minimal.fit_baselines":
            Path(argv[argv.index("--output") + 1]).write_bytes(b"mock baseline")
        elif module in {"swixqsm_minimal.qc", "swixqsm_minimal.audit_splits"}:
            path = Path(argv[argv.index("--output") + 1]); path.mkdir()
            schema = "swixqsm.data_qc_report" if module.endswith(".qc") else "swixqsm.independence_report"
            (path / "report.json").write_text(json.dumps({"schema": schema, "version": 1, "status": "passed"}))
        return 0

    def test_selected_model_dispatch_no_test_evaluation_and_disabled_baselines_pending(self):
        result = server.run_server(self.save(), runner=self.runner, builder=self.builder)
        self.assertEqual(result["completed"], ["pix2pix"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][self.calls[0].index("--model") + 1], "pix2pix")
        self.assertFalse(result["formal_test_evaluation_performed"])
        self.assertEqual(result["simple_baselines"]["status"], "pending_not_fitted")
        commands = [json.loads(line) for line in (self.root / "work" / "commands.jsonl").read_text().splitlines()]
        self.assertTrue(all(record["shell"] is False for record in commands))

    def test_all_and_reviewer_suite_are_serial_distinct_recipes(self):
        for selector, expected in (("all", server.FOUR_MODELS), ("reviewer_suite", tuple(server.RECIPES))):
            config = copy.deepcopy(self.config); config["work_dir"] = "work_" + selector
            self.calls.clear()
            result = server.run_server(self.save(config), selector, runner=self.runner, builder=self.builder)
            dispatched = [argv[argv.index("--model") + 1] for argv in self.calls]
            self.assertEqual(dispatched, list(expected))
            self.assertEqual(result["completed"], list(expected))
            self.assertEqual({argv[argv.index("--seed") + 1] for argv in self.calls}, {"42"})
            self.assertEqual(len({argv[argv.index("--manifest") + 1] for argv in self.calls}), 1)

    def test_prepare_only_subset_and_successful_reuse_do_not_train(self):
        config = copy.deepcopy(self.config); config["manifest"]["expected_test"] = 2
        path = self.save(config)
        first = server.run_server(path, prepare_only=True, runner=self.runner, builder=self.builder)
        self.assertEqual(first["status"], "prepared")
        second = server.run_server(path, model="plain_unet", prepare_only=True, runner=self.runner,
                                   builder=lambda *_a, **_k: self.fail("Must reuse the same prepared split"))
        self.assertEqual(first["protected_split"], second["protected_split"])
        self.assertEqual(self.calls, [])
        with self.assertRaisesRegex(ValueError, "expected_test=1773"):
            server.run_server(path, runner=self.runner, builder=self.builder)

    def test_not_ready_manifest_blocks_before_audit_or_neural_training(self):
        def incomplete(_root, output, **_kwargs):
            Path(output).mkdir()
            record = {"ready": False, "counts": {"train": 22, "test": 16},
                      "issues": [{"code": "missing_role", "message": "SWI brain masks missing"}]}
            (Path(output) / "inventory.json").write_text(json.dumps(record))
            return record
        result = server.run_server(self.save(), runner=self.runner, builder=incomplete)
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["training_started"])
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "work" / "prepared.json").exists())
        with self.assertRaisesRegex(FileExistsError, "new work_dir"):
            server.run_server(self.save(), runner=self.runner, builder=self.builder)

    def test_protected_test_cannot_become_model_selection_validation(self):
        record = self.builder("unused", self.root / "metadata", expected_train=2, expected_test=2, seed=42)
        manifest = Path(record["artifacts"]["split_manifest"])
        rows = server._read_csv(manifest)
        next(row for row in rows if row["split"] == "test")["split"] = "val"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "test membership"):
            server.assert_protected_split(record, record["artifacts"]["original_manifest"], manifest,
                                          record["artifacts"]["split_record"])

    def test_reuse_rejects_tampered_artifact_or_changed_common_training_settings(self):
        path = self.save()
        server.run_server(path, prepare_only=True, runner=self.runner, builder=self.builder)
        altered = copy.deepcopy(self.config); altered["training"]["epochs"] = 2
        with self.assertRaisesRegex(ValueError, "configuration"):
            server.run_server(self.save(altered), prepare_only=True, runner=self.runner, builder=self.builder)
        self.save()
        inventory = self.root / "work" / "manifest" / "inventory_cases.csv"
        inventory.write_text(inventory.read_text() + "tampered\n")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            server.run_server(path, prepare_only=True, runner=self.runner, builder=self.builder)

    def test_reuse_rejects_changed_implementation_inventory(self):
        path = self.save()
        server.run_server(path, prepare_only=True, runner=self.runner, builder=self.builder)
        with mock.patch.object(server, "_code_hashes", return_value={"changed.py": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "implementation changed"):
                server.run_server(path, prepare_only=True, runner=self.runner, builder=self.builder)

    def test_quality_and_default_identity_audit_run_before_training(self):
        contract = self.root / "contract.json"; contract.write_text("{}")
        config = copy.deepcopy(self.config); config["quality"] = {"contract": "contract.json"}; config.pop("independence")
        result = server.run_server(self.save(config), runner=self.runner, builder=self.builder)
        self.assertEqual([argv[2] for argv in self.calls], ["swixqsm_minimal.qc", "swixqsm_minimal.audit_splits", "swixqsm_minimal.train"])
        qc = self.calls[0]
        self.assertEqual(qc[qc.index("--split") + 1:qc.index("--qsm-units")], ["train", "val"])
        self.assertIn("--qc-report", self.calls[-1]); self.assertIn("--independence-report", self.calls[-1])
        self.assertEqual(result["status"], "training_completed")

    def test_failed_quality_report_stops_before_identity_audit_or_training(self):
        contract = self.root / "contract.json"; contract.write_text("{}")
        config = copy.deepcopy(self.config); config["quality"] = {"contract": "contract.json"}

        def failed_report(argv, **kwargs):
            self.runner(argv, **kwargs)
            path = Path(argv[argv.index("--output") + 1]) / "report.json"
            report = json.loads(path.read_text()); report["status"] = "failed"; path.write_text(json.dumps(report))
            return 0

        with self.assertRaisesRegex(ValueError, "completely passed"):
            server.run_server(self.save(config), runner=failed_report, builder=self.builder)
        self.assertEqual([argv[2] for argv in self.calls], ["swixqsm_minimal.qc"])
        self.assertFalse((self.root / "work" / "prepared.json").exists())

    def test_optional_simple_fitting_is_before_models_and_requires_real_space_declaration(self):
        config = copy.deepcopy(self.config)
        config["simple_baselines"] = {"enabled": True, "reference_id": "declared_CSF", "space_id": "declared_common_space", "units": "ppb"}
        result = server.run_server(self.save(config), runner=self.runner, builder=self.builder)
        self.assertEqual([argv[2] for argv in self.calls], ["swixqsm_minimal.fit_baselines", "swixqsm_minimal.train"])
        self.assertEqual(result["simple_baselines"]["status"], "mean_linear_fitted_train_only")
        self.assertFalse(result["simple_baselines"]["test_evaluated"])
        config["simple_baselines"].pop("space_id")
        with self.assertRaisesRegex(ValueError, "space_id"):
            server.load_config(self.save(config))

    def test_config_unknown_keys_nonfinite_seed_conflict_and_argument_injection_rejected(self):
        invalid = []
        for field, value in (("resume", "old.pt"), ("model", "plain_unet"), ("lambda_adv", 999), ("epochs", 1.5)):
            config = copy.deepcopy(self.config); config["training"][field] = value; invalid.append(config)
        config = copy.deepcopy(self.config); config["training"]["qsm_scale"] = float("nan"); invalid.append(config)
        config = copy.deepcopy(self.config); config["manifest"]["seed"] = 99; invalid.append(config)
        config = copy.deepcopy(self.config); config["manifest"]["qsm_name"] = "../outside.nii.gz"; invalid.append(config)
        config = copy.deepcopy(self.config); config["quality"]["report"] = "absent.json"; invalid.append(config)
        for config in invalid:
            with self.subTest(config=config), self.assertRaises((ValueError, FileNotFoundError)):
                server.load_config(self.save(config))
        self.assertFalse((self.root / "work").exists())

    def test_existing_experiment_is_not_overwritten_or_implicitly_resumed(self):
        path = self.save()
        server.run_server(path, runner=self.runner, builder=self.builder)
        previous_calls = len(self.calls)
        with self.assertRaisesRegex(FileExistsError, "--resume"):
            server.run_server(path, runner=self.runner, builder=self.builder)
        self.assertEqual(len(self.calls), previous_calls)

    def test_default_runner_uses_argument_list_and_shell_false(self):
        command = ["python", "-m", "example", "--value", "literal ; $(no-shell)"]
        with mock.patch.object(server.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
            self.assertEqual(server._default_runner(command, log_path=self.root / "child.log", cwd=str(self.root)), 0)
        self.assertEqual(run.call_args.args[0], command)
        self.assertIs(run.call_args.kwargs["shell"], False)

    def test_cpu_prepare_only_accepts_desired_cuda_and_null_optional_lists(self):
        config = copy.deepcopy(self.config)
        config["training"]["device"] = "cuda"
        config["manifest"].update(original_train_ids=None, original_test_ids=None, metadata_csv=None)
        with mock.patch("torch.cuda.is_available", return_value=False):
            result = server.run_server(self.save(config), prepare_only=True, runner=self.runner, builder=self.builder)
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(self.calls, [])
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                server.run_server(self.save(config), runner=self.runner, builder=self.builder)

    def test_segment_is_not_reported_as_completed_full_training(self):
        config = copy.deepcopy(self.config)
        config["training"].update(epochs=3, max_epochs_this_run=1)
        result = server.run_server(self.save(config), runner=self.runner, builder=self.builder)
        self.assertEqual(result["status"], "partial_training")
        self.assertEqual(result["completed"], [])
        self.assertEqual(result["outcomes"][0]["status"], "needs_explicit_resume")
        self.assertIn("--resume", result["outcomes"][0]["resume_argv"])

    def test_real_metadata_builder_prepare_only_reads_no_image_content(self):
        config = copy.deepcopy(self.config)
        config["manifest"].update(expected_test=1, input_mask_source="Synthetic SWI-only declaration")
        for folder, identities in (("train", ("subjT000", "subjT001")), ("validate", ("subjT002",))):
            for identity in identities:
                directory = self.root / "source" / folder / identity
                directory.mkdir(parents=True)
                for name in ("SWI.nii.gz", "QSM.nii.gz", "brain_mask.nii.gz", "QSM_mask.nii.gz"):
                    (directory / name).write_bytes(b"not a NIfTI, metadata-only fixture")
        with mock.patch("nibabel.load", side_effect=AssertionError("No image decoding during metadata preparation")):
            result = server.run_server(self.save(config), prepare_only=True, runner=self.runner)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["protected_split"]["group_counts"], {"train": 1, "val": 1, "test": 1})

    def test_enabled_input_masks_runs_preinventory_preparer_and_final_builder_in_order(self):
        config = copy.deepcopy(self.config)
        config["manifest"]["expected_test"] = 2
        config["input_masks"] = {"enabled": True, "skull_stripped": True, "fill_holes": True, "valid_mask_policy": "intersect"}
        stages = []

        def builder(data_root, output, **kwargs):
            stages.append("inventory" if kwargs["inventory_only"] else "final_manifest")
            record = self.builder(data_root, output, **kwargs)
            if kwargs["inventory_only"]:
                record["ready"] = False
                record["issues"] = [{"code": "missing_role_file", "role": "swi_mask_path"}, {"code": "missing_input_mask_source"}]
                (Path(output) / "inventory.json").write_text(json.dumps(record))
            return record

        def masks(manifest, output, **kwargs):
            stages.append("masks")
            self.assertEqual(kwargs, {"skull_stripped": True, "fill_holes": True, "valid_mask_policy": "intersect"})
            output = Path(output); output.mkdir()
            rows = server._read_csv(manifest)
            cases = []
            for row in rows:
                path = output / (row["case_id"] + "_input_mask.nii.gz")
                path.write_bytes(b"mock SWI-derived support")
                valid = output / (row["case_id"] + "_valid_mask.nii.gz"); valid.write_bytes(b"mock derived validity")
                provenance = output / (row["case_id"] + "_input_mask.json"); provenance.write_text("{}")
                valid_provenance = output / (row["case_id"] + "_valid_mask.json"); valid_provenance.write_text("{}")
                cases.append({"input_mask_path": str(path), "input_mask_sha256": server._sha(path),
                              "final_valid_mask_path": str(valid), "final_valid_mask_sha256": server._sha(valid),
                              "provenance_path": str(provenance), "provenance_sha256": server._sha(provenance),
                              "valid_mask_provenance_path": str(valid_provenance),
                              "valid_mask_provenance_sha256": server._sha(valid_provenance)})
                row["swi_mask_path"] = str(path)
                row["mask_path"] = str(valid)
                row["input_mask_source"] = "Mock SWI-only support; no QSM target read"
            metadata = output / "metadata.csv"
            with metadata.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            report = output / "report.json"; report.write_text(json.dumps({"status": "mock_generated", "cases": cases}))
            return {"metadata_csv": str(metadata), "report_path": str(report), "status": "mock_generated", "cases": cases}

        result = server.run_server(self.save(config), prepare_only=True, runner=self.runner, builder=builder, mask_preparer=masks)
        self.assertEqual(stages, ["inventory", "masks", "final_manifest"])
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(self.calls, [])

    def test_input_masks_cannot_repair_missing_validity_mask_or_bad_membership(self):
        config = copy.deepcopy(self.config)
        config["manifest"]["expected_test"] = 2
        config["input_masks"] = {"enabled": True, "skull_stripped": True}

        def bad_builder(data_root, output, **kwargs):
            record = self.builder(data_root, output, **kwargs)
            record.update(ready=False, issues=[{"code": "missing_role_file", "role": "mask_path"}])
            (Path(output) / "inventory.json").write_text(json.dumps(record))
            return record

        result = server.run_server(self.save(config), prepare_only=True, runner=self.runner, builder=bad_builder,
                                   mask_preparer=lambda *_a, **_k: self.fail("Cannot generate missing QSM validity support"))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stage"], "before_input_masks")
        self.assertEqual(self.calls, [])

    def test_mask_derivation_cannot_rewrite_targets_or_reference_metadata(self):
        inventory = self.builder("unused", self.root / "metadata", expected_train=2, expected_test=2, seed=42)
        original = Path(inventory["artifacts"]["original_manifest"])
        rows = server._read_csv(original)
        rows[0]["qsm_path"] = str(self.root / "wrong_target.nii.gz")
        altered = self.root / "altered.csv"
        with altered.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "image targets"):
            server._assert_mask_derivation(original, altered)

    def _real_support_preparation(self):
        import nibabel as nib
        import numpy as np
        config = copy.deepcopy(self.config)
        config["manifest"]["expected_test"] = 1
        config["input_masks"] = {"enabled": True, "skull_stripped": True,
                                 "fill_holes": True, "valid_mask_policy": "intersect"}
        for folder, identities in (("train", ("subjT000", "subjT001")), ("validate", ("subjT002",))):
            for identity in identities:
                directory = self.root / "source" / folder / identity
                directory.mkdir(parents=True)
                swi = np.zeros((8, 8, 2), np.float32); swi[2:6, 2:6, :] = 1
                valid = (swi > 0).astype(np.uint8); valid[0, 0, 0] = 1
                for name, array in (("SWI.nii.gz", swi), ("QSM_mask.nii.gz", valid)):
                    image = nib.Nifti1Image(array, np.diag([1., 1., 2., 1.])); image.header.set_xyzt_units("mm")
                    nib.save(image, directory / name)
                (directory / "QSM.nii.gz").write_bytes(b"QSM labels must never be opened in this preparation probe")
        original_load = nib.load

        def check_load(path, *args, **kwargs):
            if Path(path).name == "QSM.nii.gz":
                self.fail("QSM target was read while preparing SWI-only support")
            return original_load(path, *args, **kwargs)

        with mock.patch.object(nib, "load", side_effect=check_load):
            result = server.run_server(self.save(config), prepare_only=True, runner=self.runner)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["protected_split"]["group_counts"], {"train": 1, "val": 1, "test": 1})
        prepared = json.loads((self.root / "work" / "prepared.json").read_text())
        return config, prepared

    def test_real_swi_only_mask_preparation_and_builder_preserve_test_and_never_load_qsm(self):
        config, prepared = self._real_support_preparation()
        mask_record = prepared["input_masks"]
        self.assertEqual(mask_record["n_cases"], 3)
        self.assertTrue(all(case["excluded_valid_voxels"] == 1 for case in mask_record["cases"]))
        self.assertFalse(any(Path(path).name == "QSM.nii.gz" for path in mask_record["source_sha256"]))
        # The derived masks are final manifest roles. Their original source
        # validity masks, and test SWI, must also invalidate prepared reuse.
        sources = [self.root / "source" / "train" / "subjT000" / "QSM_mask.nii.gz",
                   self.root / "source" / "validate" / "subjT002" / "SWI.nii.gz"]
        import os
        for source in sources:
            stat = source.stat()
            try:
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + subjT000))
                with self.assertRaisesRegex(ValueError, "Input-mask source metadata changed"):
                    server.run_server(self.save(config), prepare_only=True, runner=self.runner)
            finally:
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def _assert_generated_test_files_protected(self, fields):
        import os
        config, prepared = self._real_support_preparation()
        case = next(case for case in prepared["input_masks"]["cases"] if case["original_split"] == "test")
        for field in fields:
            path = Path(case[field])
            self.assertIn(str(path), prepared["fixed_file_sha256"])
            original, stat = path.read_bytes(), path.stat()
            try:
                path.unlink()
                with self.subTest(field=field, change="deleted"), self.assertRaises(FileNotFoundError):
                    server.run_server(self.save(config), prepare_only=True, runner=self.runner)
                # Preserve size and timestamp so only the content binding, not
                # the ordinary-change stat detector, can reject this rewrite.
                changed = bytes([original[0] ^ 1]) + original[1:]
                path.write_bytes(changed)
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                with self.subTest(field=field, change="same_stat_rewrite"), self.assertRaisesRegex(ValueError, "report SHA256"):
                    server.run_server(self.save(config), prepare_only=True, runner=self.runner)
            finally:
                path.write_bytes(original)
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def test_reuse_rejects_deleted_or_rewritten_generated_test_masks(self):
        self._assert_generated_test_files_protected(("input_mask_path", "final_valid_mask_path"))

    def test_reuse_rejects_deleted_or_rewritten_generated_test_provenance(self):
        self._assert_generated_test_files_protected(("provenance_path", "valid_mask_provenance_path"))

    def test_generated_mask_report_hash_is_verified_before_preparation_is_committed(self):
        path = self.root / "mask.nii.gz"; path.write_bytes(b"generated mask bytes")
        provenance = self.root / "mask.json"; provenance.write_text("{}")
        case = {"input_mask_path": str(path), "input_mask_sha256": "0" * 64,
                "provenance_path": str(provenance), "provenance_sha256": server._sha(provenance)}
        report = self.root / "report.json"; report.write_text(json.dumps({"cases": [case]}))
        with self.assertRaisesRegex(ValueError, "report SHA256"):
            server._generated_mask_bindings({"report_path": str(report), "cases": [case]}, "preserve")


if __name__ == "__main__":
    unittest.main()

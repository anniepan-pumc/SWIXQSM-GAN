"""Flat server preparation with the real builder and metadata-only fake images."""
import copy
import csv
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from .. import run_server as server


class FlatServerConfigTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swix-flat-server-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = self.root / "flat_nifti"
        self.data.mkdir()
        self.results = self.root / "existing_results"
        self.results.mkdir()
        self.sentinel = self.results / "old_result.txt"
        self.sentinel.write_bytes(b"Preserve pre-existing results")
        self.train_cases = ["100_visit1", "100_visit2", "300", "500"]
        self.test_cases = ["200_visit1", "400"]
        for index, case in enumerate(self.train_cases + self.test_cases):
            self.case(self.data / case, nested=index % 2 == 0)
        self.lists = self.root / "lists"
        self.lists.mkdir()
        (self.lists / "train.csv").write_text("participant_id\n100\n300\n500\n")
        (self.lists / "test.csv").write_text("participant_id\n200\n400\n")
        self.config = {
            "schema": "swixqsm.server_training", "version": 1,
            "data_root": "flat_nifti", "work_dir": "existing_results/new_run",
            "model": "plain_unet",
            "manifest": {"layout": "flat", "expected_train": 3, "expected_test": 2,
                         "val_fraction": 1 / 3, "qsm_units": "ppb",
                         "original_train_ids": "lists/train.csv",
                         "original_test_ids": "lists/test.csv",
                         "input_mask_source": "Synthetic pre-existing input-only mask"},
            "training": {"units": "ppb", "qsm_scale": 100., "size": 32,
                         "epochs": 1, "device": "cpu", "seed": 42},
            "quality": {"allow_unverified_reason": "Synthetic metadata preparation only"},
            "independence": {"allow_unverified_reason": "Synthetic IDs only"},
        }
        # Other agents may update production files during these integration
        # tests. Split/manifest provenance is real; pin only the code inventory.
        patch = mock.patch.object(server, "_code_hashes", return_value=server._code_hashes())
        patch.start()
        self.addCleanup(patch.stop)

    @staticmethod
    def case(directory, *, nested=False):
        directory.mkdir(parents=True)
        for name in ("SWI.nii.gz", "brain_mask.nii.gz"):
            (directory / name).write_bytes(b"Intentionally invalid NIfTI: metadata only")
        target = directory / "QSM" if nested else directory
        target.mkdir(exist_ok=True)
        for name in ("QSM.nii.gz", "QSM_mask.nii.gz"):
            (target / name).write_bytes(b"Intentionally invalid NIfTI: metadata only")

    def save(self, config=None):
        path = self.root / "config.json"
        path.write_text(json.dumps(self.config if config is None else config))
        return path

    @contextmanager
    def no_image_or_child_access(self):
        original_open = Path.open

        def metadata_only_open(path, *args, **kwargs):
            if path.name.endswith((".nii", ".nii.gz")):
                raise AssertionError("Server metadata preparation must not open image bytes")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", metadata_only_open), \
                mock.patch("nibabel.load", side_effect=AssertionError("No image headers or arrays")), \
                mock.patch.object(server, "_default_runner", side_effect=AssertionError("No child programs")):
            yield

    @staticmethod
    def rows(path):
        with Path(path).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_real_flat_builder_preserves_exact_test_and_draws_val_only_from_train(self):
        path = self.save()
        with self.no_image_or_child_access():
            result = server.run_server(path, prepare_only=True)
        self.assertEqual(result["status"], "prepared")
        self.assertFalse(result["training_started"])
        self.assertEqual(self.sentinel.read_bytes(), b"Preserve pre-existing results")
        work = self.results / "new_run"
        prepared = json.loads((work / "prepared.json").read_text())
        inventory = json.loads(Path(prepared["artifacts"]["inventory"]).read_text())
        self.assertEqual(inventory["layout"], "flat")
        self.assertTrue(inventory["original_membership_verified"])
        self.assertEqual((inventory["image_header_reads"], inventory["image_array_reads"]), (0, 0))
        original = self.rows(prepared["artifacts"]["original_manifest"])
        split = self.rows(prepared["artifacts"]["manifest"])
        old_test = [row for row in original if row["split"] == "test"]
        new_test = [row for row in split if row["split"] == "test"]
        self.assertEqual(old_test, new_test)
        self.assertEqual({row["case_id"] for row in new_test}, set(self.test_cases))
        self.assertEqual({row["participant_id"] for row in new_test}, {"200", "400"})
        self.assertEqual({row["case_id"] for row in split if row["split"] in {"train", "val"}}, set(self.train_cases))
        self.assertTrue(all(row["original_split"] == "train" for row in split if row["split"] == "val"))
        self.assertEqual(len({row["split"] for row in split if row["participant_id"] == "100"}), 1)
        self.assertEqual(prepared["protected_split"]["group_counts"], {"train": 2, "val": 1, "test": 2})
        for filename in ("train.csv", "test.csv"):
            self.assertIn(str(self.lists / filename), prepared["configuration_source_sha256"])
        with self.no_image_or_child_access():
            reused = server.run_server(path, model="pix2pix", prepare_only=True)
        self.assertEqual(reused["protected_split"], result["protected_split"])
        self.assertFalse((work / "experiments").exists())

    def test_flat_requires_both_frozen_membership_lists_before_any_output(self):
        for missing in [("original_train_ids",), ("original_test_ids",),
                        ("original_train_ids", "original_test_ids")]:
            config = copy.deepcopy(self.config)
            for key in missing:
                config["manifest"].pop(key)
            with self.subTest(missing=missing), self.no_image_or_child_access(), \
                    self.assertRaisesRegex(ValueError, "both original_train_ids and original_test_ids"):
                server.run_server(self.save(config), prepare_only=True)
            self.assertFalse((self.results / "new_run").exists())

    def test_flat_work_directory_cannot_be_input_root_or_descendant(self):
        for output in [self.data, self.data / "new_output", self.data / "100_visit1" / "results"]:
            config = copy.deepcopy(self.config)
            config["work_dir"] = str(output)
            with self.subTest(output=output), self.no_image_or_child_access(), \
                    self.assertRaisesRegex(ValueError, "data root|input data root"):
                server.run_server(self.save(config), prepare_only=True)
            self.assertFalse((output / "server_plan.json").exists())
        alias = self.root / "input_alias"
        alias.symlink_to(self.data, target_is_directory=True)
        config = copy.deepcopy(self.config); config["work_dir"] = str(alias / "new_output")
        with self.no_image_or_child_access(), self.assertRaisesRegex(ValueError, "input data root"):
            server.run_server(self.save(config), prepare_only=True)

    def test_default_split_dirs_layout_remains_compatible_with_real_preparation(self):
        data = self.root / "split_nifti"
        for index, case in enumerate(self.train_cases):
            self.case(data / "train" / case, nested=index % 2 == 0)
        for case in self.test_cases:
            self.case(data / "validate" / case)
        config = copy.deepcopy(self.config)
        config["data_root"] = str(data)
        config["manifest"].pop("layout")
        with self.no_image_or_child_access():
            result = server.run_server(self.save(config), prepare_only=True)
        work = Path(result["work_dir"])
        prepared = json.loads((work / "prepared.json").read_text())
        inventory = json.loads(Path(prepared["artifacts"]["inventory"]).read_text())
        self.assertEqual(inventory["layout"], "split_dirs")
        rows = self.rows(prepared["artifacts"]["manifest"])
        self.assertEqual({row["case_id"] for row in rows if row["split"] == "test"}, set(self.test_cases))
        self.assertTrue(all("/validate/" in row["swi_path"] for row in rows if row["split"] == "test"))

    def test_unknown_flat_participant_blocks_instead_of_silently_reassigning(self):
        self.case(self.data / "999")
        with self.no_image_or_child_access():
            result = server.run_server(self.save(), prepare_only=True)
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["training_started"])
        self.assertTrue(any(issue["code"] == "unassigned_flat_participant" for issue in result["issues"]))
        inventory_path = Path(result["inventory"])
        draft = self.rows(inventory_path.parent / "original_train_test.draft.csv")
        self.assertEqual(next(row for row in draft if row["case_id"] == "999")["split"], "unassigned")
        self.assertEqual({row["case_id"] for row in draft if row["split"] == "test"}, set(self.test_cases))
        self.assertFalse((inventory_path.parent / "split").exists())
        self.assertFalse((self.results / "new_run" / "prepared.json").exists())


if __name__ == "__main__":
    unittest.main()

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..build_manifest import build_manifest, main as build_main


class BuildManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.data = self.root/"primary_cohort"
        (self.data/"train").mkdir(parents=True)
        (self.data/"validate").mkdir()
        self.case("train", "100")
        self.case("train", "200", nested=True)
        self.case("validate", "300", nested=True)
        self.kwargs = dict(qsm_units="ppb", input_mask_source="Synthetic input-only SWI brain extraction",
                           expected_train=2, expected_test=1)

    def tearDown(self):
        self.temporary.cleanup()

    def case(self, split, identifier, *, nested=False):
        directory = self.data/split/identifier
        directory.mkdir(parents=True)
        for name in ("SWI.nii.gz", "brain_mask.nii.gz"):
            (directory/name).write_bytes(b"Not a NIfTI: inventory must never read this content")
        qsm_dir = directory/"QSM" if nested else directory
        qsm_dir.mkdir(exist_ok=True)
        for name in ("QSM.nii.gz", "QSM_mask.nii.gz"):
            (qsm_dir/name).write_bytes(b"Not a NIfTI: file existence metadata only")
        return directory

    def run_builder(self, name="out", **changes):
        kwargs = {**self.kwargs, **changes}
        return build_manifest(self.data, self.root/name, **kwargs)

    def read_csv(self, relative):
        with (self.root/relative).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def metadata(self, rows, name="metadata.csv"):
        directory = self.root/"metadata"
        directory.mkdir(exist_ok=True)
        path = directory/name
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def make_flat(self):
        flat = self.root/"flat"
        flat.mkdir()
        for split in ("train", "validate"):
            for directory in (self.data/split).iterdir():
                directory.rename(flat/directory.name)
        train_ids, test_ids = self.root/"train_ids.txt", self.root/"test_ids.txt"
        train_ids.write_text("200\n100\n")  # File/directory sorting never assigns membership.
        test_ids.write_text("300\n")
        return flat, train_ids, test_ids

    def run_flat(self, flat, train_ids, test_ids, name="flat_out", **changes):
        return build_manifest(flat, self.root/name, **{**self.kwargs, "layout": "flat",
                              "original_train_ids": train_ids, "original_test_ids": test_ids, **changes})

    def test_flat_and_nested_roles_ready_no_nifti_open_and_protected_test(self):
        original_open = Path.open

        def no_image_open(path, *args, **kwargs):
            if path.name.endswith((".nii", ".nii.gz")):
                raise AssertionError("Image content must never be opened")
            return original_open(path, *args, **kwargs)

        with patch("nibabel.load", side_effect=AssertionError("No NIfTI headers or arrays")), \
                patch.object(Path, "open", no_image_open):
            record = self.run_builder()
        self.assertTrue(record["ready"])
        self.assertEqual(record["qc_status"], "not_evaluated")
        self.assertEqual((record["image_header_reads"], record["image_array_reads"]), (0, 0))
        original = self.read_csv("out/original_train_test.csv")
        split = self.read_csv("out/split/manifest.csv")
        self.assertEqual(len(original), 3)
        old_test = [row for row in original if row["split"] == "test"]
        new_test = [row for row in split if row["split"] == "test"]
        self.assertEqual(old_test, new_test)
        self.assertEqual(old_test[0]["case_id"], "300")
        self.assertTrue(old_test[0]["qsm_path"].endswith("/validate/300/QSM/QSM.nii.gz"))
        self.assertEqual({row["split"] for row in split if row["case_id"] != "300"}, {"train", "val"})
        self.assertEqual(record["new_split_group_counts"], {"train": 1, "val": 1, "test": 1})
        self.assertTrue(record["test_unchanged_by_new_split"])
        self.assertEqual(record["test_participant_ids_sorted_sha256"], hashlib.sha256(b"300\n").hexdigest())
        self.assertFalse(record["original_membership_verified"])

    def test_missing_test_qsm_retained_draft_only_never_drop(self):
        (self.data/"validate/300/QSM/QSM.nii.gz").unlink()
        record = self.run_builder()
        self.assertFalse(record["ready"])
        self.assertEqual(record["counts"]["test"]["case_directories"], 1)
        self.assertEqual(record["test_participant_ids"], ["300"])
        rows = self.read_csv("out/inventory_cases.csv")
        test = next(row for row in rows if row["split"] == "test")
        self.assertEqual(test["qsm_path"], "")
        self.assertTrue(any(item["code"] == "missing_role_file" and item["case_id"] == "300"
                            for item in record["issues"]))
        self.assertTrue((self.root/"out/original_train_test.draft.csv").exists())
        self.assertFalse((self.root/"out/original_train_test.csv").exists())
        self.assertFalse((self.root/"out/split").exists())

    def test_metadata_aliases_cannot_hide_same_original_eid_in_both_splits(self):
        self.case("validate", "100_visit2")
        metadata = self.metadata([dict(case_id="100_visit2", original_split="test",
                                       subject_id="different_alias", global_subject_id="different_global")])
        record = self.run_builder(expected_test=2, metadata_csv=metadata)
        self.assertFalse(record["ready"])
        self.assertTrue(any(item["code"] == "participant_identity_crosses_splits"
                            for item in record["issues"]))
        self.assertFalse((self.root/"out/split").exists())

    def test_all_1773_test_directories_stay_in_inventory_even_when_qsm_missing(self):
        # Real files are unnecessary to prove complete-case filtering is absent.
        for value in range(10000, 11772):
            (self.data/"validate"/str(value)).mkdir()
        record = self.run_builder(expected_test=1773)
        self.assertFalse(record["ready"])
        self.assertEqual(record["counts"]["test"]["case_directories"], 1773)
        self.assertEqual(len(record["test_participant_ids"]), 1773)
        rows = self.read_csv("out/inventory_cases.csv")
        self.assertEqual(sum(row["split"] == "test" for row in rows), 1773)
        self.assertFalse((self.root/"out/split").exists())

    def test_inventory_only_never_calls_split_even_complete(self):
        with patch("swixqsm_minimal.build_manifest.create_split", side_effect=AssertionError("No split in inventory mode")):
            record = self.run_builder(inventory_only=True)
        self.assertFalse(record["ready"])
        self.assertTrue(record["data_complete"])
        self.assertEqual(record["split_not_created_reason"], "inventory_only")
        self.assertFalse((self.root/"out/original_train_test.csv").exists())

    def test_missing_declarations_block_but_template_never_guesses(self):
        record = self.run_builder(qsm_units=None, input_mask_source=None)
        self.assertFalse(record["ready"])
        self.assertEqual(len(self.read_csv("out/inventory_cases.csv")), 3)
        template = self.read_csv("out/case_metadata.template.csv")
        for row in template:
            self.assertEqual(row["qsm_units"], "")
            self.assertEqual(row["input_mask_source"], "")
            self.assertEqual(row["registration_review"], "pending")
            self.assertEqual(row["qsm_reference"], "")
            self.assertEqual(row["qsm_reference_target_ppb"], "")
        self.assertNotIn("split", template[0])

    def test_metadata_per_case_declarations_and_all_paths_rebased(self):
        rows = []
        for case_id, split in (("100", "train"), ("200", "train"), ("300", "test")):
            rows.append(dict(case_id=case_id, original_split=split, qsm_units="ppm",
                             input_mask_source="actual per-case input-only extraction",
                             qsm_reference="declared-reference", qsm_reference_source="actual methods record",
                             registration_review="pending", registration_landmarks_path=f"evidence/{case_id}.json",
                             custom_evidence_paths=json.dumps([f"evidence/{case_id}.png"])))
        metadata = self.metadata(rows)
        record = self.run_builder(qsm_units=None, input_mask_source=None, metadata_csv=metadata)
        self.assertTrue(record["ready"])
        output = self.read_csv("out/original_train_test.csv")
        for row in output:
            self.assertEqual(row["qsm_units"], "ppm")
            self.assertEqual(row["registration_review"], "pending")
            self.assertEqual(row["registration_landmarks_path"], str(metadata.parent/"evidence"/(row["case_id"]+".json")))
            self.assertEqual(json.loads(row["custom_evidence_paths"]), [str(metadata.parent/"evidence"/(row["case_id"]+".png"))])

    def test_ambiguous_candidates_block_until_explicit_metadata_selects(self):
        directory = self.data/"train/100/QSM"
        directory.mkdir()
        (directory/"QSM.nii.gz").write_bytes(b"another candidate")
        (directory/"QSM_mask.nii.gz").write_bytes(b"another candidate")
        first = self.run_builder()
        self.assertFalse(first["ready"])
        self.assertEqual(sum(issue["code"] == "ambiguous_role_files" for issue in first["issues"]), 2)
        metadata = self.metadata([dict(case_id="100", original_split="train",
                                      qsm_path="../data/train/100/QSM/QSM.nii.gz",
                                      mask_path="../data/train/100/QSM/QSM_mask.nii.gz")])
        second = self.run_builder("resolved", metadata_csv=metadata)
        self.assertTrue(second["ready"])
        row = next(row for row in self.read_csv("resolved/original_train_test.csv") if row["case_id"] == "100")
        self.assertEqual(row["qsm_path"], str(directory/"QSM.nii.gz"))

    def test_original_id_set_validation_retains_missing_and_reports_unexpected(self):
        train_ids = self.root/"train_ids.txt"
        train_ids.write_text("100\n200\n")
        test_ids = self.root/"test_ids.csv"
        test_ids.write_text("eid\n300\n")
        verified = self.run_builder(original_train_ids=train_ids, original_test_ids=test_ids)
        self.assertTrue(verified["ready"])
        self.assertTrue(verified["original_membership_verified"])
        test_ids.write_text("eid\n400\n")
        mismatch = self.run_builder("mismatch", original_train_ids=train_ids, original_test_ids=test_ids)
        self.assertFalse(mismatch["ready"])
        codes = {issue["code"] for issue in mismatch["issues"]}
        self.assertTrue({"missing_declared_participants", "unexpected_participants"}.issubset(codes))
        rows = self.read_csv("mismatch/inventory_cases.csv")
        missing = next(row for row in rows if row["case_id"] == "400")
        self.assertEqual(missing["directory_present"], "False")
        self.assertEqual(missing["split"], "test")
        self.assertEqual(mismatch["test_participant_ids"], ["300", "400"])
        self.assertEqual(mismatch["actual_test_participant_ids"], ["300"])

    def test_duplicate_metadata_unknown_case_and_protected_split_override(self):
        row = dict(case_id="100", original_split="train", qsm_units="ppb")
        with self.assertRaisesRegex(ValueError, "Duplicate metadata"):
            self.run_builder(metadata_csv=self.metadata([row, row]))
        with self.assertRaisesRegex(ValueError, "cannot override split"):
            self.run_builder(metadata_csv=self.metadata([{**row, "split": "test"}]))
        unknown = self.run_builder(metadata_csv=self.metadata([dict(case_id="999", original_split="test")]))
        self.assertFalse(unknown["ready"])
        self.assertTrue(any(issue["code"] == "unknown_metadata_case" for issue in unknown["issues"]))
        self.assertIn("999", {row["case_id"] for row in self.read_csv("out/inventory_cases.csv")})

    def test_cross_split_identity_duplicate_paths_and_mask_role_reuse_block(self):
        original = dict(case_id="300", original_split="test", global_subject_id="cohort:100")
        record = self.run_builder(metadata_csv=self.metadata([original]))
        self.assertFalse(record["ready"])
        self.assertTrue(any(issue["code"] == "global_identity_crosses_splits" for issue in record["issues"]))
        shared = self.run_builder("shared", metadata_csv=self.metadata([
            dict(case_id="200", original_split="train", swi_path=str(self.data/"train/100/SWI.nii.gz"))]))
        self.assertFalse(shared["ready"])
        self.assertTrue(any(issue["code"] == "duplicate_role_path" for issue in shared["issues"]))
        bad_mask = self.run_builder("mask", metadata_csv=self.metadata([
            dict(case_id="300", original_split="test", swi_mask_path=str(self.data/"validate/300/QSM/QSM_mask.nii.gz"))]))
        self.assertFalse(bad_mask["ready"])
        self.assertTrue(any(issue["code"] == "roles_share_file" for issue in bad_mask["issues"]))

    def test_visits_grouped_and_metadata_alias_identity_kept_together(self):
        self.case("train", "100_repeat")
        record = self.run_builder()
        self.assertTrue(record["ready"])
        self.assertEqual(record["counts"]["train"]["case_directories"], 3)
        self.assertEqual(record["counts"]["train"]["global_subjects"], 2)
        rows = self.read_csv("out/split/manifest.csv")
        matched = [row for row in rows if row["participant_id"] == "100"]
        self.assertEqual(len(matched), 2)
        self.assertEqual(len({row["split"] for row in matched}), 1)

    def test_unsafe_role_names_rejected_and_hidden_files_ignored(self):
        for value in ("../outside.nii.gz", "/absolute.nii.gz", "QSM/*.nii.gz", "QSM/../QSM.nii.gz", "a\\b.nii.gz"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Unsafe"):
                self.run_builder(qsm_name=value)
        (self.data/"train/.hidden").mkdir()
        (self.data/"validate/._300").write_bytes(b"AppleDouble metadata")
        (self.data/"train/notes.txt").write_text("not a case")
        record = self.run_builder()
        self.assertTrue(record["ready"])
        self.assertEqual(len(record["ignored_entries"]), 3)

    def test_root_overlap_same_case_and_candidate_symlink_escape(self):
        self.case("validate", "100")
        with self.assertRaisesRegex(ValueError, "Same case_id"):
            self.run_builder(expected_test=2)
        (self.data/"validate/100/SWI.nii.gz").unlink()
        (self.data/"validate/100/brain_mask.nii.gz").unlink()
        (self.data/"validate/100/QSM.nii.gz").unlink()
        (self.data/"validate/100/QSM_mask.nii.gz").unlink()
        (self.data/"validate/100").rmdir()
        outside = self.root/"outside.nii.gz"
        outside.write_bytes(b"explicit external content")
        (self.data/"train/100/QSM.nii.gz").unlink()
        (self.data/"train/100/QSM.nii.gz").symlink_to(outside)
        record = self.run_builder()
        self.assertFalse(record["ready"])
        self.assertTrue(any(issue["code"] == "candidate_escapes_case" for issue in record["issues"]))
        other = self.root/"overlap"
        other.mkdir()
        (other/"train").symlink_to(self.data/"train", target_is_directory=True)
        (other/"validate").symlink_to(self.data/"train", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_manifest(other, self.root/"other_out", **self.kwargs)

    def test_default_expected_counts_are_strict_and_output_refuses_overwrite(self):
        record = build_manifest(self.data, self.root/"strict", qsm_units="ppb", input_mask_source="actual input-only extraction")
        self.assertFalse(record["ready"])
        self.assertTrue(any(issue["code"] == "unexpected_global_subject_count" and issue["expected"] == 1773 for issue in record["issues"]))
        with self.assertRaises(FileExistsError):
            build_manifest(self.data, self.root/"strict", **self.kwargs)
        with self.assertRaisesRegex(ValueError, "inside original"):
            build_manifest(self.data, self.data/"train/new_output", **self.kwargs)
        for changes in ({"qsm_units": "unknown"}, {"seed": -1}, {"val_fraction": 0}, {"expected_train": True}, {"subject_pattern": "["}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_builder(**changes)

    def test_custom_regex_missing_list_member_does_not_overwrite_invalid_directory(self):
        # A custom pattern rejects the existing numeric-only case names.
        # Its declared participant still gets a separate missing-member row.
        ids = self.root/"ids.txt"
        ids.write_text("sub-100\n")
        record = self.run_builder(subject_pattern=r"^sub-(?P<participant>[0-9]+)$",
                                  original_train_ids=ids)
        self.assertFalse(record["ready"])
        rows = self.read_csv("out/inventory_cases.csv")
        self.assertIn("100", {row["case_id"] for row in rows})
        self.assertIn("missing-participant:100", {row["case_id"] for row in rows})
        original = next(row for row in rows if row["case_id"] == "100")
        self.assertEqual(original["directory_present"], "True")

    def test_flat_layout_assigns_only_from_lists_and_preserves_test_visits(self):
        self.case("train", "100_visit2")
        self.case("validate", "300_visit2")
        flat, train_ids, test_ids = self.make_flat()
        original_open = Path.open

        def no_image_open(path, *args, **kwargs):
            if path.name.endswith((".nii", ".nii.gz")):
                raise AssertionError("Flat inventory must never read image content")
            return original_open(path, *args, **kwargs)

        with patch("nibabel.load", side_effect=AssertionError("No header reads")), \
                patch.object(Path, "open", no_image_open):
            record = self.run_flat(flat, train_ids, test_ids)
        self.assertTrue(record["ready"])
        self.assertTrue(record["original_membership_verified"])
        self.assertEqual(record["layout"], "flat")
        self.assertEqual(record["membership_policy"], "explicit_original_id_lists")
        self.assertFalse(record["validate_directory_is_test"])
        self.assertEqual(record["test_participant_ids"], ["300"])
        self.assertEqual(record["unassigned_case_ids"], [])
        self.assertEqual(record["original_id_list_sources"]["train"]["sha256"], hashlib.sha256(train_ids.read_bytes()).hexdigest())
        self.assertEqual((record["image_header_reads"], record["image_array_reads"]), (0, 0))
        original = self.read_csv("flat_out/original_train_test.csv")
        split = self.read_csv("flat_out/split/manifest.csv")
        self.assertEqual([r for r in original if r["split"] == "test"], [r for r in split if r["split"] == "test"])
        self.assertEqual({r["case_id"] for r in split if r["split"] == "test"}, {"300", "300_visit2"})
        self.assertEqual(len({r["split"] for r in split if r["participant_id"] == "100"}), 1)
        self.assertEqual(record["new_split_group_counts"], {"train": 1, "val": 1, "test": 1})
        self.assertTrue(all(r["registration_review"] == "pending" for r in self.read_csv("flat_out/case_metadata.template.csv")))

    def test_flat_layout_requires_two_disjoint_original_lists(self):
        flat, train_ids, test_ids = self.make_flat()
        for train, test in ((None, None), (train_ids, None), (None, test_ids)):
            with self.subTest(train=train, test=test), self.assertRaisesRegex(ValueError, "requires both"):
                self.run_flat(flat, train, test, inventory_only=True)
        self.assertFalse((self.root/"flat_out").exists())
        test_ids.write_text("100\n300\n")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_flat(flat, train_ids, test_ids)
        self.assertFalse((self.root/"flat_out").exists())

    def test_flat_unknown_directories_and_missing_members_block_without_guessing(self):
        flat, train_ids, test_ids = self.make_flat()
        (flat/"300").rename(flat/"999")
        (flat/"notes-directory").mkdir()
        (flat/".hidden").mkdir()
        (flat/"README.txt").write_text("ordinary file, not a case")
        record = self.run_flat(flat, train_ids, test_ids)
        self.assertFalse(record["ready"])
        self.assertFalse(record["original_membership_verified"])
        self.assertEqual(record["unassigned_case_ids"], ["999", "notes-directory"])
        self.assertEqual(record["test_participant_ids"], ["300"])
        self.assertEqual(record["actual_test_participant_ids"], [])
        codes = {item["code"] for item in record["issues"]}
        self.assertTrue({"unassigned_flat_participant", "invalid_case_identity", "missing_declared_participants"}.issubset(codes))
        rows = self.read_csv("flat_out/inventory_cases.csv")
        unknown = next(row for row in rows if row["case_id"] == "999")
        missing = next(row for row in rows if row["case_id"] == "300")
        self.assertEqual(unknown["original_split"], "unassigned")
        self.assertEqual(unknown["split"], "unassigned")
        self.assertEqual(unknown["directory_present"], "True")
        self.assertEqual(missing["directory_present"], "False")
        self.assertEqual(missing["split"], "test")
        self.assertEqual(len(record["ignored_entries"]), 2)
        self.assertFalse((self.root/"flat_out/split").exists())

    def test_flat_metadata_cannot_reassign_original_test_or_split_person_visits(self):
        self.case("train", "100_visit2")
        flat, train_ids, test_ids = self.make_flat()
        wrong = self.metadata([dict(case_id="300", original_split="train", subject_id="alias", global_subject_id="alias")])
        record = self.run_flat(flat, train_ids, test_ids, metadata_csv=wrong)
        self.assertFalse(record["ready"])
        self.assertTrue(any(item["code"] == "unknown_metadata_case" for item in record["issues"]))
        rows = self.read_csv("flat_out/original_train_test.draft.csv")
        self.assertEqual(next(r for r in rows if r["case_id"] == "300" and r["original_split"] == "test")["split"], "test")
        split_person = self.metadata([dict(case_id="100_visit2", original_split="train", global_subject_id="false_other_person")], name="split-person.csv")
        second = self.run_flat(flat, train_ids, test_ids, name="flat_split_person", metadata_csv=split_person)
        self.assertFalse(second["ready"])
        self.assertTrue(any(item["code"] == "participant_global_conflict" for item in second["issues"]))
        self.assertFalse((self.root/"flat_split_person/split").exists())

    def test_flat_output_rejects_every_input_root_descendant_and_symlink_alias(self):
        flat, train_ids, test_ids = self.make_flat()
        for destination in (flat, flat/"new", flat/"arbitrary/deep/output", flat/"100/artifacts"):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "outside the entire input"):
                build_manifest(flat, destination, layout="flat", original_train_ids=train_ids,
                               original_test_ids=test_ids, **self.kwargs)
        external = self.root/"external"
        external.mkdir()
        (flat/"external_link").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside the entire input"):
            build_manifest(flat, flat/"external_link/new_output", layout="flat", original_train_ids=train_ids,
                           original_test_ids=test_ids, **self.kwargs)
        self.assertFalse((external/"new_output").exists())

    def test_flat_broken_link_is_not_silently_ignored_as_an_ordinary_file(self):
        flat, train_ids, test_ids = self.make_flat()
        (flat/"999").symlink_to(self.root/"missing-case", target_is_directory=True)
        record = self.run_flat(flat, train_ids, test_ids)
        self.assertFalse(record["ready"])
        self.assertFalse(record["original_membership_verified"])
        self.assertTrue(any(item["code"] == "unclassified_flat_entry" for item in record["issues"]))
        self.assertEqual(record["unassigned_case_ids"], ["999"])
        self.assertEqual(record["ignored_entries"], [])

    def test_flat_custom_pattern_matches_participant_in_case_and_id_list(self):
        flat, train_ids, test_ids = self.make_flat()
        for identifier in ("100", "200", "300"):
            (flat/identifier).rename(flat/f"sub-{identifier}_ses-2")
        train_ids.write_text("sub-200_ses-1\nsub-100_ses-1\n")
        test_ids.write_text("sub-300_ses-1\n")
        record = self.run_flat(flat, train_ids, test_ids, subject_pattern=r"^sub-(?P<participant>[0-9]+)_ses-[0-9]+$")
        self.assertTrue(record["ready"])
        self.assertEqual(record["test_participant_ids"], ["300"])
        test_row = next(row for row in self.read_csv("flat_out/original_train_test.csv") if row["split"] == "test")
        self.assertEqual(test_row["case_id"], "sub-300_ses-2")

    def test_flat_ambiguous_and_missing_files_still_block_and_cli_selects_layout(self):
        flat, train_ids, test_ids = self.make_flat()
        duplicate = flat/"100/QSM"
        duplicate.mkdir()
        (duplicate/"QSM.nii.gz").write_bytes(b"second candidate, not real image")
        first = self.run_flat(flat, train_ids, test_ids)
        self.assertFalse(first["ready"])
        self.assertTrue(any(item["code"] == "ambiguous_role_files" for item in first["issues"]))
        (duplicate/"QSM.nii.gz").unlink()
        (flat/"300/QSM/QSM.nii.gz").unlink()
        second = self.run_flat(flat, train_ids, test_ids, name="flat_missing")
        self.assertFalse(second["ready"])
        self.assertEqual(second["test_participant_ids"], ["300"])
        argv = ["build_manifest", "--data-root", str(flat), "--output", str(self.root/"flat_cli"),
                "--layout", "flat", "--original-train-ids", str(train_ids), "--original-test-ids", str(test_ids),
                "--expected-train", "2", "--expected-test", "1", "--qsm-units", "ppb",
                "--input-mask-source", "synthetic input-only source", "--inventory-only"]
        with patch("sys.argv", argv), patch("sys.stdout", io.StringIO()):
            build_main()
        self.assertEqual(json.loads((self.root/"flat_cli/inventory.json").read_text())["layout"], "flat")


if __name__ == "__main__":
    unittest.main()

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ..split_training import create_split
from ..util.quality import QC_PATHS


class SplitTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root/"original"
        self.source.mkdir()
        self.manifest = self.source/"manifest.csv"
        self.rows = [self.row(f"train-{i}", f"global-{i}", "train", i) for i in range(10)]
        self.rows += [self.row("protected-test", "test-global", "test", 100)]

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def row(sid, gid, split, visit):
        return {"subject_id": sid, "global_subject_id": gid, "split": split,
                "cohort": "synthetic", "case_id": f"visit-{visit}", "qsm_units": "ppb",
                **{key: f"absent-{visit}/{key}.nii.gz" for key in
                   ("swi_path", "qsm_path", "swi_mask_path", "mask_path")}}

    def write_manifest(self, rows=None):
        rows = self.rows if rows is None else rows
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)

    def read_output(self, name="split"):
        with (self.root/name/"manifest.csv").open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_only_original_train_is_split_test_rows_unchanged_and_no_image_reads(self):
        self.write_manifest()
        original_bytes = self.manifest.read_bytes()
        with patch("nibabel.load", side_effect=AssertionError("Images must never be opened")), \
                patch.object(Path, "is_file", side_effect=AssertionError("No input existence checks")):
            record = create_split(self.manifest, self.root/"split", val_fraction=.21, seed=42)
        result = self.read_output()
        self.assertEqual(record["group_counts"], {"train": 7, "val": 3, "test": 1})
        self.assertEqual([row["subject_id"] for row in result], [row["subject_id"] for row in self.rows])
        test = [row for row in result if row["split"] == "test"]
        self.assertEqual(len(test), 1)
        expected = dict(self.rows[-1])
        for key in ("swi_path", "qsm_path", "swi_mask_path", "mask_path"):
            expected[key] = str((self.source/expected[key]).resolve())
        self.assertEqual(test[0], expected)
        self.assertEqual(record["test_resolved_rows_sha256_before"], record["test_resolved_rows_sha256_after"])
        self.assertEqual(record["source_manifest_sha256"], hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual(record["output_manifest_sha256"], hashlib.sha256((self.root/"split/manifest.csv").read_bytes()).hexdigest())
        self.assertEqual(record["image_header_reads"], 0)
        self.assertEqual(record["image_array_reads"], 0)
        self.assertEqual(self.manifest.read_bytes(), original_bytes)

    def test_visits_and_cross_cohort_aliases_group_on_global_subject_id(self):
        extra = self.row("alias-across-cohort", "global-0", "train", 101)
        extra["cohort"] = "other-cohort"
        self.rows += [extra, self.row("train-0", "global-0", "train", 102)]
        self.write_manifest()
        record = create_split(self.manifest, self.root/"split", val_fraction=.5)
        matched = [row for row in self.read_output() if row["global_subject_id"] == "global-0"]
        self.assertEqual(len(matched), 3)
        self.assertEqual(len({row["split"] for row in matched}), 1)
        self.assertEqual(record["subject_id_aliases_by_global_subject_id"]["global-0"],
                         ["alias-across-cohort", "train-0"])
        self.assertEqual(sum(record["group_counts"].values()), 11)

    def test_seed_and_row_order_independent_membership(self):
        self.write_manifest()
        first = create_split(self.manifest, self.root/"split", val_fraction=.4, seed=12)
        self.write_manifest(list(reversed(self.rows)))
        second = create_split(self.manifest, self.root/"reordered", val_fraction=.4, seed=12)
        third = create_split(self.manifest, self.root/"different_seed", val_fraction=.4, seed=13)
        self.assertEqual(first["group_ids"], second["group_ids"])
        self.assertNotEqual(first["group_ids"]["val"], third["group_ids"]["val"])
        ids = sorted(row["global_subject_id"] for row in self.rows if row["split"] == "train")
        expected = sorted(np.random.default_rng(12).permutation(np.asarray(ids, dtype=object))[:4].tolist())
        self.assertEqual(first["group_ids"]["val"], expected)

    def test_all_optional_qc_reference_and_evidence_paths_are_rebased_without_open(self):
        for row in self.rows:
            row.update({key: f"metadata/{key}.json" for key in QC_PATHS})
            row["custom_review_path"] = "other/review.json"
            row["evidence_paths"] = json.dumps(["qc/one.png", str(self.root/"outside.png")])
            row["notes"] = "validate is a folder name, not a split declaration"
        self.rows[-1]["swi_path"] = "validate/kept_test_swi.nii.gz"
        self.rows[0]["qsm_path"] = "~/absent-home-image.nii.gz"
        self.write_manifest()
        with patch("nibabel.load", side_effect=AssertionError("No header/array reads")):
            record = create_split(self.manifest, self.root/"split")
        result = self.read_output()
        for row in result:
            for key in (*QC_PATHS, "custom_review_path"):
                self.assertTrue(Path(row[key]).is_absolute())
                self.assertEqual(row[key], str((self.source/self.rows[0][key]).resolve()))
            self.assertEqual(json.loads(row["evidence_paths"]),
                             [str(self.source/"qc/one.png"), str(self.root/"outside.png")])
            self.assertEqual(row["notes"], self.rows[0]["notes"])
        self.assertEqual(result[-1]["split"], "test")
        self.assertEqual(result[0]["qsm_path"], str(Path("~/absent-home-image.nii.gz").expanduser().resolve()))
        self.assertTrue(set(QC_PATHS).issubset(record["absolute_path_columns"]))

    def test_cross_split_global_identity_or_local_subject_and_inconsistent_ids_rejected(self):
        for key, value in (("global_subject_id", "global-0"), ("subject_id", "train-0")):
            rows = copy.deepcopy(self.rows)
            rows[-1][key] = value
            self.write_manifest(rows)
            with self.assertRaisesRegex(ValueError, "crosses|multiple"):
                create_split(self.manifest, self.root/"split")
            self.assertFalse((self.root/"split").exists())
        rows = copy.deepcopy(self.rows)
        rows[1]["subject_id"] = rows[0]["subject_id"]
        self.write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "multiple global_subject_id"):
            create_split(self.manifest, self.root/"split")

    def test_existing_val_missing_global_empty_paths_and_bad_evidence_list_rejected(self):
        changes = [("split", "val"), ("global_subject_id", ""),
                   ("subject_id", "  "), ("mask_path", ""), ("evidence_paths", "not-json")]
        for key, value in changes:
            with self.subTest(key=key):
                rows = copy.deepcopy(self.rows)
                rows[0][key] = value
                self.write_manifest(rows)
                with self.assertRaises(ValueError):
                    create_split(self.manifest, self.root/"split")
                self.assertFalse((self.root/"split").exists())

    def test_invalid_fractions_and_seeds_rejected(self):
        self.write_manifest()
        for fraction in (0, 1, -.1, float("nan"), float("inf"), True, "bad"):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                create_split(self.manifest, self.root/"split", val_fraction=fraction)
        for seed in (-1, 2**32, .3, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                create_split(self.manifest, self.root/"split", seed=seed)

    def test_small_pool_and_rounding_clamp(self):
        self.write_manifest([self.rows[0], self.rows[-1]])
        with self.assertRaisesRegex(ValueError, "at least two"):
            create_split(self.manifest, self.root/"split")
        self.write_manifest([self.rows[0], self.rows[1], self.rows[-1]])
        record = create_split(self.manifest, self.root/"split", val_fraction=.99)
        self.assertEqual(record["n_val_before_clamp"], 2)
        self.assertEqual(record["group_counts"], {"train": 1, "val": 1, "test": 1})

    def test_existing_output_even_empty_directory_not_overwritten(self):
        self.write_manifest()
        output = self.root/"split"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            create_split(self.manifest, output)
        self.assertEqual(list(output.iterdir()), [])

    def test_duplicate_swi_and_malformed_csv_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["swi_path"] = rows[0]["swi_path"]
        self.write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "Duplicate SWI"):
            create_split(self.manifest, self.root/"split")
        self.manifest.write_text("subject_id,subject_id\np1,p2\n")
        with self.assertRaisesRegex(ValueError, "unique"):
            create_split(self.manifest, self.root/"split")
        self.write_manifest()
        with self.manifest.open("a") as stream:
            stream.write("incomplete,row\n")
        with self.assertRaisesRegex(ValueError, "mismatched"):
            create_split(self.manifest, self.root/"split")


if __name__ == "__main__":
    unittest.main()

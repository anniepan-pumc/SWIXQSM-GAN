"""Tiny synthetic range probes; corrupt/missing holdouts must never be opened."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

from .. import fit_metric_range as fitting
from ..fit_normalization import training_identity


class MetricRangeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swix-range-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def case(self, case_id, values, *, units="ppm", mask=None):
        values = np.asarray(values, dtype=np.float32).reshape(-1, 1, 1)
        row = dict(subject_id=case_id, case_id=case_id, split="train", qsm_units=units)
        for role in ("swi_path", "qsm_path", "swi_mask_path", "mask_path"):
            path = self.root / f"{case_id}_{role}.nii.gz"
            data = (values if role == "qsm_path" else
                    np.asarray(mask, dtype=np.float32).reshape(values.shape) if role == "mask_path" and mask is not None
                    else np.ones_like(values))
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            row[role] = str(path)
        return row

    def test_cli_reads_train_only_and_records_identity_hashes(self):
        row = self.case("train", [-2, 3, 1e6], mask=[1, 1, 0])
        rows = [row]
        forbidden = set()
        for split in ("val", "test"):
            holdout = dict(subject_id=split, case_id=split, split=split, qsm_units="invalid-holdout-unit")
            for role in ("swi_path", "qsm_path", "swi_mask_path", "mask_path"):
                path = self.root / f"{split}_{role}.nii.gz"
                if split == "val":
                    path.write_bytes(b"corrupt holdout, must not inspect header")
                holdout[role] = str(path); forbidden.add(str(path))
            rows.append(holdout)
        manifest = self.root / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
        output = self.root / "range.json"
        original_load, original_snapshot = nib.load, fitting._file_snapshot
        touched = []

        def checked_load(path, *args, **kwargs):
            self.assertNotIn(str(path), forbidden)
            self.assertIn(str(path), {row["qsm_path"], row["mask_path"]})
            touched.append(str(path))
            return original_load(path, *args, **kwargs)

        def checked_snapshot(path):
            self.assertNotIn(str(path), forbidden)
            return original_snapshot(path)

        with mock.patch("nibabel.load", side_effect=checked_load), \
                mock.patch.object(fitting, "_file_snapshot", side_effect=checked_snapshot):
            fitting.main(["--manifest", str(manifest), "--output", str(output), "--units", "ppb",
                          "--allow-unverified-data", "Synthetic deferred QC", "--allow-unverified-independence", "Synthetic IDs"])
        result = json.loads(output.read_text())
        self.assertEqual((result["train_min"], result["train_max"], result["data_range"]), (-2000., 3000., 5000.))
        self.assertEqual(result["train_identity_sha256"], training_identity([row]))
        self.assertEqual(result["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertEqual(result["data_qc"]["status"], "unverified")
        self.assertEqual(set(touched), {row["qsm_path"], row["mask_path"]})
        self.assertEqual(set(result["inputs"][0]["files"]), {"qsm_path", "mask_path"})

    def test_mixed_units_use_one_global_range(self):
        rows = [self.case("a", [-2, 1]), self.case("b", [-1000, 5000], units="ppb")]
        result = fitting.fit_metric_range(rows, units="ppm")
        self.assertEqual((result["train_min"], result["train_max"], result["data_range"]), (-2., 5., 7.))

    def test_rejects_constant_empty_nonfinite_and_no_train(self):
        for name, values, mask in [("constant", [2, 2], None), ("empty", [-2, 3], [0, 0]),
                                   ("nonfinite", [float("nan"), 3], None)]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                fitting.fit_metric_range([self.case(name, values, mask=mask)], units="ppm")
        with self.assertRaises(ValueError):
            fitting.fit_metric_range([{"split": "test"}], units="ppm")

    def test_save_never_overwrites_files_or_links(self):
        result = fitting.fit_metric_range([self.case("a", [-2, 3])], units="ppm")
        output = self.root / "range.json"; fitting.save_metric_range(result, output)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            fitting.save_metric_range(result, output)
        self.assertEqual(output.read_bytes(), before)
        link = self.root / "linked.json"; link.symlink_to(output)
        with self.assertRaises(FileExistsError):
            fitting.save_metric_range(result, link)

    def test_stored_is_explicitly_unverified_and_cannot_convert_to_ppb(self):
        row = self.case("stored", [-2, 3], units="stored")
        result = fitting.fit_metric_range([row], units="stored")
        self.assertEqual((result["data_range"], result["units"], result["unit_status"]), (5., "stored", "unverified"))
        with self.assertRaises(ValueError):
            fitting.fit_metric_range([row], units="ppb")


if __name__ == "__main__":
    unittest.main()

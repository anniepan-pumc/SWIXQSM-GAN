import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from ..benchmark_io import main as benchmark_main


class IOBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.rows, candidates = [], []
        for index in range(3):
            directory = self.root/f"person-{index}"
            directory.mkdir()
            shape = (32, 32, 10)
            swi = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)+index*30
            qsm = swi/100-3
            mask = np.ones(shape, dtype=np.float32)
            mask[:2] = 0
            row = {"subject_id": f"person-{index}", "split": "train", "qsm_units": "ppb"}
            for key, value in (("swi_path", swi), ("qsm_path", qsm),
                               ("swi_mask_path", mask), ("mask_path", mask)):
                path = directory/f"{key}.nii.gz"
                image = nib.Nifti1Image(value, np.eye(4))
                image.header.set_xyzt_units("mm")
                nib.save(image, path)
                row[key] = str(path)
            self.rows.append(row)
            candidates.append({"SWI.nii.gz": row["swi_path"], "QSM.nii.gz": row["qsm_path"],
                               "QSM_mask.nii.gz": row["mask_path"]})
        self.manifest = self.root/"manifest.csv"
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.raw = self.root/"raw_candidates.json"
        self.raw.write_text(json.dumps({"candidates": candidates}))

    def tearDown(self):
        self.temporary.cleanup()

    def run_cli(self, name, options):
        output = self.root/f"{name}_benchmark.json"
        argv = ["benchmark", *options, "--output", str(output)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            benchmark_main()
        return output, json.loads(output.read_text())

    def assert_comparison(self, report):
        self.assertEqual(report["schema"], "swixqsm.io_benchmark")
        self.assertEqual(report["data_qc"]["status"], "not_assessed")
        self.assertTrue(report["comparison"]["all_selected_indices_equal"])
        self.assertTrue(report["comparison"]["all_per_index_values_equal"])
        first, second = report["results"]
        self.assertEqual(first["ordered_per_index_hashes"], second["ordered_per_index_hashes"])
        self.assertEqual(first["ordered_value_manifest_sha256"], second["ordered_value_manifest_sha256"])
        self.assertEqual(first["traversal"]["n_samples"], 24)
        self.assertEqual(second["traversal"]["n_samples"], 24)
        self.assertGreater(report["comparison"]["traversal_nifti_decodes"]["fraction_reduction"], 0)
        self.assertGreater(report["comparison"]["traversal_volume_loads"]["fraction_reduction"], 0)
        self.assertGreaterEqual(first["startup"]["wall_seconds"], 0)
        self.assertGreaterEqual(second["traversal"]["wall_seconds"], 0)
        self.assertEqual(second["traversal"]["io_stats"]["cache"]["max_items"], 2)
        self.assertEqual(second["traversal"]["io_stats"]["cache"]["max_bytes"], 256*1024*1024)
        json.dumps(report, allow_nan=False)

    def test_manifest_and_raw_modes_preserve_values_and_do_not_modify_sources_or_rng(self):
        initial = {row[key]: hashlib.sha256(Path(row[key]).read_bytes()).hexdigest()
                   for row in self.rows for key in ("swi_path", "qsm_path", "swi_mask_path", "mask_path")}
        rng = torch.get_rng_state().clone()
        _, manifest = self.run_cli("manifest", ["--manifest", str(self.manifest), "--size", "32",
                                                "--qsm-scale", "10", "--qsm-offset", "0", "--units", "ppb"])
        _, raw = self.run_cli("raw", ["--raw-probe-json", str(self.raw)])
        self.assert_comparison(manifest)
        self.assert_comparison(raw)
        self.assertEqual(manifest["mode"], "prepared_manifest")
        self.assertIsNotNone(manifest["configuration"]["normalization"])
        self.assertEqual(raw["mode"], "raw_three_file_probe")
        self.assertIsNone(raw["configuration"]["normalization"])
        self.assertIsNone(raw["configuration"]["split"])
        self.assertIsNone(raw["configuration"]["size"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for path, digest in initial.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)

    def test_group_tail_partials_keep_exact_coverage(self):
        _, report = self.run_cli("partial", ["--raw-probe-json", str(self.raw), "--batch-size", "5"])
        self.assert_comparison(report)
        self.assertEqual(report["results"][0]["traversal"]["n_batches"], 5)
        self.assertEqual(report["results"][1]["traversal"]["n_batches"], 6)
        self.assertEqual(report["results"][1]["traversal"]["batch_sizes"], [5, 5, 5, 1, 5, 3])

    def test_raw_mode_is_capped_at_three_volumes(self):
        data = json.loads(self.raw.read_text())
        data["candidates"].append({role: "/not/read/fourth-volume.nii.gz"
                                   for role in ("SWI.nii.gz", "QSM.nii.gz", "QSM_mask.nii.gz")})
        self.raw.write_text(json.dumps(data))
        _, report = self.run_cli("capped", ["--raw-probe-json", str(self.raw)])
        self.assertEqual(report["source"]["volumes_used"], 3)
        self.assertEqual(report["source"]["candidates_or_split_volumes_available"], 4)
        self.assert_comparison(report)

    def test_explicit_volume_limit_bounds_image_access_in_both_modes(self):
        for mode, options in (("manifest", ["--manifest", str(self.manifest), "--size", "32",
                                            "--qsm-scale", "10", "--qsm-offset", "0", "--units", "ppb"]),
                               ("raw", ["--raw-probe-json", str(self.raw)])):
            with self.subTest(mode=mode):
                _, report = self.run_cli(mode+"_limited", [*options, "--max-volumes", "1"])
                self.assertEqual(report["source"]["volumes_used"], 1)
                self.assertEqual(report["source"]["candidates_or_split_volumes_available"], 3)
                self.assertEqual(report["configuration"]["max_volumes"], 1)
                self.assertTrue(report["comparison"]["all_per_index_values_equal"])
                for policy in report["results"]:
                    self.assertEqual(policy["traversal"]["n_samples"], 8)
                    self.assertEqual(policy["traversal"]["io_stats"]["cache"]["loads"], 1)

    def test_output_refuses_overwrite_and_invalid_cli_arguments(self):
        output = self.root/"existing.json"
        output.write_text("preserve existing result")
        invalid = [(["--raw-probe-json", str(self.raw)], output),
                   (["--raw-probe-json", str(self.raw), "--units", "ppm"], self.root/"invalid.json"),
                   (["--manifest", str(self.manifest)], self.root/"invalid.json"),
                   (["--raw-probe-json", str(self.raw), "--group-volumes", "3"], self.root/"invalid.json"),
                   (["--raw-probe-json", str(self.raw), "--cache-mib", "inf"], self.root/"invalid.json"),
                   (["--raw-probe-json", str(self.raw), "--cache-mib", "1e308"], self.root/"invalid.json"),
                   (["--raw-probe-json", str(self.raw), "--cache-mib", "1e-20"], self.root/"invalid.json"),
                   (["--raw-probe-json", str(self.raw), "--slices-per-volume", "0"], self.root/"invalid.json"),
                   (["--raw-probe-json", str(self.raw), "--max-volumes", "0"], self.root/"invalid.json"),
                   (["--raw-probe-json", str(self.raw), "--seed", "-1"], self.root/"invalid.json")]
        for options, path in invalid:
            argv = ["benchmark", *options, "--output", str(path)]
            with self.subTest(options=options), patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    benchmark_main()
                self.assertEqual(stopped.exception.code, 2)
        self.assertEqual(output.read_text(), "preserve existing result")
        self.assertFalse((self.root/"invalid.json").exists())


if __name__ == "__main__":
    unittest.main()

"""Synthetic-only checks for lazy indexing and instance prepared-volume caches."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from ..util.data import PairedSlices, load_volume, normalize_swi, pad_slice, require_grid
from ..util.normalization import QSMNormalization


class DataIOTest(unittest.TestCase):
    IO_KEYS = ("index_volume_decodes", "index_header_reads", "prepared_volume_decodes",
               "volume_preparations", "volume_decodes")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.shape = (11, 13, 4)
        self.affine = np.array([[.8, 0, 0, 12], [0, .9, 0, -7], [0, 0, 1.5, 3], [0, 0, 0, 1.]])
        self.normalization = QSMNormalization(.08, .025, "ppm")
        self.size = 32

    def write_image(self, path, array, *, affine=None, units="mm", slope=None, intercept=None):
        image = nib.Nifti1Image(array, self.affine if affine is None else affine)
        image.header.set_xyzt_units(units)
        if slope is not None:
            image.header.set_slope_inter(slope, intercept)
        nib.save(image, path)

    def case(self, sid, *, split="train", qsm_units="ppm", shape=None):
        shape = self.shape if shape is None else shape
        values = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
        brain = np.ones(shape, dtype=np.uint8)
        brain[0], brain[-1] = 0, 0
        brain[:, 0], brain[:, -1] = 0, 0
        valid = brain.copy()
        valid[:, :, 1::2] = 0  # Only z=0 and z=2 have reliable reference support.
        valid[1:3, 1:4, 0] = 0
        row = {"subject_id": sid, "split": split, "qsm_units": qsm_units}
        for key in ("swi_path", "qsm_path", "swi_mask_path", "mask_path"):
            row[key] = str(self.root / f"{sid}_{key}.nii.gz")
        self.write_image(row["swi_path"], values - 100, slope=1.5, intercept=37.)
        if qsm_units == "ppm":
            self.write_image(row["qsm_path"], values - 200, slope=.001, intercept=.025)
        else:
            self.write_image(row["qsm_path"], values - 200, slope=1., intercept=25.)
        self.write_image(row["swi_mask_path"], brain)
        self.write_image(row["mask_path"], valid)
        return row

    def literal_preload(self, row):
        """Original eager preprocessing math, independently of Dataset/cache."""
        ref, swi = load_volume(row["swi_path"])
        arrays = []
        for key in ("qsm_path", "swi_mask_path", "mask_path"):
            image, array = load_volume(row[key])
            require_grid(ref, image)
            arrays.append(array)
        qsm, brain, mask = arrays
        brain, mask = brain > 0, mask > 0
        target = self.normalization.normalize(qsm, row.get("qsm_units", self.normalization.units))
        return normalize_swi(swi, brain), target.astype(np.float32), mask.astype(np.float32)

    @staticmethod
    def spy(calls):
        def load(path):
            calls.append(str(Path(path).resolve()))
            return load_volume(path)
        return load

    def assert_io_consistent(self, dataset):
        statistics = dataset.io_stats
        for key in self.IO_KEYS:
            self.assertIn(key, statistics)
            self.assertIsInstance(statistics[key], int)
            self.assertGreaterEqual(statistics[key], 0)
        self.assertEqual(statistics["volume_decodes"],
                         statistics["index_volume_decodes"] + statistics["prepared_volume_decodes"])
        self.assertEqual(statistics["cache"], dataset.cache_stats)

    def test_bitwise_original_math_mixed_units_offset_scaling_and_sparse_padding(self):
        rows = [self.case("ppm", qsm_units="ppm"), self.case("ppb", qsm_units="ppb")]
        expected = [self.literal_preload(row) for row in rows]
        dataset = PairedSlices(rows, "train", self.size, self.normalization,
                               cache_items=2, cache_bytes=1024 * 1024)
        self.assertEqual(dataset.index, [(0, 0), (0, 2), (1, 0), (1, 2)])
        self.assertEqual(len(dataset), 4)
        for item, (volume_index, z) in enumerate(dataset.index):
            actual = dataset[item]
            wanted = tuple(pad_slice(array[:, :, z], self.size) for array in expected[volume_index])
            self.assertEqual(actual[3], rows[volume_index]["subject_id"])
            for tensor, reference in zip(actual[:3], wanted):
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(tuple(tensor.shape), (1, self.size, self.size))
                self.assertTrue(torch.equal(tensor, reference), "Lazy I/O/cache changed literal preprocessing bits")
                self.assertEqual(int(torch.count_nonzero(tensor[:, self.shape[1]:, :])), 0)
                self.assertEqual(int(torch.count_nonzero(tensor[:, :, self.shape[0]:])), 0)
        loaded_swi = load_volume(rows[0]["swi_path"])[1]
        raw = np.arange(np.prod(self.shape), dtype=np.int16).reshape(self.shape) - 100
        np.testing.assert_array_equal(loaded_swi, raw.astype(np.float32) * 1.5 + 37.)
        self.assertTrue(any(float(array.min()) < 0 for _, array, _ in expected))
        self.assert_io_consistent(dataset)

    def test_constructor_decodes_only_validity_then_first_volume_once(self):
        row = self.case("lazy")
        calls = []
        with patch("swixqsm_minimal.util.data.load_volume", side_effect=self.spy(calls)):
            dataset = PairedSlices([row], "train", self.size, self.normalization,
                                   cache_items=1, cache_bytes=1024 * 1024)
            self.assertEqual(calls, [row["mask_path"]])
            initial = dataset.io_stats
            self.assertEqual(initial["index_header_reads"], 4)
            self.assertEqual(initial["index_volume_decodes"], 1)
            self.assertEqual(initial["prepared_volume_decodes"], 0)
            self.assertEqual(initial["volume_preparations"], 0)
            self.assertEqual(initial["volume_decodes"], 1)
            self.assertEqual(dataset.cache_stats["current_items"], 0)
            calls.clear()
            first = dataset[0]
            self.assertEqual(calls, [row[key] for key in ("swi_path", "qsm_path", "swi_mask_path", "mask_path")])
            dataset[1]
            repeated = dataset[0]
            self.assertEqual(len(calls), 4)
            for actual, original in zip(repeated[:3], first[:3]):
                self.assertTrue(torch.equal(actual, original))
        self.assertEqual(dataset.io_stats["prepared_volume_decodes"], 4)
        self.assertEqual(dataset.io_stats["volume_preparations"], 1)
        self.assertEqual(dataset.io_stats["volume_decodes"], 5)
        self.assertEqual(dataset.cache_stats["misses"], 1)
        self.assertEqual(dataset.cache_stats["hits"], 2)
        self.assert_io_consistent(dataset)

    def test_train_and_val_instance_capacity_isolation(self):
        train_row = self.case("train")
        val_rows = [self.case("val_a", split="val"), self.case("val_b", split="val")]
        rows = [train_row, *val_rows]
        calls = []
        with patch("swixqsm_minimal.util.data.load_volume", side_effect=self.spy(calls)):
            train = PairedSlices(rows, "train", self.size, self.normalization, cache_items=1, cache_bytes=1024 * 1024)
            val = PairedSlices(rows, "val", self.size, self.normalization, cache_items=1, cache_bytes=1024 * 1024)
            self.assertEqual(calls, [train_row["mask_path"], *[row["mask_path"] for row in val_rows]])
            train[0]
            val[0]
            val[2]  # Its own first val volume is evicted; train stays resident.
            count = len(calls)
            train[1]
            self.assertEqual(len(calls), count)
        self.assertEqual(train.cache_stats["loads"], 1)
        self.assertEqual(train.cache_stats["evictions"], 0)
        self.assertEqual(val.cache_stats["loads"], 2)
        self.assertEqual(val.cache_stats["evictions"], 1)
        self.assertEqual(train.cache_stats["current_items"], 1)
        self.assertEqual(val.cache_stats["current_items"], 1)
        self.assert_io_consistent(train)
        self.assert_io_consistent(val)

    def test_clear_cache_stats_reset_and_fresh_decode_without_reindexing(self):
        row = self.case("clear")
        dataset = PairedSlices([row], "train", self.size, self.normalization,
                               cache_items=1, cache_bytes=1024 * 1024)
        dataset[0]
        dataset[1]
        before = dataset.io_stats
        dataset.clear_cache()
        for key in self.IO_KEYS:
            self.assertEqual(dataset.io_stats[key], before[key])
        self.assertEqual(dataset.cache_stats["current_items"], 0)
        self.assertEqual(dataset.cache_stats["loads"], 1)
        dataset[0]
        self.assertEqual(dataset.io_stats["prepared_volume_decodes"], 8)
        self.assertEqual(dataset.io_stats["volume_preparations"], 2)
        dataset.clear_cache(reset_stats=True)
        for key in self.IO_KEYS:
            self.assertEqual(dataset.io_stats[key], 0)
        for key in ("hits", "misses", "loads", "evictions", "current_items", "current_bytes", "peak_bytes"):
            self.assertEqual(dataset.cache_stats[key], 0)
        calls = []
        with patch("swixqsm_minimal.util.data.load_volume", side_effect=self.spy(calls)):
            dataset[0]
        self.assertEqual(len(calls), 4)
        self.assertEqual(dataset.io_stats["index_header_reads"], 0)
        self.assertEqual(dataset.io_stats["index_volume_decodes"], 0)
        self.assertEqual(dataset.io_stats["prepared_volume_decodes"], 4)
        self.assertEqual(dataset.io_stats["volume_preparations"], 1)
        snapshot = dataset.io_stats
        snapshot["volume_decodes"] = 999
        snapshot["cache"]["hits"] = 999
        self.assertEqual(dataset.io_stats["volume_decodes"], 4)
        self.assertEqual(dataset.cache_stats["hits"], 0)
        self.assert_io_consistent(dataset)

    def test_cached_owner_arrays_readonly_but_padded_tensors_do_not_alias(self):
        row = self.case("immutable")
        dataset = PairedSlices([row], "train", self.size, self.normalization,
                               cache_items=1, cache_bytes=1024 * 1024)
        prepared = dataset.volume(0)
        original = tuple(array.copy() for array in prepared)
        for array in prepared:
            self.assertEqual(array.dtype, np.dtype("float32"))
            self.assertFalse(array.flags.writeable)
        tensors = dataset[0]
        expected = tuple(tensor.clone() for tensor in tensors[:3])
        for tensor in tensors[:3]:
            tensor.fill_(123.)
        repeated = dataset[0]
        for actual, wanted in zip(repeated[:3], expected):
            self.assertTrue(torch.equal(actual, wanted))
        for array, wanted in zip(prepared, original):
            np.testing.assert_array_equal(array, wanted)

    def test_reference_subset_content_error_is_deferred_to_first_access(self):
        row = self.case("subset")
        valid = load_volume(row["mask_path"])[1]
        valid[0, 0, 0] = 1  # Outside the input brain mask; headers alone cannot see it.
        self.write_image(row["mask_path"], valid)
        dataset = PairedSlices([row], "train", self.size, self.normalization,
                               cache_items=1, cache_bytes=1024 * 1024)
        self.assertEqual(dataset.io_stats["prepared_volume_decodes"], 0)
        with self.assertRaises(ValueError):
            dataset[0]
        self.assertEqual(dataset.io_stats["prepared_volume_decodes"], 4)
        self.assertEqual(dataset.io_stats["volume_preparations"], 0)
        self.assertEqual(dataset.cache_stats["current_items"], 0)
        self.assertEqual(dataset.cache_stats["load_failures"], 1)
        self.assert_io_consistent(dataset)

    def test_swi_and_qsm_nonfinite_values_not_decoded_until_first_access(self):
        for role, expected_attempts in (("swi_path", 1), ("qsm_path", 2)):
            row = self.case(f"nonfinite_{role}")
            values = load_volume(row[role])[1]
            values[1, 1, 0] = np.nan
            self.write_image(row[role], values)
            calls = []
            with patch("swixqsm_minimal.util.data.load_volume", side_effect=self.spy(calls)):
                dataset = PairedSlices([row], "train", self.size, self.normalization,
                                       cache_items=1, cache_bytes=1024 * 1024)
                self.assertEqual(calls, [row["mask_path"]])
                with self.assertRaisesRegex(ValueError, "finite"):
                    dataset[0]
            self.assertEqual(dataset.io_stats["prepared_volume_decodes"], expected_attempts)
            self.assertEqual(dataset.io_stats["volume_preparations"], 0)
            self.assertEqual(dataset.cache_stats["current_items"], 0)
            self.assert_io_consistent(dataset)

    def test_header_geometry_dimensions_size_and_qsm_units_fail_in_constructor(self):
        row = self.case("affine")
        image, values = load_volume(row["qsm_path"])
        shifted = image.affine.copy()
        shifted[0, 3] += 1
        self.write_image(row["qsm_path"], values, affine=shifted)
        with self.assertRaises(ValueError):
            PairedSlices([row], "train", self.size, self.normalization)
        row = self.case("four_d")
        self.write_image(row["qsm_path"], np.ones(self.shape + (1,), dtype=np.float32))
        with self.assertRaises(ValueError):
            PairedSlices([row], "train", self.size, self.normalization)
        row = self.case("too_large", shape=(33, 13, 4))
        with self.assertRaises(ValueError):
            PairedSlices([row], "train", self.size, self.normalization)
        row = self.case("units")
        row["qsm_units"] = "unknown"
        calls = []
        with patch("swixqsm_minimal.util.data.load_volume", side_effect=self.spy(calls)):
            with self.assertRaisesRegex(ValueError, "unit"):
                PairedSlices([row], "train", self.size, self.normalization)
        self.assertNotIn(row["swi_path"], calls)
        self.assertNotIn(row["qsm_path"], calls)

    def test_small_byte_budget_bypasses_without_changing_preprocessing(self):
        row = self.case("bypass", qsm_units="ppb")
        expected = self.literal_preload(row)
        calls = []
        with patch("swixqsm_minimal.util.data.load_volume", side_effect=self.spy(calls)):
            dataset = PairedSlices([row], "train", self.size, self.normalization,
                                   cache_items=1, cache_bytes=1)
            calls.clear()
            for item in (0, 1, 0):
                _, z = dataset.index[item]
                actual = dataset[item]
                for tensor, array in zip(actual[:3], expected):
                    self.assertTrue(torch.equal(tensor, pad_slice(array[:, :, z], self.size)))
        self.assertEqual(len(calls), 12)  # Four actual file decodes on each miss.
        self.assertEqual(dataset.io_stats["prepared_volume_decodes"], 12)
        self.assertEqual(dataset.io_stats["volume_preparations"], 3)
        self.assertEqual(dataset.cache_stats["loads"], 3)
        self.assertEqual(dataset.cache_stats["hits"], 0)
        self.assertEqual(dataset.cache_stats["oversized"], 3)
        self.assertEqual(dataset.cache_stats["bypasses"], 3)
        self.assertEqual(dataset.cache_stats["evictions"], 0)
        self.assertEqual(dataset.cache_stats["current_items"], 0)
        self.assertEqual(dataset.cache_stats["current_bytes"], 0)
        self.assert_io_consistent(dataset)


if __name__ == "__main__":
    unittest.main()

import copy
import gc
import json
import pickle
import unittest
import weakref
from collections import Counter

import numpy as np

from ..util.volume_cache import InstanceVolumeCache


class VolumeCacheTest(unittest.TestCase):
    def test_lru_hits_and_read_counts(self):
        cache = InstanceVolumeCache(max_items=2, max_bytes=128)
        calls = Counter()

        def load(key):
            calls[key] += 1
            return (np.arange(4, dtype=np.float32),)

        first = cache.get("a", lambda: load("a"))
        cache.get("b", lambda: load("b"))
        self.assertIs(cache.get("a", lambda: load("a")), first)
        cache.get("c", lambda: load("c"))  # b is oldest.
        self.assertIs(cache.get("a", lambda: load("a")), first)
        cache.get("b", lambda: load("b"))
        self.assertEqual(calls, {"a": 1, "b": 2, "c": 1})
        stats = cache.stats
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 4)
        self.assertEqual(stats["loads"], 4)
        self.assertEqual(stats["evictions"], 2)
        self.assertEqual(stats["current_items"], 2)
        self.assertEqual(stats["current_bytes"], 32)

    def test_byte_capacity_counts_every_array_and_evicts_before_insert(self):
        cache = InstanceVolumeCache(max_items=10, max_bytes=64)
        arrays = (np.arange(4, dtype=np.float32), np.arange(2, dtype=np.float64))
        self.assertEqual(sum(array.nbytes for array in arrays), 32)
        cache.get(1, lambda: arrays)
        cache.get(2, lambda: arrays)
        self.assertEqual(cache.stats["current_bytes"], 64)
        cache.get(3, lambda: (np.zeros(4, dtype=np.float32),))
        self.assertEqual(cache.stats["current_bytes"], 48)
        self.assertEqual(cache.stats["current_items"], 2)
        self.assertEqual(cache.stats["evictions"], 1)
        self.assertEqual(cache.stats["peak_bytes"], 64)
        self.assertLessEqual(cache.stats["current_bytes"], cache.max_bytes)
        # Logical nbytes deliberately counts aliased buffers per stored tuple.
        self.assertFalse(cache.stats["rss_limit"])
        self.assertIn("temporaries", cache.stats["memory_limitations"])

    def test_oversized_bypass_preserves_other_entries_and_lru(self):
        cache = InstanceVolumeCache(max_items=2, max_bytes=32)
        first, second = (np.arange(4, dtype=np.float32),), (np.ones(4, dtype=np.float32),)
        cache.get("a", lambda: first)
        cache.get("b", lambda: second)
        oversized = (np.arange(9, dtype=np.float32),)
        calls = []

        def large():
            calls.append(1)
            return oversized

        self.assertIs(cache.get("large", large), oversized)
        self.assertIs(cache.get("large", large), oversized)
        self.assertEqual(len(calls), 2)
        stats = cache.stats
        self.assertEqual(stats["oversized"], 2)
        self.assertEqual(stats["bypasses"], 2)
        self.assertEqual(stats["loads"], 4)
        self.assertEqual(stats["evictions"], 0)
        self.assertEqual(stats["current_items"], 2)
        self.assertEqual(stats["current_bytes"], 32)
        # Oversized misses did not change b's recency or evict a.
        self.assertIs(cache.get("a", lambda: self.fail("a must still be resident")), first)
        self.assertIs(cache.get("b", lambda: self.fail("b must still be resident")), second)

    def test_loader_exception_and_invalid_output_never_leave_partial_entries(self):
        cache = InstanceVolumeCache(1, 32)
        original = (np.ones(4, dtype=np.float32),)
        cache.get("old", lambda: original)

        def broken():
            raise RuntimeError("synthetic load failed")

        with self.assertRaisesRegex(RuntimeError, "load failed"):
            cache.get("new", broken)
        self.assertIs(cache.get("old", lambda: self.fail("old entry must survive failure")), original)
        for invalid in ([], (), (np.zeros(1), "not an array"), np.zeros(2)):
            with self.subTest(value=type(invalid)), self.assertRaises(TypeError):
                cache.get("invalid", lambda: invalid)
        stats = cache.stats
        self.assertEqual(stats["load_failures"], 5)
        self.assertEqual(stats["load_attempts"], 6)
        self.assertEqual(stats["loads"], 1)
        self.assertEqual(stats["current_items"], 1)
        self.assertEqual(stats["evictions"], 0)
        valid = (np.arange(2, dtype=np.float32),)
        self.assertIs(cache.get("new", lambda: valid), valid)
        self.assertEqual(cache.stats["evictions"], 1)

    def test_instances_do_not_share_capacity_entries_or_keep_self_alive(self):
        one, two = InstanceVolumeCache(1, 64), InstanceVolumeCache(1, 64)
        a, b = (np.ones(4),), (np.zeros(4),)
        one.get("same_key", lambda: a)
        two.get("same_key", lambda: b)
        self.assertIs(one.get("same_key", lambda: self.fail("one miss")), a)
        self.assertIs(two.get("same_key", lambda: self.fail("two miss")), b)
        one.get("other", lambda: (np.arange(4),))
        self.assertEqual(two.stats["evictions"], 0)
        self.assertEqual(two.stats["current_items"], 1)
        reference = weakref.ref(one)
        del one
        gc.collect()
        self.assertIsNone(reference())

    def test_pickle_and_deepcopy_are_cold_and_small(self):
        cache = InstanceVolumeCache(3, 2 * 1024 * 1024)
        arrays = (np.ones(1024 * 128, dtype=np.float64),)
        cache.get("large resident", lambda: arrays)
        cache.get("large resident", lambda: self.fail("must hit"))
        original_stats = cache.stats
        serialized = pickle.dumps(cache)
        self.assertLess(len(serialized), 2048)
        for restored in (pickle.loads(serialized), copy.deepcopy(cache)):
            self.assertEqual(restored.max_items, cache.max_items)
            self.assertEqual(restored.max_bytes, cache.max_bytes)
            for key in ("hits", "misses", "loads", "load_attempts", "evictions", "current_items", "current_bytes", "peak_bytes"):
                self.assertEqual(restored.stats[key], 0)
            fresh = (np.zeros(2),)
            self.assertIs(restored.get("large resident", lambda: fresh), fresh)
            self.assertEqual(restored.stats["loads"], 1)
        self.assertEqual(cache.stats, original_stats)
        np.testing.assert_array_equal(arrays[0], np.ones_like(arrays[0]))

    def test_clear_retains_or_resets_counters_without_mutating_arrays(self):
        cache = InstanceVolumeCache(2, 64)
        first = (np.arange(4, dtype=np.float32),)
        original = first[0].copy()
        cache.get("a", lambda: first)
        cache.get("a", lambda: self.fail("must hit"))
        cache.clear()
        stats = cache.stats
        self.assertEqual(stats["current_items"], 0)
        self.assertEqual(stats["current_bytes"], 0)
        self.assertEqual(stats["peak_bytes"], 16)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["loads"], 1)
        self.assertEqual(stats["cleared_items"], 1)
        self.assertEqual(stats["cleared_bytes"], 16)
        self.assertEqual(stats["clears"], 1)
        cache.get("a", lambda: first)
        cache.clear(reset_stats=True)
        for key in ("hits", "misses", "loads", "load_attempts", "load_failures", "evictions", "oversized", "bypasses",
                    "clears", "cleared_items", "cleared_bytes", "current_items", "current_bytes", "peak_items", "peak_bytes"):
            self.assertEqual(cache.stats[key], 0)
        np.testing.assert_array_equal(first[0], original)

    def test_cache_never_casts_writes_or_changes_input_flags(self):
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        readonly = source[:, :, ::2]
        readonly.setflags(write=False)
        arrays = (source, readonly, np.ones((1, 2), dtype=np.uint8))
        copies = [array.copy() for array in arrays]
        flags = [array.flags.writeable for array in arrays]
        dtypes = [array.dtype for array in arrays]
        cache = InstanceVolumeCache(1, 1024)
        self.assertIs(cache.get(("subject", 1), lambda: arrays), arrays)
        self.assertIs(cache.get(("subject", 1), lambda: self.fail("must hit")), arrays)
        cache.get("next", lambda: (np.zeros(2),))
        cache.clear()
        for array, original, writeable, dtype in zip(arrays, copies, flags, dtypes):
            np.testing.assert_array_equal(array, original)
            self.assertEqual(array.flags.writeable, writeable)
            self.assertEqual(array.dtype, dtype)

    def test_capacity_types_keys_and_stats_snapshot(self):
        for kwargs in ({"max_items": 0}, {"max_items": -1}, {"max_items": True}, {"max_items": 1.5},
                       {"max_bytes": 0}, {"max_bytes": -2}, {"max_bytes": False}, {"max_bytes": 1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InstanceVolumeCache(**kwargs)
        cache = InstanceVolumeCache(np.int64(1), np.int64(32))
        before = cache.stats
        for key in ([], {}, ([],)):
            with self.subTest(key=key), self.assertRaisesRegex(TypeError, "hashable"):
                cache.get(key, lambda: (np.zeros(1),))
        with self.assertRaisesRegex(TypeError, "callable"):
            cache.get("x", None)
        self.assertEqual(cache.stats, before)
        snapshot = cache.stats
        snapshot["hits"] = 999
        self.assertEqual(cache.stats["hits"], 0)
        json.dumps(cache.stats, allow_nan=False)
        with self.assertRaises(ValueError):
            cache.clear(reset_stats=1)
        # Zero-byte ndarray entries still obey the item limit.
        empty = (np.empty((0,), dtype=np.float32),)
        cache.get(None, lambda: empty)
        cache.get(0, lambda: empty)
        self.assertEqual(cache.stats["current_items"], 1)
        self.assertEqual(cache.stats["current_bytes"], 0)
        self.assertEqual(cache.stats["evictions"], 1)


if __name__ == "__main__":
    unittest.main()

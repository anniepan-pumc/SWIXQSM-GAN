import copy
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ..util.sampling import CacheAwareBatchSampler


class MetadataOnlyDataset:
    def __init__(self, counts, subjects=None):
        self.rows = [{"subject_id": subject} for subject in
                     (subjects or [f"person-{index}" for index in range(len(counts))])]
        self.index = [(volume, z) for volume, count in enumerate(counts) for z in range(count)]

    def __getitem__(self, index):
        raise AssertionError("Sampler must never read an image/sample")

    def __len__(self):
        raise AssertionError("Sampler uses index metadata, not dataset callbacks")


class CacheAwareSamplingTest(unittest.TestCase):
    def test_every_index_once_and_bounded_working_set_with_unmerged_group_tails(self):
        dataset = MetadataOnlyDataset([6]*5)
        sampler = CacheAwareBatchSampler(dataset, 5, group_volumes=2)
        batches = list(sampler)
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(30)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([len(batch) for batch in batches], [5, 5, 2, 5, 5, 2, 5, 1])
        self.assertEqual(len(sampler), 8)
        working_sets = []
        for grouped_batches in (batches[:3], batches[3:6], batches[6:]):
            volumes = {dataset.index[index][0] for batch in grouped_batches for index in batch}
            self.assertLessEqual(len(volumes), 2)
            self.assertTrue(all(volumes.isdisjoint(previous) for previous in working_sets))
            working_sets.append(volumes)
        self.assertTrue(all(1 <= len(batch) <= 5 for batch in batches))

    def test_seed_and_epoch_reproducibility_and_nontrivial_epoch_changes(self):
        dataset = MetadataOnlyDataset([7, 9, 4, 6, 8, 3])
        a = CacheAwareBatchSampler(dataset, 4, seed=123)
        b = CacheAwareBatchSampler(dataset, 4, seed=123)
        first = list(a)
        self.assertEqual(first, list(a))
        self.assertEqual(first, list(b))
        a.set_epoch(2)
        b.set_epoch(2)
        self.assertEqual(list(a), list(b))
        self.assertNotEqual(first, list(a))
        a.set_epoch(0)
        self.assertEqual(first, list(a))

    def test_global_torch_python_numpy_rng_unchanged_even_when_len_is_requested(self):
        dataset = MetadataOnlyDataset([9, 4, 8, 3])
        torch.manual_seed(17)
        random.seed(18)
        np.random.seed(19)
        torch_state = torch.get_rng_state().clone()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        sampler = CacheAwareBatchSampler(dataset, 3, seed=999)
        sampler.set_epoch(7)
        expected_length = len(sampler)
        self.assertEqual(len(list(sampler)), expected_length)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(random.getstate(), python_state)
        current = np.random.get_state()
        self.assertEqual(current[0], numpy_state[0])
        np.testing.assert_array_equal(current[1], numpy_state[1])
        self.assertEqual(current[2:], numpy_state[2:])

    def test_len_matches_actual_partial_batches_for_each_epoch(self):
        dataset = MetadataOnlyDataset([1, 2, 3, 4, 5, 9])
        sampler = CacheAwareBatchSampler(dataset, 4, group_volumes=2)
        for epoch in range(8):
            sampler.set_epoch(epoch)
            self.assertEqual(len(sampler), len(list(sampler)))
            self.assertEqual(sorted(index for batch in sampler for index in batch), list(range(24)))

    def test_same_subject_visits_are_contiguous_in_volume_schedule(self):
        dataset = MetadataOnlyDataset([2]*6, ["p1", "p2", "p1", "p3", "p2", "p1"])
        sampler = CacheAwareBatchSampler(dataset, 10, group_volumes=1, seed=91)
        batches = list(sampler)
        labels = [dataset.rows[dataset.index[batch[0]][0]]["subject_id"] for batch in batches]
        for subject in set(labels):
            positions = [i for i, label in enumerate(labels) if label == subject]
            self.assertEqual(positions, list(range(min(positions), max(positions)+1)))
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(12)))

    def test_shuffle_false_is_sequential_for_standard_index_and_epoch_independent(self):
        dataset = MetadataOnlyDataset([3, 5, 2])
        sampler = CacheAwareBatchSampler(dataset, 4, group_volumes=1, shuffle=False)
        expected = [[0, 1, 2], [3, 4, 5, 6], [7], [8, 9]]
        self.assertEqual(list(sampler), expected)
        sampler.set_epoch(100)
        self.assertEqual(list(sampler), expected)
        self.assertEqual(len(sampler), len(expected))

    def test_index_and_rows_unchanged_and_no_dataset_callback(self):
        dataset = MetadataOnlyDataset([2, 4])
        before = copy.deepcopy((dataset.index, dataset.rows))
        sampler = CacheAwareBatchSampler(dataset, 3, group_volumes=20)
        list(sampler)
        self.assertEqual((dataset.index, dataset.rows), before)
        # group_volumes is an explicit caller setting, never guessed/clamped to a cache.
        self.assertEqual(sampler.group_volumes, 20)

    def test_invalid_arguments_and_nonfinite_indices_refused(self):
        dataset = MetadataOnlyDataset([2, 3])
        for kwargs in ({"batch_size": 0}, {"batch_size": True}, {"batch_size": 2.5},
                       {"batch_size": 2, "group_volumes": 0}, {"batch_size": 2, "group_volumes": float("inf")},
                       {"batch_size": 2, "seed": -1}, {"batch_size": 2, "seed": 2**64},
                       {"batch_size": 2, "shuffle": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CacheAwareBatchSampler(dataset, **kwargs)
        for index in ([], [(0, float("inf"))], [(float("nan"), 0)], [(0, -1)], [(9, 0)],
                      [(0, 0), (0, 0)], [(0, 0, 1)], [(True, 0)]):
            bad = SimpleNamespace(index=index, rows=dataset.rows)
            with self.subTest(index=index), self.assertRaises(ValueError):
                CacheAwareBatchSampler(bad, 2)
        with self.assertRaises(ValueError):
            CacheAwareBatchSampler(SimpleNamespace(index=[(0, 0)], rows=[{"subject_id": ""}]), 1)
        sampler = CacheAwareBatchSampler(dataset, 2)
        for epoch in (-1, float("inf"), True):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                sampler.set_epoch(epoch)


if __name__ == "__main__":
    unittest.main()

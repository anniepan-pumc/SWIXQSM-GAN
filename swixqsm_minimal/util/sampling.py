"""Optional bounded-volume slice batching for future training runs.

This sampler preserves one visit per effective dataset slice per epoch. It does
not preserve the old optimization trajectory: groups can introduce additional
partial batches, and nearby batches have a different subject/visit composition.
"""

from collections.abc import Mapping, Sequence
from numbers import Integral

import torch
from torch.utils.data import Sampler


def _integer(value, name, *, positive=False, bounded=False):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < (1 if positive else 0) or (bounded and value > 2**64-1):
        raise ValueError(f"{name} is outside its permitted integer range")
    return value


class CacheAwareBatchSampler(Sampler[list[int]]):
    """Batch slice indices within a bounded working set of complete volumes.

    Uses only ``dataset.index`` tuples (volume_i,z) and ``dataset.rows`` subject
    metadata. It never calls dataset.__getitem__, reads images, or inspects
    another split. The caller must check group_volumes against the *actual*
    volume-cache capacity; this class does not guess a cache size.

    With shuffle=True, a CPU-local torch.Generator(seed+epoch modulo 2^64)
    randomizes subjects, then visits within each subject. Same-subject volumes
    are consecutive in the volume schedule. Consecutive groups contain at most
    group_volumes volumes, and slices are shuffled only inside that group.
    Each group finishes with its own partial batch; leftovers never cross a
    group boundary. Subjects can mix within a group when a boundary falls
    between subjects, so this is neither equal-subject weighting nor oversampling.

    shuffle=False uses volume first-appearance order and each volume's original
    dataset-index order. For the ordinary volume-contiguous PairedSlices.index,
    this is sequential slice order. Validation can retain its existing standard
    sequential loader instead of using this optional sampler.
    """

    def __init__(self, dataset, batch_size, *, group_volumes=2, seed=42, shuffle=True):
        self.batch_size = _integer(batch_size, "batch_size", positive=True)
        self.group_volumes = _integer(group_volumes, "group_volumes", positive=True)
        self.seed = _integer(seed, "seed", bounded=True)
        if type(shuffle) is not bool:
            raise ValueError("shuffle must be boolean")
        self.shuffle = shuffle
        self.epoch = 0
        index, rows = getattr(dataset, "index", None), getattr(dataset, "rows", None)
        if (not isinstance(index, Sequence) or isinstance(index, (str, bytes)) or not index
                or not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows):
            raise ValueError("Sampler requires nonempty finite dataset.index and dataset.rows sequences")
        self._volume_indices = {}
        self._volume_order = []
        self._subjects = {}
        seen = set()
        for position, item in enumerate(index):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Each dataset index must be a (volume_i,z) pair")
            volume = _integer(item[0], "volume_i")
            z = _integer(item[1], "z")
            if volume >= len(rows):
                raise ValueError("Dataset volume_i exceeds dataset.rows")
            if (volume, z) in seen:
                raise ValueError("Dataset index repeats the same volume/slice pair")
            seen.add((volume, z))
            if volume not in self._volume_indices:
                row = rows[volume]
                if not isinstance(row, Mapping) or not isinstance(row.get("subject_id"), str) or not row["subject_id"].strip():
                    raise ValueError("Every sampled volume needs a nonempty subject_id")
                self._volume_indices[volume] = []
                self._volume_order.append(volume)
                self._subjects.setdefault(row["subject_id"].strip(), []).append(volume)
            self._volume_indices[volume].append(position)
        self.n_slices = len(index)
        self.n_volumes = len(self._volume_order)

    def set_epoch(self, epoch):
        """Choose a reproducible epoch without changing any global RNG state."""
        self.epoch = _integer(epoch, "epoch", bounded=True)

    def _groups(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed+self.epoch) % 2**64)
        if self.shuffle:
            subjects = list(self._subjects)
            order = []
            for subject_index in torch.randperm(len(subjects), generator=generator).tolist():
                volumes = self._subjects[subjects[subject_index]]
                for visit_index in torch.randperm(len(volumes), generator=generator).tolist():
                    order.append(volumes[visit_index])
        else:
            order = self._volume_order
        return [order[start:start+self.group_volumes]
                for start in range(0, len(order), self.group_volumes)], generator

    def __iter__(self):
        groups, generator = self._groups()
        for volumes in groups:
            indices = [index for volume in volumes for index in self._volume_indices[volume]]
            if self.shuffle:
                indices = [indices[position] for position in torch.randperm(len(indices), generator=generator).tolist()]
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start+self.batch_size]

    def __len__(self):
        groups, _ = self._groups()
        return sum((sum(len(self._volume_indices[volume]) for volume in group)+self.batch_size-1)
                   // self.batch_size for group in groups)

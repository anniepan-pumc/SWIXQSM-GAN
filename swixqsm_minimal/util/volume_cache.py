"""Instance-local LRU for already prepared volume arrays, never a disk cache.

The byte budget counts sum(array.nbytes) for resident prepared tuples only.
It is not an RSS limit: decoding/normalization temporaries, caller-held arrays,
tensor copies, and backing storage of array views need separate memory. Cache
operations never write, resize, cast, or change flags on caller arrays. Callers
must treat resident prepared arrays as immutable; cache hits return references.
"""
from collections import OrderedDict
from numbers import Integral

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer (not bool)")
    return int(value)


class InstanceVolumeCache:
    """Bound resident prepared volumes by both item count and logical bytes.

    Each instance owns its entries and LRU order. ``loader`` runs only on a
    miss and must return a nonempty tuple of ndarrays. An oversized result is
    returned but never retained and does not evict existing entries. Errors
    leave old entries unchanged. There is no process-global state or persistent
    disk storage, and pickle/deepcopy starts the receiving instance cold.

    This cache provides no inter-thread synchronization. Worker processes have
    separate instances; pickle deliberately excludes entries and runtime stats.
    """
    def __init__(self, max_items=2, max_bytes=256 * 1024 * 1024):
        self._max_items = _positive_integer(max_items, "max_items")
        self._max_bytes = _positive_integer(max_bytes, "max_bytes")
        self._entries = OrderedDict()
        self._current_bytes = 0
        self._reset_stats()

    @property
    def max_items(self):
        return self._max_items

    @property
    def max_bytes(self):
        return self._max_bytes

    def __len__(self):
        return len(self._entries)

    def _reset_stats(self):
        self._counters = {name: 0 for name in ("hits", "misses", "load_attempts", "loads", "load_failures",
                                               "evictions", "oversized", "bypasses", "clears",
                                               "cleared_items", "cleared_bytes")}
        self._peak_bytes = self._current_bytes
        self._peak_items = len(self._entries)

    @property
    def stats(self):
        """Return a detached metadata snapshot; mutating it cannot alter cache."""
        return {**self._counters, "current_items": len(self._entries),
                "current_bytes": self._current_bytes, "peak_items": self._peak_items,
                "peak_bytes": self._peak_bytes, "max_items": self.max_items, "max_bytes": self.max_bytes,
                "scope": "instance-local memory cache of prepared ndarray tuples only",
                "byte_accounting": "sum ndarray.nbytes per resident tuple; logical bytes, not unique backing allocations",
                "rss_limit": False,
                "memory_limitations": "Excludes decode/normalization temporaries, caller-held results, tensor copies and additional backing storage of views",
                "returned_arrays": "Shared references; cache never mutates arrays; caller must not resize or mutate resident data"}

    @staticmethod
    def _prepared(value):
        if not isinstance(value, tuple) or not value or any(not isinstance(array, np.ndarray) for array in value):
            raise TypeError("Volume loader must return a nonempty tuple of numpy ndarrays")
        return value, sum(int(array.nbytes) for array in value)

    def get(self, key, loader):
        """Get prepared tuple, loading once on a miss without writing its data.

        ``loads`` counts successful, valid loader results, including bypasses.
        ``load_attempts`` also includes exceptions/invalid results; these count
        as ``load_failures`` and never insert or evict a partial entry.
        """
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("Volume cache keys must be hashable") from exc
        if not callable(loader):
            raise TypeError("Volume cache loader must be callable")
        if key in self._entries:
            self._counters["hits"] += 1
            self._entries.move_to_end(key)
            return self._entries[key][0]
        self._counters["misses"] += 1
        self._counters["load_attempts"] += 1
        try:
            value, nbytes = self._prepared(loader())
        except BaseException:
            self._counters["load_failures"] += 1
            raise
        self._counters["loads"] += 1
        if nbytes > self.max_bytes:
            self._counters["oversized"] += 1
            self._counters["bypasses"] += 1
            return value
        while (len(self._entries) >= self.max_items
               or self._current_bytes + nbytes > self.max_bytes):
            _, (_, removed_bytes) = self._entries.popitem(last=False)
            self._current_bytes -= removed_bytes
            self._counters["evictions"] += 1
        self._entries[key] = (value, nbytes)
        self._current_bytes += nbytes
        self._peak_bytes = max(self._peak_bytes, self._current_bytes)
        self._peak_items = max(self._peak_items, len(self._entries))
        return value

    def clear(self, reset_stats=False):
        """Release cached references, optionally resetting all runtime counters."""
        if type(reset_stats) is not bool:
            raise ValueError("reset_stats must be a boolean")
        items, nbytes = len(self._entries), self._current_bytes
        self._entries.clear()
        self._current_bytes = 0
        if reset_stats:
            self._reset_stats()
        else:
            self._counters["clears"] += 1
            self._counters["cleared_items"] += items
            self._counters["cleared_bytes"] += nbytes

    def __getstate__(self):
        # In particular, do not pickle ndarrays or runtime access counters when
        # a Dataset is sent to spawned workers or copied for another instance.
        return {"schema": "swixqsm.instance_volume_cache", "version": 1,
                "max_items": self.max_items, "max_bytes": self.max_bytes}

    def __setstate__(self, state):
        if (not isinstance(state, dict) or set(state) != {"schema", "version", "max_items", "max_bytes"}
                or state["schema"] != "swixqsm.instance_volume_cache"
                or type(state["version"]) is not int or state["version"] != 1):
            raise ValueError("Unsupported volume-cache pickle state")
        self.__init__(max_items=state["max_items"], max_bytes=state["max_bytes"])

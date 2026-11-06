"""Explicit paired manifest, float32 intensities, padding without resizing."""
import csv
from copy import deepcopy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from .normalization import QSMNormalization, SWI_SPEC, normalize_swi, unit_factor
from .volume_cache import InstanceVolumeCache


def read_manifest(path, *, validate_inputs=True):
    """Parse identities/paths; the QC reporter can defer input errors to its report."""
    path = Path(path).resolve()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"subject_id", "split", "swi_path", "qsm_path", "swi_mask_path", "mask_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest requires {sorted(required)}")
    splits, seen = {}, set()
    for row in rows:
        if not row["subject_id"] or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Each row needs a subject_id and train/val/test split")
        sid = row["subject_id"]
        if sid in splits and splits[sid] != row["split"]:
            raise ValueError(f"Subject leakage across splits: {sid}")
        splits[sid] = row["split"]
        if validate_inputs and "qsm_units" in row:
            unit_factor(row["qsm_units"], row["qsm_units"])
        for key in required - {"subject_id", "split"}:
            p = Path(row[key])
            p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
            if validate_inputs and not p.is_file():
                raise FileNotFoundError(p)
            row[key] = str(p)
        # Reusing one input under a different patient alias is also invalid.
        if row["swi_path"] in seen:
            raise ValueError("Duplicate SWI path; keep one reference per input in this baseline")
        seen.add(row["swi_path"])
    return rows


def load_volume(path):
    image = nib.load(path)
    array = image.get_fdata(dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"Expected finite 3D NIfTI: {path}")
    return image, array


def require_grid(reference, image):
    if reference.shape != image.shape or not np.allclose(reference.affine, image.affine, atol=1e-4, rtol=0):
        raise ValueError("Input/reference/masks must already be registered on the same grid")


def pad_slice(array, size):
    # NIfTI [X,Y,Z] -> network [Y,X], matching the legacy permute(2,1,0).
    array = array.T
    h, w = array.shape
    if max(h, w) > size:
        raise ValueError(f"Grid {array.shape} exceeds --size {size}; increase size (no automatic crop)")
    return torch.from_numpy(np.pad(array, ((0, size-h), (0, size-w))).copy()).unsqueeze(0)


class PairedSlices(Dataset):
    """Mask-indexed slices with an instance-local, bounded prepared-volume cache.

    Source files must remain immutable for the lifetime of this dataset. Rebuild
    the dataset after changing images/metadata; this is not a persistent cache.
    Indexing checks geometry and reads only validity-mask values. Full finite,
    brain-mask support and normalization checks run when a volume is accessed;
    lazy indexing does not replace the separate pretraining data-QC gate.
    """
    def __init__(self, rows, split, size, qsm_scale, *, cache_items=2,
                 cache_bytes=256 * 1024 * 1024):
        self.normalization = qsm_scale if isinstance(qsm_scale, QSMNormalization) else QSMNormalization(qsm_scale)
        if size < 32 or size % 8:
            raise ValueError("size must be >=32 and divisible by 8; scale must be finite and positive")
        self.rows = [dict(r) for r in rows if r["split"] == split]
        self.size, self.scale = size, self.normalization.scale
        self._swi_spec = deepcopy(SWI_SPEC)
        self._cache = InstanceVolumeCache(cache_items, cache_bytes)
        self._io = self._empty_io()
        self.index = []
        for i, row in enumerate(self.rows):
            unit_factor(row.get("qsm_units", self.normalization.units), self.normalization.units)
            self._io["index_header_reads"] += 1
            reference = nib.load(row["swi_path"])
            if len(reference.shape) != 3:
                raise ValueError(f"Expected 3D NIfTI: {row['swi_path']}")
            if max(reference.shape[:2]) > self.size:
                raise ValueError("Increase --size to fit the prepared in-plane grid")
            for key in ("qsm_path", "swi_mask_path"):
                self._io["index_header_reads"] += 1
                require_grid(reference, nib.load(row[key]))
            self._io["index_header_reads"] += 1
            image, mask = self._decode(row["mask_path"], indexing=True)
            require_grid(reference, image)
            mask = mask > 0
            if not mask.any():
                raise ValueError("Valid reference mask must be nonempty and contained in SWI brain mask")
            for z in range(mask.shape[2]):
                if mask[:, :, z].any():
                    self.index.append((i, z))
        if not self.index:
            raise ValueError(f"No valid slices for {split}")

    @staticmethod
    def _empty_io():
        return dict(index_header_reads=0, index_volume_decodes=0,
                    prepared_volume_decodes=0, volume_decodes=0, volume_preparations=0)

    @property
    def cache_stats(self):
        return self._cache.stats

    @property
    def io_stats(self):
        """Application reads/preparations, not physical disk/page-cache traffic.

        Header reads count the four indexing grid checks (the validity decode
        includes its header). Decodes include attempts, including failures.
        Prepared-volume decodes also open headers via load_volume.
        """
        return {**self._io, "cache": self.cache_stats}

    def clear_cache(self, reset_stats=False):
        self._cache.clear(reset_stats=reset_stats)
        if reset_stats:
            self._io = self._empty_io()

    def _decode(self, path, *, indexing=False):
        self._io["volume_decodes"] += 1
        self._io["index_volume_decodes" if indexing else "prepared_volume_decodes"] += 1
        return load_volume(path)

    def volume(self, i):
        row = self.rows[i]
        key = (tuple(row[k] for k in ("swi_path", "qsm_path", "swi_mask_path", "mask_path")),
               row.get("qsm_units", self.normalization.units),
               self.normalization.scale, self.normalization.offset, self.normalization.units,
               json.dumps(self._swi_spec, sort_keys=True))
        return self._cache.get(key, lambda: self._prepare_volume(row))

    def _prepare_volume(self, row):
        ref, swi = self._decode(row["swi_path"])
        arrays = []
        for key in ("qsm_path", "swi_mask_path", "mask_path"):
            img, array = self._decode(row[key])
            require_grid(ref, img)
            arrays.append(array)
        qsm, brain, mask = arrays
        brain, mask = brain > 0, mask > 0
        if not mask.any() or np.any(mask & ~brain):
            raise ValueError("Valid reference mask must be nonempty and contained in SWI brain mask")
        if max(swi.shape[:2]) > self.size:
            raise ValueError("Increase --size to fit the prepared in-plane grid")
        target = self.normalization.normalize(qsm, row.get("qsm_units", self.normalization.units))
        if not np.isfinite(target).all():
            raise ValueError("QSM normalization overflow; check units and fixed scale")
        prepared = (normalize_swi(swi, brain, spec=self._swi_spec),
                    target.astype(np.float32), mask.astype(np.float32))
        # __getitem__ copies into padded tensors; never expose writable cached
        # arrays that another slice could inadvertently modify in place.
        for array in prepared:
            array.flags.writeable = False
        self._io["volume_preparations"] += 1
        return prepared

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, z = self.index[index]
        x, y, m = self.volume(i)
        return tuple(pad_slice(a[:, :, z], self.size) for a in (x, y, m)) + (self.rows[i]["subject_id"],)

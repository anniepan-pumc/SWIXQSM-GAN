"""Read-only I/O/value probe; no model, training, or data-QC certification."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch

from .util.data import PairedSlices, load_volume, read_manifest
from .util.normalization import QSMNormalization, normalization_metadata
from .util.sampling import CacheAwareBatchSampler
from .util.volume_cache import InstanceVolumeCache


RAW_ROLES = ("SWI.nii.gz", "QSM.nii.gz", "QSM_mask.nii.gz")


class _RawProbeSlices:
    """Only supplied three-role arrays; no normalization or inferred units."""

    def __init__(self, rows, *, cache_items, cache_bytes):
        self.rows = rows
        self.cache = InstanceVolumeCache(max_items=cache_items, max_bytes=cache_bytes)
        self._reads = {role: 0 for role in RAW_ROLES}
        self.index = []
        for volume in range(len(rows)):
            _, _, mask = self.volume(volume)
            self.index.extend((volume, z) for z in range(mask.shape[2]) if np.any(mask[:, :, z] > 0))
        if not self.index:
            raise ValueError("Raw probe has no positive-mask slices")

    def _load(self, volume):
        arrays = []
        for role in RAW_ROLES:
            self._reads[role] += 1
            _, array = load_volume(self.rows[volume][role])
            arrays.append(array)
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError("Raw probe needs matching 3D array shapes for shared slice indexing; no resampling is performed")
        return tuple(arrays)

    def volume(self, volume):
        return self.cache.get(volume, lambda: self._load(volume))

    def __getitem__(self, index):
        volume, z = self.index[index]
        return tuple(array[:, :, z] for array in self.volume(volume)) + (self.rows[volume]["subject_id"],)

    def __len__(self):
        return len(self.index)

    @property
    def io_stats(self):
        return {"nifti_reads": sum(self._reads.values()), "reads_by_role": dict(self._reads),
                "cache": self.cache.stats,
                "read_definition": "Calls to finite-float32 NIfTI decode; no claim about OS physical disk reads"}

    def clear_cache(self, reset_stats=True):
        self.cache.clear(reset_stats=reset_stats)
        if reset_stats:
            self._reads = {role: 0 for role in RAW_ROLES}


def _limit_slices(dataset, maximum):
    by_volume = {}
    for index in dataset.index:
        by_volume.setdefault(index[0], []).append(index)
    chosen = []
    for values in by_volume.values():
        positions = np.linspace(0, len(values)-1, min(len(values), maximum), dtype=np.int64)
        chosen.extend(values[int(position)] for position in positions)
    dataset.index = chosen
    return [{"index": index, "volume_i": int(volume), "z": int(z),
             "subject_id": dataset.rows[volume]["subject_id"]}
            for index, (volume, z) in enumerate(chosen)]


def _sample_hash(sample):
    """Hash returned values one sample at a time; never retain tensors/arrays."""
    digest = hashlib.sha256(b"swixqsm.io_returned_slice_float32_v1\0")
    for role, value in enumerate(sample[:3]):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        values = np.ascontiguousarray(value, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError("I/O value probe encountered nonfinite returned slice values")
        header = {"role_index": role, "shape": list(values.shape), "dtype": "little_endian_float32"}
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode()+b"\0")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _index_digest(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()+b"\n")
    return digest.hexdigest()


def _run_policy(factory, policy, args):
    started = time.perf_counter()
    dataset = factory()
    available = len(dataset.index)
    index_manifest = _limit_slices(dataset, args.slices_per_volume)
    startup_seconds = time.perf_counter()-started
    startup_stats = dataset.io_stats
    dataset.clear_cache(reset_stats=True)
    if policy == "global_random_slices":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        indices = torch.randperm(len(dataset.index), generator=generator).tolist()
        batches = [indices[start:start+args.batch_size] for start in range(0, len(indices), args.batch_size)]
    else:
        batches = CacheAwareBatchSampler(dataset, args.batch_size, group_volumes=args.group_volumes,
                                        seed=args.seed, shuffle=True)
    hashes, batch_sizes = {}, []
    access_seconds = hash_seconds = 0.
    started = time.perf_counter()
    for batch in batches:
        batch_sizes.append(len(batch))
        for index in batch:
            if index in hashes:
                raise RuntimeError("Benchmark sampler repeated an index")
            access_started = time.perf_counter()
            sample = dataset[index]
            access_seconds += time.perf_counter()-access_started
            hash_started = time.perf_counter()
            hashes[index] = _sample_hash(sample)
            hash_seconds += time.perf_counter()-hash_started
            del sample
    traversal_seconds = time.perf_counter()-started
    if sorted(hashes) != list(range(len(dataset.index))):
        raise RuntimeError("Benchmark sampler omitted one or more selected indices")
    value_manifest = [{**record, "sha256": hashes[record["index"]]} for record in index_manifest]
    return {"policy": policy, "startup": {"wall_seconds": startup_seconds, "io_stats": startup_stats,
                                          "valid_slices_before_limit": available},
            "traversal": {"wall_seconds": traversal_seconds, "dataset_access_seconds": access_seconds,
                          "value_hash_seconds": hash_seconds, "io_stats": dataset.io_stats,
                          "n_samples": len(hashes), "n_batches": len(batch_sizes),
                          "n_partial_batches": sum(size < args.batch_size for size in batch_sizes),
                          "batch_sizes": batch_sizes},
            "index_manifest_sha256": _index_digest(index_manifest),
            "ordered_value_manifest_sha256": _index_digest(value_manifest),
            "ordered_per_index_hashes": value_manifest}


def _raw_rows(path, maximum=3):
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("candidates"), list) or not document["candidates"]:
        raise ValueError("Raw probe JSON must provide a nonempty candidates list")
    rows = []
    for index, candidate in enumerate(document["candidates"][:min(maximum, 3)]):
        if not isinstance(candidate, dict) or any(not isinstance(candidate.get(role), str) or not candidate[role].strip()
                                                  for role in RAW_ROLES):
            raise ValueError("Each raw candidate must provide SWI.nii.gz, QSM.nii.gz and QSM_mask.nii.gz paths")
        row = {"subject_id": f"raw-probe-{index+1}"}
        for role in RAW_ROLES:
            supplied = Path(candidate[role]).expanduser()
            row[role] = str((path.parent/supplied if not supplied.is_absolute() else supplied).resolve())
        rows.append(row)
    return rows, len(document["candidates"])


def _count_comparison(results, name):
    counts = []
    for result in results:
        stats = result["traversal"]["io_stats"]
        counts.append(stats["cache"]["loads"] if name == "volume_loads"
                      else stats.get("volume_decodes", stats.get("nifti_reads")))
    baseline, grouped = counts
    ratio = grouped/baseline if baseline else None
    return {"global_random": baseline, "cache_aware": grouped,
            "cache_aware_over_global_random": ratio,
            "fraction_reduction": 1-ratio if ratio is not None else None,
            "status": "defined" if ratio is not None else "zero_baseline_count"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Four-role manifest, existing PairedSlices preprocessing only")
    source.add_argument("--raw-probe-json", help="Unverified three-file candidates; at most three volumes, no normalization/units inferred")
    parser.add_argument("--output", required=True, help="New JSON file; never overwrites a result")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="Manifest mode only")
    parser.add_argument("--size", type=int, default=416, help="Manifest PairedSlices padding size only")
    parser.add_argument("--qsm-scale", type=float, help="Manifest-only explicit physical target scale")
    parser.add_argument("--qsm-offset", type=float, help="Manifest-only explicit physical target offset, e.g. 0")
    parser.add_argument("--units", choices=["ppm", "ppb"], help="Manifest-only explicit physical target units")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-volumes", type=int, default=2)
    parser.add_argument("--cache-mib", type=float, default=256.)
    parser.add_argument("--group-volumes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--slices-per-volume", type=int, default=8)
    parser.add_argument("--max-volumes", type=int, default=3,
                        help="Limit image access (default 3); raw probe always has a hard cap of 3")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error("Benchmark JSON already exists")
    if (args.cache_volumes < 1 or args.group_volumes < 1 or args.group_volumes > args.cache_volumes
            or args.batch_size < 1 or args.slices_per_volume < 1 or args.max_volumes < 1
            or not 0 <= args.seed <= 2**64-1
            or not math.isfinite(args.cache_mib) or args.cache_mib <= 0):
        parser.error("Require positive cache/group/batch/slice limits, group<=cache, and a valid nonnegative seed")
    cache_byte_value = args.cache_mib*1024*1024
    if not math.isfinite(cache_byte_value):
        parser.error("Cache MiB converts to a nonfinite byte budget")
    cache_bytes = int(cache_byte_value)
    if cache_bytes < 1:
        parser.error("Cache byte budget is less than one byte")
    normalization = None
    if args.manifest:
        if any(value is None for value in (args.qsm_scale, args.qsm_offset, args.units)):
            parser.error("Manifest mode requires explicit --qsm-scale, --qsm-offset and --units")
        normalization = QSMNormalization(args.qsm_scale, args.qsm_offset, args.units)
        source_path = Path(args.manifest).expanduser().resolve()
        rows = read_manifest(source_path)
        selected_rows = [row for row in rows if row["split"] == args.split]
        if not selected_rows:
            parser.error("Manifest contains no selected-split volumes")
        source_rows = selected_rows[:args.max_volumes]
        factory = lambda: PairedSlices(source_rows, args.split, args.size, normalization,
                                       cache_items=args.cache_volumes, cache_bytes=cache_bytes)
        mode = "prepared_manifest"
        available_volumes = len(selected_rows)
    else:
        if any(value is not None for value in (args.qsm_scale, args.qsm_offset, args.units)):
            parser.error("Raw probe cannot normalize targets or infer/declare physical units")
        if args.split != "train" or args.size != 416:
            parser.error("--split and --size apply only to manifest mode")
        source_path = Path(args.raw_probe_json).expanduser().resolve()
        source_rows, available_volumes = _raw_rows(source_path, args.max_volumes)
        factory = lambda: _RawProbeSlices(source_rows, cache_items=args.cache_volumes, cache_bytes=cache_bytes)
        mode = "raw_three_file_probe"
    results = [_run_policy(factory, policy, args) for policy in ("global_random_slices", "cache_aware_volume_groups")]
    identical_index = results[0]["index_manifest_sha256"] == results[1]["index_manifest_sha256"]
    identical_values = results[0]["ordered_per_index_hashes"] == results[1]["ordered_per_index_hashes"]
    report = {"schema": "swixqsm.io_benchmark", "version": 1, "status": "complete", "mode": mode,
              "data_qc": {"status": "not_assessed", "reason": "Read-only I/O/value probe, not registration/reference/independence certification"},
              "source": {"path": str(source_path), "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                         "volumes_used": len(source_rows), "candidates_or_split_volumes_available": available_volumes,
                         "rows": source_rows},
              "configuration": {"seed": args.seed, "cache_volumes": args.cache_volumes,
                                "cache_mib": args.cache_mib, "cache_bytes": cache_bytes,
                                "group_volumes": args.group_volumes, "batch_size": args.batch_size,
                                "slices_per_volume": args.slices_per_volume,
                                "max_volumes": args.max_volumes,
                                "slice_selection": "At most limit uniformly spaced positions from each volume's valid third-axis slices",
                                "split": args.split if args.manifest else None,
                                "size": args.size if args.manifest else None,
                                "normalization": normalization_metadata(normalization) if normalization is not None else None},
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)},
              "execution_order": [result["policy"] for result in results], "results": results,
              "comparison": {"all_selected_indices_equal": identical_index, "all_per_index_values_equal": identical_values,
                             "traversal_volume_loads": _count_comparison(results, "volume_loads"),
                             "traversal_nifti_decodes": _count_comparison(results, "nifti_decodes"),
                             "count_scope": "Traversal application calls only; index startup is reported separately, OS disk reads are not measured",
                             "value_space": "Existing PairedSlices returned tensors" if args.manifest else "Decoded source float32 slices, no normalization",
                             "equivalence_scope": "Returned selected slice values and exact coverage only; no model/accuracy/optimization equivalence"},
              "limitations": [
                  "OS page cache is uncontrolled; startup reads index masks (and all three raw-probe roles), and later policies run after earlier ones. These are not cold-disk speed estimates.",
                  "Traversal wall time includes existing loading/preprocessing/slice access, local sampling and value hashing; NIfTI decode call counts are not physical disk read counts.",
                  "Each policy has a separate instance-local memory cache, cleared after index startup. The byte bound covers resident prepared ndarray tuples, not process RSS or decode temporaries.",
                  "A byte budget can retain fewer volumes than the item limit; group_volumes alone does not guarantee the working set fits in bytes.",
                  "Group-tail partial batches change future optimization trajectories. No training, model inference, checkpoint selection or accuracy comparison is performed.",
                  "Raw candidates have no supplied SWI brain mask or verified QSM units. Positive QSM_mask chooses probe slices only; equal array shape does not prove registration.",
                  "Source images must remain immutable. This probe records the source-list file SHA and returned sample digests, not full before/after SHA256 of every input image.",
              ]}
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".swixqsm-io-", dir=output.parent, delete=False) as stream:
        temporary_path = Path(stream.name)
        stream.write(payload)
    try:
        # Hard-link creation is atomic and refuses an existing destination.
        output.hardlink_to(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"I/O benchmark complete. Selected values identical: {identical_index and identical_values}. Report: {output}")
    if not (identical_index and identical_values):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

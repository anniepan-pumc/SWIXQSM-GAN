"""Training: --model NAME selects the new comparison workflow; --list-models lists choices.

Without --model or --resume the earlier minimal training interface is retained.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .util.data import PairedSlices, read_manifest
from .util.engine import PatchDiscriminator, load_generator, train_step, validate
from .model import UNetSmall
from .util.normalization import normalization_metadata
from .util.normalization_workflow import (add_training_normalization_arguments, bind_training_units,
                                          resolve_training_normalization)
from .util.quality_gate import add_quality_arguments, check_data_quality, prepare_quality_rows
from .util.independence_gate import add_independence_arguments, check_independence
from .util.training_state import TrainingProgress
from .util.patch_loss import adversarial_spec
from .util.training_diagnostics import EpochDiagnostics
from .util.sampling import CacheAwareBatchSampler


def epoch_io(before, after):
    """Separate interval counters from cumulative cache residency/peaks."""
    counters = ("hits", "misses", "load_attempts", "loads", "load_failures", "evictions", "oversized", "bypasses")
    return {"read_delta": {key: after[key] - before[key] for key in after if key != "cache"},
            "cache_counter_delta": {key: after["cache"][key] - before["cache"][key] for key in counters},
            "after": after}


def main():
    if any(arg in {"--model", "--list-models", "--resume"} or arg.startswith("--model=")
           or arg.startswith("--resume=") for arg in sys.argv[1:]):
        from .train_models import main as comparison_main
        return comparison_main()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True, help="New, nonexisting experiment directory")
    p.add_argument("--gen-checkpoint")
    p.add_argument("--reset-head", action="store_true")
    add_training_normalization_arguments(p, required=True)
    p.add_argument("--units", choices=["ppm", "ppb", "stored"], required=True, help="Output QSM units; stored preserves the unverified original numerical scale")
    p.add_argument("--size", type=int, default=416)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sampling-mode", choices=["volume_grouped", "slice_random"], default="volume_grouped",
                   help="Future runs: shuffle slices within bounded volume groups, or globally across all volumes")
    p.add_argument("--cache-volumes", type=int, default=2, help="Maximum prepared volumes per dataset instance")
    p.add_argument("--cache-mib", type=float, default=256., help="Prepared-array cache budget per dataset; not a total RAM limit")
    p.add_argument("--volume-group-size", type=int, default=2, help="Volumes per shuffled group; must fit cache-volumes")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-disc", type=float, default=1e-4)
    p.add_argument("--adv-weight", type=float, default=0.01)
    p.add_argument("--adversarial-mode", choices=["valid_patch", "all_patches"], default="valid_patch",
                   help="Future training: exclude empty receptive fields, or explicitly retain the historical all-patch mean")
    p.add_argument("--diagnostic-batches", type=int, default=4,
                   help="Maximum sampled training batches per epoch for physical output distributions; 0 disables sampling")
    p.add_argument("--warmup-epochs", type=int, default=2, help="L1-only epochs")
    p.add_argument("--freeze-bn", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reference-id", help="Optional consistency check against the QC contract's reference ID")
    add_quality_arguments(p)
    add_independence_arguments(p, required=True)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.warmup_epochs < 0:
        p.error("Invalid epoch/batch/warmup values")
    if args.diagnostic_batches < 0:
        p.error("diagnostic-batches must be nonnegative")
    if (args.cache_volumes < 1 or args.volume_group_size < 1 or not math.isfinite(args.cache_mib)
            or not math.isfinite(args.cache_mib * 1024 * 1024) or args.cache_mib < 1 / (1024 * 1024)):
        p.error("Cache and volume-group sizes must be positive; cache-mib must fit at least one byte")
    if args.sampling_mode == "volume_grouped" and args.volume_group_size > args.cache_volumes:
        p.error("volume-group-size must not exceed cache-volumes")
    if args.seed < 0 or args.seed >= 2**32:
        p.error("seed must be in [0, 2**32) for NumPy and local data generators")
    if any(not math.isfinite(v) or v <= 0 for v in (args.lr, args.lr_disc)) or not math.isfinite(args.adv_weight) or args.adv_weight < 0:
        p.error("Learning rates must be positive; adv-weight nonnegative")
    try:
        progress = TrainingProgress(args.epochs, args.warmup_epochs, args.adv_weight)
    except ValueError as exc:
        p.error(str(exc))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = prepare_quality_rows(read_manifest(args.manifest, validate_inputs=False), args.manifest)
    data_independence = check_independence(rows, args, checkpoint_path=args.gen_checkpoint)
    consumed_rows = [row for row in rows if row["split"] in {"train", "val"}]
    data_qc = check_data_quality(consumed_rows, args, reference_id=args.reference_id)
    data_reference = data_qc.get("data_reference")
    consumed_rows = bind_training_units(consumed_rows, args, data_qc)
    normalization, normalization_source = resolve_training_normalization(
        args, consumed_rows, data_qc, data_independence)
    norm_metadata = normalization_metadata(normalization)
    cache_kwargs = dict(cache_items=args.cache_volumes, cache_bytes=int(args.cache_mib * 1024 * 1024))
    train = PairedSlices(consumed_rows, "train", args.size, normalization, **cache_kwargs)
    val = PairedSlices(consumed_rows, "val", args.size, normalization, **cache_kwargs)
    # Loader bookkeeping and sampling have their own RNG, leaving model Dropout
    # RNG unaffected by iterator construction or sampler length inspection.
    data_generator = torch.Generator().manual_seed(args.seed)
    slice_generator = torch.Generator().manual_seed(args.seed)
    validation_generator = torch.Generator().manual_seed(args.seed)
    batch_sampler = (CacheAwareBatchSampler(train, args.batch_size, group_volumes=args.volume_group_size, seed=args.seed)
                     if args.sampling_mode == "volume_grouped" else None)
    training_loader = (DataLoader(train, batch_sampler=batch_sampler, num_workers=0, generator=data_generator)
                       if batch_sampler is not None else
                       DataLoader(train, batch_size=args.batch_size, sampler=RandomSampler(train, generator=slice_generator),
                                  num_workers=0, generator=data_generator))
    loaders = [training_loader, DataLoader(val, batch_size=args.batch_size, shuffle=False,
                                         num_workers=0, generator=validation_generator)]
    sampling_policy = {
        "schema": "swixqsm.slice_sampling", "version": 1, "mode": args.sampling_mode,
        "seed": args.seed, "epoch_order": "local generator seeded seed+zero_based_epoch; no model RNG consumption",
        "coverage": "each effective slice exactly once per epoch; no oversampling or equal-subject weighting",
        "volume_group_size": args.volume_group_size if batch_sampler is not None else None,
        "batch_size_upper_bound": args.batch_size,
        "partial_batches": "one tail per volume group" if batch_sampler is not None else "one epoch tail",
        "validation_order": "unchanged sequential dataset index", "num_workers": 0,
        "cache": {**cache_kwargs, "scope": "per dataset instance; train/val independent",
                  "budget": "resident prepared arrays only, not total RAM/RSS; no disk cache",
                  "capacity_caveat": "If group bytes exceed budget, eviction/bypass can cause repeat decodes; consult I/O counters"},
        "input_lifetime": "immutable source files/metadata for dataset lifetime; rebuild after any input change",
        "training_scope": "future runs only; changed batch composition/RNG isolation does not reproduce historical optimization",
        "io_counter_scope": "dataset indexing and traversal only; excludes upstream QC/independence scans and OS physical disk traffic",
    }
    device = torch.device(args.device)
    gen = UNetSmall(normalize=False)
    if args.gen_checkpoint:
        load_generator(gen, args.gen_checkpoint, args.reset_head)
    gen.to(device)
    disc = PatchDiscriminator().to(device)
    training_objective = {
        "schema": "swixqsm.training_objective", "version": 1,
        "generator": "masked_l1 + actual_adv_weight * adversarial_bce",
        "discriminator": "0.5 * (real_bce + fake_bce)",
        "l1_reduction": "sum(abs(pred-target)*mask)/sum(mask); normalized QSM values",
        "adversarial": adversarial_spec(disc, args.adversarial_mode),
        "scope": "This run only; no claim about the objective used for source checkpoints",
    }
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_disc, betas=(0.5, 0.999))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update(torch_version=str(torch.__version__), numpy_version=np.__version__,
                  generator_output="linear", normalization=norm_metadata,
                  qsm_scale=normalization.scale, qsm_offset=normalization.offset,
                  normalization_source=normalization_source,
                  manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  data_qc=data_qc, data_reference=data_reference,
                  data_independence=data_independence,
                  initialization=data_independence.get("initialization"),
                  training_objective=training_objective,
                  sampling_policy=sampling_policy,
                  indexing_io={"train": train.io_stats, "val": val.io_stats},
                  diagnostics_policy={"max_distribution_batches": args.diagnostic_batches,
                                      "sampling": "first bounded batches of each epoch's shuffled training loader",
                                      "distribution_support": "positive training validity mask; not the entire input brain",
                                      "representativeness": "bounded early batches, possibly few volume groups; not a random whole-epoch distribution sample",
                                      "loss_ratio": "ratio of mean batch weighted adversarial loss to mean batch L1; not a gradient ratio"},
                  selection_policy={"split": "val", "metric": "stored participant-equal MAE" if args.units == "stored" else "physical participant-equal MAE",
                                    "global_checkpoint": "best.pt may belong to warmup; consult training_state",
                                    "phase_checkpoints": "best_l1.pt, best_warmup.pt, best_gan.pt when that phase runs",
                                    "phase_scope": "current run only; initialization exposure recorded separately"})
    if args.gen_checkpoint:
        config["initial_checkpoint_sha256"] = hashlib.sha256(Path(args.gen_checkpoint).read_bytes()).hexdigest()
    (output/"config.json").write_text(json.dumps(config, indent=2))
    (output/"manifest_resolved.json").write_text(json.dumps(consumed_rows, indent=2))
    for epoch in range(args.epochs):
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        data_generator.manual_seed(args.seed + epoch)
        slice_generator.manual_seed(args.seed + epoch)
        io_before = {"train": train.io_stats, "val": val.io_stats}
        gen.train()
        if args.freeze_bn:
            for module in gen.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
        disc.train()
        weight = progress.weight(epoch)
        diagnostics = EpochDiagnostics(max_distribution_batches=args.diagnostic_batches)
        for batch_index, (x, y, mask, subjects) in enumerate(loaders[0]):
            _, _, step = train_step(gen, disc, opt_g, opt_d, *(a.to(device) for a in (x, y, mask)), weight,
                                    adversarial_mode=args.adversarial_mode, return_diagnostics=True,
                                    diagnostics_normalization=normalization if batch_index < args.diagnostic_batches else None)
            progress.complete_step(epoch)
            diagnostics.update(step["fields"])
            if step["distribution"] is not None:
                step["distribution"].update(training_batch_index=batch_index, subject_ids=list(subjects))
                diagnostics.add_distribution(step["distribution"])
        training_diagnostics = diagnostics.summary()
        score, per_patient = validate(gen, loaders[1], device, normalization)
        training_io = {"epoch_index": epoch, "sampling_mode": args.sampling_mode,
                       "train": epoch_io(io_before["train"], train.io_stats),
                       "val": epoch_io(io_before["val"], val.io_stats)}
        if not math.isfinite(score):
            raise RuntimeError("Nonfinite validation MAE; refusing to save model")
        improved_files = progress.record_validation(epoch, score)
        training_state = progress.snapshot(epoch, units=args.units)
        best = training_state["global_best"]["score"]
        state = dict(generator=gen.state_dict(), discriminator=disc.state_dict(),
                     optimizer_g=opt_g.state_dict(), optimizer_d=opt_d.state_dict(),
                     epoch=epoch+1, val_mae=score, best_val_mae=best,
                     best_val_mae_scope="global_current_run", config=config, normalization=norm_metadata,
                     data_qc=data_qc, data_reference=data_reference,
                     data_independence=data_independence, training_state=training_state,
                     training_objective=training_objective, training_diagnostics=training_diagnostics,
                     training_io=training_io, sampling_policy=sampling_policy,
                     torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        for name in ["last.pt", *improved_files]:
            temp = output/(name+".tmp")
            role = ("last" if name == "last.pt" else "global_best" if name == "best.pt"
                    else f"current_run_{training_state['current_run_phase']}_best")
            torch.save({**state, "checkpoint_role": role, "checkpoint_selection_score": score}, temp)
            temp.replace(output/name)
        selection = {"training_state": training_state, "data_independence": data_independence,
                     "initialization": config["initialization"], "selection_policy": config["selection_policy"],
                     "training_objective": training_objective, "training_diagnostics": training_diagnostics,
                     "sampling_policy": sampling_policy, "training_io": training_io}
        selection_temp = output / "selection.json.tmp"
        selection_temp.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
        selection_temp.replace(output / "selection.json")
        log = dict(epoch=epoch+1, val_mae=score, units=args.units, patient_mae=per_patient, adv_weight=weight,
                   current_run_phase=training_state["current_run_phase"],
                   generator_updates=training_state["generator_updates"],
                   discriminator_updates=training_state["discriminator_updates"],
                   adversarial_generator_updates=training_state["adversarial_generator_updates"],
                   adversarial_mode=args.adversarial_mode, training_diagnostics=training_diagnostics,
                   training_io=training_io)
        with (output/"history.jsonl").open("a") as f:
            f.write(json.dumps(log, allow_nan=False)+"\n")
        print(f"Epoch {epoch+1}: validation patient MAE={score:.6g} {args.units}", flush=True)


if __name__ == "__main__":
    main()

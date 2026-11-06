"""Prospective reviewer comparisons; select a model without changing source code.

Run ``python -m swixqsm_minimal.train --list-models``. This entry point does
not reinterpret historical checkpoints or start training when listing recipes.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from .model.training_models import create_model
from .util.comparison_engine import comparison_step, objective_spec
from .util.data import PairedSlices, read_manifest
from .util.engine import validate
from .util.independence_gate import add_independence_arguments, check_independence
from .util.normalization import normalization_metadata
from .util.normalization_workflow import (add_training_normalization_arguments, bind_training_units,
                                          resolve_training_normalization)
from .util.quality_gate import check_data_quality, prepare_quality_rows
from .util.sampling import CacheAwareBatchSampler
from .util.training_diagnostics import EpochDiagnostics
from .util.training_state import TrainingProgress


RECIPES = {
    "swixqsm_gan": ("swixqsm_gan", .2, 1.),
    "fcn_gan_l1": ("fcn_gan_l1", .2, 1.),
    "plain_unet": ("plain_unet", 1., 0.),
    "pix2pix": ("pix2pix", 100., 1.),
    "swixqsm_l1_ablation": ("swixqsm_gan", .2, 0.),
    "swixqsm_lambda1": ("swixqsm_gan", 1., 1.),
    "swixqsm_lambda10": ("swixqsm_gan", 10., 1.),
    "swixqsm_lambda100": ("swixqsm_gan", 100., 1.),
    # Revision Wave C "C2": same SWIXQSM generator, strong L1 + weak adversarial term.
    "swixqsm_c2": ("swixqsm_gan", 100., .1),
}
FOUR_MODELS = tuple(RECIPES)[:4]
DEFAULTS = dict(size=416, batch_size=4, epochs=100, lr=1e-4, lr_disc=1e-4,
                lr_policy="cosine", min_lr=0., warmup_epochs=0, seed=42,
                plateau_factor=.5, plateau_patience=5, plateau_threshold=0.,
                qsm_offset=None, sampling_mode="volume_grouped", cache_volumes=2,
                cache_mib=256., volume_group_size=2, diagnostic_batches=4,
                adversarial_mode="valid_patch", early_stopping_patience=0,
                deterministic=True, num_threads=None)
OPERATIONAL = {"resume", "device", "max_epochs_this_run", "list_models", "output", "model"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def source_hashes():
    root = Path(__file__).resolve().parent
    return {str(f.relative_to(root)): sha256(f) for f in sorted(root.rglob("*.py"))
            if "test" not in f.relative_to(root).parts and "__pycache__" not in f.parts
            and f.name != "test_smoke.py"}


def runtime_versions():
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, nibabel=nib.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version())


def input_stat_signature(rows):
    """Cheap change detector; verified QC separately binds actual file hashes."""
    result = {}
    for row in rows:
        for key in ("swi_path", "qsm_path", "swi_mask_path", "mask_path"):
            path = Path(row[key])
            stat = path.stat()
            result[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def tensor_state_sha256(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0., .02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) and module.affine:
        nn.init.normal_(module.weight, 1., .02)
        nn.init.zeros_(module.bias)


class GlobalParticipantSlices(PairedSlices):
    """Keep image preparation unchanged, aggregate visits under global identity."""
    def __getitem__(self, index):
        x, y, mask, _ = super().__getitem__(index)
        volume_index, _ = self.index[index]
        return x, y, mask, self.rows[volume_index]["global_subject_id"]


class ObservedBatches:
    """Hash actual yielded index batches without an extra sampling traversal."""
    def __init__(self, source):
        self.source, self.digest, self.n_batches = source, None, 0

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        digest = hashlib.sha256()
        self.n_batches = 0
        for batch in self.source:
            batch = [int(i) for i in batch]
            digest.update(json.dumps(batch, separators=(",", ":")).encode() + b"\n")
            self.n_batches += 1
            yield batch
        self.digest = digest.hexdigest()


def identity_checks(rows):
    groups = {}
    for row in rows:
        gid = row.get("global_subject_id", "").strip()
        if not gid:
            raise ValueError("New comparisons require a nonempty global_subject_id in every row")
        row["global_subject_id"] = gid
        if gid in groups and groups[gid] != row["split"]:
            raise ValueError("Global participant crosses train/val/test: " + gid)
        groups[gid] = row["split"]
    if not {"train", "val"}.issubset(set(groups.values())):
        raise ValueError("New training needs participant-disjoint train AND val; split original train first. The 1773-person test set cannot select checkpoints.")


def rng_state():
    name, keys, position, gaussian, cached = np.random.get_state()
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=[name, keys.tolist(), position, gaussian, cached],
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(value):
    torch.set_rng_state(value["torch"])
    random.setstate(value["python"])
    name, keys, position, gaussian, cached = value["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    if value["cuda"] is not None and torch.cuda.is_available():
        if len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device inventory changed; exact RNG restoration is unavailable")
        torch.cuda.set_rng_state_all(value["cuda"])


def restore_progress(progress, history):
    for index, record in enumerate(history):
        if record["epoch"] != index + 1 or record["n_train_batches"] < 1:
            raise ValueError("Resume history is not a complete ordered epoch sequence")
        for _ in range(record["n_train_batches"]):
            progress.complete_step(index)
        progress.record_validation(index, record["val_mae"])
    return progress


def early_stopping_reached(state, patience):
    """Count nonimproving epochs within L1 or GAN, excluding warmup."""
    phase = state["current_run_phase"]
    if not patience or phase == "warmup":
        return False
    best = state["best_by_phase"][phase]
    return best is not None and state["epoch"] - best["epoch"] >= patience


def make_lr_scheduler(optimizer, args, *, period):
    """Use one scheduler per optimizer; plateau observes physical val MAE."""
    if args.lr_policy == "constant":
        return None
    if args.lr_policy == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, period, eta_min=args.min_lr)
    if args.lr_policy == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.plateau_factor,
            patience=args.plateau_patience, threshold=args.plateau_threshold,
            threshold_mode="abs", min_lr=args.min_lr)
    raise ValueError("Unknown learning-rate policy: " + args.lr_policy)


def step_epoch_schedulers(args, epoch, score, training_state, opt_g, sched_g, sched_d):
    """Advance after validation, returning the possibly phase-reset G scheduler.

    The epoch-boundary rule needs no transient flag: a resumed first GAN epoch
    resets exactly as an uninterrupted run does. Resetting plateau statistics
    deliberately keeps the optimizer's current LR, including warmup reductions.
    """
    if not math.isfinite(score):
        raise RuntimeError("Nonfinite validation MAE; refusing to advance learning-rate schedulers")
    reset = (args.lr_policy == "plateau" and sched_g is not None
             and args.warmup_epochs > 0 and epoch == args.warmup_epochs
             and training_state["current_run_phase"] == "gan")
    if reset:
        sched_g = make_lr_scheduler(opt_g, args, period=args.epochs)
    d_updated = training_state["current_epoch_updates"]["discriminator_updates"] > 0
    event = dict(policy=args.lr_policy, generator_stepped=sched_g is not None,
                 discriminator_stepped=sched_d is not None and d_updated,
                 generator_reset=reset, reset_reason="warmup_to_gan" if reset else None)
    for scheduler, active in ((sched_g, sched_g is not None),
                              (sched_d, sched_d is not None and d_updated)):
        if active:
            scheduler.step(score) if args.lr_policy == "plateau" else scheduler.step()
    return sched_g, sched_d, event


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=[*RECIPES, "all", "reviewer_suite"])
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--manifest")
    p.add_argument("--output", help="New experiment directory; with --resume, its existing directory")
    p.add_argument("--resume", help="Only this new entry point's last.pt; epoch-boundary continuation")
    p.add_argument("--units", choices=["ppm", "ppb", "stored"])
    add_training_normalization_arguments(p)
    p.add_argument("--reference-id")
    for name in ("size", "batch-size", "epochs", "warmup-epochs", "seed", "cache-volumes",
                 "volume-group-size", "diagnostic-batches", "early-stopping-patience", "num-threads",
                 "max-epochs-this-run", "plateau-patience"):
        p.add_argument("--" + name, type=int)
    for name in ("lr", "lr-disc", "min-lr", "cache-mib", "lambda-l1", "lambda-adv", "plateau-factor", "plateau-threshold"):
        p.add_argument("--" + name, type=float)
    p.add_argument("--lr-policy", choices=["constant", "cosine", "plateau"])
    p.add_argument("--sampling-mode", choices=["volume_grouped", "slice_random"])
    p.add_argument("--adversarial-mode", choices=["valid_patch", "all_patches"])
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--device")
    quality = p.add_mutually_exclusive_group()
    quality.add_argument("--qc-report", help="Hash-bound report from swixqsm_minimal.qc")
    quality.add_argument("--allow-unverified-data", metavar="REASON", help="Explicit diagnostic opt-out, never QC passed")
    add_independence_arguments(p, required=False)
    return p, p.parse_args(argv)


def resolve_args(p, args, resume=None):
    original = vars(args).copy()
    previous = {} if resume is None else resume["config"]["arguments"]
    values = {key: value if value is not None else previous.get(key, DEFAULTS.get(key))
              for key, value in original.items()}
    # These operational fields must never silently inherit a previous segment.
    values["max_epochs_this_run"] = original["max_epochs_this_run"]
    values["resume"] = original["resume"]
    values["device"] = original["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(values["device"])
    if device.type not in {"cpu", "cuda"}:
        p.error("Comparison training supports CPU or CUDA; RNG replay for other backends is not implemented")
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is unavailable in this environment")
    values["model"] = original["model"] or previous.get("model")
    values["output"] = original["output"] or (str(Path(args.resume).resolve().parent) if args.resume else None)
    if (any(values.get(key) is None for key in ("model", "manifest", "output", "units"))
            or (values.get("qsm_scale") is None) == (values.get("normalization_file") is None)):
        p.error("Specify --model, --manifest, --output, --units and exactly one of --normalization-file/--qsm-scale (or --resume)")
    if values.get("normalization_file") and values.get("qsm_offset") is not None:
        p.error("--qsm-offset cannot accompany --normalization-file")
    for key in ("manifest", "output", "qc_report", "independence_report", "resume", "normalization_file"):
        if values.get(key):
            values[key] = str(Path(values[key]).resolve())
    for a, b in (("qc_report", "allow_unverified_data"), ("independence_report", "allow_unverified_independence")):
        if bool(values.get(a)) == bool(values.get(b)):
            p.error(f"Supply exactly one of --{a.replace('_','-')} or --{b.replace('_','-')}")
    for key in ("epochs", "batch_size", "cache_volumes", "volume_group_size"):
        if values[key] < 1:
            p.error(key + " must be positive")
    for key in ("warmup_epochs", "diagnostic_batches", "early_stopping_patience", "plateau_patience"):
        if values[key] < 0:
            p.error(key + " must be nonnegative")
    if values["size"] < 32 or values["size"] % 8 or not 0 <= values["seed"] < 2**32:
        p.error("size must be >=32 and divisible by 8; seed in [0,2**32)")
    if values["sampling_mode"] == "volume_grouped" and values["volume_group_size"] > values["cache_volumes"]:
        p.error("volume-group-size must fit cache-volumes")
    for key in ("lr", "lr_disc", "cache_mib"):
        if not math.isfinite(values[key]) or values[key] <= 0:
            p.error(key + " must be finite and positive")
    if not math.isfinite(values["cache_mib"] * 1024 * 1024) or values["cache_mib"] * 1024 * 1024 < 1:
        p.error("cache-mib must fit at least one byte and a finite budget")
    if not math.isfinite(values["min_lr"]) or not 0 <= values["min_lr"] <= min(values["lr"], values["lr_disc"]):
        p.error("min-lr must lie between zero and both initial learning rates")
    if not math.isfinite(values["plateau_factor"]) or not 0 < values["plateau_factor"] < 1:
        p.error("plateau-factor must be finite and strictly between zero and one")
    if not math.isfinite(values["plateau_threshold"]) or values["plateau_threshold"] < 0:
        p.error("plateau-threshold must be finite and nonnegative, in training physical units")
    for key in ("lambda_l1", "lambda_adv"):
        if values[key] is not None and (not math.isfinite(values[key]) or values[key] < 0):
            p.error(key + " must be finite and nonnegative")
    if values["num_threads"] is not None and values["num_threads"] < 1:
        p.error("num-threads must be positive")
    if values["max_epochs_this_run"] is not None and values["max_epochs_this_run"] < 1:
        p.error("max-epochs-this-run must be positive")
    if resume is not None:
        if values["model"] != previous["model"] or values["output"] != previous["output"]:
            p.error("Resume must keep the exact recipe and experiment directory")
        conflicts = [key for key in previous if key not in OPERATIONAL and values.get(key) != previous[key]]
        if conflicts:
            p.error("Resume cannot change the training protocol: " + ", ".join(conflicts))
    return argparse.Namespace(**values)


def run_one(args, *, resume=None):
    architecture, default_l1, default_adv = RECIPES[args.model]
    l1 = default_l1 if args.lambda_l1 is None else args.lambda_l1
    adv = default_adv if args.lambda_adv is None else args.lambda_adv
    if architecture == "plain_unet" and adv != 0:
        raise ValueError("plain_unet has no discriminator; lambda-adv must be zero")
    if args.model == "swixqsm_l1_ablation" and (l1 != .2 or adv != 0):
        raise ValueError("The strict .2-L1/zero-adversarial recipe is fixed; use swixqsm_gan to override coefficients")
    progress = TrainingProgress(args.epochs, args.warmup_epochs, adv)
    output = Path(args.output)
    if resume is None and output.exists():
        raise FileExistsError("Experiment already exists; choose a new output or explicitly resume last.pt")
    manifest_hash = sha256(args.manifest)
    code = source_hashes()
    if resume is not None and (manifest_hash != resume["config"]["manifest_sha256"]
                               or code != resume["config"]["source_sha256"]):
        raise ValueError("Resume manifest or training source changed")
    if resume is not None and runtime_versions() != resume["config"].get("runtime_versions"):
        raise ValueError("Resume runtime versions changed; numeric replay is not established across environments")
    rows = prepare_quality_rows(read_manifest(args.manifest, validate_inputs=False), args.manifest)
    identity_checks(rows)
    independence = check_independence(rows, args, checkpoint_path=None)
    consumed = [r for r in rows if r["split"] in {"train", "val"}]
    signatures = input_stat_signature(consumed)
    if resume is not None and signatures != resume["config"].get("input_stat_signature"):
        raise ValueError("Resume training/validation input file metadata changed")
    qc = check_data_quality(consumed, args, reference_id=args.reference_id)
    consumed = bind_training_units(consumed, args, qc)
    normalization, normalization_source = resolve_training_normalization(args, consumed, qc, independence)
    if resume is not None and (normalization_metadata(normalization) != resume["normalization"]
            or normalization_source != resume["config"].get("normalization_source", normalization_source)):
        raise ValueError("Resume normalization parameters or fitting artifact changed")
    seed_all(args.seed)
    gen, disc, model_spec = create_model(architecture, image_size=args.size)
    # Separate initialization streams prevent the presence of D from changing
    # the initial generator or the post-initialization training Dropout stream.
    seed_all(args.seed)
    gen.apply(initialize)
    if disc is not None:
        seed_all((args.seed + 104729) % 2**32)
        disc.apply(initialize)
    initial_sha = tensor_state_sha256(gen.state_dict())
    model_spec["initialization"] = {"policy": "common normal(0,.02) Conv/ConvTranspose/Linear; BN normal(1,.02); zero biases",
                                     "generator_seed": args.seed, "discriminator_seed": (args.seed + 104729) % 2**32,
                                     "generator_initial_state_sha256": initial_sha,
                                     "pretrained": False}
    seed_all(args.seed)
    device = torch.device(args.device)
    gen.to(device)
    if disc is not None:
        disc.to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(.5, .999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_disc, betas=(.5, .999)) if disc is not None else None
    sched_g = make_lr_scheduler(opt_g, args, period=args.epochs)
    sched_d = (make_lr_scheduler(opt_d, args, period=args.epochs - args.warmup_epochs)
               if opt_d is not None and adv > 0 else None)
    objective = objective_spec(disc, model_spec, l1, adv, args.adversarial_mode, normalization.scale, units=args.units)
    cache = dict(cache_items=args.cache_volumes, cache_bytes=int(args.cache_mib * 1024 * 1024))
    train = GlobalParticipantSlices(consumed, "train", args.size, normalization, **cache)
    val = GlobalParticipantSlices(consumed, "val", args.size, normalization, **cache)
    sampler_rng = torch.Generator().manual_seed(args.seed)
    sampler = (CacheAwareBatchSampler(train, args.batch_size, group_volumes=args.volume_group_size, seed=args.seed)
               if args.sampling_mode == "volume_grouped" else
               BatchSampler(RandomSampler(train, generator=sampler_rng), args.batch_size, drop_last=False))
    observed = ObservedBatches(sampler)
    loader_rng, val_rng = torch.Generator().manual_seed(args.seed), torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(train, batch_sampler=observed, num_workers=0, generator=loader_rng)
    validation = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0, generator=val_rng)
    norm_spec = normalization_metadata(normalization)
    config = dict(arguments=vars(args).copy(), **vars(args))
    config.update(schema="swixqsm.comparison_training_config", version=1, recipe=args.model,
                  model_name=architecture, model_spec=model_spec, generator_output="linear", normalization=norm_spec,
                  qsm_scale=normalization.scale, qsm_offset=normalization.offset,
                  normalization_source=normalization_source,
                  manifest_sha256=manifest_hash, source_sha256=code, torch_version=str(torch.__version__),
                  runtime_versions=runtime_versions(), input_stat_signature=signatures,
                  input_stat_limitation="Size/mtime detect ordinary changes only; verified QC supplies content hashes. Opt-out data remain unverified.",
                  numpy_version=np.__version__, nibabel_version=nib.__version__, precision="float32",
                  data_qc=qc, data_reference=qc.get("data_reference"), data_independence=independence,
                  initialization=independence["initialization"], training_objective=objective,
                  sampling_policy={"mode": args.sampling_mode, "seed": args.seed, "num_workers": 0,
                      "group_volumes": args.volume_group_size, "cache": cache,
                      "coverage": "every valid slice once per epoch; global participants used for validation aggregation",
                      "trace": "SHA256 of actually yielded dataset-index batches in history; no model RNG use"},
                  augmentation={"enabled": False, "policy": "none for all comparisons; shuffling is not contrast/orientation augmentation"},
                  scheduler_policy={"name": args.lr_policy, "generator": "once per completed epoch",
                      "discriminator": "only after an epoch with actual D updates; cosine period epochs minus warmup",
                      "strict_adversarial_ablation": "no discriminator scheduler", "min_lr": args.min_lr,
                      "monitor": {"split": "val", "metric": "stored_participant_equal_mae" if args.units == "stored" else "physical_participant_equal_mae", "units": args.units, "mode": "min"}
                          if args.lr_policy == "plateau" else None,
                      "plateau": {"factor": args.plateau_factor, "patience": args.plateau_patience,
                          "threshold": args.plateau_threshold, "threshold_mode": "abs",
                          "generator_phase_reset": "reset plateau history once at warmup_to_gan; retain current optimizer LR",
                          "discriminator_phase_reset": "none; warmup never advances D scheduler"}
                          if args.lr_policy == "plateau" else None},
                  selection_policy={"split": "val", "metric": "stored_participant_equal_mae" if args.units == "stored" else "physical_participant_equal_mae", "mode": "min",
                      "test_image_access": False, "best": "global best.pt plus separately best_l1/warmup/gan.pt",
                      "early_stopping_patience": args.early_stopping_patience,
                      "early_stopping_scope": "within current L1 or GAN phase; warmup excluded"},
                  scope="Prospective revised training, not an exact historical optimization or output reproduction")
    history, start_epoch = [], 0
    if resume is not None:
        history = resume["history"]
        if (len(history) != resume["epoch"] or sha256(output / "history.jsonl") != resume["history_sha256"]
                or [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()] != history):
            raise ValueError("Resume history is incomplete or differs from last committed checkpoint")
        restore_progress(progress, history)
        if progress.snapshot(len(history) - 1, units=args.units) != resume["training_state"]:
            raise ValueError("Resume progress counters/selection differ from committed history")
        if model_spec != resume["model_spec"] or objective != resume["training_objective"]:
            raise ValueError("Resume architecture, initialization or objective differs")
        if early_stopping_reached(resume["training_state"], args.early_stopping_patience):
            raise ValueError("The prespecified validation early-stopping rule was already reached; resume cannot override it")
        gen.load_state_dict(resume["generator"], strict=True)
        opt_g.load_state_dict(resume["optimizer_g"])
        if disc is not None:
            disc.load_state_dict(resume["discriminator"], strict=True)
            opt_d.load_state_dict(resume["optimizer_d"])
        if sched_g is not None:
            sched_g.load_state_dict(resume["scheduler_g"])
            if sched_d is not None:
                sched_d.load_state_dict(resume["scheduler_d"])
        restore_rng(resume["rng"])
        start_epoch = resume["epoch"]
        if start_epoch >= args.epochs:
            raise ValueError("This experiment already reached its requested epoch budget")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "config.json", config)
        atomic_json(output / "manifest_resolved.json", consumed)
    atomic_json(output / "active_segment.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                "resume_checkpoint_sha256": sha256(args.resume) if args.resume else None,
                "start_epoch": start_epoch, "device": str(device), "arguments": vars(args)})
    stop_reason = "epoch_budget_completed"
    for epoch in range(start_epoch, args.epochs):
        if isinstance(sampler, CacheAwareBatchSampler):
            sampler.set_epoch(epoch)
        sampler_rng.manual_seed(args.seed + epoch)
        loader_rng.manual_seed(args.seed + epoch)
        val_rng.manual_seed(args.seed + epoch + 1)
        gen.train()
        diagnostics = EpochDiagnostics(args.diagnostic_batches)
        learning_rates = {"generator": opt_g.param_groups[0]["lr"],
                          "discriminator": opt_d.param_groups[0]["lr"] if opt_d else None}
        io_before = {"train": train.io_stats, "val": val.io_stats}
        for batch_index, (x, y, mask, identities) in enumerate(loader):
            step = comparison_step(gen, disc, opt_g, opt_d, *(t.to(device) for t in (x, y, mask)),
                                   l1, progress.weight(epoch), adversarial_mode=args.adversarial_mode,
                                   normalization=normalization if batch_index < args.diagnostic_batches else None)
            if not step["generator_updated"] or step["discriminator_updated"] != (progress.weight(epoch) > 0):
                raise RuntimeError("Observed optimizer updates disagree with requested training phase")
            progress.complete_step(epoch)
            diagnostics.update(step["fields"])
            if step["distribution"] is not None:
                step["distribution"].update(training_batch_index=batch_index, subject_ids=list(identities))
                diagnostics.add_distribution(step["distribution"])
        score, per_person = validate(gen, validation, device, normalization)
        if not math.isfinite(score) or any(not math.isfinite(v) for v in per_person.values()):
            raise RuntimeError("Nonfinite validation MAE; refusing to checkpoint this epoch")
        improved = progress.record_validation(epoch, score)
        training_state = progress.snapshot(epoch, units=args.units)
        sched_g, sched_d, scheduler_event = step_epoch_schedulers(
            args, epoch, score, training_state, opt_g, sched_g, sched_d)
        record = dict(epoch=epoch + 1, recipe=args.model, model_name=architecture, val_mae=score,
                      units=args.units, patient_mae=per_person, n_train_batches=observed.n_batches,
                      sampling_index_sha256=observed.digest, learning_rates=learning_rates,
                      learning_rates_next={"generator": opt_g.param_groups[0]["lr"],
                          "discriminator": opt_d.param_groups[0]["lr"] if opt_d else None},
                      scheduler_event=scheduler_event,
                      lambda_l1=l1, lambda_adv=progress.weight(epoch),
                      training_state=training_state, training_diagnostics=diagnostics.summary(),
                      io_before=io_before, io_after={"train": train.io_stats, "val": val.io_stats})
        history.append(record)
        # A complete history is atomically committed. If a later disk write
        # fails, strict resume refuses an inconsistent checkpoint/history pair.
        history_path = output / "history.jsonl"
        temp = output / "history.jsonl.tmp"
        temp.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in history))
        temp.replace(history_path)
        state = dict(schema="swixqsm.comparison_checkpoint", version=1, generator=gen.state_dict(),
                     discriminator=disc.state_dict() if disc is not None else None,
                     optimizer_g=opt_g.state_dict(), optimizer_d=opt_d.state_dict() if opt_d else None,
                     scheduler_g=sched_g.state_dict() if sched_g else None,
                     scheduler_d=sched_d.state_dict() if sched_d else None,
                     epoch=epoch + 1, val_mae=score, config=config, model_spec=model_spec,
                     normalization=norm_spec, data_qc=qc, data_reference=qc.get("data_reference"),
                     data_independence=independence, training_state=training_state,
                     training_objective=objective, training_diagnostics=record["training_diagnostics"],
                     history=history, history_sha256=sha256(history_path), rng=rng_state())
        for filename in [*improved, "last.pt"]:
            role = "last" if filename == "last.pt" else "global_best" if filename == "best.pt" else "current_run_" + filename[5:-3] + "_best"
            temp = output / (filename + ".tmp")
            torch.save(dict(state, checkpoint_role=role, checkpoint_selection_score=score), temp)
            temp.replace(output / filename)
        atomic_json(output / "selection.json", dict(training_state=training_state, model_spec=model_spec,
                    training_objective=objective, data_independence=independence, selection_policy=config["selection_policy"]))
        print(f"[{args.model}] epoch {epoch + 1}/{args.epochs}: val participant MAE={score:.7g} {args.units}; "
              f"phase={training_state['current_run_phase']}; G/D updates={training_state['generator_updates']}/{training_state['discriminator_updates']}", flush=True)
        if early_stopping_reached(training_state, args.early_stopping_patience):
            stop_reason = "validation_patience_exhausted"
            break
        if args.max_epochs_this_run is not None and epoch + 1 - start_epoch >= args.max_epochs_this_run and epoch + 1 < args.epochs:
            stop_reason = "segment_epoch_budget; resume last.pt with the original total epochs"
            break
    result = dict(recipe=args.model, model_name=architecture, directory=str(output),
                  epochs_completed=len(history), stop_reason=stop_reason,
                  generator_initial_state_sha256=initial_sha, training_state=training_state,
                  best_checkpoint=str(output / "best.pt"), formal_test_evaluation_performed=False)
    atomic_json(output / "run_result.json", result)
    return result


def main(argv=None):
    p, initial = parse_args(argv)
    if initial.list_models:
        print(json.dumps({"recipes": {name: {"architecture": values[0], "lambda_l1": values[1], "lambda_adv": values[2]}
                                      for name, values in RECIPES.items()},
                          "all": list(FOUR_MODELS), "reviewer_suite": list(RECIPES),
                          "loss_value_space": "fixed normalized QSM; physical L1 coefficient = lambda_l1/qsm_scale",
                          "non_neural_baselines": "zero + train_mean + voxelwise linear via fit_baselines/evaluate"}, indent=2))
        return
    resume = torch.load(initial.resume, map_location="cpu", weights_only=True) if initial.resume else None
    if resume is not None and (resume.get("schema") != "swixqsm.comparison_checkpoint" or resume.get("version") != 1
                               or resume.get("checkpoint_role") != "last"):
        p.error("Resume requires a last.pt produced by this comparison entry point")
    args = resolve_args(p, initial, resume)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.deterministic
    if args.model not in {"all", "reviewer_suite"}:
        return run_one(args, resume=resume)
    if resume is not None:
        p.error("Resume individual last.pt experiments, not a multi-model suite")
    if args.lambda_l1 is not None or args.lambda_adv is not None:
        p.error("Suite recipes have fixed distinct coefficients; override coefficients in a named single-model experiment")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    names = FOUR_MODELS if args.model == "all" else tuple(RECIPES)
    atomic_json(root / "suite_plan.json", {"schema": "swixqsm.comparison_suite", "version": 1,
                "recipes": list(names), "manifest_sha256": sha256(args.manifest), "seed": args.seed,
                "execution": "serial, one selected model at a time", "test_selection": False})
    results = []
    for name in names:
        one = argparse.Namespace(**vars(args))
        one.model, one.output = name, str(root / name)
        results.append(run_one(one))
        atomic_json(root / "suite_progress.json", {"planned": list(names), "completed": results})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    main()

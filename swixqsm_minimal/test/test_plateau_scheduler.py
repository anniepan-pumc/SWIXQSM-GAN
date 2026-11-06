"""Physical validation-driven LR scheduling, using synthetic data only."""
import argparse
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import torch
from torch import nn

from .. import train_models


def scheduler_args(**changes):
    values = dict(lr_policy="plateau", epochs=8, warmup_epochs=0, min_lr=.01,
                  plateau_factor=.5, plateau_patience=0, plateau_threshold=0.)
    values.update(changes)
    return argparse.Namespace(**values)


class PlateauSchedulerTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def schedulers(self, args):
        g = torch.optim.Adam([nn.Parameter(torch.tensor(1.))], lr=.1)
        d = torch.optim.Adam([nn.Parameter(torch.tensor(1.))], lr=.1)
        return (g, d, train_models.make_lr_scheduler(g, args, period=args.epochs),
                train_models.make_lr_scheduler(d, args, period=args.epochs - args.warmup_epochs))

    def advance(self, args, epoch, score, bundle, *, phase="gan", d_updates=1):
        g, d, sg, sd = bundle
        # Complete real optimizer updates before a cosine step as in run_one.
        for opt, active in ((g, True), (d, d_updates > 0)):
            if active:
                opt.param_groups[0]["params"][0].grad = torch.tensor(.1)
                opt.step()
                opt.zero_grad()
        state = {"current_run_phase": phase,
                 "current_epoch_updates": {"discriminator_updates": d_updates}}
        sg, sd, event = train_models.step_epoch_schedulers(args, epoch, score, state, g, sg, sd)
        return (g, d, sg, sd), event

    def test_stall_reduces_improvement_retains_and_minimum_is_respected(self):
        args = scheduler_args(plateau_patience=1, min_lr=.03)
        bundle = self.schedulers(args)
        actual = []
        for epoch, score in enumerate([1., 1., 1., .8, .8, .8, .8, .8]):
            bundle, event = self.advance(args, epoch, score, bundle)
            actual.append(bundle[0].param_groups[0]["lr"])
            self.assertTrue(event["discriminator_stepped"])
        self.assertEqual(actual, [.1, .1, .05, .05, .05, .03, .03, .03])
        self.assertEqual(bundle[1].param_groups[0]["lr"], .03)
        self.assertEqual(bundle[2].threshold_mode, "abs")

    def test_absolute_threshold_is_a_physical_mae_difference(self):
        args = scheduler_args(plateau_threshold=.1)
        bundle = self.schedulers(args)
        bundle, _ = self.advance(args, 0, 1., bundle)
        # A .05 improvement is below the requested .1 physical-unit threshold.
        bundle, _ = self.advance(args, 1, .95, bundle)
        self.assertEqual(bundle[0].param_groups[0]["lr"], .05)
        bundle, _ = self.advance(args, 2, .8, bundle)
        self.assertEqual(bundle[0].param_groups[0]["lr"], .05)

    def test_warmup_resets_g_history_once_retains_lr_and_d_skips_inactive_epochs(self):
        args = scheduler_args(warmup_epochs=2)
        bundle = self.schedulers(args)
        initial_d = copy.deepcopy(bundle[3].state_dict())
        for epoch in range(2):
            bundle, event = self.advance(args, epoch, .1, bundle, phase="warmup", d_updates=0)
            self.assertFalse(event["generator_reset"])
            self.assertFalse(event["discriminator_stepped"])
            self.assertEqual(bundle[3].state_dict(), initial_d)
        self.assertEqual(bundle[0].param_groups[0]["lr"], .05)
        bundle, event = self.advance(args, 2, 2., bundle)
        self.assertTrue(event["generator_reset"])
        self.assertEqual(event["reset_reason"], "warmup_to_gan")
        self.assertEqual(bundle[2].best, 2.)
        self.assertEqual(bundle[0].param_groups[0]["lr"], .05)
        self.assertEqual(bundle[1].param_groups[0]["lr"], .1)
        d_before = copy.deepcopy(bundle[3].state_dict())
        bundle, event = self.advance(args, 3, 2., bundle, d_updates=0)
        self.assertFalse(event["generator_reset"])
        self.assertFalse(event["discriminator_stepped"])
        self.assertEqual(bundle[3].state_dict(), d_before)
        self.assertEqual(bundle[0].param_groups[0]["lr"], .025)
        # L1-only training has no warmup-to-GAN reset, even with warmup > 0.
        l1_bundle = self.schedulers(args)
        for epoch in range(3):
            l1_bundle, event = self.advance(args, epoch, .1, l1_bundle, phase="l1", d_updates=0)
            self.assertFalse(event["generator_reset"])
        self.assertEqual(l1_bundle[0].param_groups[0]["lr"], .025)

    def test_constant_and_cosine_keep_their_existing_schedules(self):
        for policy in ("constant", "cosine"):
            args = scheduler_args(lr_policy=policy, epochs=4, warmup_epochs=2, min_lr=0.)
            bundle = self.schedulers(args)
            for epoch in range(4):
                bundle, event = self.advance(args, epoch, 100. - epoch, bundle,
                    phase="warmup" if epoch < 2 else "gan", d_updates=int(epoch >= 2))
                self.assertFalse(event["generator_reset"])
            expected = .1 if policy == "constant" else 0.
            self.assertAlmostEqual(bundle[0].param_groups[0]["lr"], expected)
            self.assertAlmostEqual(bundle[1].param_groups[0]["lr"], expected)

    def test_nonfinite_monitor_is_rejected_without_changing_scheduler(self):
        args = scheduler_args()
        bundle = self.schedulers(args)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            state = copy.deepcopy(bundle[2].state_dict())
            with self.assertRaisesRegex(RuntimeError, "Nonfinite validation MAE"):
                self.advance(args, 0, invalid, bundle)
            self.assertEqual(bundle[2].state_dict(), state)

    def test_cli_defaults_and_invalid_plateau_protocol(self):
        base = ["--model", "plain_unet", "--manifest", "/tmp/synthetic.csv", "--output", "/tmp/synthetic_exp",
                "--units", "ppm", "--qsm-scale", "1", "--allow-unverified-data", "synthetic",
                "--allow-unverified-independence", "synthetic"]
        parser, initial = train_models.parse_args(base)
        args = train_models.resolve_args(parser, initial)
        self.assertEqual((args.lr_policy, args.plateau_factor, args.plateau_patience,
                          args.plateau_threshold), ("cosine", .5, 5, 0.))
        for flag, value in [("factor", "0"), ("factor", "1"), ("factor", "nan"),
                            ("patience", "-1"), ("threshold", "-.1"), ("threshold", "inf")]:
            parser, initial = train_models.parse_args(base + ["--lr-policy", "plateau", "--plateau-" + flag, value])
            with self.assertRaises(SystemExit):
                train_models.resolve_args(parser, initial)


class TinyGenerator(nn.Module):
    """Test substitute: stochastic updates exercise RNG/optimizer resume."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 2, 1), nn.Dropout2d(.2), nn.Conv2d(2, 1, 1))

    def forward(self, value):
        return self.net(value)


class TinyDiscriminator(nn.Module):
    kind = "global_qsm_only"

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, swi, qsm):
        return self.conv(qsm).mean((2, 3))


def tiny_models(architecture, *, image_size):
    return TinyGenerator(), TinyDiscriminator(), {"model_name": architecture, "image_size": image_size,
                                                 "generator_output": "linear", "test_substitute": True}


class PlateauWorkflowTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for index, split in enumerate(("train", "val", "test")):
            row = dict(subject_id=split, global_subject_id=split, split=split, qsm_units="ppm")
            values = np.arange(32 * 32 * 2, dtype=np.float32).reshape(32, 32, 2)
            arrays = {"swi_path": values + index, "qsm_path": .01 * np.sin(values),
                      "mask_path": np.ones_like(values), "swi_mask_path": np.ones_like(values)}
            for key, array in arrays.items():
                path = self.root / f"{split}_{key}.nii.gz"
                if split == "test":
                    path.write_bytes(b"held-out test must never be opened")
                else:
                    image = nib.Nifti1Image(array, np.diag([1., 1., 2., 1.]))
                    image.header.set_xyzt_units("mm")
                    nib.save(image, path)
                row[key] = str(path)
            rows.append(row)
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.model_patch = mock.patch.object(train_models, "create_model", side_effect=tiny_models)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def cli(self, output, *extra):
        return ["--model", "swixqsm_gan", "--manifest", str(self.manifest), "--output", str(output),
                "--size", "32", "--units", "ppm", "--qsm-scale", ".1", "--batch-size", "2",
                "--epochs", "6", "--warmup-epochs", "2", "--lr", ".01", "--lr-disc", ".01",
                "--lr-policy", "plateau", "--plateau-patience", "0", "--min-lr", ".001",
                "--device", "cpu", "--num-threads", "1", "--diagnostic-batches", "1",
                "--allow-unverified-data", "synthetic only", "--allow-unverified-independence", "synthetic only", *extra]

    def run_with_scores(self, argv, scores):
        values = iter(scores)
        def validation(*unused):
            score = next(values)
            return score, {"val": score}
        with mock.patch.object(train_models, "validate", side_effect=validation):
            return train_models.main(argv)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for one, other in zip(left, right):
                self.assert_nested_equal(one, other)
        else:
            self.assertEqual(left, right)

    def test_complete_epoch_resume_at_warmup_boundary_and_inside_gan_matches(self):
        scores = [.1, .1, 1., 1., .8, .8]
        continuous = self.root / "continuous"
        self.run_with_scores(self.cli(continuous), scores)
        expected = torch.load(continuous / "last.pt", weights_only=True)
        self.assertEqual(expected["config"]["scheduler_policy"]["monitor"],
                         {"split": "val", "metric": "physical_participant_equal_mae", "units": "ppm", "mode": "min"})
        history = expected["history"]
        self.assertEqual([h["scheduler_event"]["generator_reset"] for h in history],
                         [False, False, True, False, False, False])
        self.assertEqual([h["learning_rates"]["generator"] for h in history],
                         [.01, .01, .005, .005, .0025, .0025])
        self.assertEqual([h["learning_rates_next"]["generator"] for h in history],
                         [.01, .005, .005, .0025, .0025, .00125])
        self.assertEqual([h["scheduler_event"]["discriminator_stepped"] for h in history],
                         [False, False, True, True, True, True])
        self.assertEqual(expected["scheduler_g"]["best"], .8)
        self.assertEqual(expected["scheduler_d"]["best"], .8)
        for boundary in (2, 4):
            segmented = self.root / f"segmented_{boundary}"
            self.run_with_scores(self.cli(segmented, "--max-epochs-this-run", str(boundary)), scores[:boundary])
            self.run_with_scores(["--resume", str(segmented / "last.pt"), "--device", "cpu"], scores[boundary:])
            actual = torch.load(segmented / "last.pt", weights_only=True)
            for key in ("generator", "discriminator", "optimizer_g", "optimizer_d", "scheduler_g", "scheduler_d",
                        "training_state", "rng"):
                self.assert_nested_equal(expected[key], actual[key])
            for one, other in zip(history, actual["history"]):
                for key in ("learning_rates", "learning_rates_next", "scheduler_event", "val_mae", "training_state",
                            "sampling_index_sha256", "training_diagnostics"):
                    self.assert_nested_equal(one[key], other[key])
            # Protocol changes cannot be smuggled into a resumed segment.
            with self.assertRaises(SystemExit):
                train_models.main(["--resume", str(segmented / "last.pt"), "--plateau-factor", ".2"])

    def test_nonfinite_validation_never_commits_an_epoch(self):
        output = self.root / "nonfinite"
        with self.assertRaisesRegex(RuntimeError, "Nonfinite validation MAE"):
            self.run_with_scores(self.cli(output), [float("nan")])
        self.assertFalse((output / "last.pt").exists())
        self.assertFalse((output / "history.jsonl").exists())
        self.assertEqual(json.loads((output / "config.json").read_text())["arguments"]["lr_policy"], "plateau")


if __name__ == "__main__":
    unittest.main()

"""Loss/support integration checks using temporary synthetic tensors only."""
import copy
import json
import unittest
from unittest import mock

import torch

from ..model import UNetSmall
from ..model.discriminator import PatchDiscriminator
from ..util import engine
from ..util.normalization import QSMNormalization
from ..util.patch_loss import patch_bce, patch_support
from ..util.training_diagnostics import EpochDiagnostics


class AdversarialTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20261017)
        self.gen, self.disc = UNetSmall(normalize=False), PatchDiscriminator()
        self.x = torch.rand(2, 1, 128, 128)
        self.y = torch.randn_like(self.x)
        self.mask = torch.zeros_like(self.x)
        self.mask[:, :, 56:72, 56:72] = 1

    @staticmethod
    def optimizers(gen, disc):
        return torch.optim.SGD(gen.parameters(), lr=.001), torch.optim.SGD(disc.parameters(), lr=.001)

    def test_all_gan_terms_share_support_and_g_forward_bn_update_happen_once(self):
        opt_g, opt_d = self.optimizers(self.gen, self.disc)
        captured = []
        before_bn = {name: module.num_batches_tracked.clone() for name, module in self.gen.named_modules()
                     if isinstance(module, torch.nn.BatchNorm2d)}

        def capture(module, args, output):
            output.retain_grad()
            captured.append(output)

        hook = self.gen.register_forward_hook(capture)
        try:
            with mock.patch.object(engine, "patch_bce", wraps=patch_bce) as bce:
                g_loss, d_loss, details = engine.train_step(self.gen, self.disc, opt_g, opt_d,
                    self.x, self.y, self.mask, .01, return_diagnostics=True,
                    diagnostics_normalization=QSMNormalization(.1, .02, "ppm"))
            self.assertEqual(bce.call_count, 3)
            weights = [call.args[2] for call in bce.call_args_list]
            self.assertTrue(all(value is weights[0] for value in weights))
            self.assertTrue(torch.equal(weights[0], patch_support(self.mask, self.disc)))
            self.assertEqual([call.args[1] for call in bce.call_args_list], [1., 0., 1.])
        finally:
            hook.remove()
        self.assertEqual(len(captured), 1)
        for name, module in self.gen.named_modules():
            if name in before_bn:
                self.assertEqual(int(module.num_batches_tracked - before_bn[name]), 1)
        self.assertEqual(float(captured[0].grad[self.mask == 0].abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in self.disc.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in self.disc.parameters()))
        fields = details["fields"]
        support_count = int(weights[0].sum())
        self.assertLess(support_count, weights[0].numel())
        self.assertEqual(fields["adv"]["count"], support_count)
        for name in ("d_real", "d_fake", "d_total", "d_real_probability", "d_fake_probability"):
            self.assertEqual(fields[name]["count"], support_count)
        pooled = EpochDiagnostics()
        pooled.update(fields)
        pooled.add_distribution(details["distribution"])
        summary = pooled.summary()
        metrics = summary["metrics"]
        self.assertAlmostEqual(g_loss, metrics["l1_batch"]["mean"] + metrics["weighted_adv_batch"]["mean"], places=6)
        self.assertAlmostEqual(d_loss, .5 * (metrics["d_real"]["mean"] + metrics["d_fake"]["mean"]), places=7)
        self.assertEqual(details["distribution"]["units"], "ppm")
        self.assertEqual(details["distribution"]["n_valid_voxels"], int(self.mask.sum()))
        json.dumps(summary, allow_nan=False)

    def test_diagnostics_reuse_prediction_without_changing_updates_or_rng(self):
        with_diag_g, with_diag_d = copy.deepcopy(self.gen), copy.deepcopy(self.disc)
        opt_g, opt_d = self.optimizers(self.gen, self.disc)
        other_g, other_d = self.optimizers(with_diag_g, with_diag_d)
        torch.manual_seed(77)
        old_return = engine.train_step(self.gen, self.disc, opt_g, opt_d, self.x, self.y, self.mask, .01)
        rng_without = torch.get_rng_state().clone()
        torch.manual_seed(77)
        new_return = engine.train_step(with_diag_g, with_diag_d, other_g, other_d,
            self.x, self.y, self.mask, .01, return_diagnostics=True,
            diagnostics_normalization=QSMNormalization(.1, .02, "ppm"))
        self.assertEqual(old_return, new_return[:2])
        self.assertTrue(torch.equal(rng_without, torch.get_rng_state()))
        for left, right in ((self.gen, with_diag_g), (self.disc, with_diag_d)):
            for key, value in left.state_dict().items():
                self.assertTrue(torch.equal(value, right.state_dict()[key]), key)

    def test_l1_only_does_not_execute_or_measure_discriminator(self):
        opt_g, opt_d = self.optimizers(self.gen, self.disc)
        before = copy.deepcopy(self.disc.state_dict())
        with mock.patch.object(self.disc, "forward", side_effect=AssertionError("D must not run")):
            _, d_loss, details = engine.train_step(self.gen, self.disc, opt_g, opt_d,
                self.x, self.y, self.mask, 0., return_diagnostics=True)
        self.assertEqual(d_loss, 0.)  # Compatibility return, not a measured discriminator loss.
        diagnostics = EpochDiagnostics()
        diagnostics.update(details["fields"])
        metrics = diagnostics.summary()["metrics"]
        for name in ("adv", "weighted_adv", "d_real", "d_fake", "d_total"):
            self.assertIsNone(metrics[name]["mean"])
            self.assertEqual(metrics[name]["status"], "not_observed")
        self.assertEqual(metrics["weighted_adv_batch"]["mean"], 0.)
        self.assertEqual(len(opt_d.state), 0)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.disc.state_dict()[key]))

    def test_explicit_all_patch_mode_keeps_valid_and_loss_denominators_distinct(self):
        opt_g, opt_d = self.optimizers(self.gen, self.disc)
        _, _, details = engine.train_step(self.gen, self.disc, opt_g, opt_d,
            self.x, self.y, self.mask, .01, adversarial_mode="all_patches", return_diagnostics=True)
        fields = details["fields"]
        self.assertLess(fields["valid_patch_fraction"]["sum"], fields["valid_patch_fraction"]["count"])
        self.assertEqual(fields["loss_patch_fraction"]["sum"], fields["loss_patch_fraction"]["count"])
        self.assertEqual(fields["adv"]["count"], fields["loss_patch_fraction"]["count"])
        self.assertEqual(fields["adv"]["reduction"], "all_patch_mean")

    def test_invalid_support_is_rejected_before_any_model_update(self):
        opt_g, opt_d = self.optimizers(self.gen, self.disc)
        before_g, before_d = copy.deepcopy(self.gen.state_dict()), copy.deepcopy(self.disc.state_dict())
        with mock.patch.object(self.gen, "forward", side_effect=AssertionError("Invalid support must fail first")):
            for mask in (torch.zeros_like(self.mask), torch.full_like(self.mask, -1.), torch.full_like(self.mask, float("nan"))):
                with self.subTest(value=mask.flatten()[0]), self.assertRaises(ValueError):
                    engine.train_step(self.gen, self.disc, opt_g, opt_d, self.x, self.y, mask, .01)
        for model, before in ((self.gen, before_g), (self.disc, before_d)):
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_nonfinite_inputs_fail_consistently_with_or_without_diagnostics(self):
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=.001)
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=.001)
        before_g, before_d = copy.deepcopy(self.gen.state_dict()), copy.deepcopy(self.disc.state_dict())
        for with_diagnostics in (False, True):
            for field in ("x", "y"):
                values = {"x": self.x.clone(), "y": self.y.clone()}
                values[field][0, 0, 0, 0] = float("nan")
                with self.subTest(diagnostics=with_diagnostics, field=field), \
                        mock.patch.object(self.gen, "forward", side_effect=AssertionError("Invalid inputs must fail first")), \
                        self.assertRaisesRegex(ValueError, "must be finite"):
                    engine.train_step(self.gen, self.disc, opt_g, opt_d, values["x"], values["y"], self.mask, 0.,
                                      return_diagnostics=with_diagnostics)
        for model, before in ((self.gen, before_g), (self.disc, before_d)):
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]))
        self.assertEqual(len(opt_g.state), 0)
        self.assertEqual(len(opt_d.state), 0)


if __name__ == "__main__":
    unittest.main()

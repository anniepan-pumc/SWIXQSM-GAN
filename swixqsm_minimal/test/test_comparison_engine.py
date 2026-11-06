import gc
import json
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from ..model.training_models import MODEL_NAMES, Pix2Pix70Discriminator, create_model
from ..util.comparison_engine import (
    adversarial_spec, comparison_step, nominal_patch_support, objective_spec,
)
from ..util.normalization import QSMNormalization
from ..util.training_diagnostics import EpochDiagnostics


class ConstantGenerator(nn.Module):
    def __init__(self, value=0.5):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.forward_calls = 0

    def forward(self, image):
        self.forward_calls += 1
        return image * 0 + self.value


class ToyGlobalDiscriminator(nn.Module):
    kind = "global_qsm_only"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.forward_calls = 0

    def forward(self, swi, qsm):
        self.forward_calls += 1
        return qsm.mean((2, 3)) * self.weight


class CheckingSGD(torch.optim.SGD):
    def __init__(self, params, lr=0., check=None):
        super().__init__(params, lr=lr)
        self.step_calls = 0
        self.check = check

    def step(self, closure=None):
        if self.check is not None:
            self.check()
        self.step_calls += 1
        return super().step(closure)


class ComparisonEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        gc.collect()

    def arrays(self, size=32):
        return torch.zeros(2, 1, size, size), torch.full((2, 1, size, size), 2.), torch.ones(2, 1, size, size)

    def test_analytic_objective_weighted_diagnostics_and_gradient_isolation(self):
        generator, discriminator = ConstantGenerator(), ToyGlobalDiscriminator()
        opt_g = CheckingSGD(generator.parameters(), check=lambda: self.assertTrue(
            all(parameter.grad is None for parameter in discriminator.parameters())))
        opt_d = CheckingSGD(discriminator.parameters(), check=lambda: self.assertTrue(
            all(parameter.grad is None for parameter in generator.parameters())))
        x, y, mask = self.arrays()
        mask.zero_()
        mask[:, :, 10, 10] = 7.  # Positive membership, not a seven-fold probability weight.
        result = comparison_step(generator, discriminator, opt_g, opt_d, x, y, mask, .2, 1.)
        real_logit, fake_logit = torch.tensor(2. / 1024), torch.tensor(.5 / 1024)
        d_real = float(F.softplus(-real_logit))
        d_fake = float(F.softplus(fake_logit))
        adv = float(F.softplus(-fake_logit))
        self.assertAlmostEqual(result["losses"]["l1"], 1.5)
        self.assertAlmostEqual(result["losses"]["weighted_l1"], .3)
        self.assertAlmostEqual(result["losses"]["d_real"], d_real)
        self.assertAlmostEqual(result["losses"]["d_fake"], d_fake)
        self.assertAlmostEqual(result["losses"]["d_total"], .5 * (d_real + d_fake))
        self.assertAlmostEqual(result["losses"]["g_total"], .3 + adv)
        self.assertEqual(result["fields"]["l1"]["count"], 2)
        self.assertAlmostEqual(result["fields"]["l1_batch"]["sum"], .3)
        self.assertEqual((generator.forward_calls, discriminator.forward_calls), (1, 3))
        self.assertEqual((opt_g.step_calls, opt_d.step_calls), (1, 1))
        self.assertTrue(result["generator_updated"] and result["discriminator_updated"])
        self.assertTrue(all(parameter.grad is None for parameter in discriminator.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in discriminator.parameters()))
        diagnostics = EpochDiagnostics()
        diagnostics.update(result["fields"])
        ratio = diagnostics.summary()["weighted_adv_to_l1"]["value"]
        self.assertAlmostEqual(ratio, adv / .3, places=6)
        json.dumps(result, allow_nan=False)

    def test_real_default_width_models_one_step(self):
        x, y, mask = self.arrays()
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                torch.manual_seed(3)
                generator, discriminator, _ = create_model(model_name, image_size=32)
                opt_g = CheckingSGD(generator.parameters(), lr=1e-4)
                opt_d = CheckingSGD(discriminator.parameters(), lr=1e-4) if discriminator is not None else None
                hooks, forward_counts = [], {"g": 0, "d": 0}

                def count(name):
                    def hook(*_):
                        forward_counts[name] += 1
                    return hook

                hooks.append(generator.register_forward_hook(count("g")))
                if discriminator is not None:
                    hooks.append(discriminator.register_forward_hook(count("d")))
                bn = next((layer for layer in discriminator.modules() if isinstance(layer, nn.BatchNorm2d)), None) if discriminator else None
                bn_before = int(bn.num_batches_tracked) if bn is not None else None
                result = comparison_step(generator, discriminator, opt_g, opt_d, x, y, mask,
                                         .2, 1. if discriminator is not None else 0.)
                self.assertEqual(forward_counts, {"g": 1, "d": 3 if discriminator is not None else 0})
                self.assertEqual(opt_g.step_calls, 1)
                self.assertTrue(all(math.isfinite(value) for value in result["losses"].values()))
                self.assertTrue(all(torch.isfinite(parameter).all() for parameter in generator.parameters()))
                if bn is not None:
                    self.assertEqual(int(bn.num_batches_tracked) - bn_before, 3)
                if discriminator is not None:
                    self.assertEqual(opt_d.step_calls, 1)
                    self.assertTrue(all(parameter.grad is None for parameter in discriminator.parameters()))
                for hook in hooks:
                    hook.remove()
                del generator, discriminator, opt_g, opt_d, result, bn

    def test_plain_no_discriminator_and_warmup_no_discriminator_calls_or_support(self):
        x, y, mask = self.arrays()
        for discriminator in (None, ToyGlobalDiscriminator()):
            generator = ConstantGenerator()
            optimizer = CheckingSGD(generator.parameters())
            with patch("swixqsm_minimal.util.comparison_engine.nominal_patch_support",
                       side_effect=AssertionError("Support must not be computed")):
                result = comparison_step(generator, discriminator, optimizer, None, x, y, mask, 2., 0.)
            self.assertEqual(result["losses"]["g_total"], 3.)
            self.assertFalse(result["discriminator_updated"])
            self.assertEqual(result["forward_counts"]["discriminator"], 0)
            self.assertEqual(result["fields"]["adv"]["count"], 0)
            self.assertEqual(result["fields"]["valid_patch_fraction"]["count"], 0)
            if discriminator is not None:
                self.assertEqual(discriminator.forward_calls, 0)

    def test_nominal_rf70_support_and_batchnorm_limitation_are_explicit(self):
        discriminator = Pix2Pix70Discriminator()
        support_mask = torch.zeros(1, 1, 128, 128)
        support_mask[0, 0, 64, 64] = 1
        support = nominal_patch_support(support_mask, discriminator)
        # Interior output footprints have index bounds 8*i-23 .. 8*i+46.
        # Pixel 64 belongs precisely to output indices 3..10 on either axis.
        expected = torch.zeros(1, 1, 14, 14)
        expected[:, :, 3:11, 3:11] = 1
        torch.testing.assert_close(support, expected, rtol=0., atol=0.)
        self.assertFalse(support.requires_grad)
        spec = adversarial_spec(discriminator)
        self.assertEqual(spec["nominal_convolutional_receptive_field"], [70, 70])
        self.assertEqual(spec["convolutional_output_stride"], [8, 8])
        self.assertFalse(spec["background_gradient_isolation"])
        self.assertIn("BatchNorm", spec["normalization_dependency"])
        self.assertIn("direct all-empty", spec["limitation"])
        discriminator.net.append(nn.MaxPool2d(2))
        with self.assertRaisesRegex(ValueError, "Unsupported conditional discriminator layer"):
            nominal_patch_support(support_mask, discriminator)

    def test_all_patches_and_valid_patch_use_same_masked_input_but_different_loss_support(self):
        x, y, mask = self.arrays(size=128)
        mask.zero_()
        mask[:, :, 64, 64] = 1
        for mode, expected_count in (("valid_patch", 2 * 64), ("all_patches", 2 * 196)):
            generator, discriminator = ConstantGenerator(), Pix2Pix70Discriminator()
            opt_g = CheckingSGD(generator.parameters())
            opt_d = CheckingSGD(discriminator.parameters())
            observed = []
            hook = discriminator.register_forward_pre_hook(lambda _, inputs: observed.append(inputs[1].detach().clone()))
            result = comparison_step(generator, discriminator, opt_g, opt_d, x, y, mask, 1., 1.,
                                     adversarial_mode=mode)
            hook.remove()
            self.assertEqual(result["fields"]["adv"]["count"], expected_count)
            self.assertEqual(result["fields"]["valid_patch_fraction"]["sum"], 2 * 64)
            self.assertTrue(all(int((value != 0).sum()) == 2 for value in observed))

    def test_invalid_input_and_objective_rejected_before_updates(self):
        x, y, mask = self.arrays()
        generator = ConstantGenerator()
        optimizer = CheckingSGD(generator.parameters())
        for l1, adv in ((0., 0.), (-1., 0.), (float("nan"), 0.), (1., 1.), (True, 0.)):
            with self.subTest(l1=l1, adv=adv), self.assertRaises(ValueError):
                comparison_step(generator, None, optimizer, None, x, y, mask, l1, adv)
        empty = mask.clone()
        empty[1].zero_()
        with self.assertRaisesRegex(ValueError, "Every training slice"):
            comparison_step(generator, None, optimizer, None, x, y, empty, 1., 0.)
        nonfinite = y.clone()
        nonfinite[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            comparison_step(generator, None, optimizer, None, x, nonfinite, mask, 1., 0.)
        for keyword in ({"amp": "fp16"}, {"scaler": object()}):
            with self.assertRaisesRegex(ValueError, "FP32 only"):
                comparison_step(generator, None, optimizer, None, x, y, mask, 1., 0., **keyword)
        with self.assertRaisesRegex(ValueError, "must not have opt_d"):
            comparison_step(generator, None, optimizer, optimizer, x, y, mask, 1., 0.)
        self.assertEqual(optimizer.step_calls, 0)
        self.assertEqual(generator.forward_calls, 0)

    def test_nonfinite_prediction_or_gradient_fails_without_silent_skip(self):
        x, y, mask = self.arrays()
        generator = ConstantGenerator(float("nan"))
        discriminator = ToyGlobalDiscriminator()
        opt_g, opt_d = CheckingSGD(generator.parameters()), CheckingSGD(discriminator.parameters())
        with self.assertRaisesRegex(RuntimeError, "generator prediction"):
            comparison_step(generator, discriminator, opt_g, opt_d, x, y, mask, 1., 1.)
        self.assertEqual((opt_g.step_calls, opt_d.step_calls), (0, 0))
        self.assertEqual(discriminator.forward_calls, 0)

        generator = ConstantGenerator()
        opt_g = CheckingSGD(generator.parameters(), lr=.01)
        hook = generator.value.register_hook(lambda gradient: gradient * float("inf"))
        before = generator.value.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "nonfinite gradients"):
            comparison_step(generator, None, opt_g, None, x, y, mask, 1., 0.)
        hook.remove()
        self.assertEqual(opt_g.step_calls, 0)
        torch.testing.assert_close(generator.value, before)
        self.assertIsNone(generator.value.grad)

    def test_nonfinite_discriminator_gradient_prevents_both_updates(self):
        x, y, mask = self.arrays()
        generator, discriminator = ConstantGenerator(), ToyGlobalDiscriminator()
        opt_g, opt_d = CheckingSGD(generator.parameters()), CheckingSGD(discriminator.parameters())
        hook = discriminator.weight.register_hook(lambda gradient: gradient * float("inf"))
        with self.assertRaisesRegex(RuntimeError, "discriminator backward: nonfinite gradients"):
            comparison_step(generator, discriminator, opt_g, opt_d, x, y, mask, 1., 1.)
        hook.remove()
        self.assertEqual((opt_g.step_calls, opt_d.step_calls), (0, 0))
        self.assertIsNone(discriminator.weight.grad)
        self.assertTrue(discriminator.weight.requires_grad)

    def test_late_generator_failure_is_fatal_after_canonical_discriminator_step(self):
        x, y, mask = self.arrays()
        generator, discriminator = ConstantGenerator(), ToyGlobalDiscriminator()
        opt_g, opt_d = CheckingSGD(generator.parameters()), CheckingSGD(discriminator.parameters())
        hook = generator.value.register_hook(lambda gradient: gradient * float("inf"))
        with self.assertRaisesRegex(RuntimeError, "generator backward after discriminator update"):
            comparison_step(generator, discriminator, opt_g, opt_d, x, y, mask, 1., 1.)
        hook.remove()
        self.assertEqual((opt_g.step_calls, opt_d.step_calls), (0, 1))
        self.assertTrue(discriminator.weight.requires_grad)
        self.assertIsNone(discriminator.weight.grad)

    def test_distribution_and_physical_scale_objective_metadata(self):
        x, y, mask = self.arrays()
        generator = ConstantGenerator()
        normalization = QSMNormalization(scale=.1, offset=.02, units="ppm")
        result = comparison_step(generator, None, CheckingSGD(generator.parameters()), None,
                                 x, y, mask, .2, 0., normalization=normalization)
        self.assertEqual(result["distribution"]["units"], "ppm")
        self.assertAlmostEqual(result["distribution"]["error"]["mae"], .15, places=6)
        spec = objective_spec(None, {"model_name": "plain_unet"}, .2, 0., "valid_patch", .1)
        self.assertEqual(spec["lambda_l1_per_physical_unit"], 2.)
        self.assertIn("weighted L1", spec["diagnostic_l1_batch"])
        self.assertEqual(spec["adversarial"]["kind"], "none")
        json.dumps(spec, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

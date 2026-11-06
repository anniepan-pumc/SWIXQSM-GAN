import copy
import json
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from ..model.discriminator import PatchDiscriminator
from ..util.patch_loss import adversarial_spec, patch_bce, patch_support


def brute_support(mask, discriminator):
    """Independent explicit input-index tracing through each Conv2d."""
    batch, _, height, width = mask.shape
    locations = [[{(row, column)} if bool(mask[0, 0, row, column] > 0) else set()
                  for column in range(width)] for row in range(height)]
    for layer in discriminator.net:
        if not isinstance(layer, nn.Conv2d):
            continue
        kh, kw = layer.kernel_size
        sh, sw = layer.stride
        ph, pw = layer.padding
        dh, dw = layer.dilation
        oh = (height+2*ph-dh*(kh-1)-1)//sh+1
        ow = (width+2*pw-dw*(kw-1)-1)//sw+1
        output = [[set() for _ in range(ow)] for _ in range(oh)]
        for row in range(oh):
            for column in range(ow):
                for ki in range(kh):
                    for kj in range(kw):
                        hi, wi = row*sh-ph+ki*dh, column*sw-pw+kj*dw
                        if 0 <= hi < height and 0 <= wi < width:
                            output[row][column].update(locations[hi][wi])
        locations, height, width = output, oh, ow
    return torch.tensor([[bool(value) for value in row] for row in locations], dtype=torch.float32)[None, None]


class PatchLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261017)
        self.discriminator = PatchDiscriminator()

    def test_exact_topology_spec(self):
        spec = adversarial_spec(self.discriminator)
        self.assertEqual(spec["receptive_field"], [38, 38])
        self.assertEqual(spec["stride"], [8, 8])
        self.assertEqual(spec["first_center_input_index"], [3.5, 3.5])
        self.assertEqual(spec["first_center_from_input_edge"], [4., 4.])
        self.assertEqual(spec["nominal_input_index_offsets"], {"minimum": [-15., -15.], "maximum": [22., 22.]})
        self.assertEqual(adversarial_spec(self.discriminator, "all_patches")["mode"], "all_patches")
        json.dumps(spec, allow_nan=False)

    def test_single_voxels_and_boundaries_match_brute_index_tracing(self):
        for row, column in ((0, 0), (0, 31), (31, 31), (15, 15), (16, 8), (22, 23)):
            with self.subTest(location=(row, column)):
                mask = torch.zeros(1, 1, 32, 32)
                mask[0, 0, row, column] = .25
                actual = patch_support(mask, self.discriminator)
                self.assertTrue(torch.equal(actual, brute_support(mask, self.discriminator)))

    def test_full_mask_preserves_edge_patches_and_shape_matches_logits(self):
        mask = torch.ones(2, 1, 40, 48, dtype=torch.float64, requires_grad=True)
        support = patch_support(mask, self.discriminator)
        with torch.no_grad():
            logits = self.discriminator(mask.float(), mask.float())
        self.assertEqual(support.shape, logits.shape)
        self.assertTrue(torch.equal(support, torch.ones_like(support)))
        self.assertFalse(support.requires_grad)
        self.assertIsNone(support.grad_fn)

    def test_nominal_rf_and_stride_single_voxel_geometry(self):
        mask = torch.zeros(1, 1, 128, 128)
        mask[0, 0, 48, 40] = 1.
        support = patch_support(mask, self.discriminator)
        expected = torch.zeros_like(support)
        for row in range(support.shape[2]):
            for column in range(support.shape[3]):
                expected[0, 0, row, column] = (8*row-15 <= 48 <= 8*row+22
                                               and 8*column-15 <= 40 <= 8*column+22)
        self.assertTrue(torch.equal(support, expected))

    def test_support_tracks_actual_kernel_dilation_and_stride(self):
        discriminator = nn.Module()
        discriminator.net = nn.Sequential(nn.Conv2d(2, 4, 3, padding=2, dilation=2),
                                           nn.LeakyReLU(.2), nn.Conv2d(4, 1, 3, stride=2))
        mask = torch.zeros(1, 1, 32, 32)
        mask[0, 0, 7, 9] = 1.
        self.assertTrue(torch.equal(patch_support(mask, discriminator), brute_support(mask, discriminator)))
        spec = adversarial_spec(discriminator)
        self.assertEqual(spec["receptive_field"], [7, 7])
        self.assertEqual(spec["stride"], [2, 2])

    def test_background_perturbations_do_not_affect_loss_or_gradients(self):
        mask = torch.zeros(1, 1, 96, 96)
        mask[:, :, 32:40, 32:40] = 1.
        support = patch_support(mask, self.discriminator)
        first = torch.randn(support.shape, dtype=torch.float64, requires_grad=True)
        second = first.detach().clone()
        second[support == 0] = float("nan")
        second.requires_grad_()
        a = patch_bce(first, 1., support)
        b = patch_bce(second, 1., support)
        a.backward()
        b.backward()
        self.assertEqual(float(a.detach()), float(b.detach()))
        self.assertTrue(torch.equal(first.grad, second.grad))
        self.assertEqual(float(first.grad[support == 0].abs().sum()), 0.)

    def test_stride_aligned_distant_padding_does_not_dilute_loss_or_gradient(self):
        mask = torch.zeros(1, 1, 96, 96)
        mask[:, :, 32:40, 32:40] = 1.
        original_support = patch_support(mask, self.discriminator)
        padded_support = patch_support(F.pad(mask, (0, 32, 0, 32)), self.discriminator)
        self.assertEqual(float(original_support.sum()), float(padded_support.sum()))
        original = torch.randn(original_support.shape, dtype=torch.float64, requires_grad=True)
        padded = F.pad(original.detach(), (0, 4, 0, 4), value=50.).requires_grad_()
        a, b = patch_bce(original, 0., original_support), patch_bce(padded, 0., padded_support)
        a.backward()
        b.backward()
        self.assertEqual(float(a.detach()), float(b.detach()))
        self.assertTrue(torch.equal(original.grad, padded.grad[:, :, :12, :12]))
        self.assertEqual(float(padded.grad[:, :, 12:].abs().sum()), 0.)
        self.assertEqual(float(padded.grad[:, :, :, 12:].abs().sum()), 0.)

    def test_all_ones_matches_legacy_mean_and_selected_bce_hand_calculation(self):
        logits = torch.randn(3, 1, 5, 7, dtype=torch.float64, requires_grad=True)
        target = torch.zeros_like(logits)
        actual = patch_bce(logits, target, torch.ones_like(logits))
        expected = F.binary_cross_entropy_with_logits(logits, target)
        self.assertEqual(float(actual.detach()), float(expected.detach()))
        selected = torch.tensor([[[[0., 2., -1.]]]], dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([[[[1., 0., .1]]]], requires_grad=True)
        actual = patch_bce(selected, 1., weights)
        expected = (math.log(2.)+math.log1p(math.exp(1.)))/2
        self.assertAlmostEqual(float(actual.detach()), expected, places=14)
        actual.backward()
        self.assertIsNone(weights.grad)
        self.assertEqual(float(selected.grad[0, 0, 0, 1]), 0.)

    def test_entire_batch_patch_count_not_equal_slice_average(self):
        logits = torch.tensor([[[[0., 0., 0.]]], [[[2., 2., 2.]]]], dtype=torch.float64)
        weights = torch.tensor([[[[1., 0., 0.]]], [[[1., 1., 1.]]]])
        actual = patch_bce(logits, 0., weights)
        expected = (math.log(2.)+3*math.log1p(math.exp(2.)))/4
        self.assertAlmostEqual(float(actual.detach()), expected, places=14)

    def test_empty_support_refused(self):
        mask = torch.zeros(1, 1, 32, 32)
        weights = patch_support(mask, self.discriminator)
        self.assertEqual(float(weights.sum()), 0.)
        with self.assertRaisesRegex(ValueError, "empty"):
            patch_bce(torch.zeros_like(weights), 1., weights)

    def test_invalid_masks_shapes_targets_and_devices_refused(self):
        for mask in (torch.zeros(1, 32, 32), torch.zeros(1, 2, 32, 32),
                     torch.full((1, 1, 32, 32), -1.), torch.full((1, 1, 32, 32), float("nan")),
                     torch.full((1, 1, 32, 32), float("inf"))):
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                patch_support(mask, self.discriminator)
        logits = torch.zeros(1, 1, 4, 4)
        with self.assertRaises(ValueError):
            patch_bce(logits, 1., torch.ones(1, 1, 3, 4))
        with self.assertRaises(ValueError):
            patch_bce(logits, 2., torch.ones_like(logits))
        with self.assertRaises(ValueError):
            patch_bce(logits, torch.zeros(1, 1, 3, 4), torch.ones_like(logits))
        meta_discriminator = copy.deepcopy(self.discriminator).to("meta")
        with self.assertRaisesRegex(ValueError, "same device"):
            patch_support(torch.ones(1, 1, 32, 32), meta_discriminator)

    def test_unknown_normalization_or_spatial_layers_cannot_silently_change_support(self):
        for layer in (nn.BatchNorm2d(1), nn.AvgPool2d(2), nn.AdaptiveAvgPool2d(1), nn.Identity()):
            discriminator = copy.deepcopy(self.discriminator)
            discriminator.net.append(layer)
            with self.subTest(layer=type(layer).__name__):
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    patch_support(torch.ones(1, 1, 32, 32), discriminator)
                with self.assertRaises(ValueError):
                    adversarial_spec(discriminator)


if __name__ == "__main__":
    unittest.main()

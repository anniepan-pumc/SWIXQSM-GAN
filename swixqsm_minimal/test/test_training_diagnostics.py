import copy
import json
import unittest

import numpy as np
import torch

from ..util.normalization import QSMNormalization
from ..util.training_diagnostics import EpochDiagnostics, masked_distribution


def field(total, count, reduction="batch_mean"):
    return {"sum": total, "count": count, "reduction": reduction}


class DistributionTest(unittest.TestCase):
    def test_valid_only_exact_stats_errors_and_no_mutation(self):
        pred = np.array([1., 3., 999., 5., 7., 999., 9., 999.]).reshape(2, 2, 2)
        target = np.array([0., 2., -999., 4., 6., -999., 8., -999.]).reshape(2, 2, 2)
        mask = np.array([1, 1, 0, 1, 1, 0, 1, 0]).reshape(2, 2, 2)
        originals = [value.copy() for value in (pred, target, mask)]
        result = masked_distribution(pred, target, mask)
        values = pred[mask > 0]
        self.assertEqual(result["status"], "defined")
        self.assertEqual(result["n_valid_voxels"], 5)
        self.assertEqual(result["prediction"]["mean"], values.mean())
        self.assertEqual(result["prediction"]["sd"], values.std(ddof=0))
        for key, probability in (("p01", .01), ("p50", .5), ("p99", .99)):
            self.assertEqual(result["prediction"][key], np.quantile(values, probability, method="linear"))
        self.assertEqual(result["error"], {"bias": 1., "mae": 1., "rmse": 1.})
        self.assertIsNone(result["units"])
        self.assertEqual(result["value_space"], "as_supplied")
        self.assertAlmostEqual(result["nmse"], 5 / float(np.square(target[mask > 0]).sum()))
        self.assertEqual(result["zero_baseline_nmse"], 1)
        for value, original in zip((pred, target, mask), originals):
            np.testing.assert_array_equal(value, original)

    def test_tensor_is_detached_and_physical_offset_precedes_energy(self):
        pred = torch.tensor([[[[-1., 0.], [1., 2.]]]], requires_grad=True)
        target = torch.zeros_like(pred, requires_grad=True)
        mask = torch.ones_like(pred)
        norm = QSMNormalization(.1, .02, "ppm")
        result = masked_distribution(pred, target, mask, norm)
        expected = pred.detach().numpy() * .1 + .02
        self.assertEqual(result["units"], "ppm")
        self.assertEqual(result["value_space"], "physical")
        self.assertAlmostEqual(result["prediction"]["mean"], expected.mean(), places=7)
        self.assertAlmostEqual(result["target"]["mean"], .02)
        self.assertAlmostEqual(result["target_energy"], 4 * .02**2)
        self.assertAlmostEqual(result["error"]["mae"], .1)
        self.assertIsNone(pred.grad)
        self.assertIsNone(target.grad)
        self.assertTrue(pred.requires_grad)
        self.assertEqual(result["sd_ratio_status"], "zero_target_sd")

    def test_zero_denominators_empty_support_and_masked_nonfinite(self):
        zero = np.zeros((2, 2, 2))
        one = np.ones_like(zero)
        result = masked_distribution(one, zero, one)
        self.assertEqual(result["nmse_status"], "zero_target_energy")
        self.assertIsNone(result["nmse"])
        self.assertIsNone(result["zero_baseline_nmse"])
        self.assertEqual(result["sd_ratio_status"], "zero_target_sd")
        empty = masked_distribution(one, zero, zero)
        self.assertEqual(empty["status"], "empty_support")
        self.assertIsNone(empty["prediction"]["p50"])
        self.assertIsNone(empty["error"]["mae"])
        pred, target, mask = one.copy(), one.copy(), one.copy()
        pred[0, 0, 0], target[0, 0, 0], mask[0, 0, 0] = np.nan, np.inf, 0
        self.assertEqual(masked_distribution(pred, target, mask)["status"], "defined")
        mask[0, 0, 0] = 1
        failed = masked_distribution(pred, target, mask)
        self.assertEqual(failed["status"], "nonfinite_valid_values")
        self.assertEqual(failed["n_nonfinite_valid_pairs"], 1)
        self.assertIsNone(failed["error"]["mae"])
        for result in (result, empty, failed):
            json.dumps(result, allow_nan=False)

    def test_numpy_tensor_equivalence_and_shape_errors(self):
        values = np.arange(24., dtype=np.float32).reshape(2, 3, 4)
        mask = np.ones_like(values)
        left = masked_distribution(values, values / 2, mask)
        right = masked_distribution(torch.from_numpy(values), torch.from_numpy(values / 2), torch.from_numpy(mask))
        self.assertEqual(left, right)
        for inputs in ((values[:, :, 0], values[:, :, 0], mask[:, :, 0]),
                       (values, values, mask[:1]),
                       (np.zeros((0, 2, 3)),) * 3):
            with self.subTest(shapes=[value.shape for value in inputs]), self.assertRaises(ValueError):
                masked_distribution(*inputs)
        mask[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "mask"):
            masked_distribution(values, values, mask)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            masked_distribution(values, values, -np.ones_like(values))
        with self.assertRaisesRegex(ValueError, "normalization"):
            masked_distribution(values, values, np.ones_like(values), .1)


class EpochDiagnosticsTest(unittest.TestCase):
    def test_sum_count_pooling_and_actual_batch_ratio(self):
        diagnostics = EpochDiagnostics()
        diagnostics.update({"l1": field(100, 100, "valid_voxel_mean"),
                            "adv": field(4, 8, "valid_patch_mean"),
                            "weighted_adv": field(.4, 8, "valid_patch_mean"),
                            "l1_batch": field(1, 1), "weighted_adv_batch": field(.05, 1),
                            "g_total": field(1.05, 1),
                            "valid_patch_fraction": field(8, 10, "fraction")})
        diagnostics.update({"l1": field(9, 1, "valid_voxel_mean"),
                            "adv": field(2, 2, "valid_patch_mean"),
                            "weighted_adv": field(.2, 2, "valid_patch_mean"),
                            "l1_batch": field(9, 1), "weighted_adv_batch": field(.1, 1),
                            "g_total": field(9.1, 1),
                            "valid_patch_fraction": field(2, 10, "fraction")})
        result = diagnostics.summary()
        self.assertAlmostEqual(result["metrics"]["l1"]["mean"], 109 / 101)
        self.assertAlmostEqual(result["metrics"]["adv"]["mean"], .6)
        self.assertEqual(result["metrics"]["valid_patch_fraction"]["mean"], .5)
        ratio = result["weighted_adv_to_l1"]
        self.assertAlmostEqual(ratio["value"], .15 / 10)
        self.assertEqual(ratio["scope"], "ratio_of_epoch_batch_means")
        self.assertEqual(ratio["numerator_count"], 2)
        self.assertEqual(ratio["denominator_count"], 2)
        self.assertIn("not gradient", ratio["interpretation"])

    def test_l1_only_adversarial_not_observed_is_not_fake_zero(self):
        diagnostics = EpochDiagnostics()
        fields = {name: field(0, 0, "valid_patch_mean") for name in
                  ("adv", "weighted_adv", "d_real", "d_fake", "d_total", "d_real_probability",
                   "d_fake_probability", "d_fake_for_g_probability")}
        fields.update(l1=field(20, 10, "valid_voxel_mean"), l1_batch=field(2, 1),
                      weighted_adv_batch=field(0, 1), g_total=field(2, 1))
        diagnostics.update(fields)
        result = diagnostics.summary()
        self.assertIsNone(result["metrics"]["d_total"]["mean"])
        self.assertEqual(result["metrics"]["d_total"]["status"], "not_observed")
        self.assertEqual(result["weighted_adv_to_l1"]["value"], 0)
        self.assertEqual(result["weighted_adv_to_l1"]["status"], "defined")
        json.dumps(result, allow_nan=False)

    def test_ratio_undefined_zero_l1_missing_or_mismatched_counts(self):
        for numerator, expected in ((0, "zero_l1_and_weighted_adv"), (2, "zero_l1")):
            diagnostics = EpochDiagnostics()
            diagnostics.update({"l1_batch": field(0, 1), "weighted_adv_batch": field(numerator, 1)})
            ratio = diagnostics.summary()["weighted_adv_to_l1"]
            self.assertIsNone(ratio["value"])
            self.assertEqual(ratio["status"], expected)
        diagnostics = EpochDiagnostics()
        self.assertEqual(diagnostics.summary()["weighted_adv_to_l1"]["status"], "missing_operand")
        diagnostics.update({"l1_batch": field(1, 2), "weighted_adv_batch": field(1, 1)})
        self.assertEqual(diagnostics.summary()["weighted_adv_to_l1"]["status"], "mismatched_batch_counts")
        fallback = EpochDiagnostics()
        fallback.update({"l1": field(10, 10, "valid_voxel_mean"), "weighted_adv": field(2, 10, "valid_patch_mean")})
        self.assertAlmostEqual(fallback.summary()["weighted_adv_to_l1"]["value"], .2)

    def test_distribution_storage_is_bounded_and_quantiles_never_averaged(self):
        diagnostics = EpochDiagnostics(max_distribution_batches=2)
        records = []
        for value in (0., 20., 100.):
            array = np.full((2, 2, 2), value)
            record = masked_distribution(array, array, np.ones_like(array))
            records.append(copy.deepcopy(record))
            kept = diagnostics.add_distribution(record)
            self.assertEqual(kept, value != 100.)
            record["prediction"]["p50"] = -999
        result = diagnostics.summary()["distributions"]
        self.assertEqual(result["n_batches_offered"], 3)
        self.assertEqual(result["n_batches_recorded"], 2)
        self.assertEqual(result["n_valid_voxels_recorded"], 16)
        self.assertEqual(result["batches"], records[:2])
        self.assertIsNone(result["epoch_quantiles"])
        self.assertIn("not full-epoch", result["scope"])
        result["batches"][0]["prediction"]["p50"] = -888
        self.assertEqual(diagnostics.summary()["distributions"]["batches"][0]["prediction"]["p50"], 0)
        disabled = EpochDiagnostics(0)
        self.assertFalse(disabled.add_distribution(records[0]))
        self.assertEqual(disabled.summary()["distributions"]["n_batches_recorded"], 0)

    def test_fractional_counts_finite_metadata_and_transactional_validation(self):
        diagnostics = EpochDiagnostics()
        diagnostics.update({"custom": field(torch.tensor(2.), torch.tensor(.5), "weighted_mean")})
        self.assertEqual(diagnostics.summary()["metrics"]["custom"]["mean"], 4)
        before = diagnostics.summary()
        invalid = [{"custom": field(1, 1, "different_reduction")},
                   {"a": field(2, 1), "b": field(float("nan"), 1)},
                   {"a": field(2, 0)}, {"a": field(0, -1)},
                   {"a": {"sum": 1, "count": 1}}, {},
                   {"a": field(True, 1)}]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                diagnostics.update(fields)
            self.assertEqual(diagnostics.summary(), before)
        json.dumps(diagnostics.summary(), allow_nan=False)
        for maximum in (-1, True, .5):
            with self.assertRaises(ValueError):
                EpochDiagnostics(maximum)
        with self.assertRaises(ValueError):
            diagnostics.add_distribution({"prediction": {"mean": 0}})

    def test_all_patch_reduction_and_extreme_denominator_stay_explicit(self):
        diagnostics = EpochDiagnostics()
        diagnostics.update({"d_total": field(2, 8, "all_patch_mean"),
                            "l1": field(1, 1e-320, "valid_voxel_mean"),
                            "weighted_adv": field(1, 1, "all_patch_mean")})
        result = diagnostics.summary()
        self.assertEqual(result["metrics"]["d_total"]["reduction"], "all_patch_mean")
        self.assertEqual(result["metrics"]["d_total"]["mean"], .25)
        self.assertIsNone(result["metrics"]["l1"]["mean"])
        self.assertEqual(result["metrics"]["l1"]["status"], "numerical_overflow")
        self.assertEqual(result["weighted_adv_to_l1"]["status"], "operand_numerical_overflow")
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

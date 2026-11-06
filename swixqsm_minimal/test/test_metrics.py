import json
import math
import unittest

import numpy as np

from ..util.metrics import _physical_log, physical_metrics


class PhysicalMetricsTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261017)
        self.reference = self.rng.normal(5., 12., size=(25, 27, 21))
        self.mask = np.ones(self.reference.shape, dtype=bool)

    def test_identity_and_strict_json(self):
        metrics = physical_metrics(self.reference, self.reference, self.mask, data_range=100.)
        for name in ("mae", "mse", "rmse", "bias", "nmse", "hfen_rmse", "hfen_relative"):
            self.assertEqual(metrics[name], 0.)
        self.assertAlmostEqual(metrics["ssim"], 1., places=12)
        self.assertIsNone(metrics["psnr_db"])
        self.assertEqual(metrics["metric_status"]["psnr_db"], "infinite")
        self.assertEqual(metrics["ssim_centers"], (25-6)*(27-6)*21)
        self.assertEqual(metrics["hfen_centers"], (25-12)*(27-12)*(21-12))
        json.dumps(metrics, allow_nan=False)

    def test_physical_zero_prediction_has_nmse_one(self):
        metrics = physical_metrics(self.reference, np.zeros_like(self.reference), self.mask, data_range=100.)
        self.assertEqual(metrics["nmse"], 1.)
        self.assertEqual(metrics["hfen_relative"], 1.)
        self.assertAlmostEqual(metrics["mae"], float(np.abs(self.reference).mean()))
        self.assertAlmostEqual(metrics["bias"], float(-self.reference.mean()))
        expected = 10*math.log10(100.**2 / np.mean(self.reference**2))
        self.assertAlmostEqual(metrics["psnr_db"], expected)

    def test_offset_bias_preserves_physical_values(self):
        metrics = physical_metrics(self.reference, self.reference+7.5, self.mask, data_range=100.)
        for name in ("mae", "rmse", "bias"):
            self.assertAlmostEqual(metrics[name], 7.5)
        self.assertAlmostEqual(metrics["mse"], 7.5**2)
        self.assertAlmostEqual(metrics["pred_mean"]-metrics["ref_mean"], 7.5)
        self.assertAlmostEqual(metrics["pred_std"], metrics["ref_std"])
        self.assertLess(metrics["ssim"], 1.)
        self.assertAlmostEqual(metrics["hfen_rmse"], 0., places=12)

    def test_hfen_constant_offset_is_not_high_frequency_on_thick_slices(self):
        reference = np.full((51, 51, 21), 20.)
        metrics = physical_metrics(reference, reference+7.5, np.ones_like(reference),
                                   spacing=(.8, .8, 3.), data_range=100.)
        self.assertEqual(metrics["hfen_rmse"], 0.)
        self.assertIsNone(metrics["hfen_relative"])
        self.assertEqual(metrics["metric_status"]["hfen_relative"],
                         "zero_reference_high_frequency_energy")
        self.assertEqual(metrics["sufficient_statistics"]["hfen_reference_energy"], 0.)
        # Verify the corrected kernel itself, independently of the analytic
        # constant-map shortcut in physical_metrics.
        response = _physical_log(reference, (1.5/.8, 1.5/.8, 1.5/3.), (.8, .8, 3.))
        self.assertLess(float(np.max(np.abs(response[8:-8, 8:-8, 2:-2]))), 1e-12)
        # The error also has zero HF energy when the underlying map varies.
        variable = reference+self.rng.normal(size=reference.shape)
        metrics = physical_metrics(variable, variable+7.5, np.ones_like(reference),
                                   spacing=(.8, .8, 3.))
        self.assertAlmostEqual(metrics["hfen_rmse"], 0., places=12)
        self.assertAlmostEqual(metrics["hfen_relative"], 0., places=12)

    def test_unit_scaling(self):
        prediction = .8*self.reference + 3.
        original = physical_metrics(self.reference, prediction, self.mask, data_range=100.)
        scaled = physical_metrics(self.reference*1000, prediction*1000, self.mask, data_range=100000.)
        for name in ("mae", "rmse", "bias", "ref_mean", "pred_mean", "ref_std", "pred_std", "hfen_rmse"):
            self.assertAlmostEqual(scaled[name]/1000, original[name], places=10)
        self.assertAlmostEqual(scaled["mse"]/1e6, original["mse"], places=10)
        for name in ("nmse", "psnr_db", "ssim", "hfen_relative"):
            self.assertAlmostEqual(scaled[name], original[name], places=10)

    def test_outside_mask_values_have_no_effect(self):
        mask = np.zeros_like(self.mask)
        mask[2:-2, 2:-2, 2:-2] = True
        prediction = self.reference + self.rng.normal(size=self.reference.shape)
        baseline = physical_metrics(self.reference, prediction, mask, data_range=100.)
        reference, perturbed = self.reference.copy(), prediction.copy()
        reference[~mask] = np.nan
        perturbed[~mask] = np.inf
        actual = physical_metrics(reference, perturbed, mask, data_range=100.)
        self.assertEqual(baseline, actual)
        self.assertGreater(actual["hfen_centers"], 0)
        json.dumps(actual, allow_nan=False)

    def test_zero_reference_denominators(self):
        reference = np.zeros_like(self.reference)
        metrics = physical_metrics(reference, np.ones_like(reference)*2, self.mask, data_range=100.)
        self.assertIsNone(metrics["nmse"])
        self.assertEqual(metrics["metric_status"]["nmse"], "zero_reference_energy")
        self.assertIsNone(metrics["hfen_relative"])
        self.assertEqual(metrics["metric_status"]["hfen_relative"], "zero_reference_high_frequency_energy")
        self.assertEqual(metrics["mae"], 2.)
        json.dumps(metrics, allow_nan=False)

    def test_missing_or_invalid_range_not_inferred(self):
        for peak in (None, 0, -2, float("nan"), float("inf"), "invalid"):
            with self.subTest(data_range=peak):
                metrics = physical_metrics(self.reference, self.reference, self.mask, data_range=peak)
                self.assertIsNone(metrics["psnr_db"])
                self.assertIsNone(metrics["ssim"])
                expected = "missing_data_range" if peak is None else "invalid_data_range"
                self.assertEqual(metrics["metric_status"]["ssim"], expected)
                json.dumps(metrics, allow_nan=False)

    def test_spatially_insufficient_support_does_not_flatten(self):
        reference = np.arange(125.).reshape(5, 5, 5)
        metrics = physical_metrics(reference, reference+1, np.ones_like(reference), data_range=100.)
        self.assertEqual(metrics["mae"], 1.)
        for name in ("ssim", "hfen_rmse", "hfen_relative"):
            self.assertIsNone(metrics[name])
            self.assertEqual(metrics["metric_status"][name], "insufficient_support")
        self.assertEqual(metrics["ssim_centers"], 0)
        self.assertEqual(metrics["hfen_centers"], 0)
        for name in ("ssim_sum", "ssim_count", "hfen_error_energy", "hfen_reference_energy", "hfen_count"):
            self.assertEqual(metrics["sufficient_statistics"][name], 0)
        # Axial SSIM can still be valid on one slice, whereas 3D HFEN cannot.
        one_slice = np.ones((11, 11, 1))
        metrics = physical_metrics(one_slice, one_slice, one_slice, data_range=10.)
        self.assertEqual(metrics["ssim_centers"], 25)
        self.assertAlmostEqual(metrics["ssim"], 1.)
        self.assertIsNone(metrics["hfen_rmse"])

    def test_hfen_uses_physical_spacing_and_millimetre_sigma(self):
        spacing = (.5, 1., 2.)
        x = (np.arange(61)-30)*spacing[0]
        reference = np.broadcast_to(np.cos(.2*x)[:, None, None], (61, 31, 17)).copy()
        metrics = physical_metrics(reference, np.zeros_like(reference), np.ones_like(reference),
                                   spacing=spacing, hfen_sigma_mm=1.5)
        self.assertEqual(metrics["parameters"]["hfen_radius_voxels"], [12, 6, 3])
        self.assertEqual(metrics["hfen_centers"], 37*19*11)
        # Analytic LoG of a cosine, allowing finite/discrete Gaussian kernel error.
        expected = .2**2*np.exp(-.5*(1.5*.2)**2)*np.sqrt(np.mean(np.cos(.2*x[12:-12])**2))
        self.assertAlmostEqual(metrics["hfen_rmse"]/expected, 1., delta=.02)

    def test_sufficient_statistics_allow_unequal_visit_aggregation(self):
        references = [np.full((15, 15, 15), 2.), np.full((17, 17, 17), 3.)]
        predictions = [references[0]+2, references[1]-1]
        visits = [physical_metrics(ref, pred, np.ones_like(ref), data_range=100.)
                  for ref, pred in zip(references, predictions)]
        stats = {key: sum(visit["sufficient_statistics"][key] for visit in visits)
                 for key in visits[0]["sufficient_statistics"]}
        all_ref = np.concatenate([ref.ravel() for ref in references])
        all_pred = np.concatenate([pred.ravel() for pred in predictions])
        error = all_pred-all_ref
        self.assertEqual(stats["n"], len(all_ref))
        self.assertAlmostEqual(stats["abs_error_sum"]/stats["n"], float(np.abs(error).mean()))
        self.assertAlmostEqual(stats["signed_error_sum"]/stats["n"], float(error.mean()))
        self.assertAlmostEqual(stats["squared_error_sum"]/stats["reference_energy"],
                               float(np.sum(error**2)/np.sum(all_ref**2)))
        self.assertAlmostEqual(stats["reference_sum"], float(all_ref.sum()))
        self.assertAlmostEqual(stats["prediction_sum"], float(all_pred.sum()))
        self.assertAlmostEqual(stats["prediction_energy"], float(np.sum(all_pred**2)))
        expected_ssim = sum(visit["ssim"]*visit["ssim_centers"] for visit in visits)/stats["ssim_count"]
        self.assertAlmostEqual(stats["ssim_sum"]/stats["ssim_count"], expected_ssim)
        expected_hfen = sum(visit["hfen_rmse"]**2*visit["hfen_centers"] for visit in visits)
        self.assertAlmostEqual(stats["hfen_error_energy"], expected_hfen)
        self.assertEqual(stats["hfen_count"], sum(visit["hfen_centers"] for visit in visits))

    def test_invalid_domains_rejected(self):
        with self.assertRaises(ValueError):
            physical_metrics(self.reference, self.reference, np.zeros_like(self.mask))
        bad = self.reference.copy()
        bad[3, 3, 3] = np.nan
        with self.assertRaises(ValueError):
            physical_metrics(bad, self.reference, self.mask)
        for spacing in ((1, 0, 1), (1, 1), (1, float("nan"), 1)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                physical_metrics(self.reference, self.reference, self.mask, spacing=spacing)


if __name__ == "__main__":
    unittest.main()

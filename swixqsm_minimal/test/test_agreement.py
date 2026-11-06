import json
import math
import unittest

import numpy as np

from ..util.agreement import paired_agreement


class PairedAgreementTest(unittest.TestCase):
    def test_identity_definitions_and_json(self):
        reference = np.arange(1., 9.)
        actual = paired_agreement(reference, reference, bootstrap_reps=100)
        self.assertEqual(actual["n_participants"], 8)
        for name in ("bias", "difference_sd", "loa_lower", "loa_upper"):
            self.assertEqual(actual["estimates"][name], 0.)
        self.assertAlmostEqual(actual["estimates"]["pearson_r"], 1.)
        self.assertAlmostEqual(actual["estimates"]["icc_a1"], 1.)
        self.assertEqual(actual["definitions"]["icc"]["model"], "two_way_mixed")
        self.assertEqual(actual["definitions"]["icc"]["definition"], "absolute_agreement")
        self.assertEqual(actual["definitions"]["icc"]["measurement"], "single")
        json.dumps(actual, allow_nan=False)

    def test_analytic_constant_bias_fixture_absolute_single_icc(self):
        # Independently derived ANOVA for x=(1,2,3,4), y=x+2:
        # MSR=10/3, MSC=8, MSE=0, k=2, n=4 => ICC=5/11.
        # Consistency ICC would incorrectly be 1; average-measures ICC 5/8.
        reference = np.arange(1., 5.)
        actual = paired_agreement(reference, reference+2, bootstrap_reps=0)
        self.assertEqual(actual["estimates"]["bias"], 2.)
        self.assertEqual(actual["estimates"]["difference_sd"], 0.)
        self.assertEqual(actual["estimates"]["loa_lower"], 2.)
        self.assertEqual(actual["estimates"]["loa_upper"], 2.)
        self.assertAlmostEqual(actual["estimates"]["pearson_r"], 1.)
        self.assertAlmostEqual(actual["estimates"]["icc_a1"], 5/11)

    def test_analytic_proportional_bias_fixture(self):
        # x=(1,2,3,4), y=2x: MSR=15/2, MSC=25/2, MSE=5/6.
        # ICC=(20/3)/(85/6)=8/17; sample SD of y-x=sqrt(5/3).
        reference = np.arange(1., 5.)
        actual = paired_agreement(reference, 2*reference, bootstrap_reps=0)
        sd = math.sqrt(5/3)
        self.assertEqual(actual["estimates"]["bias"], 2.5)
        self.assertAlmostEqual(actual["estimates"]["difference_sd"], sd)
        self.assertAlmostEqual(actual["estimates"]["loa_lower"], 2.5-1.96*sd)
        self.assertAlmostEqual(actual["estimates"]["loa_upper"], 2.5+1.96*sd)
        self.assertAlmostEqual(actual["estimates"]["pearson_r"], 1.)
        self.assertAlmostEqual(actual["estimates"]["icc_a1"], 8/17)

    def test_negative_icc_is_not_truncated(self):
        # Three anti-ordered participants: MSR=MSC=0, MSE=2 => ICC=-3.
        actual = paired_agreement([-1., 0., 1.], [1., 0., -1.], bootstrap_reps=0)
        self.assertAlmostEqual(actual["estimates"]["icc_a1"], -3.)
        self.assertAlmostEqual(actual["estimates"]["pearson_r"], -1.)

    def test_small_n_preserves_pairs_and_reports_status(self):
        for n in (0, 1, 2):
            with self.subTest(n=n):
                reference = np.arange(n, dtype=float)
                actual = paired_agreement(reference, reference+3, bootstrap_reps=30)
                self.assertEqual(actual["n_participants"], n)
                self.assertIsNone(actual["estimates"]["icc_a1"])
                self.assertEqual(actual["metric_status"]["icc_a1"], "insufficient_n_for_icc_policy")
                self.assertEqual(actual["bootstrap"]["metrics"]["icc_a1"]["status"], "point_estimate_undefined")
                if n:
                    self.assertEqual(actual["estimates"]["bias"], 3.)
                    if n < 3:
                        interval = actual["bootstrap"]["metrics"]["bias"]
                        self.assertEqual(interval["status"], "insufficient_n_for_bootstrap_policy")
                        self.assertIsNone(interval["lower"])
                        self.assertIsNone(interval["upper"])
                        self.assertEqual(interval["valid_replicates"], 30)
                else:
                    self.assertIsNone(actual["estimates"]["bias"])
                if n < 2:
                    for metric in ("difference_sd", "loa_lower", "loa_upper", "pearson_r"):
                        self.assertIsNone(actual["estimates"][metric])
                for metric in actual["bootstrap"]["metrics"].values():
                    self.assertEqual(metric["valid_replicates"]+metric["undefined_replicates"], 30)
                    self.assertIsNone(metric["lower"])
                    self.assertIsNone(metric["upper"])
                self.assertEqual(actual["bootstrap"]["minimum_participants_for_ci"], 3)
                json.dumps(actual, allow_nan=False)

    def test_constant_and_degenerate_cases(self):
        identical = paired_agreement([2.]*5, [2.]*5, bootstrap_reps=20)
        self.assertIsNone(identical["estimates"]["pearson_r"])
        self.assertEqual(identical["metric_status"]["pearson_r"], "constant_both")
        self.assertIsNone(identical["estimates"]["icc_a1"])
        self.assertEqual(identical["metric_status"]["icc_a1"], "degenerate_denominator")
        shifted = paired_agreement([2.]*5, [3.]*5, bootstrap_reps=0)
        self.assertEqual(shifted["estimates"]["icc_a1"], 0.)
        self.assertEqual(shifted["estimates"]["bias"], 1.)
        for reference, prediction, reason in (([1.]*3, [1., 2., 3.], "constant_reference"),
                                               ([1., 2., 3.], [1.]*3, "constant_prediction")):
            result = paired_agreement(reference, prediction, bootstrap_reps=0)
            self.assertIsNone(result["estimates"]["pearson_r"])
            self.assertEqual(result["metric_status"]["pearson_r"], reason)
            self.assertAlmostEqual(result["estimates"]["icc_a1"], 0.)
        json.dumps(identical, allow_nan=False)

    def test_bootstrap_deterministic_global_rng_unchanged_and_paired(self):
        reference = np.arange(1., 7.)
        np.random.seed(7654)
        before = np.random.get_state()
        first = paired_agreement(reference, reference+2, bootstrap_reps=100, seed=19)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(first, paired_agreement(reference, reference+2, bootstrap_reps=100, seed=19))
        # Independently resampling the methods would not preserve constant bias.
        bias_ci = first["bootstrap"]["metrics"]["bias"]
        self.assertEqual((bias_ci["lower"], bias_ci["upper"]), (2., 2.))
        self.assertEqual(bias_ci["valid_replicates"], 100)
        self.assertNotEqual(first["bootstrap"],
                            paired_agreement(reference, reference+2, bootstrap_reps=100, seed=20)["bootstrap"])

    def test_bootstrap_percentiles_match_independent_bias_resampling(self):
        reference = np.array([1., 3., 4., 9.])
        prediction = np.array([2., 2., 8., 10.])
        seed, reps, confidence = 31, 123, .8
        actual = paired_agreement(reference, prediction, bootstrap_reps=reps, seed=seed, confidence=confidence)
        rng = np.random.default_rng(seed)
        biases = [float((prediction-reference)[rng.integers(0, 4, size=4)].mean()) for _ in range(reps)]
        expected = np.quantile(biases, [.1, .9], method="linear")
        interval = actual["bootstrap"]["metrics"]["bias"]
        self.assertAlmostEqual(interval["lower"], expected[0])
        self.assertAlmostEqual(interval["upper"], expected[1])
        self.assertEqual(actual["definitions"]["loa_multiplier"], 1.96)
        ordinary = paired_agreement(reference, prediction, bootstrap_reps=0, confidence=.95)
        self.assertEqual(actual["estimates"], ordinary["estimates"])

    def test_partial_defined_replicates_are_counted_and_labeled(self):
        actual = paired_agreement([0., 1., 2.], [0., 1., 2.], bootstrap_reps=200, seed=3)
        for metric in ("pearson_r", "icc_a1"):
            interval = actual["bootstrap"]["metrics"][metric]
            self.assertEqual(interval["status"], "partial_defined")
            self.assertGreater(interval["valid_replicates"], 0)
            self.assertGreater(interval["undefined_replicates"], 0)
            self.assertEqual(sum(interval["undefined_reason_counts"].values()), interval["undefined_replicates"])
            self.assertAlmostEqual(interval["lower"], 1.)
            self.assertAlmostEqual(interval["upper"], 1.)
        self.assertEqual(actual["bootstrap"]["metrics"]["bias"]["status"], "defined")

    def test_disabled_and_one_replicate_no_ci(self):
        for reps, status in ((0, "disabled"), (1, "insufficient_defined_replicates")):
            actual = paired_agreement([1., 2., 3., 4.], [2., 4., 5., 6.], bootstrap_reps=reps)
            interval = actual["bootstrap"]["metrics"]["bias"]
            self.assertEqual(interval["status"], status)
            self.assertIsNone(interval["lower"])
            self.assertIsNone(interval["upper"])

    def test_common_unit_scaling_translation_and_order(self):
        reference = np.array([1., 2., 4., 5., 9.])
        prediction = np.array([2., 1., 5., 8., 9.])
        baseline = paired_agreement(reference, prediction, bootstrap_reps=0)
        scaled = paired_agreement(reference*1000+1e10, prediction*1000+1e10, bootstrap_reps=0)
        reversed_pairs = paired_agreement(reference[::-1], prediction[::-1], bootstrap_reps=0)
        for name in ("bias", "difference_sd", "loa_lower", "loa_upper"):
            self.assertAlmostEqual(scaled["estimates"][name]/1000, baseline["estimates"][name])
        for name in ("pearson_r", "icc_a1"):
            self.assertAlmostEqual(scaled["estimates"][name], baseline["estimates"][name])
        for name in baseline["estimates"]:
            self.assertAlmostEqual(reversed_pairs["estimates"][name], baseline["estimates"][name])

    def test_extreme_finite_inputs_are_strict_json(self):
        actual = paired_agreement([-1e308, 0., 1e308], [1e308, 0., -1e308], bootstrap_reps=10)
        self.assertEqual(actual["estimates"]["bias"], 0.)
        self.assertIsNone(actual["estimates"]["difference_sd"])
        self.assertEqual(actual["metric_status"]["difference_sd"], "numerical_overflow")
        self.assertAlmostEqual(actual["estimates"]["icc_a1"], -3.)
        json.dumps(actual, allow_nan=False)

    def test_invalid_inputs_do_not_silently_delete_pairs(self):
        invalid = (([1., np.nan], [1., 2.]), ([1., 2.], [1., np.inf]),
                   ([[1., 2.]], [[1., 2.]]), ([1.], [1., 2.]),
                   ([1+2j], [1.]), (["1"], ["1"]), ([True], [True]))
        for reference, prediction in invalid:
            with self.subTest(reference=reference, prediction=prediction), self.assertRaises(ValueError):
                paired_agreement(reference, prediction, bootstrap_reps=0)
        for kwargs in ({"bootstrap_reps": -1}, {"bootstrap_reps": 2.5}, {"bootstrap_reps": True},
                       {"seed": -1}, {"seed": 1.5}, {"seed": True},
                       {"confidence": 0}, {"confidence": 1}, {"confidence": float("nan")},
                       {"confidence": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                paired_agreement([1., 2., 3.], [1., 2., 3.], **kwargs)


if __name__ == "__main__":
    unittest.main()

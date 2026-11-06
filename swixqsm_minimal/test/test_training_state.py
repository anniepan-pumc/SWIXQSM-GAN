import io
import json
import math
import unittest

import torch

from ..util.training_state import TrainingProgress


class TrainingStateTest(unittest.TestCase):
    def test_global_warmup_best_and_separate_gan_best(self):
        progress = TrainingProgress(epochs=2, warmup_epochs=1, adv_weight=.01)
        self.assertEqual(progress.phase(0), "warmup")
        self.assertEqual(progress.weight(0), 0)
        progress.complete_step(0)
        progress.complete_step(0)
        self.assertEqual(progress.record_validation(0, 1), ["best.pt", "best_warmup.pt"])
        progress.complete_step(1)
        self.assertEqual(progress.phase(1), "gan")
        self.assertEqual(progress.weight(1), .01)
        self.assertEqual(progress.record_validation(1, 2), ["best_gan.pt"])
        state = progress.snapshot(1)
        self.assertEqual(state["global_best"]["phase"], "warmup")
        self.assertEqual(state["global_best"]["epoch_index"], 0)
        self.assertEqual(state["global_best"]["score"], 1)
        self.assertEqual(state["best_by_phase"]["gan"]["score"], 2)
        self.assertIsNone(state["best_by_phase"]["l1"])
        self.assertEqual(state["current_run_phase"], "gan")
        self.assertEqual(state["generator_updates"], 3)
        self.assertEqual(state["discriminator_updates"], 1)
        self.assertEqual(state["adversarial_generator_updates"], 1)
        self.assertEqual(state["updates_by_phase"]["warmup"]["generator_updates"], 2)
        self.assertEqual(state["updates_by_phase"]["warmup"]["discriminator_updates"], 0)

    def test_gan_request_must_reach_adversarial_phase(self):
        for epochs, warmup in ((1, 1), (2, 3), (2, 2)):
            with self.subTest(epochs=epochs, warmup=warmup), self.assertRaisesRegex(ValueError, "epochs > warmup"):
                TrainingProgress(epochs, warmup, .01)

    def test_l1_only_is_current_run_l1_regardless_of_warmup(self):
        for warmup in (0, 1, 20):
            progress = TrainingProgress(2, warmup, 0)
            progress.complete_step(0)
            self.assertEqual(progress.phase(0), "l1")
            self.assertEqual(progress.weight(0), 0)
            self.assertEqual(progress.record_validation(0, 4), ["best.pt", "best_l1.pt"])
            progress.complete_step(1)
            self.assertEqual(progress.record_validation(1, 3), ["best.pt", "best_l1.pt"])
            state = progress.snapshot(1)
            self.assertEqual(state["current_run_phase"], "l1")
            self.assertEqual(state["discriminator_updates"], 0)
            self.assertFalse(state["gan_training_observed"])
            self.assertEqual(state["phase_scope"], "current_run_only; source history separate")
            self.assertFalse(state["source_history_inferred"])
            self.assertIsNone(state["best_by_phase"]["warmup"])
            self.assertIsNone(state["best_by_phase"]["gan"])

    def test_gan_without_warmup_and_real_step_requirement(self):
        progress = TrainingProgress(2, 0, .02)
        self.assertEqual(progress.phase(0), "gan")
        self.assertEqual(progress.weight(0), .02)
        for operation in (lambda: progress.record_validation(0, 2), lambda: progress.snapshot(0)):
            with self.assertRaisesRegex(ValueError, "actual adversarial updates"):
                operation()
        progress.complete_step(0)
        self.assertEqual(progress.record_validation(0, 2), ["best.pt", "best_gan.pt"])
        self.assertEqual(progress.snapshot(0)["current_epoch_updates"]["discriminator_updates"], 1)
        # A previous GAN epoch cannot authorize saving an untrained later epoch.
        with self.assertRaisesRegex(ValueError, "actual adversarial updates"):
            progress.record_validation(1, 1)
        with self.assertRaisesRegex(ValueError, "actual adversarial updates"):
            progress.snapshot(1)

    def test_ties_retain_global_owner_but_new_phase_gets_a_best(self):
        progress = TrainingProgress(3, 1, .1)
        progress.complete_step(0)
        progress.record_validation(0, 1)
        progress.complete_step(1)
        self.assertEqual(progress.record_validation(1, 1), ["best_gan.pt"])
        progress.complete_step(2)
        self.assertEqual(progress.record_validation(2, 1), [])
        state = progress.snapshot(2)
        self.assertEqual(state["global_best"]["phase"], "warmup")
        self.assertEqual(state["best_by_phase"]["gan"]["epoch_index"], 1)

    def test_nonfinite_validation_never_changes_selection(self):
        progress = TrainingProgress(2, 0, 0)
        progress.complete_step(0)
        before = progress.snapshot(0)
        for score in (float("nan"), float("inf"), -float("inf"), True, "1"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                progress.record_validation(0, score)
            self.assertEqual(progress.snapshot(0), before)
        self.assertEqual(progress.record_validation(0, .5), ["best.pt", "best_l1.pt"])

    def test_input_ranges_and_chronological_updates(self):
        for args in ((0, 0, 0), (2, -1, 0), (True, 0, 0), (2.5, 0, 0),
                     (2, 0, -1), (2, 0, float("nan")), (2, 0, float("inf")), (2, 0, True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                TrainingProgress(*args)
        progress = TrainingProgress(3, 0, 0)
        for epoch in (-1, 3, True, 1.2):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                progress.phase(epoch)
        progress.complete_step(0)
        progress.record_validation(0, 2)
        with self.assertRaisesRegex(ValueError, "after"):
            progress.complete_step(0)
        with self.assertRaisesRegex(ValueError, "already"):
            progress.record_validation(0, 1)
        progress.complete_step(1)
        with self.assertRaisesRegex(ValueError, "earlier"):
            progress.complete_step(0)
        with self.assertRaisesRegex(ValueError, "earlier"):
            progress.snapshot(0)

    def test_snapshot_is_plain_finite_isolated_metadata(self):
        progress = TrainingProgress(2, 1, .01)
        initial = progress.snapshot(0)
        self.assertIsNone(initial["global_best"])
        self.assertEqual(initial["best_by_phase"], {"l1": None, "warmup": None, "gan": None})
        encoded = json.dumps(initial, allow_nan=False)
        self.assertNotIn("Infinity", encoded)
        progress.complete_step(0)
        progress.record_validation(0, 1.5)
        state = progress.snapshot(0)
        payload = io.BytesIO()
        torch.save({"training_state": state}, payload)
        payload.seek(0)
        self.assertEqual(torch.load(payload, weights_only=True)["training_state"], state)
        state["global_best"]["score"] = -1
        state["updates_by_phase"]["warmup"]["generator_updates"] = 100
        unchanged = progress.snapshot(0)
        self.assertEqual(unchanged["global_best"]["score"], 1.5)
        self.assertEqual(unchanged["updates_by_phase"]["warmup"]["generator_updates"], 1)
        self.assertTrue(math.isfinite(unchanged["global_best"]["score"]))


if __name__ == "__main__":
    unittest.main()

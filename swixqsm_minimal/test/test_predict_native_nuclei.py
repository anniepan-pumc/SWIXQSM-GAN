"""Real SWI-only inference from tiny constant checkpoints, without training."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import torch

from .. import predict_native_nuclei as pipeline
from ..model import UNetSmall
from ..test.infer import infer_case as real_infer_case
from ..util.nifti import sidecar_path
from ..util.normalization import QSMNormalization, normalization_metadata


class PredictNativeNucleiTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix="swix-predict-native-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.swi_shape, self.qsm_shape = (32, 32, 2), (20, 18, 3)
        swi = np.arange(np.prod(self.swi_shape), dtype=np.float32).reshape(self.swi_shape) + 10
        self.brain = np.ones(self.swi_shape, dtype=np.float32)
        self.brain[0] = 0
        self.swi_entry = self.save("external_swi.nii.gz", swi, (.8, .9, 2.))
        self.brain_entry = self.save("t1_mapped_brain_candidate.nii.gz", self.brain, (.8, .9, 2.))
        self.qsm_entry = self.save("native_qsm.nii.gz", np.full(self.qsm_shape, 10.), (1.2, 1.1, 2.5))
        self.reliability_entry = self.save("native_qsm_reliability.nii.gz", np.ones(self.qsm_shape), (1.2, 1.1, 2.5))
        qsm_left, swi_left = np.zeros(self.qsm_shape), np.zeros(self.swi_shape)
        qsm_left[5:7, 5:7, :] = 385
        swi_left[4:6, 4:6, :] = 385
        # Only RN_L is complete in both grids. RN bilateral lacks R; other
        # nuclei lack all components and must remain ineligible, never imputed.
        self.registration = dict(case_id="synthetic_external_case", cohort="synthetic_external", subject_id="synthetic_person",
            registration_status="candidate_completed_pending_QC", protocol_revision="scanner_identity_initialization_v2",
            inputs=dict(swi=self.swi_entry, qsm_grid=self.qsm_entry, qsm_reliability_mask=self.reliability_entry),
            targets=dict(gre=dict(labels=dict(left=self.save("gre_left.nii.gz", qsm_left, (1.2, 1.1, 2.5)),
                                             right=self.save("gre_right.nii.gz", np.zeros(self.qsm_shape), (1.2, 1.1, 2.5)))),
                         swi=dict(mapped_brain=self.brain_entry,
                                  labels=dict(left=self.save("swi_left.nii.gz", swi_left, (.8, .9, 2.)),
                                              right=self.save("swi_right.nii.gz", np.zeros(self.swi_shape), (.8, .9, 2.))))))
        self.registration_path = self.root / "registration.json"
        self.registration_path.write_text(json.dumps(self.registration))
        self.cases = [dict(case_id=self.registration["case_id"], cohort=self.registration["cohort"],
                          subject_id=self.registration["subject_id"], registration_report=self.registration_path.name)]
        self.cases_path = self.root / "cases.json"
        self.cases_path.write_text(json.dumps(self.cases))
        # Trained checkpoints may live under a directory named example_training;
        # this synthetic directory holds weights only, never restricted-cohort images.
        self.checkpoint = self.root / "example_training" / "models" / "plain_unet.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint_relative = str(self.checkpoint.relative_to(self.root))
        self.save_checkpoint(self.checkpoint)
        self.models_path = self.root / "checkpoints.csv"
        self.write_models([dict(model="plain_unet", checkpoint_path=self.checkpoint_relative, expected_phase="l1")])
        self.reason = "Synthetic external test; units/reference/registration/ROI review deferred"

    def save(self, name, values, spacing):
        path = self.root / name
        image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), np.diag([*spacing, 1.]))
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        return dict(path=path.name, sha256=pipeline.sha(path))

    @staticmethod
    def save_checkpoint(path, *, recipe="plain_unet", phase="l1", adversarial_updates=0, units="stored"):
        generator = UNetSmall(normalize=False)
        for parameter in generator.parameters():
            parameter.data.zero_()
        generator.final[0].bias.data.fill_(-2.)
        normalization = normalization_metadata(QSMNormalization(scale=4., offset=5., units=units))
        torch.save(dict(generator=generator.state_dict(), normalization=normalization,
                        config=dict(size=32, recipe=recipe, generator_output="linear", normalization=normalization,
                                    units=units, qsm_scale=4., qsm_offset=5.),
                        training_state=dict(current_run_phase=phase, adversarial_generator_updates=adversarial_updates),
                        data_qc=dict(status="unverified", reason="Synthetic constant checkpoint; no training performed")), path)

    def write_models(self, rows):
        with self.models_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("model", "checkpoint_path", "expected_phase"))
            writer.writeheader()
            writer.writerows(rows)

    def run_pipeline(self, output, expected_models=("plain_unet",)):
        return pipeline.run(self.cases_path, self.models_path, output, source_root=self.root,
                            reason=self.reason, device="cpu", expected_models=expected_models)

    @staticmethod
    def csv_rows(path):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))

    def assert_failed(self, output, n_completed=0):
        status = json.loads((output / "run_status.json").read_text())
        self.assertEqual(status["status"], "failed_partial_predictions_preserved")
        self.assertEqual(len(status["completed_predictions"]), n_completed)
        self.assertIsInstance(status["error"], dict)
        self.assertFalse(status["training_performed"])
        self.assertFalse((output / "metrics").exists())
        return status

    def test_real_stored_inverse_cli_single_model_and_swi_only_inference(self):
        output = self.root / "success"
        original_sha = pipeline.sha(self.checkpoint)
        image_reads, infer_inputs = [], []
        inside_infer = False
        actual_load = nib.load
        allowed = {(self.root / entry["path"]).resolve() for entry in (self.swi_entry, self.brain_entry)}

        def read_image(path, *args, **kwargs):
            path = Path(path).resolve()
            image_reads.append((inside_infer, path))
            if inside_infer:
                self.assertIn(path, allowed, "Reference QSM/labels must never enter generator inference")
            return actual_load(path, *args, **kwargs)

        def infer(context, swi, mask, destination, **kwargs):
            nonlocal inside_infer
            self.assertEqual({Path(swi).resolve(), Path(mask).resolve()}, allowed)
            self.assertEqual(kwargs["output_units"], "stored")
            infer_inputs.append((Path(swi), Path(mask)))
            inside_infer = True
            try:
                return real_infer_case(context, swi, mask, destination, **kwargs)
            finally:
                inside_infer = False

        arguments = ["predict-native", "--cases", str(self.cases_path), "--checkpoints", str(self.models_path),
                     "--source-root", str(self.root), "--output", str(output), "--device", "cpu", "--num-threads", "1",
                     "--expected-model", "plain_unet", "--allow-unverified-data", self.reason]
        with mock.patch("nibabel.load", side_effect=read_image), mock.patch.object(pipeline, "infer_case", side_effect=infer), \
                mock.patch("swixqsm_minimal.train_models.main", side_effect=AssertionError("No training")), \
                mock.patch.object(sys, "argv", arguments):
            pipeline.main()
        self.assertEqual(len(infer_inputs), 1)
        self.assertEqual({path for inside, path in image_reads if inside}, allowed)
        self.assertIn((False, (self.root / self.qsm_entry["path"]).resolve()), image_reads)
        self.assertEqual(pipeline.sha(self.checkpoint), original_sha)
        status = json.loads((output / "run_status.json").read_text())
        self.assertEqual(status["status"], "completed_QC_pending")
        self.assertEqual(status["expected_models"], ["plain_unet"])
        self.assertEqual(len(status["completed_predictions"]), 1)
        self.assertFalse(status["training_performed"])
        self.assertFalse(status["restricted_cohort_images_read"])
        products = json.loads((output / "evaluation_cases.json").read_text())
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0]["expected_checkpoint_sha256"], original_sha)
        provenance = json.loads(Path(products[0]["prediction_provenance"]).read_text())
        prediction_path = Path(provenance["prediction_path"])
        values = nib.load(prediction_path).get_fdata()
        np.testing.assert_array_equal(values[self.brain > 0], -3.)  # -2 * 4 + 5, exactly once.
        np.testing.assert_array_equal(values[self.brain <= 0], 0.)
        self.assertTrue(provenance["inverse_transform_applied_once"])
        self.assertEqual(provenance["checkpoint_sha256"], original_sha)
        self.assertEqual(provenance["input_swi_sha256"], self.swi_entry["sha256"])
        metadata = json.loads(sidecar_path(prediction_path).read_text())
        self.assertEqual((metadata["Units"], metadata["ValueSpace"], metadata["UnitStatus"]),
                         ("stored", "restored_stored_values", "unverified_original_storage"))
        self.assertTrue(metadata["InverseNormalization"]["applied"])
        self.assertEqual(metadata["CheckpointSHA256"], original_sha)
        report = json.loads((output / "metrics/report.json").read_text())
        self.assertEqual((report["n_cases"], report["n_models_available"], report["n_measurements_including_zero"]), (1, 1, 30))
        self.assertTrue(report["no_voxel_error_across_grids"])
        self.assertTrue(report["research_only"])
        self.assertFalse(report["formal_accuracy_claim"])
        rows = self.csv_rows(output / "metrics/native_roi_per_case.csv")
        rn = next(row for row in rows if row["model"] == "plain_unet" and row["region"] == "RN" and row["side"] == "L")
        self.assertEqual(rn["status"], "numeric_candidate_QC_pending")
        self.assertEqual((int(rn["n_reference_voxels"]), int(rn["n_prediction_voxels"])), (12, 8))
        self.assertEqual(float(rn["reference_mean_raw"]), 10.)
        self.assertEqual(float(rn["prediction_mean_raw"]), -3.)
        self.assertEqual(float(rn["roi_mean_abs_error_raw"]), 13.)
        self.assertNotIn("mae", rn)  # Two native grids, not matched voxel errors.
        bilateral = next(row for row in rows if row["model"] == "plain_unet" and row["region"] == "RN" and row["side"] == "bilateral")
        self.assertEqual(bilateral["status"], "missing_or_incomplete_reliable_ROI")
        self.assertEqual(bilateral["reference_all_components_present"], "False")
        summary = next(row for row in self.csv_rows(output / "metrics/native_roi_summary.csv")
                       if row["model"] == "plain_unet" and row["region"] == "RN" and row["side"] == "L")
        self.assertEqual(summary["n_defined_participants"], "1")
        self.assertEqual(summary["icc_a1"], "")
        self.assertEqual(summary["confidence_intervals_status"], "disabled_small_pilot_pending_QC")

    def test_missing_default_or_explicit_models_reject_before_inference(self):
        for index, expected in enumerate((None, ["plain_unet", "pix2pix"])):
            output = self.root / f"missing_{index}"
            with mock.patch.object(pipeline, "load_inference_context", side_effect=AssertionError("No inference preflight")):
                with self.assertRaisesRegex(ValueError, "every expected model"):
                    self.run_pipeline(output, expected_models=expected)
            self.assertFalse(output.exists())

    def test_wrong_phase_and_unobserved_gan_reject_with_failed_status(self):
        self.write_models([dict(model="plain_unet", checkpoint_path=self.checkpoint_relative, expected_phase="gan")])
        output = self.root / "wrong_phase"
        with mock.patch.object(pipeline, "infer_case", side_effect=AssertionError("Phase must fail before inference")):
            with self.assertRaisesRegex(ValueError, "current-run gan"):
                self.run_pipeline(output)
        self.assert_failed(output)
        self.save_checkpoint(self.checkpoint, phase="gan", adversarial_updates=0)
        output = self.root / "unobserved_gan"
        with self.assertRaisesRegex(ValueError, "adversarial.*updates"):
            self.run_pipeline(output)
        self.assert_failed(output)

    def test_duplicate_weight_bytes_and_mislabeled_recipe_are_rejected(self):
        duplicate = self.root / "same_bytes_different_filename.pt"
        duplicate.write_bytes(self.checkpoint.read_bytes())
        self.write_models([dict(model="plain_unet", checkpoint_path=self.checkpoint_relative, expected_phase="l1"),
                           dict(model="pix2pix", checkpoint_path=duplicate.name, expected_phase="l1")])
        output = self.root / "duplicates"
        with mock.patch.object(pipeline, "load_inference_context", side_effect=AssertionError("Reject repeated SHA before loading")):
            with self.assertRaisesRegex(ValueError, "distinct checkpoint bytes"):
                self.run_pipeline(output, expected_models=["plain_unet", "pix2pix"])
        self.assertFalse(output.exists())
        self.write_models([dict(model="pix2pix", checkpoint_path=self.checkpoint_relative, expected_phase="l1")])
        output = self.root / "wrong_recipe"
        with self.assertRaisesRegex(ValueError, "recipe.*model name"):
            self.run_pipeline(output, expected_models=["pix2pix"])
        self.assert_failed(output)

    def test_mid_run_failure_preserves_completed_prediction_without_completed_claim(self):
        second = copy.deepcopy(self.registration)
        second.update(case_id="second_failure_only_case", subject_id="second_failure_only_person")
        second_path = self.root / "second_registration.json"
        second_path.write_text(json.dumps(second))
        # Reused synthetic image paths only exercise failure bookkeeping; no
        # two-case metrics are allowed to be published by this test.
        cases = [*self.cases, dict(case_id=second["case_id"], cohort=second["cohort"], subject_id=second["subject_id"],
                                   registration_report=second_path.name)]
        self.cases_path.write_text(json.dumps(cases))
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Synthetic second-case inference failure")
            return real_infer_case(*args, **kwargs)

        output = self.root / "partial"
        with mock.patch.object(pipeline, "infer_case", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "second-case inference failure"):
                self.run_pipeline(output)
        status = self.assert_failed(output, n_completed=1)
        pp = Path(status["completed_predictions"][0]["prediction_provenance"])
        self.assertTrue(pp.is_file())
        self.assertTrue(Path(json.loads(pp.read_text())["prediction_path"]).is_file())
        self.assertFalse((output / "evaluation_cases.json").exists())

    def test_changed_swi_during_inference_and_physical_checkpoint_do_not_complete(self):
        def change_swi(context, swi, mask, destination, **kwargs):
            result = real_infer_case(context, swi, mask, destination, **kwargs)
            image = nib.load(swi)
            changed = nib.Nifti1Image(image.get_fdata().astype(np.float32) + 1., image.affine, image.header)
            nib.save(changed, swi)
            return result

        output = self.root / "changed_input"
        with mock.patch.object(pipeline, "infer_case", side_effect=change_swi):
            with self.assertRaisesRegex(ValueError, "SWI/mask changed during inference"):
                self.run_pipeline(output)
        self.assert_failed(output)
        # Restore the synthetic SWI/hash binding before testing intensity units.
        values = np.arange(np.prod(self.swi_shape), dtype=np.float32).reshape(self.swi_shape) + 10
        self.swi_entry = self.save("external_swi.nii.gz", values, (.8, .9, 2.))
        self.registration["inputs"]["swi"] = self.swi_entry
        self.registration_path.write_text(json.dumps(self.registration))
        self.save_checkpoint(self.checkpoint, units="ppm")
        output = self.root / "physical_checkpoint"
        with mock.patch.object(pipeline, "infer_case", side_effect=AssertionError("No invented stored/ppm conversion")):
            with self.assertRaisesRegex(ValueError, "inventing a conversion"):
                self.run_pipeline(output)
        self.assert_failed(output)


if __name__ == "__main__":
    unittest.main()

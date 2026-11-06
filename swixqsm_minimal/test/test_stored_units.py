"""Synthetic checks for unverified numerical values; no cohort data are read."""
import argparse
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

from .. import evaluate, fit_normalization, run_server, train_models
from ..build_manifest import build_manifest
from ..model import UNetSmall
from ..predict_split import main as predict_main
from ..restore_scale import main as restore_main
from ..test.infer import main as infer_main
from ..util.baselines import fit_baselines
from ..util.data import read_manifest
from ..util.evaluation import evaluate_rows
from ..util.nifti import save_qsm, sidecar_path
from ..util.normalization import QSMNormalization, from_checkpoint, normalization_metadata, unit_factor
from ..util.normalization_workflow import physical_output_metadata, resolve_training_normalization
from ..util.quality_gate import bind_input_units
from ..util.regions import RegionEvaluator
from ..validate import main as validate_main


class StoredUnitsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix="swix-stored-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = []
        base = np.arange(32 * 32 * 2, dtype=np.float32).reshape(32, 32, 2)
        self.targets = {}
        for index, split in enumerate(("train", "val", "test")):
            target = base / 100 - 8 + index
            self.targets[split] = target
            row = dict(subject_id=f"s{index}", global_subject_id=f"g{index}", case_id=f"c{index}",
                       split=split, qsm_units="stored", qsm_reference="unverified_original_reference")
            for role, values in dict(swi_path=base + index, qsm_path=target,
                                     swi_mask_path=np.ones_like(base), mask_path=np.ones_like(base)).items():
                path = self.save(f"{split}_{role}.nii.gz", values)
                row[role] = str(path)
            self.rows.append(row)
        self.manifest = self.root / "manifest.csv"
        self.write_manifest(self.rows, self.manifest)
        self.artifact = fit_normalization.fit_training_normalization(self.rows, units="stored", manifest_path=self.manifest)
        self.normalization_path = self.root / "normalization.json"
        fit_normalization.save_fitted_normalization(self.artifact, self.normalization_path)
        self.norm = QSMNormalization.from_dict(self.artifact["normalization"]["qsm"])
        generator = UNetSmall(normalize=False)
        for parameter in generator.parameters():
            parameter.data.zero_()
        generator.final[0].bias.data.fill_(-2.)
        self.checkpoint = self.root / "oracle.pt"
        self.spec = normalization_metadata(self.norm)
        self.checkpoint_payload = dict(generator=generator.state_dict(), normalization=self.spec,
            config=dict(size=32, units="stored", generator_output="linear", normalization=self.spec),
            data_qc=dict(status="unverified", reason="Synthetic stored-value test"))
        torch.save(self.checkpoint_payload, self.checkpoint)
        self.quality = ["--allow-unverified-data", "Synthetic QC deferred"]
        self.independence = ["--allow-unverified-independence", "Synthetic fixture identities"]

    def save(self, name, values):
        path = self.root / name
        image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), np.diag([.8, .9, 1.5, 1]))
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        return path

    @staticmethod
    def write_manifest(rows, path):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
            writer.writeheader()
            writer.writerows(rows)

    def test_no_physical_conversion_or_clipping_is_possible(self):
        values = np.array([-1000., -.5, 0., 8000.], dtype=np.float32)
        self.assertEqual(unit_factor("stored", "stored"), 1.)
        np.testing.assert_allclose(self.norm.denormalize(self.norm.normalize(values)), values, atol=.001)
        for physical in ("ppm", "ppb"):
            for source, target in (("stored", physical), (physical, "stored")):
                with self.assertRaisesRegex(ValueError, "cannot be converted"):
                    unit_factor(source, target)
        meta = physical_output_metadata(self.norm)
        self.assertEqual((meta["Units"], meta["ValueSpace"], meta["UnitStatus"]),
                         ("stored", "restored_stored_values", "unverified_original_storage"))
        self.assertEqual(meta["InverseNormalization"]["output_value_space"], "restored_stored_values")
        self.assertEqual(unit_factor("ppm", "ppb"), 1000.)

    def test_fitting_is_train_only_and_retains_stored_provenance(self):
        for row in self.rows[1:]:
            for role in ("swi_path", "qsm_path", "swi_mask_path", "mask_path"):
                Path(row[role]).write_bytes(b"Unreadable holdout image")
        loaded_rows = read_manifest(self.manifest, validate_inputs=False)
        result = fit_normalization.fit_training_normalization(loaded_rows, units="stored")
        self.assertAlmostEqual(result["normalization"]["qsm"]["offset"], np.mean(self.targets["train"], dtype=np.float64))
        self.assertAlmostEqual(result["normalization"]["qsm"]["scale"], np.std(self.targets["train"], dtype=np.float64))
        self.assertEqual(result["unit_status"], "unverified_original_storage")
        self.assertEqual(result["data_qc"]["status"], "unverified")
        self.assertEqual({item["split"] for item in result["inputs"]}, {"train"})
        self.assertEqual(read_manifest(self.manifest)[0]["qsm_units"], "stored")

    def test_stored_cannot_masquerade_as_verified_physical_qc(self):
        with self.assertRaisesRegex(ValueError, "physical-unit QC"):
            fit_normalization.fit_training_normalization(self.rows, units="stored", data_qc={"status": "verified"})
        with self.assertRaisesRegex(ValueError, "Verified QC"):
            bind_input_units(self.rows, {"status": "verified"})
        args = argparse.Namespace(units="stored", normalization_file=None, qsm_scale=2., qsm_offset=0.)
        with self.assertRaisesRegex(ValueError, "physical-unit QC"):
            resolve_training_normalization(args, [], {"status": "verified"}, {})
        forged = copy.deepcopy(self.checkpoint_payload)
        forged["data_qc"] = {"status": "verified"}
        with self.assertRaisesRegex(ValueError, "verified physical-unit"):
            from_checkpoint(forged)
        with self.assertRaisesRegex(ValueError, "unverified restored_stored_values"):
            save_qsm(self.root / "falsely_physical.nii.gz", self.targets["test"], nib.load(self.rows[2]["swi_path"]),
                     {"Units": "stored", "ValueSpace": "physical_qsm"})

    def test_infer_validate_and_restore_apply_one_numeric_inverse(self):
        row = self.rows[2]
        output = self.root / "prediction.nii.gz"
        infer_main(["--checkpoint", str(self.checkpoint), "--swi", row["swi_path"], "--swi-mask", row["swi_mask_path"],
                    "--output", str(output), "--output-units", "stored", *self.quality])
        expected = self.norm.denormalize(-2.)
        np.testing.assert_allclose(nib.load(output).get_fdata(), expected, atol=2e-6)
        metadata = json.loads(sidecar_path(output).read_text())
        self.assertEqual(metadata["ValueSpace"], "restored_stored_values")
        self.assertEqual(metadata["DataQC"]["status"], "unverified")
        validation = self.root / "validation.json"
        validate_main(["--checkpoint", str(self.checkpoint), "--manifest", str(self.manifest), "--split", "test",
                       "--output", str(validation), "--output-units", "stored", *self.quality, *self.independence])
        result = json.loads(validation.read_text())
        self.assertEqual(result["UnitStatus"], "unverified_original_storage")
        self.assertAlmostEqual(result["mae"], float(np.abs(expected - self.targets["test"]).mean()), places=5)
        raw = self.save("raw.nii.gz", np.full_like(self.targets["test"], -2.))
        restored = self.root / "restored.nii.gz"
        restore_main(["--checkpoint", str(self.checkpoint), "--input", str(raw), "--input-value-space", "network_normalized",
                      "--swi-mask", row["swi_mask_path"], "--output", str(restored), "--output-units", "stored"])
        np.testing.assert_array_equal(nib.load(restored).get_fdata(), nib.load(output).get_fdata())
        with self.assertRaisesRegex(ValueError, "refusing double inverse"):
            restore_main(["--checkpoint", str(self.checkpoint), "--input", str(output), "--input-value-space", "network_normalized",
                          "--swi-mask", row["swi_mask_path"], "--output", str(self.root / "twice.nii.gz")])

    def test_batch_prediction_and_per_case_evaluation_never_claim_physical_units(self):
        predictions = self.root / "predictions"
        predict_main(["--checkpoint", str(self.checkpoint), "--manifest", str(self.manifest), "--split", "test",
                      "--output", str(predictions), "--output-units", "stored", "--reference-id", "unverified_original_reference",
                      *self.quality, *self.independence])
        run = json.loads((predictions / "prediction_run.json").read_text())
        self.assertEqual(run["unit_status"], "unverified_original_storage")
        evaluation = self.root / "evaluation"
        argv = ["evaluate", "--manifest", str(predictions / "evaluation_manifest.csv"), "--split", "test", "--output", str(evaluation),
                "--units", "stored", "--qsm-units", "stored", "--reference-id", "unverified_original_reference",
                "--data-range", "40", "--save-predictions", "--distribution-diagnostics", *self.quality, *self.independence]
        with mock.patch.object(sys, "argv", argv):
            evaluate.main()
        for name in ("per_case.csv", "per_subject.csv"):
            with (evaluation / name).open(newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual({record["units"] for record in records}, {"stored"})
            self.assertEqual({record["value_space"] for record in records}, {"restored_stored_values"})
            self.assertEqual({record["unit_status"] for record in records}, {"unverified_original_storage"})
            self.assertEqual({record["method"] for record in records}, {"zero", "prediction"})
        summary = json.loads((evaluation / "summary.json").read_text())
        self.assertEqual(summary["config"]["unit_status"], "unverified_original_storage")
        self.assertIn("numeric zero", summary["config"]["zero_baseline"])
        distributions = json.loads((evaluation / "distribution_diagnostics.json").read_text())
        self.assertEqual({record["statistics"]["value_space"] for record in distributions["cases"]}, {"restored_stored_values"})
        for sidecar in (evaluation / "predictions").glob("*.json"):
            self.assertEqual(json.loads(sidecar.read_text())["UnitStatus"], "unverified_original_storage")

    def test_stored_predictions_must_be_explicit_and_not_physical(self):
        row = dict(self.rows[2])
        path = self.save("bad_prediction.nii.gz", self.targets["test"])
        row["prediction_path"] = str(path)
        row["prediction_units"] = "stored"
        row["prediction_reference"] = "unverified_original_reference"
        for metadata in ({"Units": "stored", "ValueSpace": "physical_qsm"},
                         {"Units": "ppb", "ValueSpace": "restored_stored_values"},
                         {"Units": "stored", "ValueSpace": "network_normalized"}):
            sidecar_path(path).write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                evaluate_rows([row], units="stored", reference_id="unverified_original_reference")
        with self.assertRaisesRegex(ValueError, "anatomical baselines"):
            evaluate_rows([row], units="stored", reference_id="unverified_original_reference", baseline_bundle=object())
        with self.assertRaisesRegex(ValueError, "baseline fitting"):
            fit_baselines([], units="stored", reference_id="unknown", space_id="unknown")
        with self.assertRaisesRegex(ValueError, "ROI physical-unit"):
            RegionEvaluator([], "not-read.json", "not-read.csv", units="stored", reference_id="unknown")

    def test_server_and_manifest_accept_stored_only_with_unverified_policy(self):
        config = dict(schema="swixqsm.server_training", version=1, data_root="source", work_dir="work", model="plain_unet",
                      manifest=dict(expected_train=2, expected_test=1773, qsm_units="stored"),
                      quality=dict(allow_unverified_reason="QC and units deferred"), independence=dict(run_audit=True),
                      training=dict(units="stored", qsm_units="stored", qsm_scale=1., size=32, epochs=1, batch_size=2, device="cpu"))
        config_path = self.root / "server.json"
        config_path.write_text(json.dumps(config))
        loaded = run_server.load_config(config_path)
        self.assertEqual(loaded["training"]["units"], "stored")
        self.assertTrue(loaded["independence"]["run_audit"])
        config["simple_baselines"] = dict(enabled=True, units="ppb", reference_id="unknown", space_id="unknown")
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "baseline fitting"):
            run_server.load_config(config_path)
        data_root = self.root / "inventory_source"
        for split, identifier in (("train", "101"), ("train", "102"), ("validate", "103")):
            case = data_root / split / identifier
            case.mkdir(parents=True)
            for name in ("SWI.nii.gz", "QSM.nii.gz", "brain_mask.nii.gz", "QSM_mask.nii.gz"):
                (case / name).write_bytes(b"Inventory metadata only; not a NIfTI")
        with mock.patch("nibabel.load", side_effect=AssertionError("Inventory must not open images")):
            record = build_manifest(data_root, self.root / "inventory", qsm_units="stored", input_mask_source="Synthetic SWI mask",
                                    expected_train=2, expected_test=1)
        self.assertTrue(record["ready"])
        self.assertEqual(record["qc_status"], "not_evaluated")

    def test_training_records_stored_selection_and_supports_resume(self):
        output = self.root / "training"
        train_models.main(["--model", "plain_unet", "--manifest", str(self.manifest), "--output", str(output),
                           "--normalization-file", str(self.normalization_path), "--units", "stored", "--size", "32",
                           "--batch-size", "2", "--epochs", "2", "--max-epochs-this-run", "1", "--warmup-epochs", "0",
                           "--num-threads", "1", "--device", "cpu", *self.quality, *self.independence])
        first = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(first["training_state"]["selection_metric"], "stored_participant_equal_mae")
        self.assertNotIn("physical_l1_relation", first["training_objective"])
        self.assertEqual(first["training_objective"]["unit_status"], "unverified_original_storage")
        train_models.main(["--resume", str(output / "last.pt"), "--max-epochs-this-run", "1"])
        final = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(final["epoch"], 2)
        self.assertEqual(final["normalization"]["qsm"]["units"], "stored")
        self.assertEqual(final["data_qc"]["status"], "unverified")
        self.assertEqual(final["training_state"]["selection_metric"], "stored_participant_equal_mae")


if __name__ == "__main__":
    unittest.main()

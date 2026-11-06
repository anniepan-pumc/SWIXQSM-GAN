"""Synthetic batch inference, provenance, target isolation and atomic publishing."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import torch

from .. import evaluate, predict_split
from ..model import UNetSmall
from ..test import infer
from ..util.independence import build_independence_report
from ..util.independence_gate import check_independence
from ..util.nifti import sidecar_path
from ..util.normalization import QSMNormalization, normalization_metadata
from ..util.quality import build_report
from ..util.quality_gate import prepare_quality_rows
from ..util.reference import reference_identity


class PredictSplitTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.norm = QSMNormalization(scale=.2, offset=-.1, units="ppm")
        self.reference = {"id": "synthetic_zero", "method": "mean", "region": "synthetic_region",
                          "target_ppb": 0., "tolerance_ppb": .001, "require_mask": True}
        self.contract = dict(schema="swixqsm.data_contract", version=1,
            geometry=dict(axis_codes=["R", "A", "S"], slice_axis=2,
                          spacing_mm_min=[.79, .89, 1.49], spacing_mm_max=[.81, .91, 1.51],
                          max_obliquity_deg=1.), reference=self.reference, registration={"require_review": True})
        self.rows, self.targets = [], []
        shape = (32, 32, 3)
        raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        self.brain = np.ones(shape, np.float32)
        self.brain[0] = 0
        reference_mask = np.zeros(shape, np.float32)
        reference_mask[2:5, 2:5, :] = 1
        for index, split in enumerate(("train", "val", "test", "test")):
            row = dict(subject_id=f"subject{index}", global_subject_id=f"person{index}", split=split,
                       case_id="../../unsafe/name" if index == 2 else f"case{index}",
                       qsm_units="ppm", qsm_reference="synthetic_zero", qsm_reference_method="mean",
                       qsm_reference_region="synthetic_region", qsm_reference_source="synthetic test only",
                       registration_review="accepted", registration_reviewer="synthetic test fixture",
                       registration_review_date="2026-10-18", prediction_path="old_other_model.nii.gz",
                       prediction_units="ppb", prediction_reference="old_reference")
            qsm = (.05 + .01 * np.sin(raw / 100 + index)).astype(np.float32)
            qsm[reference_mask > 0] = 0
            self.targets.append(qsm)
            arrays = dict(swi_path=raw + index, qsm_path=qsm, swi_mask_path=self.brain,
                          mask_path=self.brain, qsm_reference_mask_path=reference_mask)
            for key, values in arrays.items():
                path = self.root / f"{index}_{key}.nii.gz"
                image = nib.Nifti1Image(values, np.diag([.8, .9, 1.5, 1.]))
                image.header.set_xyzt_units("mm")
                nib.save(image, path)
                row[key] = str(path)
            self.rows.append(row)
        self.manifest = self.root / "manifest.csv"
        self.write_manifest()
        self.qc_path, self.independence_path = self.root / "qc.json", self.root / "independence.json"
        self.make_reports()
        spec = normalization_metadata(self.norm)
        generator = UNetSmall(normalize=False)
        for parameter in generator.parameters():
            parameter.data.zero_()
        generator.final[0].bias.data.fill_(-2.)
        self.checkpoint = self.root / "frozen.pt"
        self.state = dict(generator=generator.state_dict(), normalization=spec,
            config=dict(size=32, generator_output="linear", units="ppm", qsm_scale=.2, qsm_offset=-.1,
                        normalization=spec),
            training_state={"current_run_phase": "gan", "adversarial_generator_updates": 7},
            data_qc={"status": "verified", "contract": self.contract},
            data_reference=reference_identity(self.reference),
            data_independence=self.independence, checkpoint_role="current_run_gan_best")
        torch.save(self.state, self.checkpoint)

    def write_manifest(self):
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def make_reports(self):
        report = build_report(self.rows, self.contract)
        self.assertEqual(report["n_passed"], len(self.rows), report)
        self.qc_path.write_text(json.dumps(report, allow_nan=False))
        independence = build_independence_report(prepare_quality_rows(self.rows, self.manifest))
        self.assertEqual(independence["status"], "passed", independence)
        self.independence_path.write_text(json.dumps(independence, allow_nan=False))
        args = type("Args", (), dict(independence_report=str(self.independence_path),
                                     allow_unverified_independence=None))()
        self.independence = check_independence(prepare_quality_rows(self.rows, self.manifest), args,
                                               content_splits=("train",), check_initialization=False)

    def cli(self, output, *, verified=True, split="test", extra=()):
        flags = (["--qc-report", str(self.qc_path), "--independence-report", str(self.independence_path)]
                 if verified else ["--allow-unverified-data", "synthetic only", "--allow-unverified-independence", "synthetic only"])
        return ["--checkpoint", str(self.checkpoint), "--manifest", str(self.manifest),
                "--split", split, "--output", str(output), "--reference-id", "synthetic_zero",
                "--expected-phase", "gan", *flags, *extra]

    def predictions(self, output):
        with (output / "evaluation_manifest.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        return rows, [row for row in rows if row["prediction_path"]]

    def test_load_once_selected_split_only_no_qsm_decode_and_same_reports_evaluate(self):
        # Full manifest membership remains intact, but other split images are
        # corrupt. Only selected SWI/mask may even be opened by nibabel here.
        for row in self.rows[:2]:
            for key in ("swi_path", "qsm_path", "swi_mask_path", "mask_path", "qsm_reference_mask_path"):
                Path(row[key]).write_bytes(b"unselected split must not be opened")
        allowed = {str(Path(row[key]).resolve()) for row in self.rows[2:] for key in ("swi_path", "swi_mask_path")}
        actual_load = nib.load
        def selected_load(path, *args, **kwargs):
            self.assertIn(str(Path(path).resolve()), allowed, "QSM or another split was opened")
            return actual_load(path, *args, **kwargs)
        output = self.root / "predictions"
        with mock.patch.object(infer, "load_checkpoint_snapshot", wraps=infer.load_checkpoint_snapshot) as load_checkpoint, \
             mock.patch.object(infer, "generator_from_checkpoint", wraps=infer.generator_from_checkpoint) as load_model, \
             mock.patch.object(nib, "load", side_effect=selected_load):
            payload = predict_split.main(self.cli(output, extra=["--output-units", "ppb"]))
        self.assertEqual(load_checkpoint.call_count, 1)
        self.assertEqual(load_model.call_count, 1)
        self.assertEqual(payload["n_predictions"], 2)
        all_rows, selected = self.predictions(output)
        self.assertEqual(len(all_rows), 4)
        self.assertEqual([r["subject_id"] for r in all_rows], [r["subject_id"] for r in self.rows])
        self.assertTrue(all(not r["prediction_path"] and not r["prediction_units"] for r in all_rows[:2]))
        for row in selected:
            path = Path(row["prediction_path"])
            self.assertEqual(path.parent, output / "predictions")
            self.assertEqual(len(path.name), 31)  # 24 hex characters + .nii.gz
            self.assertTrue(path.is_file())
            values = nib.load(path).get_fdata()
            np.testing.assert_allclose(values[self.brain > 0], -500., atol=1e-4)
            np.testing.assert_array_equal(values[self.brain <= 0], 0.)
            meta = json.loads(sidecar_path(path).read_text())
            self.assertEqual(meta["Units"], "ppb")
            self.assertEqual(meta["ValueSpace"], "physical_qsm")
            self.assertTrue(meta["InverseNormalization"]["applied"])
            self.assertEqual(meta["TrainingState"]["current_run_phase"], "gan")
            self.assertFalse(meta["InferenceDataQC"]["qsm_voxels_decoded_here"])
            self.assertEqual(meta["CheckpointSHA256"], hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        # The output full manifest keeps the original audit identity. evaluate
        # may now read the selected QSM targets, using the same two reports.
        evaluation = self.root / "evaluation"
        argv = ["evaluate", "--manifest", str(output / "evaluation_manifest.csv"), "--split", "test",
                "--output", str(evaluation), "--units", "ppb", "--reference-id", "synthetic_zero",
                "--expected-prediction-phase", "gan", "--qc-report", str(self.qc_path),
                "--independence-report", str(self.independence_path), "--data-range", "1000"]
        with mock.patch("sys.argv", argv):
            evaluate.main()
        summary = json.loads((evaluation / "summary.json").read_text())
        self.assertEqual(summary["config"]["data_independence"]["status"], "verified")
        self.assertEqual(summary["config"]["data_qc"]["status"], "verified")
        self.assertEqual(set(summary["summary"]), {"zero", "prediction"})
        wanted = np.mean([np.abs(self.targets[i][self.brain > 0].astype(np.float64) * 1000 + 500).mean() for i in (2, 3)])
        self.assertAlmostEqual(summary["summary"]["prediction"]["all"]["metrics"]["mae"]["mean"], wanted, places=4)

    def test_ppm_default_and_explicit_ppb_preserve_signed_inverse_exactly_once(self):
        # Unverified inference still needs an explicit source-unit declaration;
        # fallback must not mutate non-prediction metadata in the copied CSV.
        for row in self.rows:
            row.pop("qsm_units")
        self.write_manifest()
        for units, expected in ((None, -.5), ("ppb", -500.)):
            output = self.root / (units or "default")
            extra = ["--qsm-units", "ppm"] + (["--output-units", units] if units else [])
            predict_split.main(self.cli(output, verified=False, split="val", extra=extra))
            all_rows, selected = self.predictions(output)
            self.assertEqual(len(selected), 1)
            self.assertNotIn("qsm_units", all_rows[0])
            values = nib.load(selected[0]["prediction_path"]).get_fdata()
            np.testing.assert_allclose(values[self.brain > 0], expected, atol=1e-4)
            self.assertEqual(selected[0]["prediction_units"], units or "ppm")

    def test_phase_and_qc_conflicts_never_publish(self):
        output = self.root / "wrong_phase"
        args = self.cli(output)
        args[args.index("--expected-phase") + 1] = "warmup"
        with self.assertRaisesRegex(ValueError, "phase"):
            predict_split.main(args)
        self.assertFalse(output.exists())
        row = self.rows[2]
        row["qsm_reference_source"] = "changed after QC"
        self.write_manifest()
        stale = self.root / "stale_metadata"
        args = self.cli(stale, verified=False)
        start = args.index("--allow-unverified-data")
        args[start:start + 2] = ["--qc-report", str(self.qc_path)]
        with self.assertRaisesRegex(ValueError, "QC metadata"):
            predict_split.main(args)
        self.assertFalse(stale.exists())

    def test_stale_qsm_bytes_are_rejected_without_loading_qsm(self):
        Path(self.rows[2]["qsm_path"]).write_bytes(b"changed QSM byte content")
        output = self.root / "stale_bytes"
        # Use an explicit independence opt-out so QC supplies the failure.
        args = self.cli(output, verified=False)
        start = args.index("--allow-unverified-data")
        args[start:start + 2] = ["--qc-report", str(self.qc_path)]
        with self.assertRaisesRegex(ValueError, "QC input hash changed"):
            predict_split.main(args)
        self.assertFalse(output.exists())

    def test_checkpoint_reference_qc_and_independence_must_match(self):
        for label, mutate, pattern in (
            ("reference", lambda state: state["data_reference"].update(region="another_region"), "reference"),
            ("qc", lambda state: state["data_qc"].update(status="unverified"), "verified training data QC"),
            ("independence", lambda state: state["data_independence"].update(manifest_identity_sha256="0" * 64), "different identity"),
        ):
            state = torch.load(self.checkpoint, weights_only=True)
            mutate(state)
            torch.save(state, self.checkpoint)
            output = self.root / label
            with self.assertRaisesRegex(ValueError, pattern):
                predict_split.main(self.cli(output))
            self.assertFalse(output.exists())
            torch.save(self.state, self.checkpoint)

    def test_mid_batch_failure_cleans_staging_and_publishes_nothing(self):
        output = self.root / "failed"
        calls = 0
        original = predict_split.infer_case
        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic failure after first prediction")
            return original(*args, **kwargs)
        with mock.patch.object(predict_split, "infer_case", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                predict_split.main(self.cli(output))
        self.assertEqual(calls, 2)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".swixqsm-predict-split-*")), [])

    def test_manifest_reference_fallback_stays_unverified_and_gan_requires_updates(self):
        state = torch.load(self.checkpoint, weights_only=True)
        state.pop("data_reference")
        torch.save(state, self.checkpoint)
        output = self.root / "declared_reference"
        args = self.cli(output, verified=False, split="val")
        start = args.index("--reference-id")
        del args[start:start + 2]
        predict_split.main(args)
        _, selected = self.predictions(output)
        metadata = json.loads(sidecar_path(selected[0]["prediction_path"]).read_text())
        self.assertEqual(metadata["ReferenceID"], "synthetic_zero")
        self.assertIsNone(metadata["DataReference"])
        self.assertEqual(metadata["DataQC"]["status"], "unverified")
        state["training_state"]["adversarial_generator_updates"] = 0
        torch.save(state, self.checkpoint)
        rejected = self.root / "no_gan_updates"
        with self.assertRaisesRegex(ValueError, "no recorded.*adversarial"):
            predict_split.main(self.cli(rejected, verified=False))
        self.assertFalse(rejected.exists())

    def test_duplicate_case_ids_and_existing_output_are_rejected(self):
        self.rows[3]["case_id"] = self.rows[2]["case_id"]
        self.write_manifest()
        output = self.root / "duplicate"
        with self.assertRaisesRegex(ValueError, "case_id"):
            predict_split.main(self.cli(output, verified=False))
        self.assertFalse(output.exists())
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "keep.txt"
        marker.write_text("unchanged")
        with self.assertRaises(SystemExit):
            predict_split.main(self.cli(existing, verified=False))
        self.assertEqual(marker.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()

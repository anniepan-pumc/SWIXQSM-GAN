import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from ..evaluate_native_nuclei import evaluate, sha, summarize


class NativeNucleiTest(unittest.TestCase):
    def test_dual_grid_mean_error_is_not_voxel_mae_and_hash_changes_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(name, array, spacing):
                path = root / name
                nib.save(nib.Nifti1Image(np.asarray(array, dtype=np.float32), np.diag([spacing, spacing, spacing, 1.])), path)
                return {"path": str(path), "sha256": sha(path)}
            reference = np.zeros((4, 4, 4)); reference[0, 0, :2] = [2., 4.]
            prediction = np.zeros((3, 3, 3)); prediction[0, 0, :2] = [8., 4.]
            left = np.zeros_like(reference); left[0, 0, :2] = 385
            right = np.zeros_like(reference)
            right[1, 0, 0] = 385
            reference[1, 0, 0] = 2.
            left[2, 0, 0] = 310
            pleft = np.zeros_like(prediction); pleft[0, 0, :2] = 385
            pleft[2, 0, 0] = 310
            pright = np.zeros_like(prediction)
            qsm = save("qsm.nii.gz", reference, 1.)
            swi = save("swi.nii.gz", prediction, 2.)
            mask = save("mask.nii.gz", np.ones_like(reference), 1.)
            report = {"case_id": "external__1", "cohort": "external", "subject_id": "1",
                      "registration_status": "candidate_completed_pending_QC", "protocol_revision": "scanner_identity_initialization_v2",
                      "inputs": {"qsm_grid": qsm, "swi": swi, "qsm_reliability_mask": mask},
                      "targets": {"gre": {"labels": {"left": save("left.nii.gz", left, 1.), "right": save("right.nii.gz", right, 1.)}},
                                  "swi": {"labels": {"left": save("pleft.nii.gz", pleft, 2.), "right": save("pright.nii.gz", pright, 2.)}}}}
            rp = root / "registration.json"; rp.write_text(json.dumps(report))
            provenance = {"cohort": "external", "subject_id": "1", "checkpoint_sha256": "a" * 64,
                          "prediction_path": swi["path"], "prediction_sha256": swi["sha256"]}
            pp = root / "provenance.json"; pp.write_text(json.dumps(provenance))
            cases = [{"case_id": "external__1", "cohort": "external", "subject_id": "1",
                      "registration_report": str(rp), "prediction_provenance": str(pp)}]
            result = evaluate(cases, root / "results", source_root=root, reason="Deferred by user")
            self.assertEqual(result["n_measurements_including_zero"], 30)
            import csv
            with (root / "results/native_roi_per_case.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            rn = next(r for r in rows if r["region"] == "RN" and r["side"] == "L" and r["model"] != "zero")
            self.assertEqual(float(rn["roi_mean_abs_error_raw"]), 3.)
            self.assertNotIn("mae", rn)
            self.assertEqual(rn["formal_accuracy_claim"], "False")
            zero_rn = next(r for r in rows if r["region"] == "RN" and r["side"] == "R" and r["model"] == "zero")
            model_rn = next(r for r in rows if r["region"] == "RN" and r["side"] == "R" and r["model"] != "zero")
            self.assertEqual(zero_rn["status"], "numeric_candidate_QC_pending")
            self.assertEqual(zero_rn["n_prediction_voxels"], zero_rn["n_reference_voxels"])
            self.assertEqual(model_rn["status"], "missing_or_incomplete_reliable_ROI")
            whole_sn = next(r for r in rows if r["region"] == "SN" and r["side"] == "L" and r["model"] != "zero")
            self.assertEqual(whole_sn["status"], "missing_or_incomplete_reliable_ROI")
            self.assertEqual(whole_sn["prediction_all_components_present"], "False")
            with self.assertRaisesRegex(ValueError, "expected model"):
                evaluate(cases, root / "wrong_models", source_root=root, reason="Deferred", expected_models=["missing"])
            provenance["prediction_sha256"] = "0" * 64; pp.write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, "hash changed"):
                evaluate(cases, root / "changed", source_root=root, reason="Deferred")

    def test_cohorts_are_not_pooled_and_excluded_rois_remain_counted(self):
        rows = []
        for cohort, bias in (("a", 2.), ("b", -10.)):
            for i in range(3):
                rows.append({"cohort": cohort, "model": "m", "region": "RN", "side": "bilateral",
                             "status": "numeric_candidate_QC_pending" if i < 2 else "missing_or_incomplete_reliable_ROI",
                             "reference_mean_raw": float(i), "prediction_mean_raw": i + bias,
                             "mean_difference_raw": bias, "roi_mean_abs_error_raw": abs(bias),
                             "roi_mean_squared_error_raw": bias ** 2, "zero_roi_mean_abs_error_raw": float(i)})
        summaries = summarize(rows)
        self.assertEqual(len(summaries), 2)
        self.assertEqual([s["bias"] for s in summaries], [2., -10.])
        self.assertTrue(all(s["n_defined_participants"] == 2 and s["n_excluded_participants"] == 1 for s in summaries))
        self.assertTrue(all(s["icc_a1"] is None for s in summaries))


if __name__ == "__main__":
    unittest.main()

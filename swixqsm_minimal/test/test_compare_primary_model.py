"""Synthetic volumes only: a less-smoothed candidate must pass the texture criterion."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
from scipy import ndimage

from .. import compare_primary_model


REGISTRATION = Path(__file__).resolve().parents[1] / "preregistration" / "primary_model_c2_vs_plain_20261029.json"


def save(path, array):
    image = nib.Nifti1Image(array.astype(np.float32), np.diag([.8, .8, 3., 1.]))
    image.header.set_xyzt_units("mm")
    nib.save(image, str(path))


class ComparePrimaryModelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        registration = json.loads(REGISTRATION.read_text())
        registration["internal"]["bootstrap"]["replicates"] = 500
        self.registration = self.root / "registration.json"
        self.registration.write_text(json.dumps(registration))
        self.registration_sha = hashlib.sha256(self.registration.read_bytes()).hexdigest()
        rng = np.random.default_rng(0)
        mask = np.zeros((32, 32, 4))
        mask[4:28, 4:28, :] = 1
        rows = {"sharp": [], "smooth": []}
        for index in range(6):
            sid = f"s{index}"
            reference = ndimage.gaussian_filter(rng.normal(size=mask.shape), (.6, .6, 0)) * 20 + 5
            paths = {key: self.root / f"{sid}_{key}.nii.gz" for key in ("swi", "qsm", "mask", "sharp", "smooth")}
            save(paths["swi"], rng.normal(size=mask.shape))
            save(paths["qsm"], reference)
            save(paths["mask"], mask)
            save(paths["sharp"], ndimage.gaussian_filter(reference, (.5, .5, 0)))
            save(paths["smooth"], ndimage.gaussian_filter(reference, (2., 2., 0)))
            for model in rows:
                rows[model].append(dict(case_id=f"c{index}", subject_id=sid, split="val",
                    swi_path=str(paths["swi"]), qsm_path=str(paths["qsm"]), swi_mask_path=str(paths["mask"]),
                    mask_path=str(paths["mask"]), qsm_units="stored", prediction_path=str(paths[model]),
                    prediction_units="stored", prediction_reference="unverified_original_reference"))
        self.manifests = {}
        for model, model_rows in rows.items():
            path = self.root / model / "evaluation_manifest.csv"
            path.parent.mkdir()
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(model_rows[0]))
                writer.writeheader()
                writer.writerows(model_rows)
            self.manifests[model] = str(path)

    def run_comparison(self, candidate, comparator, name):
        return compare_primary_model.main([
            "--preregistration", str(self.registration), "--preregistration-sha256", self.registration_sha,
            "--candidate", self.manifests[candidate], "--comparator", self.manifests[comparator],
            "--output", str(self.root / name)])

    def test_less_smoothed_candidate_passes_internal_criteria(self):
        summary = self.run_comparison("sharp", "smooth", "out")
        self.assertEqual(summary["n_subjects"], 6)
        self.assertTrue(all(item["passed"] for item in summary["criteria"].values()))
        self.assertEqual(summary["internal_decision"], "internal_criteria_met_external_criterion_D_pending")
        means = summary["descriptive_means"]["hf_power_ratio"]
        self.assertLess(means["comparator"], means["candidate"])
        self.assertTrue((self.root / "out" / "per_subject.csv").is_file())

    def test_more_smoothed_candidate_fails(self):
        summary = self.run_comparison("smooth", "sharp", "out")
        self.assertFalse(summary["criteria"]["B_less_oversmoothing"]["passed"])
        self.assertEqual(summary["internal_decision"], "internal_criteria_not_met_comparator_remains_primary")

    def test_rejects_modified_preregistration(self):
        with self.assertRaises(ValueError):
            compare_primary_model.main([
                "--preregistration", str(self.registration), "--preregistration-sha256", "0" * 64,
                "--candidate", self.manifests["sharp"], "--comparator", self.manifests["smooth"],
                "--output", str(self.root / "out")])


if __name__ == "__main__":
    unittest.main()

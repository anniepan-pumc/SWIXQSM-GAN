"""Save susceptibility, never SWI header intensity scaling or display limits."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def sidecar_path(path):
    path = Path(path)
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[:-7]+".json")
    if name.endswith(".nii"):
        return path.with_suffix(".json")
    raise ValueError("Output must end in .nii or .nii.gz")


def save_qsm(path, values, reference, metadata):
    path = Path(path)
    sidecar = sidecar_path(path)
    if path.exists() or sidecar.exists():
        raise FileExistsError("Prediction or JSON sidecar already exists")
    values = np.asarray(values, dtype=np.float32)
    if values.shape != reference.shape or not np.isfinite(values).all():
        raise ValueError("Prediction must be finite and match the reference grid")
    if metadata.get("Units") not in {"ppm", "ppb", "stored"}:
        raise ValueError("Output susceptibility units must be explicit")
    if metadata["Units"] == "stored":
        if (metadata.get("ValueSpace") != "restored_stored_values"
                or metadata.get("UnitStatus") != "unverified_original_storage"):
            raise ValueError("stored output requires explicit unverified restored_stored_values metadata")
        if metadata.get("DataQC", {}).get("status") in {"verified", "geometry_verified", "passed"}:
            raise ValueError("stored values cannot be certified by a physical-unit data QC contract")
    elif metadata.get("ValueSpace") == "restored_stored_values":
        raise ValueError("restored_stored_values cannot declare physical ppm/ppb units")
    # Keep spatial metadata, replace all intensity semantics of the source SWI.
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_intent("none")
    header["cal_min"], header["cal_max"] = float(values.min()), float(values.max())
    description = ("SWIXQSM; unverified stored values" if metadata["Units"] == "stored"
                   else f"SWIXQSM; susceptibility {metadata['Units']}")
    header["descrip"] = description.encode()
    image = nib.Nifti1Image(values, reference.affine, header)
    image.header.set_slope_inter(1., 0.)
    serialized = json.dumps(metadata, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)
    sidecar.write_text(serialized+"\n")

"""Resolve ground-truth QSM NIfTI path for a subject directory."""
import os


def resolve_qsm_gt_path(subject_dir):
    """
    Prefer QSM/result/QSM.nii.gz (reconstructed pipeline output), then UKB-style
    QSM/QSM.nii.gz, then QSM.nii.gz at subject root.
    """
    candidates = (
        os.path.join(subject_dir, "QSM", "result", "QSM.nii.gz"),
        os.path.join(subject_dir, "QSM", "QSM.nii.gz"),
        os.path.join(subject_dir, "QSM.nii.gz"),
    )
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

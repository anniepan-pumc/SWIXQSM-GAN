#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch-run validate_swi_qsm (PyTorch GAN) on a dataset root; GT via qsm_gt_path."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from qsm_gt_path import resolve_qsm_gt_path
from validate_swi_qsm import SWIQSMValidator


def find_subjects(data_path):
    out = []
    for subject_id in sorted(os.listdir(data_path)):
        subject_dir = os.path.join(data_path, subject_id)
        if not os.path.isdir(subject_dir):
            continue
        swi_path = os.path.join(subject_dir, "SWI.nii.gz")
        qsm_path = resolve_qsm_gt_path(subject_dir)
        if os.path.isfile(swi_path) and qsm_path:
            out.append((subject_id, swi_path, qsm_path))
    return out


def flatten_metrics(metrics, subject_id):
    flat = {"subject_id": subject_id}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                col_name = f"{k}_{k2}" if k2 else k
                if isinstance(v2, np.ndarray):
                    flat[col_name] = float(v2.item() if v2.size == 1 else v2.mean())
                elif isinstance(v2, (int, float, np.integer, np.floating)):
                    flat[col_name] = float(v2)
                else:
                    flat[col_name] = v2
        elif isinstance(v, np.ndarray):
            flat[k] = float(v.item() if v.size == 1 else v.mean())
        elif isinstance(v, (int, float, np.integer, np.floating)):
            flat[k] = float(v)
        else:
            flat[k] = v
    return flat


def main():
    parser = argparse.ArgumentParser(description="Batch GAN SWI→QSM validation")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--gen_model_path", type=str, default=None)
    parser.add_argument("--disc_model_path", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu | cuda | mps; default: auto",
    )
    parser.add_argument("--method", type=str, default="robust_minmax")
    args = parser.parse_args()

    subjects = find_subjects(args.data_path)
    if not subjects:
        print("No subjects with SWI.nii.gz + resolved QSM GT.")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects under {args.data_path}")

    validator = SWIQSMValidator(
        gen_model_path=args.gen_model_path,
        disc_model_path=args.disc_model_path,
        device=args.device,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    rows = []
    for subject_id, swi_path, qsm_path in subjects:
        try:
            metrics = validator.validate(
                swi_path=swi_path,
                qsm_path=qsm_path,
                method=args.method,
                if_mask=False,
                mask_path=None,
                output_dir=None,
                flipAxes=[False, False, False],
            )
            rows.append(flatten_metrics(metrics, subject_id))
            print(f"OK {subject_id}")
        except Exception as e:
            print(f"FAIL {subject_id}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.output_csv, index=False, encoding="utf-8")
        print(f"Saved: {args.output_csv} ({len(rows)} rows)")
    else:
        print("No successful rows.")
        sys.exit(1)


if __name__ == "__main__":
    main()

# SWIXQSM-GAN

Synthetic quantitative susceptibility mapping (QSM) from routine susceptibility-weighted imaging (SWI).

This GitHub package is the **revision codebase** used for the NeuroImage resubmission workflows. It includes the SWIXQSM U-Net / GAN training stack, physical-unit evaluation, and the named **C2** recipe (`λ_L1=100`, `λ_adv=0.1`).

**Trained weights are not included.** Provide your own checkpoints or retrain.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r swixqsm_minimal/requirements.txt
```

Python ≥ 3.10. CUDA is recommended for training; inference also runs on CPU / MPS depending on PyTorch build.

## Quick start

List recipes (includes `swixqsm_c2`):

```bash
python -m swixqsm_minimal.train --list-models
```

Train the L1-only primary U-Net or the weak-adversarial C2 variant (paths are examples):

```bash
python -m swixqsm_minimal.train \
  --model plain_unet \
  --manifest /path/to/manifest.csv \
  --output /path/to/exp/plain_unet \
  --units stored --normalization-file /path/to/qsm_normalization.json \
  --allow-unverified-data "state your QC policy" \
  --device cuda

python -m swixqsm_minimal.train \
  --model swixqsm_c2 \
  --manifest /path/to/manifest.csv \
  --output /path/to/exp/swixqsm_c2 \
  --units stored --normalization-file /path/to/qsm_normalization.json \
  --allow-unverified-data "state your QC policy" \
  --device cuda
```

Inference and evaluation entry points:

```bash
python -m swixqsm_minimal.test.infer --help
python -m swixqsm_minimal.predict_split --help
python -m swixqsm_minimal.evaluate --help
```

## Package layout

```text
SWIXQSM-GAN/
├── swixqsm_minimal/     # Train / infer / evaluate library + docs
├── configs/recipes/     # Documented named recipes (incl. C2)
├── LICENSE              # MIT
├── CITATION.cff
├── AUTHORS.md
└── README.md
```

Technical notes (normalization, QC, server training, reviewer baselines) live as Markdown under `swixqsm_minimal/`.

## What is intentionally omitted

- Model checkpoints (`.pt` / `.pth`)
- Medical imaging volumes (`.nii.gz`, DICOM, …)
- Private cohort exports, participant ID lists, and local experiment `outputs/`
- Site-specific server paths and restricted-cohort operational notes

This public tree omits restricted-cohort participant identifiers, site paths,
and trained weights.

## Citation

See `CITATION.cff`. Update the repository URL and coauthors before public archival.

## License

MIT — see `LICENSE`.

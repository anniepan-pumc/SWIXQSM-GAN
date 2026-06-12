# SWIXQSM-GAN

Synthetic quantitative susceptibility mapping from routine susceptibility-weighted MRI.

SWIXQSM-GAN is a PyTorch framework for translating susceptibility-weighted imaging
(SWI) into QSM-like maps. It was developed for research workflows where paired SWI
and reconstructed quantitative susceptibility mapping (QSM) data are available for
training, and where fast QSM-like inference from routine SWI may support biomarker
exploration.

The repository now exposes the core workflow as a GitHub-style project:

- load paired SWI/QSM NIfTI volumes;
- train a 2D slice-wise GAN with a U-Net generator and PatchGAN-style discriminator;
- generate synthetic QSM from SWI;
- evaluate image-level agreement with PSNR, SSIM, MAE, RMSE, NMSE, HFEN, FID, and
  correlation metrics;
- support downstream ROI susceptibility analysis scripts.

## Core Functionality

The SWIXQSM-GAN pipeline is built from four main components.

1. **Data loading**: `data.ukb_loader.UKBDataset` reads subject-level NIfTI files,
   selects paired cases from a CSV manifest, slices volumes when requested, resizes
   images to 256 x 256, and normalizes intensities.
2. **Generator**: `model.unet.UNetSmall` maps one-channel SWI slices to one-channel
   synthetic QSM slices.
3. **Discriminator**: `model.discriminator.Discriminator_ccgan` provides the
   adversarial PatchGAN-style signal used during training.
4. **Training and validation**: `utils.train_gan.fit2d` trains with adversarial loss
   plus L1 content loss, while `validate_swi_qsm.py` performs NIfTI inference,
   geometry restoration, optional GT alignment, and metrics.

The same components can be imported through the package facade:

```python
from swixqsm_gan.models import UNetSmall, DiscriminatorCCGAN
from swixqsm_gan.data import UKBDataset
from swixqsm_gan.training import fit2d
from swixqsm_gan.inference import SWIQSMValidator
```

## Repository Structure

```text
SWIXQSM-GAN/
├── swixqsm_gan/              # Importable package facade for core functions
├── model/                    # Minimal generator and discriminator implementations
├── data/                     # Dataset loader and local data-preparation scripts
├── utils/                    # Training loop, metrics, losses, postprocessing
├── scripts/                  # ROI analysis and figure-generation scripts
├── configs/                  # Example dataset manifests
├── validate_swi_qsm.py       # Single-subject and multi-dataset validation CLI
├── batch_validate_swi_qsm.py # Batch validation over subject folders
├── train_swixqsm_gan.py      # Public training CLI
├── requirements.txt          # pip environment
├── environment.yml           # conda environment
└── docs/
```

Large imaging data, trained checkpoints, generated outputs, and local cohort files
are intentionally excluded by `.gitignore`.

## Environment

Python 3.11 is recommended.

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Using Conda or Mamba:

```bash
conda env create -f environment.yml
conda activate swixqsm-gan
```

Training was designed for CUDA GPUs. Inference can run on CUDA, Apple MPS, or CPU;
runtime and memory use depend on volume size and batch size.

## Expected Data Layout

The training loader expects a subject-folder layout and a CSV manifest.

```text
dataset_root/
└── sub-001/
    ├── SWI.nii.gz
    └── QSM.nii.gz
```

The manifest should contain at least:

```csv
SWI,QSM,InBoth
sub-001,sub-001,1
sub-002,sub-002,1
```

For validation, `validate_swi_qsm.py` also supports a BIDS-like layout:

```text
dataset_root/
└── sub-001/
    ├── SWI/sub-001_SWI.nii.gz
    └── QSM/result/QSM.nii.gz
```

## Training

Use `train_swixqsm_gan.py` for a portable training entry point:

```bash
python train_swixqsm_gan.py \
  --data_path /path/to/dataset_root \
  --record_path /path/to/manifest.csv \
  --output_dir ./save/swixqsm_run \
  --epochs 20 \
  --batch_size 32 \
  --grad_accum_steps 1 \
  --shuffle
```

Resume or fine-tune from checkpoints:

```bash
python train_swixqsm_gan.py \
  --data_path /path/to/dataset_root \
  --record_path /path/to/manifest.csv \
  --gen_checkpoint /path/to/gen.pt \
  --disc_checkpoint /path/to/disc.pt \
  --output_dir ./save/swixqsm_finetune
```

The original `ukb_trainer.py` is preserved as a provenance script from local
development, but it contains machine-specific paths. Prefer `train_swixqsm_gan.py`
for shared or reproducible use.

## Inference and Validation

Generate and evaluate one subject:

```bash
python validate_swi_qsm.py \
  --swi_path /path/to/SWI.nii.gz \
  --qsm_path /path/to/QSM.nii.gz \
  --gen_model_path /path/to/gen.pt \
  --disc_model_path /path/to/disc.pt \
  --output_dir ./output/sub-001 \
  --save_metrics ./output/sub-001_metrics.csv
```

Generate synthetic QSM only:

```bash
python validate_swi_qsm.py \
  --swi_path /path/to/SWI.nii.gz \
  --generate_only \
  --gen_model_path /path/to/gen.pt \
  --disc_model_path /path/to/disc.pt \
  --output_dir ./output/sub-001
```

Run multiple datasets:

```bash
python validate_swi_qsm.py \
  --run_dataset ukb:/path/to/ukb_validate \
  --run_dataset bids:/path/to/csvd_bids \
  --gen_model_path /path/to/gen.pt \
  --disc_model_path /path/to/disc.pt \
  --aggregate_csv ./output/metrics_all.csv
```

## ROI Analysis

ROI-level agreement scripts are kept under `scripts/`. They were used for regional
susceptibility analysis after synthetic QSM generation. These scripts may require
project-specific cohort tables, FreeSurfer outputs, and local file paths.

## Release Notes for Public GitHub Use

Before publishing, check `docs/repository_release_checklist.md`.

Recommended public repository policy:

- do not commit protected cohort imaging data;
- do not commit raw DICOM/NIfTI files unless they are synthetic or explicitly
  shareable;
- keep trained weights outside git unless redistribution is allowed;
- replace local absolute paths in analysis scripts before release;
- add a small public demo dataset or synthetic example when available.

## Citation

If you use this repository, cite the software metadata in `CITATION.cff`.

Suggested citation:

> Pan X. SWIXQSM-GAN: Synthetic QSM from routine SWI for brain susceptibility
> biomarker assessment. Version 0.1.0. 2026.

## Data Availability

This repository contains code for model development and evaluation. It does not
include UK Biobank data, PUMCH cohort data, or protected clinical imaging files.
Users should obtain imaging data through the appropriate institutional or database
access procedures.

## Clinical Use Disclaimer

SWIXQSM-GAN is research software. Synthetic QSM should not replace fully
reconstructed QSM when absolute susceptibility quantification is required, and it
should not be used as a standalone clinical diagnostic tool without appropriate
validation and regulatory approval.

# Repository Structure

This repository separates reusable SWIXQSM-GAN code from local research
artifacts and protected imaging data.

```text
SWIXQSM-GAN/
├── swixqsm_gan/              # Importable package facade for core functions
│   ├── models/               # Generator and discriminator exports
│   ├── data/                 # Dataset loader exports
│   ├── training/             # GAN training-loop exports
│   └── inference/            # Validation/inference exports
├── model/                    # Minimal generator and discriminator implementations
├── data/                     # Dataset loaders and data-preparation scripts
├── utils/                    # Metrics, losses, postprocessing, training utilities
├── scripts/                  # ROI analysis and publication figure scripts
├── configs/                  # Example dataset manifests
├── validate_swi_qsm.py       # Single-subject and batch validation CLI
├── batch_validate_swi_qsm.py # Batch validation CLI for subject folders
├── train_swixqsm_gan.py      # Public training CLI without local hard-coded paths
├── requirements.txt          # pip environment
├── environment.yml           # conda environment
└── README.md
```

The upload-ready copy keeps only the model definitions required by
`train_swixqsm_gan.py` and `validate_swi_qsm.py`: `model/unet.py` for the
generator and `model/discriminator.py` for the discriminator.

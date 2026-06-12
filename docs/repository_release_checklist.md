# Repository Release Checklist

Use this checklist before publishing or archiving the SWIXQSM-GAN repository.

## Required Before Public Release

- [ ] Replace `https://github.com/USERNAME/SWIXQSM-GAN` in `CITATION.cff` with the final repository URL.
- [ ] Add all manuscript coauthors and contributors to `CITATION.cff`, `.zenodo.json`, and `AUTHORS.md`.
- [ ] Confirm the intended license. The current root license is MIT.
- [ ] Remove protected health information and private cohort files.
- [ ] Remove local paths from scripts or document how users should edit them.
- [ ] Upload trained weights only if sharing is permitted by institutional policy.
- [ ] Add a DOI after Zenodo archival.
- [ ] Update the README citation after manuscript publication.

## Recommended

- [ ] Add a small public demo dataset or synthetic example data.
- [ ] Add a reproducible configuration file for training and validation paths.
- [ ] Add unit or smoke tests for model loading and metric calculation.
- [ ] Add environment information for CUDA, Apple MPS, and CPU inference.


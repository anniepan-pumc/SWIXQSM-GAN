# Named training recipes

Use with:

```bash
python -m swixqsm_minimal.train --model <name> ...
```

| name | architecture | λ_L1 | λ_adv | notes |
|---|---|---:|---:|---|
| `plain_unet` | SWIXQSM U-Net | 1 | 0 | L1-only primary after A–C check |
| `swixqsm_c2` | SWIXQSM U-Net | 100 | 0.1 | revision C2 (weak adversarial) |
| `swixqsm_lambda100` | SWIXQSM U-Net | 100 | 1 | B1-style strong L1 + full adv |
| `pix2pix` | pix2pix U-Net | 100 | 1 | comparator |
| `fcn_gan_l1` | FCN (no skip) | 0.2 | 1 | architecture ablation |
| `swixqsm_gan` | SWIXQSM U-Net | 0.2 | 1 | original coefficient default |

JSON files below are documentation only; coefficients are hard-coded in
`swixqsm_minimal.train_models.RECIPES`.

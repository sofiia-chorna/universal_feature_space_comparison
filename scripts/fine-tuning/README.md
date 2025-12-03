# Fine-Tuning

Scripts and models for analyzing PET model fine-tuning experiments and tracking reconstruction errors across training epochs.

- `errors.py` – compares reconstruction errors across different fine-tuned PET variants (bespoke, pet-mad, ff, hf, ftl, hft)
- `errors-progress.py` – Tracks reconstruction error evolution during fine-tuning across training epochs for each variant
- `get_llfs_ft-progress.py` – Extracts last-layer features (LLFs) from model checkpoints at different training epochs
- `pre_steps.sh` – Setup script for ependencies before fine-tuning runs
- `pet-mad-v1.0.2.ckpt` – Pre-trained PET model checkpoint (baseline)
- `LPS.extxyz, train.extxyz, val.extxyz, test.extxyz` – Dataset files for fine-tuning experiments

In `eval` folder there are the logs of prediction errors across those PET checkpoints.

In folders `bespoke`, `ff`, `hf`, `ftl`, and `htl` there are configuration files, model checkpoints and scripts to train a PET moddel with the corresponding training strategy.

The calculated features are stored in `data/features/lips/*`. Errors are in `results/reconstruction_errors/fine-tuning/progress/*`

# Data

This directory contains datasets and precomputed feature representations used by the project plotting and analysis workflows.

The feature arrays stored under `data/features/` (for example, the `cumulants/` and `fine-tuning/` subfolders) are the ones used directly by the notebooks in `plotting/` and the analysis scripts in `scripts/`.
The `data/xyz/` folder contains ASE structure files used as inputs to feature extraction pipelines.

If you need to regenerate features, run the extraction scripts located in `scripts/last-layer-features/`.

Typical workflow:

Create or activate the model-specific conda environment (see `models/README.md` and `models/envs/`).

```bash
conda activate mace
```

Run the model-specific extraction script found inside `scripts/last-layer-features/<model>/`.

```bash
cd scripts/last-layer-features/<model>/
python extract_features.py --help
```

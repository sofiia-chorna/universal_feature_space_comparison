# Scripts

This directory contains all Python and shellscripts for data preprocessing, feature extraction, and analysis.

## Overview

```
scripts/
├── data_preprocess/          # Data preparation and filtering
├── last-layer-features/      # Extract embeddings from ML models
├── cumulants/                # Statistical moment analysis
├── dos/                      # Density of states reconstruction
├── fine-tuning/              # Model fine-tuning experiment
├── ll_vs_bb/                 # Last-layer vs. backbone comparison
├── umlips/                   # Cross UMLIP analysis
├── variants/                 # Model variant comparisons
├── src/                      # Utilities
```

## Pipeline workflows

The first step is to prepare dataset subsets for analysis by executing the following scripts:
- `get_consistent_mad_test.py` — Extract consistent test set from MAD
- `get_organic_mad_test.py` — Extract organic molecules from MAD test set
- `get_consistent_salexandria.py` — Extract consistent test set from Alexandria dataset

Next, we extract last-layer features from each ML model (see `last-layer-features/`).

Other scripts then are using them to calculate reconstruction errors. Please activate the related setup environment for error calculations:

```bash
conda activate skmatter
```

See `models/README.md` for environment setup instructions.

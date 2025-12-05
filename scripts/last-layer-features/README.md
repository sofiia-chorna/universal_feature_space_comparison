# Last-layer features extraction scripts

This directory contains helper scripts to extract last-layer features (LLFs), as well as backbone features, from trained models for several model variants.

Before running any extraction script, you must activate the corresponding conda environment for the model:

```bash
# For DPA
conda activate dpa-env

# For MACE
conda activate mace-env

# For PET
conda activate pet-env

# For UMA
conda activate uma-env
```

Then proceed with the scripts below. See `models/README.md` for environment setup instructions.

## Directory structure and scripts

- dpa/
  - `pre_steps.py` - installation steps required before running feature extraction for DPA
  - `get_llfs.py`  - script that extracts last-layer features from a DPA model checkpoint for a dataset
  - `run.sh`       - convenience wrapper to run preprocessing and the extraction pipeline for DPA

- mace/
  - `pre_steps.sh` - installation script performing setup for MACE before extraction
  - `get_llfs.py`  - script that extracts last-layer features from a MACE model checkpoint for a dataset
  - `run.sh`       - convenience wrapper to run preprocessing and the extraction pipeline for MACE

- pet/
  - `pre_steps.sh` - installation script performing setup for PET before extraction
  - `get_llfs.py`  - script that extracts last-layer and backbone features from a PET model checkpoint for a dataset
  - `run.sh`       - convenience wrapper to run preprocessing and the extraction pipeline for PET

- uma/
  - `pre_steps.sh` - installation script performing setup for UMA before extraction
  - `get_llfs.py`  - script that extracts last-layer and backbone features from a UMA model checkpoint for a dataset
  - `run.sh`       - convenience wrapper to run preprocessing and the extraction pipeline for UMA

Results of execution are saved in `data/features/mad/umlips/*`.

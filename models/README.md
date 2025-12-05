# Models & environments

This directory contains trained model checkpoints and conda environment configurations for running machine learning interatomic potentials (MLIPs) used in the feature space comparison project.

## Setting up environments

This script creates all five conda environments from the `.yml` files in `envs/`.

```bash
cd models/
bash create_envs.sh
```

## Downloading models

```bash
cd models/
bash get_models.sh
```

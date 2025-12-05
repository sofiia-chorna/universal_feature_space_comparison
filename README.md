This repository contains scripts to reproduce results of the paper "Comparing the latent features of universal
machine-learning interatomic potentials".

## Project structure

```
├── data/
│   ├── features/                  # Generated feature representations
│   └── xyz/                       # ASE files
├── models/
│   ├── envs/                      # Conda environments
│   └── models/                    # Pre-trained MLIP checkpoints
├── scripts/
│   ├── data_preprocess/          # Dataset preparation and filtering
│   ├── last-layer-features/      # Feature extraction pipelines
│   ├── cumulants/                # Cumulants experiment
│   ├── dos/                      # PET-MAD-DOS experiment
│   ├── fine-tuning/              # Fine-tuning experiment
│   ├── ll_vs_bb/                 # Last-layer vs backbone comparison
│   ├── umlips/                   # Cross-model MLIP analysis
│   └── variants/                 # Model variant comparisons
├── plotting/                     # Jupyter notebooks for visualization
├── results/                      # Generated figures and error metrics
└── src/                          # Utilities
```

## Getting started

First, create all conda environments for the different MLIPs:

```bash
cd models/
bash create_envs.sh
```

Second, download model checkpoints

```bash
cd models/
bash get_models.sh
```

This fetches pre-trained model weights for all MLIPs used in the analysis.

Next, prepare dataset subsets for analysis:

```bash
conda activate skmatter
cd scripts/data_preprocess/

python get_consistent_mad_test.py
python get_organic_mad_test.py
python get_consistent_salexandria.py
```

Generate last-layer features from each model:

```bash
cd scripts/last-layer-features/

# Extract features for each MLIP (DPA, MACE, PET, UMA)
# See individual README files in each model directory
```

Run analysis scripts to compute reconstruction errors:

```bash
conda activate skmatter

# Calculate statistical moments
cd scripts/cumulants/
bash run_calculate_cumulant.sh
bash run_errors_model.sh
bash run_errors_umlip.sh

# Other analyses
# ...
```

Finally, run Jupyter notebooks to re-create figures:

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn jupyterlab notebook

cd plotting/
jupyter notebook
```

Generated outputs are stored in the `results/` directory:

```
results/
├── figures/                       # Plots
└── reconstruction_errors/         # Quantitative metrics in JSON format
```

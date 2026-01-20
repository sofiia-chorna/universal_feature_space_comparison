# Comparing the latent features of universal machine-learning interatomic potentials
This repository contains scripts to reproduce results from the paper "Comparing the latent features of universal machine-learning interatomic potentials" (preprint: https://arxiv.org/abs/2512.05717).

## Abstract
The past few years have seen the development of "universal" machine-learning interatomic potentials (uMLIPs) capable of approximating the ground-state potential energy surface across a wide range of chemical structures and compositions with reasonable accuracy. While these models differ in the architecture and the dataset used, they share the ability to compress a staggering amount of chemical information into descriptive latent features. Herein, we systematically analyze what the different uMLIPs have learned by quantitatively assessing the relative information content of their latent features with feature reconstruction errors as metrics, and observing how the trends are affected by the choice of training set and training protocol. We find that the uMLIPs encode chemical space in significantly distinct ways, with substantial cross-model feature reconstruction errors. When variants of the same model architecture are considered, trends become dependent on the dataset, target, and training protocol of choice. We also observe that fine-tuning of a uMLIP retains a strong pre-training bias in the latent features. Finally, we discuss how atom-level features, which are directly output by MLIPs, can be compressed into global structure-level features via concatenation of progressive cumulants, each adding significantly new information about the variability across the atomic environments within a given system.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b0b0dcb8-869f-4f0f-a3a9-298c5eb62609" alt="cover" width="600">
</p>

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

## Citing
```
@misc{chorna2025comparinglatentfeaturesuniversal,
      title={Comparing the latent features of universal machine-learning interatomic potentials}, 
      author={Sofiia Chorna and Davide Tisi and Cesare Malosso and Wei Bin How and Michele Ceriotti and Sanggyu Chong},
      year={2025},
      eprint={2512.05717},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2512.05717}, 
}
```

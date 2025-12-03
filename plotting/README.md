## Plotting notebooks

The folder contains Jupyter notebooks and helper scripts for generating figures used in the paper.
To run them, install the libraries first:
`pip install numpy scipy pandas matplotlib seaborn scikit-learn jupyterlab notebook`

Generated figures are stored under `results/figures/` subfolders (e.g. `results/figures/cumulants`, `results/figures/fine-tuning`, etc.). The plotting notebooks use the JSON results of reconstruction from the  `results/reconstruction_errors/*` folders.

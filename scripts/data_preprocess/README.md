## Data preprocessing scripts

This folder contains helper scripts used to filter and prepare the XYZ files for MAD and sAlexandria datasets.

- `get_consistent_mad_test.py`. Filter out frames from `data/xyz/mad-test.xyz` that are single-atom or contain elements with atomic numbers 84 or 86. Output of execution is in `data/raw/mad-test-consistent.xyz`

- `get_consistent_salexandria.py`. Filter `data/xyz/alex_val_sub.xyz` to remove single-atom frames, frames that contain actinides and frames with no sufficiently close atoms (minimum pair distance > 6.0 Å). Output is in `data/xyz/alex_val_sub_consistent.xyz`

- `get_organic_mad_test.py`. Filter `data/xyz/mad-test-consistent.xyz` keeping only frames whose atomic numbers are a subset of a set of organic elements. Output is in `data/xyz/mad-test-consistent-organic.xyz`

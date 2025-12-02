# Last-Layer vs backbone

Reconstruction error comparison between last-layer features (LL) and backbone (BB) features.

- `uma.py` – compares UMA model LL vs BB reconstruction errors across datasets (OC20, ODAC, OMAT, OMC, OMOL).
- `pet.py`– compares PET model LL vs BB reconstruction errors across datasets.

Results are saved to `results/reconstruction_errors/ll_vs_bb/<error_type>_<model>.json` containing reconstruction errors.

Before executing the scripts, it is first necessary to calculate features using `scripts/last-layer-features/pet/get_llfs.py` for PET or `scripts/last-layer-features/uma/get_llfs.py` for UMA . Already calculated features are saved in `data/features/mad/umlips/pet` for PET and in `data/features/mad/umlips/uma` for UMA.

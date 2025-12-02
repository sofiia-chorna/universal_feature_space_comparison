# Model variants comparison

Reconstruction error analysis comparing different variant checkpoints across model architectures.

- `dpa.py` – compares DPA model variants (MPTraj, OC20M, ODAC23, Omat24, SPICE2)
- `mace.py` – compares MACE model variants (matpes-pbe, matpes-r2scan, mptraj, omat)
- `mace-mp-off.py` – compares MACE models (`mace-mp-0b3` vs `mace-off23`) on organic subset
- `mace-sizes-mp.py` – compares MACE models of different sizes of `mace-mp-0a`
- `mace-sizes-off.py` – compares MACE models of different sizes of `mace-off23`
- `pet.py` – compares uPET model variants (mad, mptraj, omatpes, omat-l, oam, omad).
- `pet-sizes.py` – compares uPET OMAT models of different sizes.
- `uma.py` – compares UMA model variants (oc20, odac, omat, omc, omol)

Results are saved to `results/reconstruction_errors/variants/<model>/*`.

The features used in the scripts are saved in `data/features/mad/umlips/*`. They were calculated using the scripts in `scripts/last-layer-features`.


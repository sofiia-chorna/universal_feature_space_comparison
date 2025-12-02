# Density of States (DOS)

Reconstruction error comparison between PET-MAD models with and without density of states (DOS).

`errors.py` compares reconstruction errors between standard PET-MAD and PET-MAD-DOS variants.

Results are saved to `results/reconstruction_errors/dos/` containing:
- `lfre_dos.json` – LFRE comparison between PET-MAD and PET-MAD-DOS.
- `gfre_dos.json` – GFRE comparison between PET-MAD and PET-MAD-DOS.

The features used in the scripts were first calculated using `scripts/last-layer-features/pet/get_llfs.py` and stored in `data/features/mad/umlips/pet/*`.

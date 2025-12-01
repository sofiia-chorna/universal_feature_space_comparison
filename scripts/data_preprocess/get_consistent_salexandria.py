import ase.io
import numpy as np

structures = ase.io.read("data/xyz/alex_val_sub.xyz", ":")
print(f"initial number of structures: {len(structures)}")

ATOMIC_NUMBERS_OUT = [89, 90, 91, 92, 93, 94]

filtered_out_single_atom = 0
filtered_out_atomic_numbers = 0
filtered_out_no_close_atoms = 0

filtered_structures = []

for i, structure in enumerate(structures):
    atomic_numbers = structure.get_atomic_numbers()

    if len(structure) == 1:
        filtered_out_single_atom += 1
        continue

    if np.isin(structure.get_atomic_numbers(), ATOMIC_NUMBERS_OUT).any():
        filtered_out_atomic_numbers += 1
        continue

    distances = structure.get_all_distances()
    mask = distances > 0
    if np.any(mask):
        min_dist = np.min(distances[mask])
        if min_dist > 6.0:
            filtered_out_no_close_atoms += 1
            continue

    filtered_structures.append(structure)

print(f"final number of structures: {len(filtered_structures)}")
ase.io.write("data/xyz/alex_val_sub_consistent.xyz", filtered_structures)

import ase.io

structures = ase.io.read("./data/xyz/mad-test.xyz", ":")
print(f"initial number of test structures {len(structures)}")

filtered_structures = []
for structure in structures:
    atomic_numbers = structure.get_atomic_numbers()

    if len(structure) != 1 and (84 not in atomic_numbers and 86 not in atomic_numbers):
        filtered_structures.append(structure)

print(f"final number of structures: {len(filtered_structures)}")

ase.io.write("./data/raw/mad-test-consistent.xyz", filtered_structures)

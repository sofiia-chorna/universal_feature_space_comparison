import ase.io
import numpy as np
from sklearn.model_selection import train_test_split


def get_random_indices(feats, seed=0):
    print(f"using seed", seed)
    full_data_len = len(next(iter(feats.values())))

    N_TEST = 1000
    rng = np.random.default_rng(seed)
    global_test_indices = rng.choice(full_data_len, size=N_TEST, replace=False)
    print(f"Selected {N_TEST} test indices")

    return global_test_indices


def get_stratified_indices(DATASET_PATH, seed=0, n_test=1000, per_structure=False):
    print(f"using seed {seed}")

    structures = ase.io.read(DATASET_PATH, ":")

    structure_subsets = [s.info.get("subset", "unknown") for s in structures]
    unique_subsets = list(set(structure_subsets))
    subset_to_idx = {subset: i for i, subset in enumerate(unique_subsets)}

    if per_structure:
        print("performing stratified split per structure.")

        all_structure_indices = np.arange(len(structures))
        numerical_labels = [subset_to_idx[label] for label in structure_subsets]

        test_size = n_test / len(all_structure_indices)

        _, test_structure_indices = train_test_split(
            all_structure_indices,
            test_size=test_size,
            random_state=seed,
            stratify=numerical_labels,
        )

        print(f"selected {len(test_structure_indices)} test structures")

        test_subsets = [structure_subsets[i] for i in test_structure_indices]
        print("subset distribution in test set (by structures):")
        for subset in unique_subsets:
            count = test_subsets.count(subset)
            print(
                f"{subset}: {count} structures ({count/len(test_structure_indices)*100:.1f}%)"
            )

        return test_structure_indices

    else:
        print("performing stratified split per atom")

        atom_to_subset = []
        atom_to_structure_idx = []

        for struct_idx, structure in enumerate(structures):
            subset = structure.info.get("subset", "unknown")
            for _ in range(len(structure)):
                atom_to_subset.append(subset)
                atom_to_structure_idx.append(struct_idx)

        unique_subsets = list(set(atom_to_subset))
        subset_to_idx = {subset: i for i, subset in enumerate(unique_subsets)}
        numerical_labels = [subset_to_idx[label] for label in atom_to_subset]

        all_atom_indices = np.arange(len(atom_to_subset))
        test_size = n_test / len(all_atom_indices)

        _, test_atom_indices = train_test_split(
            all_atom_indices,
            test_size=test_size,
            random_state=seed,
            stratify=numerical_labels,
        )

        print(f"selected {len(test_atom_indices)} test atom indices")

        test_subsets = [atom_to_subset[i] for i in test_atom_indices]
        print("subset distribution in test set (by atoms):")
        for subset in unique_subsets:
            count = test_subsets.count(subset)
            print(f"{subset}: {count} atoms ({count/len(test_atom_indices)*100:.1f}%)")

        test_structure_indices = [atom_to_structure_idx[i] for i in test_atom_indices]
        unique_test_structures = set(test_structure_indices)
        print(f"these atoms come from {len(unique_test_structures)} unique structures")

        structure_subsets = [
            structures[idx].info.get("subset", "unknown")
            for idx in unique_test_structures
        ]
        print("subset distribution in test set (by structures):")
        for subset in unique_subsets:
            count = structure_subsets.count(subset)
            print(
                f"{subset}: {count} structures ({count/len(unique_test_structures)*100:.1f}%)"
            )

        return test_atom_indices

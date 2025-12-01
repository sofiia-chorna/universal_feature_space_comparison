import ase.io
import numpy as np
import torch
from deepmd.infer import DeepPot
import sys


if len(sys.argv) < 2:
    print("Usage: python get_llfs.py [dpa_branch]")
    sys.exit(1)

task = sys.argv[1]
print("task", task)

if task not in ["Mptraj", "Omat24", "OC20M", "SPICE2", "ODAC23"]:
    raise ValueError(f"Invalid model name")

OUTPUT_PATH = f"data/features/mad/umlips/dpa/dpa-{task}"  #  or "data/features/alexandria/umlips/dpa/dpa-{task}"
print("OUTPUT_PATH", OUTPUT_PATH)

DATASET_PATH = (
    "data/xyz/mad-test-consistent.xyz"  #  or "data/xyz/alex_val_sub_consistent.xyz"
)
print("DATASET_PATH", DATASET_PATH)

MODEL_PATH = f"./models/dpa/dpa3.1_{task}.pth"


def process_deepmd_features(deepmd_features, device):
    combined_descriptors = []
    mean_descriptors = []

    for atom_features in deepmd_features:
        n_atoms = atom_features.shape[0]
        mean_features = torch.mean(atom_features, dim=0)
        std_features = (
            torch.zeros_like(mean_features)
            if n_atoms == 1
            else torch.std(atom_features, dim=0)
        )
        combined = torch.cat([mean_features, std_features], dim=0)

        combined_descriptors.append(combined)
        mean_descriptors.append(mean_features)

    return torch.stack(combined_descriptors, dim=0), torch.stack(
        mean_descriptors, dim=0
    )


def extract_deepmd_features(structures, model_path, device="cpu"):
    dp = DeepPot(model_path, device=device)
    all_atom_features = []

    for _i, structure in enumerate(structures):
        coord = structure.get_positions().reshape(1, -1)
        cell = structure.get_cell().array.reshape(1, -1)

        symbols = structure.get_chemical_symbols()
        unique_symbols = sorted(set(symbols))
        atype = [unique_symbols.index(sym) for sym in symbols]

        fit_ll_atomic = dp.eval_fitting_last_layer(coord, cell, atype, device=device)

        atom_features = torch.from_numpy(fit_ll_atomic).squeeze(0).to(device)
        all_atom_features.append(atom_features)

    return all_atom_features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    structures = ase.io.read(DATASET_PATH, ":")
    print("number of loaded structures", len(structures))

    atom_features = extract_deepmd_features(structures, MODEL_PATH, device)
    atom_feats = torch.cat(atom_features, dim=0).cpu().numpy().astype(np.float64)

    print(f"number of atom features tensors: {atom_feats.shape}")
    np.save(OUTPUT_PATH, atom_feats)

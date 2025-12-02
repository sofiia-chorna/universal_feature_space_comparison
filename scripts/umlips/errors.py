import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

import numpy as np
from src.errors import compute_model_vs_model
from src.utils import get_stratified_indices

KEY = "mad"
SAVE_DIR = "./results/reconstruction_errors/umlips/"
os.makedirs(SAVE_DIR, exist_ok=True)

# ALEXANDRIA
"""
KEY = "alexandria"
DATASET_PATH = "data/xyz/alex_val_sub_consistent.xyz"
FEAT_PATHS = {
    "dpa": "data/features/alexandria/umlips/dpa-Omat24.npy",
    "uma": "data/features/alexandria/umlips/uma-omat.npy",
    "mace": "data/features/alexandria/umlips/mace.npy",
    "petmad": "data/features/alexandria/umlips/pet.npy",
}
"""

# MAD
DATASET_PATH = "data/xyz/mad-test-consistent.xyz"
FEAT_PATHS = {
    "dpa": "data/features/mad/umlips/dpa/dpa-Omat24.npy",
    "uma": "data/features/mad/umlips/uma/uma-omat.npy",
    "mace": "data/features/mad/umlips/mace/mace.npy",
    "petmad": "data/features/mad/umlips/pet/pet-mad.npy",
}


def main(error_type="LFRE"):
    model_features = {name.upper(): np.load(path) for name, path in FEAT_PATHS.items()}

    global_test_indices = get_stratified_indices(DATASET_PATH, 0)

    out_file = os.path.join(SAVE_DIR, f"{error_type.lower()}_{KEY}.json")

    results = compute_model_vs_model(
        model_features,
        error_type=error_type,
        save_path=out_file,
        test_indices=global_test_indices,
    )

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved results to {out_file}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <LFRE|GFRE>")
        sys.exit(1)

    error_type = sys.argv[1].upper()
    main(error_type)

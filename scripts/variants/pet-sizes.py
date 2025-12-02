import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

import numpy as np
from src.errors import compute_model_vs_model
from src.utils import get_stratified_indices


KEY = "omat_mad"
SAVE_DIR = "./results/reconstruction_errors/variants/pet/sizes"
os.makedirs(SAVE_DIR, exist_ok=True)

FEAT_PATHS = {
    "xs": "data/features/mad/umlips/pet/omat-xs.npy",
    "s": "data/features/mad/umlips/pet/omat-s.npy",
    "m": "data/features/mad/umlips/pet/omat-m.npy",
    "l": "data/features/mad/umlips/pet/omat-l.npy",
}


def main(error_type="LFRE"):
    model_features = {name.upper(): np.load(path) for name, path in FEAT_PATHS.items()}

    for name, feats in model_features.items():
        print(name, feats.shape)

    global_test_indices = get_stratified_indices("data/xyz/mad-test-consistent.xyz")

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

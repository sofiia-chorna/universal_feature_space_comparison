import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

import numpy as np
from src.errors import compute_model_vs_model
from src.utils import get_stratified_indices


KEY = "lps"
SAVE_DIR = "./results/reconstruction_errors/fine-tuning/model_vs_model"
os.makedirs(SAVE_DIR, exist_ok=True)


FEAT_PATHS = {
    "bespoke": "data/features/lips/pet-fine-tuning/bespoke.npy",
    "pet-mad": "data/features/lips/pet-fine-tuning/pet-mad.npy",
    "ff": "data/features/lips/pet-fine-tuning/ff.npy",
    "hf": "data/features/lips/pet-fine-tuning/hf.npy",
    "ftl": "data/features/lips/pet-fine-tuning/ftl.npy",
    "hft": "data/features/lips/pet-fine-tuning/hft.npy",
}


def main(error_type="LFRE"):
    model_features = {name.upper(): np.load(path) for name, path in FEAT_PATHS.items()}

    for name, feats in model_features.items():
        print(name, feats.shape)

    global_test_indices = get_stratified_indices("scripts/fine-tuning/LPS.extxyz")

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

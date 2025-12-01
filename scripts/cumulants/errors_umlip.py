import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

import numpy as np
from src.errors import compute_model_vs_model
from src.utils import get_random_indices, get_stratified_indices

KEY = "mad"
SAVE_DIR = f"./results/cumulants/umlips"
os.makedirs(SAVE_DIR, exist_ok=True)


def main(error_type, order, model_a, model_b):
    print(order, model_a, model_b)
    features_a = np.load(
        f"./data/features/mad/cumulants/{order}_cumulant_{model_a}.npy"
    )
    features_b = np.load(
        f"./data/features/mad/cumulants/{order}_cumulant_{model_b}.npy"
    )

    model_features = {
        f"{order}_{model_a}": features_a,
        f"{order}_{model_b}": features_b,
    }

    dataset = ase.io.read("./data/xyz/mad-test-consistent.xyz", ":")
    global_test_indices = get_stratified_indices(DATASET_PATH, per_structure=True)

    out_file = os.path.join(
        SAVE_DIR, f"{error_type.lower()}_{KEY}_{order}_{model_a}v{model_b}.json"
    )

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
    if len(sys.argv) != 5:
        print("Usage: python errors_umlip.py <LFRE|GFRE> <order> <model_a> <model_b>")
        sys.exit(1)

    error_type = sys.argv[1].upper()
    order = int(sys.argv[2])
    model_a = sys.argv[3].lower()
    model_b = sys.argv[4].lower()
    main(error_type, order, model_a, model_b)

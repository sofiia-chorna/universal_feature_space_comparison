import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

import numpy as np
from src.errors import compute_model_vs_model
from src.utils import get_random_indices, get_stratified_indices


MODEL = "pet"  #  or "mace"

KEY = "mad"
SAVE_DIR = f"./results/cumulants/"
os.makedirs(SAVE_DIR, exist_ok=True)


def main(error_type, order_a, order_b):
    features_a = np.load(
        f"./data/features/mad/cumulants/{order_a}_cumulant_{MODEL}.npy"
    )
    features_b = np.load(
        f"./data/features/mad/cumulants/{order_b}_cumulant_{MODEL}.npy"
    )

    model_features = {
        f"{order_a}_cumulant": features_a,
        f"{order_b}_cumulant": features_b,
    }

    DATASET_PATH = "./data/xyz/mad-test-consistent.xyz"
    global_test_indices = get_stratified_indices(DATASET_PATH, per_structure=True)

    out_file = os.path.join(
        SAVE_DIR, f"{error_type.lower()}_{KEY}_{order_a}v{order_b}.json"
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
    if len(sys.argv) != 4:
        print("Usage: python errors_model.py <LFRE|GFRE> <order_a> <order_b>")
        sys.exit(1)

    error_type = sys.argv[1].upper()
    order_a = int(sys.argv[2])
    order_b = int(sys.argv[3])
    main(error_type, order_a, order_b)

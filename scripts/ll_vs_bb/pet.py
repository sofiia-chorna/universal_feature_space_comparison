import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

import numpy as np

from src.errors import compute_source_to_target
from src.utils import get_stratified_indices

KEY = "pet"
SAVE_DIR = "./results/reconstruction_errors/ll_vs_bb"
os.makedirs(SAVE_DIR, exist_ok=True)


FEAT_PATHS_LL = {
    "pet-mad_ll": "data/features/mad/umlips/pet/pet-mad.npy",
    "omatpes_ll": "data/features/mad/umlips/pet/omatpes.npy",
    "omat-l_ll": "data/features/mad/umlips/pet/omat-l.npy",
    "omad_ll": "data/features/mad/umlips/pet/omad.npy",
}

FEAT_PATHS_BB = {
    "pet-mad_bb": "data/features/mad/umlips/pet/pet-mad_bb.npy",
    "omatpes_bb": "data/features/mad/umlips/pet/omatpes_bb.npy",
    "omat-l_bb": "data/features/mad/umlips/pet/omat-l_bb.npy",
    "omad_bb": "data/features/mad/umlips/pet/omad_bb.npy",
}


def main(error_type="LFRE"):
    ll_features = {name.upper(): np.load(path) for name, path in FEAT_PATHS_LL.items()}
    bb_features = {name.upper(): np.load(path) for name, path in FEAT_PATHS_BB.items()}

    for name, feats in ll_features.items():
        print(name, "shape", feats.shape)

    for name, feats in bb_features.items():
        print(name, "len", feats.shape)

    global_test_indices = get_stratified_indices("data/xyz/mad-test-consistent.xyz")

    out_file = os.path.join(SAVE_DIR, f"{error_type.lower()}_{KEY}.json")

    results = compute_source_to_target(
        ll_features,
        bb_features,
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

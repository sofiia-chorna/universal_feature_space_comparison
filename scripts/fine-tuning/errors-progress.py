import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

import numpy as np
from src.errors import compute_model_vs_reference
from src.utils import get_stratified_indices


MAP = {
    "bespoke": {
        "target": "data/features/lips/pet-fine-tuning/bespoke.npy",
        "source": "models/pet/fine-tuning/bespoke/outputs",
        "max_epoch": 1940,
    },
    "ff": {
        "target": "data/features/lips/pet-fine-tuning/ff.npy",
        "source": "models/pet/fine-tuning/ff/outputs",
        "max_epoch": 490,
    },
    "hf": {
        "target": "data/features/lips/pet-fine-tuning/hf.npy",
        "source": "models/pet/fine-tuning/hf/outputs",
        "max_epoch": 490,
    },
    "ftl": {
        "target": "data/features/lips/pet-fine-tuning/ftl.npy",
        "source": "models/pet/fine-tuning/tl/outputs",
        "max_epoch": 990,
    },
    "htl": {
        "target": "data/features/lips/pet-fine-tuning/htl.npy",
        "source": "models/pet/fine-tuning/htl/outputs",
        "max_epoch": 990,
    },
}


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python error_lips.py <GFRE|LFRE> <bespoke|ff|hf|ftl|htl>")
        sys.exit(1)

    error_type = sys.argv[1].upper()
    if error_type not in ["GFRE", "LFRE"]:
        print("Invalid argument. Choose 'GFRE' or 'LFRE'.")
        sys.exit(1)

    CHECKPOINT_TYPE = sys.argv[2]
    CHECKPOINT_TYPE_TARGET = f"{CHECKPOINT_TYPE}_final"

    print("computing", error_type, "for", CHECKPOINT_TYPE)

    OUTPUT_DIR = f"data/features/lips/progress/{CHECKPOINT_TYPE}"

    TARGET = MAP.get(CHECKPOINT_TYPE_TARGET).get("target")
    EPOCHES = range(0, MAP.get(CHECKPOINT_TYPE).get("max_epoch") + 1, 10)

    features_dict = {
        f"epoch_{epoch}": np.load(f"{OUTPUT_DIR}/epoch_{epoch}.npy")
        for epoch in EPOCHES
    }

    target_features = {CHECKPOINT_TYPE_TARGET: np.load(TARGET)}

    print(
        f"TARGER '{CHECKPOINT_TYPE_TARGET}' shape: {target_features[CHECKPOINT_TYPE_TARGET].shape}"
    )

    global_test_indices = get_stratified_indices("scripts/fine-tuning/LPS.extxyz")

    SAVE_DIR = f"./results/reconstruction_errors/lips/progress/"
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_file = os.path.join(SAVE_DIR, f"{error_type.lower()}_{CHECKPOINT_TYPE}.json")

    results = compute_model_vs_reference(
        features_dict,
        target_features,
        error_type=error_type,
        save_path=out_file,
        test_indices=global_test_indices,
    )

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved results to {out_file}]")


if __name__ == "__main__":
    main()

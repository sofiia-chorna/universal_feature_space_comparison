import sys
import numpy as np
import torch
from ase.io import read
from fairchem.core import FAIRChemCalculator, pretrained_mlip


if len(sys.argv) < 3:
    print("Usage: python get_llfs.py <task> <BB|LL> [model_name]")
    print("Tasks: oc20, omat, omol, odac, omc")
    print("Default model_name: uma-s-1p1")
    sys.exit(1)


TASK = sys.argv[1].lower()
VARIANT = sys.argv[2].upper()
MODEL_NAME = sys.argv[3] if len(sys.argv) == 3 else "uma-s-1p1"

VALID_TASKS = ["oc20", "omat", "omol", "odac", "omc"]

if TASK not in VALID_TASKS:
    raise ValueError(f"Unknown task '{TASK}'. Valid: {VALID_TASKS}")

if VARIANT not in ["BB", "LL"]:
    raise ValueError("Second argument must be BB or LL")

print("Task:", TASK)
print("Variant:", VARIANT)
print("Model name:", MODEL_NAME)

DATASET_PATH = "data/raw/mad-test-consistent.xyz"
OUTPUT_PATH = f"data/features/mad/umlips/uma/uma-{TASK}_{VARIANT.lower()}.npy"

print("DATASET:", DATASET_PATH)
print("Output:", OUTPUT_PATH)


def extract_uma_features(
    structures, task, variant, model_name="uma-s-1p1", device="cuda"
):
    predictor = pretrained_mlip.get_predict_unit(
        model_name, device=device, expose_feat=True
    )
    calc = FAIRChemCalculator(predictor, task_name=task, expose_feat=True)

    bb_list = []
    ll_list = []

    for atoms in structures:
        atoms.calc = calc
        _ = atoms.get_potential_energy()

        if variant in ["BB", "ALL"]:
            bb = atoms.calc.results["bb_feat"].to(torch.float64).cpu()
            bb_list.append(bb)

        if variant in ["LL", "ALL"]:
            ll = atoms.calc.results["ll_feat"].to(torch.float64).cpu()
            ll_list.append(ll)

    return bb_list, ll_list


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    structures = read(DATASET_PATH, ":")
    print(f"Loaded {len(structures)} test structures")

    bb_feats, ll_feats = extract_uma_features(
        structures, task=TASK, variant=VARIANT, model_name=MODEL_NAME, device=device
    )

    if VARIANT == "BB":
        concat = torch.cat(bb_feats, dim=0).numpy()
    else:
        concat = torch.cat(ll_feats, dim=0).numpy()

    np.save(OUTPUT_PATH, concat)
    print("Saved to:", OUTPUT_PATH)

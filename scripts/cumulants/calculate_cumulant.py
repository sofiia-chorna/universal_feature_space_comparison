import sys

import ase.io
import numpy as np
from scipy.stats import moment


def central_moment_to_cumulant(central_moments):
    """
    κ1 = μ1
    κ2 = μ2
    κ3 = μ3
    κ4 = μ4 - 3μ2²
    κ5 = μ5 - 10μ2μ3
    κ6 = μ6 - 15μ2μ4 - 10μ3^2 + 30μ2^3
    κ7 = μ7 - 21μ2μ5 - 35μ3μ4 + 210μ2^2 μ3
    κ8 = μ8 - 28μ2μ6 - 56μ3μ5 - 35μ4^2 + 420μ2μ3^2 + 560μ2^2 μ4 - 630μ2^4
    """
    if len(central_moments) < 1:
        return []

    cumulants = []

    # κ1 = μ1 (mean)
    if len(central_moments) >= 1:
        cumulants.append(central_moments[0])

    # κ2 = μ2 (variance)
    if len(central_moments) >= 2:
        cumulants.append(central_moments[1])

    # κ3 = μ3
    if len(central_moments) >= 3:
        cumulants.append(central_moments[2])

    # κ4 = μ4 - 3μ2^2
    if len(central_moments) >= 4:
        κ4 = central_moments[3] - 3 * central_moments[1] ** 2
        cumulants.append(κ4)

    # κ5 = μ5 - 10μ2μ3
    if len(central_moments) >= 5:
        κ5 = central_moments[4] - 10 * central_moments[1] * central_moments[2]
        cumulants.append(κ5)

    # κ6 = μ6 - 15μ2μ4 - 10μ3^2 + 30μ2^3
    if len(central_moments) >= 6:
        κ6 = (
            central_moments[5]
            - 15 * central_moments[1] * central_moments[3]
            - 10 * central_moments[2] ** 2
            + 30 * central_moments[1] ** 3
        )
        cumulants.append(κ6)

    # κ7 = μ7 - 21μ2μ5 - 35μ3μ4 + 210μ2²μ3
    if len(central_moments) >= 7:
        κ7 = (
            central_moments[6]
            - 21 * central_moments[1] * central_moments[4]
            - 35 * central_moments[2] * central_moments[3]
            + 210 * central_moments[1] ** 2 * central_moments[2]
        )
        cumulants.append(κ7)

    # κ8 = μ8 - 28μ2μ6 - 56μ3μ5 - 35μ4² + 420μ2μ3^2 + 560μ2^2μ4 - 630μ2^4
    if len(central_moments) >= 8:
        κ8 = (
            central_moments[7]
            - 28 * central_moments[1] * central_moments[5]
            - 56 * central_moments[2] * central_moments[4]
            - 35 * central_moments[3] ** 2
            + 420 * central_moments[1] * central_moments[2] ** 2
            + 560 * central_moments[1] ** 2 * central_moments[3]
            - 630 * central_moments[1] ** 4
        )
        cumulants.append(κ8)

    return cumulants


def calculate_cumulants(block, max_order=8, axis=0):
    block = np.asarray(block, dtype=np.float64)

    central_moments = []
    for k in range(1, max_order + 1):
        if k == 1:
            moment_val = np.mean(block, axis=axis, dtype=np.float64)
        else:
            moment_val = moment(block, moment=k, axis=axis, nan_policy="omit")
        central_moments.append(moment_val)

    cumulants = central_moment_to_cumulant(central_moments)

    return cumulants


def construct_struct_feats_cumulant(atom_feats, dataset, order=8):
    atom_feats = np.asarray(atom_feats, dtype=np.float64)
    n_atoms = [len(atoms) for atoms in dataset]
    features = []
    start = 0

    for n in n_atoms:
        block = atom_feats[start : start + n].astype(np.float64)

        if n == 0:
            print("zero atoms")

        cumulants = calculate_cumulants(block, max_order=order, axis=0)

        feature_block = []
        for i, cumulant in enumerate(cumulants):
            if i == 0:
                feature_block.append(cumulant)
            elif i == 1:
                # for variance, we take square root to get std-like value
                var_feat = np.sqrt(np.abs(cumulant))
                var_feat = np.sign(cumulant) * var_feat
                feature_block.append(var_feat)
            else:
                # for higher cumulants, we take appropriate roots
                abs_cumulant = np.abs(cumulant)
                root_val = np.power(abs_cumulant, 1.0 / (i + 1))

                root_val = np.sign(cumulant) * root_val
                feature_block.append(root_val)

        combined_feat = np.concatenate(feature_block).astype(np.float64)
        features.append(combined_feat)
        start += n

    descriptors = np.vstack(features).astype(np.float64)
    return descriptors


FEAT_PATHS = {
    "pet": "./data/features/mad/umlips/pet/pet-mad.npy",
    "mace": "./data/features/mad/mace/umlips/mace-mp-b03.npy",
    "dpa": "./data/features/mad/dpa/umlips/dpa-Omat24.npy",
    "uma": "./data/features/mad/uma/umlips/uma-omat.npy",
}

if len(sys.argv) < 2:
    print("Usage: python calculate_cumulant.py [pet|mace|dpa|uma]")
    sys.exit(1)

MODEL = sys.argv[1].lower()
if MODEL not in ["mace", "pet", "dpa", "uma"]:
    raise ValueError(f"Invalid model")


def main():
    dataset = ase.io.read("./data/xyz/mad-test-consistent.xyz", ":")

    feats = np.load(FEAT_PATHS.get(MODEL)).astype(np.float64)
    print("input feature shape:", feats.shape, "dtype:", feats.dtype)

    for order in range(1, 9):
        desc = construct_struct_feats_cumulant(feats, dataset, order)
        
        OUTPUT_PATH = f"./data/features/mad/cumulants/{order}_cumulant_{MODEL}"
        np.save(OUTPUT_PATH, desc)

        print(f"saved order {order} cumulant:", desc.shape, desc.dtype)


if __name__ == "__main__":
    main()

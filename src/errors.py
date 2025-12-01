import json
import os
from collections import defaultdict

import numpy as np
from skmatter.linear_model import Ridge2FoldCV
from skmatter.metrics import global_reconstruction_error, local_reconstruction_error


def _save_partial(results, save_path):
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, save_path)


def _compute_pairwise_error(A, B, error_type, n_local_points=None, test_indices=None):
    estimator = Ridge2FoldCV(
        alphas=np.geomspace(1e-9, 1, 20),
        alpha_type="relative",
        regularization_method="cutoff",
        random_state=0x5F3759DF,
        shuffle=True,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )

    if error_type.upper() == "GFRE":
        return global_reconstruction_error(
            A, B, test_idx=test_indices, estimator=estimator
        )

    return local_reconstruction_error(
        A,
        B,
        n_local_points=n_local_points,
        test_idx=test_indices,
        n_jobs=-1,
        estimator=estimator,
    )


def compute_model_vs_model(
    model_features,
    error_type="LFRE",
    test_indices=None,
    save_path=None,
):
    results = defaultdict(dict)

    if save_path and os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)

    feat_len = len(next(iter(model_features.values())))
    if test_indices is not None and np.max(test_indices) >= feat_len:
        raise ValueError("test_indices contain indices outside feature array bounds.")

    model_names = list(model_features.keys())

    for _i, ref_name in enumerate(model_names):
        ref_feat = model_features[ref_name]
        print(f"\n--- Using {ref_name} as reference ---")

        for _j, model_name in enumerate(model_names):
            if ref_name not in results:
                results[ref_name] = {}

            if ref_name == model_name:
                model_feat = model_features[model_name]
                n_local_points = (
                    min(model_feat.shape[1], ref_feat.shape[1])
                    if error_type.upper() == "LFRE"
                    else None
                )

                print("n_local_points", n_local_points)

                err = _compute_pairwise_error(
                    model_feat, ref_feat, error_type, n_local_points, test_indices
                )

                results[ref_name][model_name] = {
                    "A": model_name,
                    "B": ref_name,
                    "A_to_B": float(err),
                    "B_to_A": float(err),
                }

                print(f"  {model_name} <=> {ref_name}: {err:.4f}")

                if save_path:
                    _save_partial(results, save_path)

                continue

            if model_name in results.get(ref_name, {}):
                continue

            # reuse symmetric result if exists
            if ref_name in results.get(model_name, {}):
                inv = results[model_name][ref_name]

                if ref_name not in results:
                    results[ref_name] = {}

                results[ref_name][model_name] = {
                    "A": model_name,
                    "B": ref_name,
                    "A_to_B": inv["B_to_A"],
                    "B_to_A": inv["A_to_B"],
                }
                continue

            model_feat = model_features[model_name]
            n_local_points = (
                min(model_feat.shape[1], ref_feat.shape[1])
                if error_type.upper() == "LFRE"
                else None
            )

            a_to_b = _compute_pairwise_error(
                model_feat, ref_feat, error_type, n_local_points, test_indices
            )
            b_to_a = _compute_pairwise_error(
                ref_feat, model_feat, error_type, n_local_points, test_indices
            )

            results[ref_name][model_name] = {
                "A": model_name,
                "B": ref_name,
                "A_to_B": float(a_to_b),
                "B_to_A": float(b_to_a),
            }

            print(f"  {model_name} => {ref_name}: {a_to_b:.4f}")
            print(f"  {ref_name} => {model_name}: {b_to_a:.4f}")

            if save_path:
                _save_partial(results, save_path)

    return dict(results)


def compute_model_vs_reference(
    model_features,
    reference_features,
    error_type="LFRE",
    n_local_points=None,
    test_indices=None,
    save_path=None,
):
    results = defaultdict(dict)

    if save_path and os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)

    feat_len = len(next(iter(model_features.values())))
    if test_indices is not None and np.max(test_indices) >= feat_len:
        raise ValueError(
            "test_indices contain indices outside the feature array bounds."
        )

    for ref_name, ref_feat in reference_features.items():
        print(f"\n--- Using {ref_name} as reference ---")
        for model_name, model_feat in model_features.items():
            if ref_name in results and model_name in results[ref_name]:
                continue

            print(f"Computing {model_name} => {ref_name}...")

            n_local_points = (
                min(model_feat.shape[1], ref_feat.shape[1])
                if error_type.upper() == "LFRE"
                else None
            )

            a_to_b = _compute_pairwise_error(
                model_feat, ref_feat, error_type, n_local_points, test_indices
            )
            b_to_a = _compute_pairwise_error(
                ref_feat, model_feat, error_type, n_local_points, test_indices
            )

            if ref_name not in results:
                results[ref_name] = {}

            results[ref_name][model_name] = {
                "A": model_name,
                "B": ref_name,
                "A_to_B": float(a_to_b),
                "B_to_A": float(b_to_a),
            }

            print(f"  {model_name} => {ref_name}: {a_to_b:.4f}")
            print(f"  {ref_name} => {model_name}: {b_to_a:.4f}")

            if save_path:
                _save_partial(results, save_path)

    return dict(results)


def compute_source_to_target(
    model_features,
    reference_features,
    error_type="LFRE",
    test_indices=None,
    save_path=None,
):
    results = defaultdict(dict)

    if save_path and os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)

    feat_len = len(next(iter(model_features.values())))
    if test_indices is not None and np.max(test_indices) >= feat_len:
        raise ValueError("test_indices contain indices outside feature array bounds.")

    ref_keys = list(reference_features.keys())
    model_keys = list(model_features.keys())

    if len(ref_keys) != len(model_keys):
        raise ValueError(
            "The number of keys in reference_features and model_features must match for paired comparison"
        )

    for i in range(len(ref_keys)):
        ref_name = ref_keys[i]
        model_name = model_keys[i]

        if ref_name in results and model_name in results[ref_name]:
            continue

        ref_feats = reference_features[ref_name]
        model_feats = model_features[model_name]

        print(f"\n--- Comparing {model_name} and {ref_name} ---")

        n_local_points = (
            min(model_feats.shape[1], ref_feats.shape[1])
            if error_type.upper() == "LFRE"
            else None
        )

        a_to_b = _compute_pairwise_error(
            model_feats, ref_feats, error_type, n_local_points, test_indices
        )
        b_to_a = _compute_pairwise_error(
            ref_feats, model_feats, error_type, n_local_points, test_indices
        )

        if ref_name not in results:
            results[ref_name] = {}

        results[ref_name][model_name] = {
            "A": model_name,
            "B": ref_name,
            "A_to_B": float(a_to_b),
            "B_to_A": float(b_to_a),
        }

        print(f"  {model_name} => {ref_name}: {a_to_b:.4f}")
        print(f"  {ref_name} => {model_name}: {b_to_a:.4f}")

        if save_path:
            _save_partial(results, save_path)

    return dict(results)

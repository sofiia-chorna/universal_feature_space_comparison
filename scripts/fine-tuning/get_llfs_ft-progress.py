import sys
import ase.io
import numpy as np
import torch
import metatomic.torch as mta
import vesin.metatomic as vesin_metatomic
from metatrain.utils.io import load_model as load_metatrain_model
import os


MAP = {
    "bespoke": {"source": "models/pet/fine-tuning/bespoke/outputs", "max_epoch": 1940},
    "ff": {"source": "models/pet/fine-tuning/ff/outputs", "max_epoch": 490},
    "hf": {"source": "models/pet/fine-tuning/hf/outputs", "max_epoch": 490},
    "ftl": {"source": "models/pet/fine-tuning/tl/outputs", "max_epoch": 990},
    "htl": {"source": "models/pet/fine-tuning/htl/outputs", "max_epoch": 990},
}

device = "cuda"


def get_llfs(model, structures):
    length_unit = "angstrom"
    batch_size = 32
    dtype = torch.float32
    check_consistency = False
    # requested_key = "mtt::aux::energy_last_layer_features" for models with attached head (tranfser learning)
    requested_key = "mtt::aux::LPS_last_layer_features"

    systems = mta.systems_to_torch(structures)
    vesin_metatomic.compute_requested_neighbors(systems, length_unit, model, "angstrom")

    systems = [s.to(dtype=dtype, device=device) for s in systems]

    options = mta.ModelEvaluationOptions(
        length_unit=length_unit,
        outputs={requested_key: mta.ModelOutput(per_atom=True)},
        selected_atoms=None,
    )

    outputs = []

    for i_sys in range(0, len(systems), batch_size):
        batch_systems = systems[i_sys : i_sys + batch_size]
        batch_outputs = model(
            batch_systems,
            options,
            check_consistency=check_consistency,
        )
        outputs.append(
            batch_outputs[requested_key].block().values.detach().cpu().numpy()
        )

    outputs_np = np.concatenate(outputs)
    print(outputs_np.shape)

    return outputs_np


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python error_lips.py <bespoke|ff|hf|htl|tl>")
        sys.exit(1)

    CHECKPOINT_TYPE = sys.argv[1]
    print(f"running for {CHECKPOINT_TYPE}")

    OUTPUT_DIR_BASE = f"data/features/lips/progress/{CHECKPOINT_TYPE}"

    MAX_EPOCH = MAP.get(CHECKPOINT_TYPE).get("max_epoch")
    SOURCE = MAP.get(CHECKPOINT_TYPE).get("source")

    frames = ase.io.read("./data/raw/lips/LPS.extxyz", ":")
    print(len(frames))

    for epoch in range(0, MAX_EPOCH + 1, 10):
        checkpoint_path = os.path.join(SOURCE, f"model_{epoch}.ckpt")
        model = load_metatrain_model(checkpoint_path).to(device)
        mts_model = model.export()

        feats = get_llfs(mts_model, frames)
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
        np.save(f"{OUTPUT_DIR_BASE}/epoch_{epoch}", feats)


if __name__ == "__main__":
    main()

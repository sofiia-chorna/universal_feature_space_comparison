import warnings

import ase.io
import numpy as np
import torch
from mace import data, tools
from mace.data import KeySpecification
from mace.modules import LLPRModel
from mace.tools import torch_geometric
from mace.tools.scripts_utils import get_dataset_from_xyz

warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)
device = tools.init_device("cuda")

DATASET_PATH = "data/xyz/mad-test-consistent.xyz"
print("DATASET_PATH", DATASET_PATH, len(ase.io.read(DATASET_PATH, ":")))

CONFIG = {
    "mp-0b3": {
        "model_path": "models/mace/mace-mp-0b3-medium.model",
        "output_path": "data/features/mad/umlips/mace/variants/mace-mp-0b3_bb",
    }
}

OUTPUT_PATH = CONFIG["output_path"]


def get_mace_descriptors(model, node_feats, invariants_only=True):
    irreps_out = model.products[0].linear.irreps_out

    num_invariant_features = 0
    for mul, ir in irreps_out:
        if ir.l == 0:
            num_invariant_features += mul

    num_interactions = len(model.interactions)

    per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
    per_layer_features[-1] = num_invariant_features

    if invariants_only:
        descriptors_list = []
        for i in range(num_interactions):
            start = i * irreps_out.dim
            end = start + num_invariant_features
            descriptors_list.append(node_feats[:, start:end])
        return torch.cat(descriptors_list, dim=-1)
    else:
        to_keep = sum(per_layer_features)
        return node_feats[:, :to_keep]


for model_name, entry in CONFIG.items():
    path = entry.get("model_path")
    model = torch.load(path, map_location=device).eval()
    wrapper = LLPRModel(model).to(device)

    atomic_numbers = model.atomic_numbers.tolist()

    stats = {
        "atomic_numbers": atomic_numbers,
        "r_max": model.r_max.item(),
    }

    key_specification_config = KeySpecification(
        info_keys={
            "energy": "energy",
            "stress": "stress",
            "head": "config_type",
        },
        arrays_keys={"forces": "forces"},
    )

    config_type_weights = {"Default": 1.0}

    z_table = tools.get_atomic_number_table_from_zs(stats["atomic_numbers"])

    collections, atomic_energies_dict = get_dataset_from_xyz(
        work_dir="",
        train_path=DATASET_PATH,
        valid_path="data/xyz/dummy_valid.xyz",
        test_path="data/xyz/dummy_test.xyz",
        valid_fraction=0,
        config_type_weights=config_type_weights,
        key_specification=key_specification_config,
    )

    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=stats["r_max"])
            for config in collections.train
        ],
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )

    all_descriptors = []
    wrapper.eval()

    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            outputs = wrapper(batch, compute_force=False, compute_stress=False)

            raw_node_feats = outputs["node_feats"]
            descriptors = get_mace_descriptors(
                model=model, node_feats=raw_node_feats, invariants_only=True
            )
            all_descriptors.append(descriptors.cpu())

    all_descriptors_final = torch.cat(all_descriptors, dim=0).numpy()

    np.save(OUTPUT_PATH, all_descriptors_final)
    print(f"Saved descriptors with shape {all_descriptors_final.shape}")

import ase.io
import numpy as np
import torch
from mace import data, tools
from mace.data import KeySpecification
from mace.modules import LLPRModel
from mace.tools import torch_geometric
from mace.tools.scripts_utils import get_dataset_from_xyz

import sys

if len(sys.argv) < 2:
    print("Usage: python run_mace.py [small|medium|large]")
    sys.exit(1)

KEY = sys.argv[1].lower()
if KEY not in ["small", "medium", "large"]:
    raise ValueError(f"Invalid model key '{KEY}'. Choose from: small, medium, large")


torch.set_default_dtype(torch.float64)
device = tools.init_device("cuda")

DATASET_PATH = "data/xyz/mad-test-consistent-organic.xyz"
print("DATASET_PATH", DATASET_PATH, len(ase.io.read(DATASET_PATH, ":")))

MODEL_DICT = {
    "small": {
        "model_path": "models/mace/MACE-OFF23_small.model",
        "output_path": "data/features/mad/umlips/mace/off23/mace-off23-small",
    },
    "medium": {
        "model_path": "models/mace/MACE-OFF23_medium.model",
        "output_path": "data/features/mad/umlips/mace/off23/mace-off23-medium",
    },
    "large": {
        "model_path": "models/mace/MACE-OFF23_large.model",
        "output_path": "data/features/mad/umlips/mace/off23/mace-off23-large",
    },
}

MODEL_PATH = MODEL_DICT[KEY]["model_path"]
OUTPUT_PATH = MODEL_DICT[KEY]["output_path"]

print("MODEL_PATH", MODEL_PATH)
print("OUTPUT_PATH", OUTPUT_PATH)

model = torch.load(MODEL_PATH, weights_only=False)
mace_llpr = LLPRModel(model).to(device)

atomic_numbers = model.atomic_numbers.tolist()
atomic_energies = model.atomic_energies_fn.atomic_energies.squeeze().tolist()

stats = {
    "atomic_numbers": atomic_numbers,
    "atomic_energies": {
        int(z): float(e) for z, e in zip(atomic_numbers, atomic_energies)
    },
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


def aggregate_features(self, ll_feats):
    ll_feats_list = torch.split(ll_feats, self.hidden_sizes_before_readout, dim=-1)
    ll_feats_list = [
        (ll_feats if is_linear else readout.non_linearity(readout.linear_1(ll_feats)))[
            :, :size
        ]
        for ll_feats, readout, size, is_linear in zip(
            ll_feats_list,
            self.orig_model.readouts.children(),
            self.hidden_sizes,
            self.readouts_are_linear,
        )
    ]
    ll_feats_cat = torch.cat(ll_feats_list, dim=-1)
    return ll_feats_cat


all_ll_feats = []

for batch in iter(train_loader):
    batch = batch.to(device)
    outputs = mace_llpr(batch)
    ll_feats = aggregate_features(mace_llpr, outputs["node_feats"])
    all_ll_feats.append(ll_feats.detach().cpu())

all_ll_feats = torch.cat(all_ll_feats, dim=0)
all_ll_feats_np = all_ll_feats.cpu().numpy()
np.save(OUTPUT_PATH, all_ll_feats)

import sys
import ase.io
import numpy as np
import torch
from mace import data, tools
from mace.data import KeySpecification
from mace.modules import LLPRModel
from mace.tools import torch_geometric
from mace.tools.scripts_utils import get_dataset_from_xyz

torch.set_default_dtype(torch.float64)


if len(sys.argv) < 3:
    print("Usage: python get_llfs_mace.py <family> <model_key>")
    print("Families: a0, off23, variants")
    sys.exit(1)

FAMILY = sys.argv[1].lower()
KEY = sys.argv[2].lower()

MODEL_REGISTRY = {
    "a0": {
        "dataset": "data/xyz/mad-test-consistent.xyz",
        "models": {
            "small": {
                "model_path": "models/mace/2023-12-10-mace-128-L0_energy_epoch-249.model",
                "output_path": "data/features/mad/umlips/mace/mp-a0/mace-mp-0a-small",
            },
            "medium": {
                "model_path": "models/mace/2023-12-03-mace-128-L1_epoch-199.model",
                "output_path": "data/features/mad/umlips/mace/mp-a0/mace-mp-0a-medium",
            },
            "large": {
                "model_path": "models/mace/2024-01-07-mace-128-L2_epoch-199.model",
                "output_path": "data/features/mad/umlips/mace/mp-a0/mace-mp-0a-large",
            },
        },
    },
    "off23": {
        "dataset": "data/xyz/mad-test-consistent-organic.xyz",
        "models": {
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
        },
    },
    "variants": {
        "dataset": "data/xyz/mad-test-consistent.xyz",
        "models": {
            "mp-0b3": {
                "model_path": "models/mace/mace-mp-0b3-medium.model",
                "output_path": "data/features/mad/umlips/mace/variants/mace-mp-0b3",
            },
            "matpes_pbe": {
                "model_path": "models/mace/MACE-matpes-pbe-omat-ft.model",
                "output_path": "data/features/mad/umlips/mace/variants/mace-matpes-pbe",
            },
            "matpes_r2scan": {
                "model_path": "models/mace/MACE-matpes-r2scan-omat-ft.model",
                "output_path": "data/features/mad/umlips/mace/variants/mace-matpes-r2scan",
            },
            "omat": {
                "model_path": "models/mace/mace-omat-0-medium.model",
                "output_path": "data/features/mad/umlips/mace/variants/mace-omat",
            },
            "mace-mh-1": {
                "model_path": "models/mace/mace-mh-1.model.omat_pbe",
                "output_path": "data/features/mad/umlips/mace/variants/mace-mh-1-omat",
            }
        },
    },
}

if FAMILY not in MODEL_REGISTRY:
    raise ValueError(f"Unknown family '{FAMILY}'")

if KEY not in MODEL_REGISTRY[FAMILY]["models"]:
    raise ValueError(f"Unknown model key '{KEY}' for family '{FAMILY}'")

CONFIG = MODEL_REGISTRY[FAMILY]["models"][KEY]
DATASET_PATH = MODEL_REGISTRY[FAMILY]["dataset"]

MODEL_PATH = CONFIG["model_path"]
OUTPUT_PATH = CONFIG["output_path"]

print("DATASET_PATH:", DATASET_PATH)
print("MODEL_PATH:", MODEL_PATH)
print("OUTPUT_PATH:", OUTPUT_PATH)

device = tools.init_device("cuda")
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

key_spec = KeySpecification(
    info_keys={"energy": "energy", "stress": "stress", "head": "config_type"},
    arrays_keys={"forces": "forces"},
)

collections, _ = get_dataset_from_xyz(
    work_dir="",
    train_path=DATASET_PATH,
    valid_path="data/xyz/dummy_valid.xyz",
    test_path="data/xyz/dummy_test.xyz",
    valid_fraction=0,
    config_type_weights={"Default": 1.0},
    key_specification=key_spec,
)

z_table = tools.get_atomic_number_table_from_zs(stats["atomic_numbers"])

train_loader = torch_geometric.dataloader.DataLoader(
    dataset=[
        data.AtomicData.from_config(config, z_table=z_table, cutoff=stats["r_max"])
        for config in collections.train
    ],
    batch_size=4,
    shuffle=False,
)


def aggregate_features(self, ll_feats):
    ll_feats_list = torch.split(ll_feats, self.hidden_sizes_before_readout, dim=-1)
    out = []
    for z, readout, size, is_linear in zip(
        ll_feats_list,
        self.orig_model.readouts.children(),
        self.hidden_sizes,
        self.readouts_are_linear,
    ):
        block = z if is_linear else readout.non_linearity(readout.linear_1(z))
        out.append(block[:, :size])
    return torch.cat(out, dim=-1)


all_ll = []

for batch in train_loader:
    batch = batch.to(device)
    out = mace_llpr(batch)
    feats = aggregate_features(mace_llpr, out["node_feats"])
    all_ll.append(feats.cpu())

all_ll = torch.cat(all_ll, dim=0)
np.save(OUTPUT_PATH, all_ll.numpy())

print("Saved:", OUTPUT_PATH)

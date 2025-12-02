import sys
import ase.io
import numpy as np
import torch
import metatomic.torch as mta
import vesin.metatomic as vesin_metatomic
from metatrain.utils.io import load_model as load_metatrain_model


if len(sys.argv) != 2:
    print("Usage: python3 get_pet_features.py <model_key>")
    sys.exit(1)

MODEL_KEY = sys.argv[1].lower()
MODEL_PATHS = {
    "pet-mad": "models/pet/pet-mad-v1.0.2.pt",
    "bespoke": "scripts/fine-tuning/bespoke/model.pt",
    "ff": "scripts/fine-tuning/ff/model.pt",
    "hf": "scripts/fine-tuning/ff/model.pt",
    "ftl": "scripts/fine-tuning/ftl/model.pt",
    "htl":  "scripts/fine-tuning/htl/model.pt",
}

if MODEL_KEY not in MODEL_PATHS:
    print(f"Error: unknown model '{MODEL_KEY}'.")
    print("Available models:", list(MODEL_PATHS.keys()))
    sys.exit(1)

MODEL_PATH = MODEL_PATHS[MODEL_KEY]
DATASET_PATH = "./data/raw/lips/LPS.extxyz"
OUTPUT_DIR = f"data/features/lips"

device = "cuda"

REQUEST_KEY = "mtt::aux::energy_last_layer_features" 
# or REQUEST_KEY = "mtt::aux::LPS_last_layer_features" if with new attached head (tranfer learning)

OUTPUT_PATH =  f"{OUTPUT_DIR}/{MODEL_KEY}.npy"

print("Model:", MODEL_KEY)
print("Model path:", MODEL_PATH)
print("Requested key:", REQUEST_KEY)
print("Output path:", OUTPUT_PATH)

model = load_metatrain_model(MODEL_PATH).to(device)
length_unit = model.capabilities().length_unit

structures = ase.io.read(DATASET_PATH, ":")
print(f"Loaded {len(structures)} structures")

systems = mta.systems_to_torch(structures)
vesin_metatomic.compute_requested_neighbors(systems, length_unit, model)
systems = [s.to(dtype=torch.float32, device=device) for s in systems]

options = mta.ModelEvaluationOptions(
    length_unit=length_unit,
    outputs={REQUEST_KEY: mta.ModelOutput(per_atom=True)},
    selected_atoms=None,
)

batch_size = 32
check_consistency = False
outputs = []

for i in range(0, len(systems), batch_size):
    batch = systems[i : i + batch_size]
    out = model(batch, options, check_consistency=check_consistency)
    feats = out[REQUEST_KEY].block().values.detach().cpu().numpy()
    outputs.append(feats)

features = np.concatenate(outputs)
np.save(OUTPUT_PATH, features)

print("Saved:", OUTPUT_PATH)

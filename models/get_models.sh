#!/bin/bash
set -e

mkdir -p models/mace
mkdir -p models/pet
mkdir -p models/dpa
mkdir -p envs

echo "=== Fetching MACE models ==="
(
    cd models/mace

    wget -nc https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model
    wget -nc https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model
    wget -nc https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model
    wget -nc https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model
    wget -nc https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model

    wget -nc https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model
    wget -nc https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model
    wget -nc https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model
)

echo "=== Fetching DPA model ==="
(
    cd models/dpa
    wget -nc https://store.aissquare.com/models/35b4ce45-4f59-4868-9fd7-a0c0f5ad9464/DPA-3.1-3M.pt
)

echo "All models downloaded successfully."


echo "=== Creating PET environment ==="

if ! conda env list | grep -q "^petenv"; then
    conda create -y -n petenv python=3.10
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate petenv

echo "=== Installing metatrain[pet] ==="
pip install -q metatrain[pet]

echo "=== Fetching PET models ==="
(
    cd models/pet

    huggingface-cli login

    wget -nc https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.0.2/models/pet-mad-v1.0.2.ckpt

    CKPTS=(
        pet-oam-l-v0.1.0.ckpt
        pet-omad-l-v0.1.0.ckpt
        pet-omat-l-v0.1.0.ckpt
        pet-omat-l-v1.0.0.ckpt
        pet-omat-m-v1.0.0.ckpt
        pet-omat-s-v1.0.0.ckpt
        pet-omat-xs-v1.0.0.ckpt
        pet-omatpes-l-v0.1.0.ckpt
        pet-spice-l-v1.0.0.ckpt
    )

    for C in "${CKPTS[@]}"; do
        echo "-- downloading $C"
        wget -nc "https://huggingface.co/lab-cosmo/upet/resolve/main/models/${C}"
    done

    echo
    echo "=== Exporting all ckpt → pt using mtt export ==="

    for C in *.ckpt; do
        OUT="${C%.ckpt}.pt"
        echo "-- exporting $C -> $OUT"
        mtt export "$C" --output "$OUT"
    done

    echo
    echo "Done. Models saved in models/pet/"
    ls -lh

)

conda deactivate
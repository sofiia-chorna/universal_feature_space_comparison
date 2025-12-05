#!/bin/bash
set -e

ENV_DIR="envs"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_DIR" ]; then
    echo "ERROR: Directory '$ENV_DIR' not found."
    exit 1
fi

shopt -s nullglob
YAML_LIST=("$ENV_DIR"/*.yml)
shopt -u nullglob

if [ ${#YAML_LIST[@]} -eq 0 ]; then
    echo "ERROR: No .yml files found in $ENV_DIR/"
    exit 1
fi

echo "Creating conda environments from $ENV_DIR/"
echo

for yaml in "${YAML_LIST[@]}"; do
    echo "Processing $yaml"

    env_name=$(grep -E '^name:' "$yaml" | awk '{print $2}')

    if [ -z "$env_name" ]; then
        echo "ERROR: Could not read name: field in $yaml"
        exit 1
    fi

    if conda env list | grep -q "^${env_name} "; then
        echo "Environment '$env_name' already exists, skipping"
    else
        echo "Creating environment '$env_name'"
        conda env create -f "$yaml"
    fi

    echo
done

echo "Done."

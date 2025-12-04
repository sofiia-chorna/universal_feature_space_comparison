#!/bin/bash
set -e

ENV_DIR="envs"
YAML_LIST=( "$ENV_DIR"/*.yaml )

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Conda environment creation script ==="

for yaml in "${YAML_LIST[@]}"; do
    env_name=$(grep -E '^name:' "${ENV_DIR}/${yaml}" | awk '{print $2}')

    if [[ -z "$env_name" ]]; then
        echo "ERROR: Could not read env name from $yaml"
        exit 1
    fi

    echo ""
    echo "=== Processing environment: $env_name ==="

    if conda env list | grep -q "^${env_name} "; then
        echo "Environment '$env_name' already exists, skipping"
    else
        echo "Creating environment '$env_name' from ${yaml}"
        conda env create -f "${ENV_DIR}/${yaml}"
        echo "Created env: $env_name"
    fi
done

echo ""
echo "All environments processed successfully"

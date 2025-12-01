module load gcc
module load cuda

export DP_VARIANT=cuda
export CUDAToolkit_ROOT=$CUDA_HOME

python3 scripts/last-layer-features/dpa/pre_steps.py

for task in Mptraj Omat24 OC20M SPICE2 ODAC23; do
    python3 scripts/last-layer-features/dpa/get_llfs.py $task
done

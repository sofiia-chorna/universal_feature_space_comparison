bash scripts/last-layer-features/mace/pre_steps.sh

family=variants
for task in mp-0b3 matpes_pbe matpes_r2scan omat; do
    python3 python get_llfs_mace.py $family $model_key
done

family=a0
for task in small medium large; do
    python3 python get_llfs_mace.py $family $model_key
done

family=off23
for task in small medium large; do
    python3 python get_llfs_mace.py $family $model_key
done

bash scripts/last-layer-features/uma/pre_steps.sh

for variant in LL BB; do
    for task in oc20 omat omol odac omc; do
        python3 scripts/last-layer-features/uma/get_llfs.py $task $variant
    done
done

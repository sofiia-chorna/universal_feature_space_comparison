bash scripts/last-layer-features/pet/pre_steps.sh

for variant in LL BB; do
    for model_key in pet-mad bespoke ff hf ftl htl omatpes omat-l omat-m omat-s omat-xs oam spice-l omad pet-mad-dos; do
        python3 scripts/last-layer-features/pet/get_llfs.py $model_key $variant
    done
done

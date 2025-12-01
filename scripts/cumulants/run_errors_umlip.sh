#!/bin/bash

order=8

for error_type in GFRE LFRE; do
    for model_a in PET MACE UMA DPA; do
        for model_b in PET MACE UMA DPA; do
            python3 errors_umlip.sh "$error_type" "$order" "$model_a" "$model_b"
        done
    done
done

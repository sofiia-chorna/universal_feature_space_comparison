#!/bin/bash

for error_type in GFRE LFRE; do
    for base_order in {1..8}; do
        for order in {1..8}; do
            python3 errors_model.py "$error_type" "$base_order" "$order"
        done
    done
done

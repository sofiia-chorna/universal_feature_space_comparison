#!/bin/bash

for model in PET MACE UMA DPA; do
    python calculate_cumulant.py "$model"
done

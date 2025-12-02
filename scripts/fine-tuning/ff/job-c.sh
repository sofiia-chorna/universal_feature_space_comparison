#!/bin/bash
#SBATCH --job-name=da_Li
#SBATCH --nodes=1
#SBATCH --time=20:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=h100 

source ~/miniforge3/bin/activate 
conda activate cartospeed_2025_9

mtt train options-c.yaml  

#!/bin/bash

#SBATCH --job-name=aggregate_era
#SBATCH --array=0-71         # Adjust this based on number of files - 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=./logs/output_%A_%a.txt

source ~/.bashrc
conda activate era

python test_interp_era.py

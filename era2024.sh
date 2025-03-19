#!/bin/bash

#SBATCH --job-name=era2024
#SBATCH --output=era2024.out
#SBATCH --error=era2024.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=32G

source ~/.bashrc
conda activate geo

python pull_era_data.py --year 2024
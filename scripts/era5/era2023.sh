#!/bin/bash

#SBATCH --job-name=era2023
#SBATCH --output=era2023.out
#SBATCH --error=era2023.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=32G

source ~/.bashrc
conda activate pull-era-data

python pull_era_data.py --year 2023
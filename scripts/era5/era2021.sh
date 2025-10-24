#!/bin/bash

#SBATCH --job-name=era2021
#SBATCH --output=era2021.out
#SBATCH --error=era2021.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=32G

source ~/.bashrc
conda activate pull-era-data

python pull_era_data.py --year 2021
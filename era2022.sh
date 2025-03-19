#!/bin/bash

#SBATCH --job-name=era2022
#SBATCH --output=era2022.out
#SBATCH --error=era2022.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=32G

source ~/.bashrc
conda activate geo

python pull_era_data.py --year 2022
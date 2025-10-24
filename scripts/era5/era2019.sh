#!/bin/bash

#SBATCH --job-name=era2019
#SBATCH --output=era2019.out
#SBATCH --error=era2019.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=32G

source ~/.bashrc
conda activate pull-era-data

python pull_era_data.py --year 2019
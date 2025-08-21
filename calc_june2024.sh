#!/bin/bash

#SBATCH --job-name=extreme
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs/output_%a.txt
#SBATCH --error=./logs/output_%a.err
#SBATCH --time=01:00:00

source gp/bin/activate

python calc_extremes.py \
    --input "/work-old/zdc6/temp_stud_pll/" \
    --year 2024 \
    --month 6
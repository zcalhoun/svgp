#!/bin/bash

#SBATCH --job-name=s_pll_2000
#SBATCH --array=66
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs_s_pll_2000/output_%a.txt
#SBATCH --error=./logs_s_pll_2000/output_%a.err
#SBATCH --time=01:30:00

source ~/.bashrc
conda activate svgp

python main.py \
    -i /work/zdc6/weather_underground/durham/combined/ \
    -o /work/zdc6/temp_s_pll/ \
    --num_epochs 40 \
    --variable tempAvg \
    --likelihood Student \
    --loss PLL \
    --num_inducing_points 2000

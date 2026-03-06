#!/bin/bash

#SBATCH --job-name=g-elbo
#SBATCH --array=60-71
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs_g_elbo/output_%a.txt
#SBATCH --error=./logs_g_elbo/output_%a.err
#SBATCH --time=01:30:00

source ~/.bashrc
conda activate svgp

python main.py \
    -i /work/zdc6/weather_underground/durham/combined/ \
    -o /work/zdc6/temp_g_elbo/ \
    --num_epochs 10 \
    --variable tempAvg \
    --likelihood Gaussian \
    --loss ELBO \
    --num_inducing_points 2000
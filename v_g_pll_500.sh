#!/bin/bash

#SBATCH --job-name=g_pll_500
#SBATCH --array=0-71
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs_g_pll_500/output_%a.txt
#SBATCH --error=./logs_g_pll_500/output_%a.err
#SBATCH --time=00:20:00

source gp/bin/activate

python main.py \
    -i /work/zdc6/weather_underground/durham/combined/ \
    -o /work/zdc6/temp_g_pll_500/ \
    --num_epochs 10 \
    --variable tempAvg \
    --likelihood Gaussian \
    --loss PLL \
    --num_inducing_points 500

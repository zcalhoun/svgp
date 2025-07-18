#!/bin/bash

#SBATCH --job-name=g_pll
#SBATCH --array=0-71
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs_g_pll/output_%a.txt
#SBATCH --error=./logs_g_pll/output_%a.err
#SBATCH --time=00:10:00

source gp/bin/activate

python main.py \
    -i /work-old/zdc6/weather_underground/durham/combined/ \
    -o /work-old/zdc6/temp_gauss_pll_no_weight/ \
    --num_epochs 10 \
    --variable tempAvg \
    --likelihood Gaussian \
    --loss PLL
